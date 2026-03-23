# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
import scipy


from autocvd import autocvd
autocvd(num_gpus=1)

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

import bayesflow as bf
import keras
from utils import train_test_split_IC_and_times

TYPE_GENERATIVE_MODEL = 'flow_matching' 
SEED = 0
N_EPOCHS = 100
BATCH_SIZE = 100_000
CHECKPOINT_DIR = './checkpoints_new/'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DATA_DIR = "./preprocessing_output/"
IC = np.load(os.path.join(DATA_DIR, "initial_conditions.npy"))
time = np.load(os.path.join(DATA_DIR, "time.npy"))
inference_conditions = ['Mstar', 'FeH', 'PMMA', 'PMMB', 'PMMM']


#Import and preprocess data for Bayesflow
train_IC, val_IC, train_time, val_time = train_test_split_IC_and_times(IC, time, test_ratio=0.1, seed=SEED)
print('Train_time shape:', train_time.shape)
print('Val_time shape:', val_time.shape)
print('Expand dimentions for broadcasting...')
train_IC = train_IC[:, None, :]
# val_IC = val_IC[:, None, :]
print(f"Train IC shape: {train_IC.shape}, Train time shape: {train_time.shape}")
print(f"Val IC shape: {val_IC.shape}, Val time shape: {val_time.shape}")
print('Repeat IC for each time step because each time step is the random variable')
train_IC = np.repeat(train_IC, train_time.shape[1], axis=1)
# val_IC = np.repeat(val_IC, val_time.shape[1], axis=1)
print(f"Train IC shape: {train_IC.shape}, Train time shape: {train_time.shape}")
print(f"Val IC shape: {val_IC.shape}, Val time shape: {val_time.shape}")
print('Reshape to 2D for training...')
train_IC = train_IC.reshape(-1, train_IC.shape[-1])
# val_IC = val_IC.reshape(-1, val_IC.shape[-1])
train_time = train_time.reshape(-1, 1)
# val_time = val_time.reshape(-1, 1)
print(f"Train IC shape: {train_IC.shape}, Train time shape: {train_time.shape}")
print(f"Val IC shape: {val_IC.shape}, Val time shape: {val_time.shape}")

#Create training and val data ditcionaries for Bayesflow
train_data = {
    'Mstar': train_IC[:, 0][:, None].reshape((-1, 1)),
    'FeH': train_IC[:, 1][:, None].reshape((-1, 1)),
    'PMMA': train_IC[:, 2][:, None].reshape((-1, 1)),
    'PMMB': train_IC[:, 3][:, None].reshape((-1, 1)),
    'PMMM': train_IC[:, 4][:, None].reshape((-1, 1)),
    'Age': train_time,
}
val_data = {
    'Mstar': val_IC[:, 0][:, None].reshape((-1, 1)),
    'FeH': val_IC[:, 1][:, None].reshape((-1, 1)),
    'PMMA': val_IC[:, 2][:, None].reshape((-1, 1)),
    'PMMB': val_IC[:, 3][:, None].reshape((-1, 1)),
    'PMMM': val_IC[:, 4][:, None].reshape((-1, 1)),
    'Age': val_time,
}

def objective(trial):
    inference_mlp_depth = trial.suggest_int("inference_mlp_depth", 2, 8)
    inference_mlp_width = trial.suggest_int("inference_mlp_width", 32, 512)
    time_embedding_dim = trial.suggest_int("inference_time_embedding_dim", 16, 64, step=2)    

    #Bayesflow workflow setup
    adapter = (
            bf.adapters.Adapter()
            .to_array()
            .convert_dtype("float64", "float32")
            .concatenate(inference_conditions, into="inference_conditions")
            .rename('Age', "inference_variables")
        )

    workflow = bf.BasicWorkflow(
            adapter=adapter,
            inference_network=bf.networks.FlowMatching(subnet_kwargs={
                                                        "widths": [inference_mlp_width] * inference_mlp_depth,
                                                        "time_embedding_dim": time_embedding_dim,
                                                        }),
            standardize=['inference_variables', 'inference_conditions'],
            # checkpoint_filepath=CHECKPOINT_DIR,
            # checkpoint_name=f"{TYPE_GENERATIVE_MODEL}_model_noOT_{N_EPOCHS}_{BATCH_SIZE}",
        )
    
    history = workflow.fit_offline(
            train_data,
            epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2,
        )

    val_samples = workflow.sample(
        num_samples = 3999,
        conditions = val_data,
        batch_size = 50,
    )
    ks_stat = []
    for i in range(val_samples['Age'].shape[0]):
        true_points = val_time[i]
        sampled_points = val_samples['Age'][i]

        ks_stat.append(scipy.stats.ks_2samp(true_points.flatten(), sampled_points.flatten()).statistic)
    
    return np.mean(ks_stat)
    
if __name__ == "__main__":
    import optuna
    from optuna.storages import JournalStorage, JournalFileStorage

    study_name = 'study_flowmatching'  # Unique identifier of the study.
    storage_name = JournalStorage(JournalFileStorage("./optuna_flowmatching.log"))
    study = optuna.create_study(direction="minimize",study_name=study_name, storage=storage_name,  load_if_exists=True)
    study.optimize(objective, n_trials=180)

    print("Best trial:")
    print("  Value (mean ks test):", study.best_value)
    print("  Params:", study.best_params)