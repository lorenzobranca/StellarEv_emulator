# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid


from autocvd import autocvd
autocvd(num_gpus=1)

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

import bayesflow as bf
import keras
from utils import train_test_split_IC_and_times

MODE = 'train'  # 'train' or 'evaluate', 'inverse_CDF'
TYPE_GENERATIVE_MODEL = 'flow_matching'  # 'flow_matching' or 'normalizing_flow'
SEED = 0
N_EPOCHS = 100
BATCH_SIZE = 100_000
CHECKPOINT_DIR = './checkpoints_202/'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DATA_DIR = "./preprocessing_new/"
IC = np.load(os.path.join(DATA_DIR, "initial_conditions.npy"))
time = np.load(os.path.join(DATA_DIR, "time.npy"))
inference_conditions = ['Mstar', 'FeH', 'PMMA', 'PMMB', 'PMMM']

#Bayesflow workflow setup
adapter = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .concatenate(inference_conditions, into="inference_conditions")
        .rename('Age', "inference_variables")
    )

#from hyperaparm 202
inference_mlp_depth= 2
inference_mlp_width= 409
inference_time_embedding_dim= 16

if TYPE_GENERATIVE_MODEL == 'flow_matching':
    workflow = bf.BasicWorkflow(
            adapter=adapter,
            inference_network=bf.networks.FlowMatching(subnet_kwargs={
                                                        "widths": [inference_mlp_width] * inference_mlp_depth,
                                                        "time_embedding_dim": inference_time_embedding_dim,
                                                        }),
            standardize=['inference_variables', 'inference_conditions'],
            checkpoint_filepath=CHECKPOINT_DIR,
            checkpoint_name=f"{TYPE_GENERATIVE_MODEL}_model_noOT_{N_EPOCHS}_{BATCH_SIZE}",
        )
else:
    pass

if MODE == 'train':
    #Import and preprocess data for Bayesflow
    train_IC, val_IC, train_time, val_time = train_test_split_IC_and_times(IC, time, test_ratio=0.1, seed=SEED)
    print('Expand dimentions for broadcasting...')
    train_IC = train_IC[:, None, :]
    val_IC = val_IC[:, None, :]
    print(f"Train IC shape: {train_IC.shape}, Train time shape: {train_time.shape}")
    print(f"Val IC shape: {val_IC.shape}, Val time shape: {val_time.shape}")
    print('Repeat IC for each time step because each time step is the random variable')
    train_IC = np.repeat(train_IC, train_time.shape[1], axis=1)
    val_IC = np.repeat(val_IC, val_time.shape[1], axis=1)
    print(f"Train IC shape: {train_IC.shape}, Train time shape: {train_time.shape}")
    print(f"Val IC shape: {val_IC.shape}, Val time shape: {val_time.shape}")
    print('Reshape to 2D for training...')
    train_IC = train_IC.reshape(-1, train_IC.shape[-1])
    val_IC = val_IC.reshape(-1, val_IC.shape[-1])
    train_time = train_time.reshape(-1, 1)
    val_time = val_time.reshape(-1, 1)
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
    print('Train and val data Bayesflow dictionaries:')
    for k in train_data:
        print(f"{k}: train shape {train_data[k].shape}, val shape {val_data[k].shape}")
        print(f"{k}: min train {train_data[k].min()}, max train {train_data[k].max()}")

    history = workflow.fit_offline(
            train_data,
            epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2,
        )
    workflow.approximator.save(os.path.join(CHECKPOINT_DIR, f'{TYPE_GENERATIVE_MODEL}_{N_EPOCHS}_{BATCH_SIZE}_final.keras'))

if MODE == 'evaluate':
    workflow.approximator = keras.models.load_model(os.path.join(CHECKPOINT_DIR, f'model_noOT_{N_EPOCHS}_{BATCH_SIZE}.keras'))

    #we need to reload the data because the shapes have changed a lot in training
    train_IC, val_IC, train_time, val_time = train_test_split_IC_and_times(IC, time, test_ratio=0.1, seed=SEED)
    validation_set = {}
    for i, k in enumerate(inference_conditions):
        validation_set[k] = val_IC[:, i].reshape(-1, 1)

    # val_index = 435
    print('Sampling true_vs_predicted')
    val_index = 0
    conditions_index  = {k: validation_set[k][:, 0][val_index].reshape(1, 1) for k in inference_conditions}
    observable_index = {}
    observable_index['Age'] = val_time[val_index].reshape(1, -1)
    sample = workflow.sample(
                            num_samples = len(observable_index['Age'].flatten()),
                            conditions  = conditions_index)
    print('samples shape:',sample['Age'].shape)
    print('Conditions used:', conditions_index)
    print('Max age of test set index:', 10**observable_index['Age'].max())
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(10**sample['Age'].flatten(), histtype='step', bins=10, label=f'Samples', color='Green')
    ax.hist(10**observable_index['Age'].flatten(), histtype='step', bins=10, label=f'Sim {val_index}', color='Red')
    ax.set_xlabel('Age (Gyr)', fontsize=20)
    # ax.set_yscale('log')
    ax.legend(fontsize=20)
    fig.savefig(f'./plots_flowmatching/true_vs_predidect_index_{val_index}.pdf')
    plt.show()

    print("Sampling for validation set")
    SAMPLE_BATCH_SIZE = 100  # number of conditions per batch/figure
    N_TOTAL_VAL = 400        # total number of validation samples to process

    for batch_idx in range(N_TOTAL_VAL // SAMPLE_BATCH_SIZE):
        start = batch_idx * SAMPLE_BATCH_SIZE
        end = start + SAMPLE_BATCH_SIZE
        val_index = np.arange(start, end)
        print(f"Batch {batch_idx}: val indices {start}-{end-1}")

        conditions_index = {k: validation_set[k][:, 0][val_index].reshape(-1, 1) for k in inference_conditions}
        observable_index = {}
        observable_index['Age'] = val_time[val_index].reshape(len(val_index), -1)
        n_times_per_sample = observable_index['Age'].shape[1]
        sample = workflow.sample(
            num_samples=n_times_per_sample,
            conditions=conditions_index,
        )
        print(f'  samples shape: {sample["Age"].shape}')

        nrows = int(np.ceil(np.sqrt(SAMPLE_BATCH_SIZE)))
        ncols = int(np.ceil(SAMPLE_BATCH_SIZE / nrows))
        fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
        for j, idx in enumerate(val_index):
            ax = fig.add_subplot(nrows, ncols, j + 1)
            ax.hist(10**sample['Age'][j].flatten(), histtype='step', bins=10, label='Samples', color='Green')
            ax.hist(10**observable_index['Age'][j].flatten(), histtype='step', bins=10, label=f'Sim {idx}', color='Red')
            ax.set_xlabel('Age (Gyr)', fontsize=8)
            ax.set_yscale('log')
            ax.set_title(f'Val {idx}', fontsize=6)
            ax.legend()
        fig.tight_layout()
        fig.savefig(f'./plots_flowmatching/true_vs_predicted_val_{start}_{end}.pdf')
        plt.close(fig)
        print(f"  Saved figure for validation indices {start}-{end-1}")
    exit()


if MODE == "inverse_CDF":    
    workflow.approximator = keras.models.load_model(
        os.path.join(CHECKPOINT_DIR, f'model_noOT_{N_EPOCHS}_{BATCH_SIZE}_final.keras')
    )

    # Get validation data
    _, val_IC, _, val_time = train_test_split_IC_and_times(IC, time, test_ratio=0.1, seed=SEED)
    validation_set = {}
    for i, k in enumerate(inference_conditions):
        validation_set[k] = val_IC[:, i].reshape(-1, 1)

    # Pick a validation sample
    val_index = 1
    conditions_index  = {k: validation_set[k][:, 0][val_index].reshape(1, 1) for k in inference_conditions}
    observable_index = {}
    observable_index['Age'] = val_time[val_index].reshape(1, -1)
    sampled_ages = workflow.sample(
                            num_samples = len(observable_index['Age'].flatten()),
                            conditions  = conditions_index)

    # sampled_ages['Age'] has shape (1, num_samples, 1) — extract and flatten to (num_samples, 1)
    sampled_ages_array = sampled_ages['Age'].reshape(-1, 1)
    n_samples = len(sampled_ages_array)

    # Replicate conditions for each sampled age
    log_prob_data = {
        k: np.tile(conditions_index[k], (n_samples, 1))  # (n_samples, 1)
        for k in inference_conditions
    }
    log_prob_data['Age'] = sampled_ages_array  # (n_samples, 1)

    # Evaluate log_prob — this goes through the adapter automatically
    log_probs = workflow.log_prob(log_prob_data)
    print(f"log_probs shape: {log_probs.shape}")
    print(f"log_probs min: {log_probs.min()}, max: {log_probs.max()}")

    # Convert to PDF values
    pdf_values = np.exp(log_probs.flatten())

    # Sort by age for CDF construction
    # sort_idx = np.argsort(sampled_ages_array.flatten())
    # sorted_ages = sampled_ages_array.flatten()[sort_idx]
    # sorted_pdf = pdf_values[sort_idx]

    # # Build CDF from sorted samples via cumulative trapezoid
    # from scipy.integrate import cumulative_trapezoid
    # from scipy.interpolate import interp1d

    # cdf_values = np.zeros_like(sorted_pdf)
    # cdf_values[1:] = cumulative_trapezoid(sorted_pdf, sorted_ages)
    # cdf_values /= cdf_values[-1]  # normalize to [0, 1]
    # print('cdf_values:', cdf_values)

    # # Build CDF interpolation: x -> u
    # cdf_fn = interp1d(sorted_ages, cdf_values, kind='cubic', bounds_error=False,
    #                   fill_value=(0.0, 1.0))

    # # Build inverse CDF (quantile function): u -> x
    # unique_mask = np.diff(cdf_values, prepend=-1) > 0
    # quantile_fn = interp1d(cdf_values[unique_mask], sorted_ages[unique_mask],
    #                        kind='cubic', bounds_error=False,
    #                        fill_value=(sorted_ages[0], sorted_ages[-1]))
    sort_idx = np.argsort(sampled_ages_array.flatten())
    sorted_ages = sampled_ages_array.flatten()[sort_idx]
    sorted_pdf = pdf_values[sort_idx]

    # Remove duplicate ages by averaging their PDF values
    unique_ages, inverse_idx = np.unique(sorted_ages, return_inverse=True)
    unique_pdf = np.zeros_like(unique_ages)
    np.add.at(unique_pdf, inverse_idx, sorted_pdf)
    counts = np.bincount(inverse_idx).astype(float)
    unique_pdf /= counts  # average PDF at duplicate points

    print(f"Unique ages: {len(unique_ages)} out of {len(sorted_ages)} samples")

    # Build CDF from unique sorted samples via cumulative trapezoid
    cdf_values = np.zeros_like(unique_pdf)
    cdf_values[1:] = cumulative_trapezoid(unique_pdf, unique_ages)
    cdf_values /= cdf_values[-1]  # normalize to [0, 1]

    # Choose interpolation kind based on number of unique points
    interp_kind = 'cubic' if len(unique_ages) >= 4 else 'linear'
    # interp_kind = 'linear'

    # Build CDF interpolation: x -> u
    cdf_fn = interp1d(unique_ages, cdf_values, kind=interp_kind, bounds_error=False,
                      fill_value=(0.0, 1.0))

    # Build inverse CDF (quantile function): u -> x
    # Ensure strict monotonicity in CDF for inversion
    unique_mask = np.diff(cdf_values, prepend=-1) > 0
    if np.sum(unique_mask) < 4:
        interp_kind_inv = 'linear'
    else:
        interp_kind_inv = 'cubic'

    quantile_fn = interp1d(cdf_values[unique_mask], unique_ages[unique_mask],
                           kind=interp_kind_inv, bounds_error=False,
                           fill_value=(unique_ages[0], unique_ages[-1]))

    # Draw new samples via inverse CDF sampling
    rng = np.random.default_rng(42)
    u = np.linspace(0, 1, 3999)  # uniform samples in [0, 1]
    inverse_cdf_samples = quantile_fn(u)

    # Determine grid bounds from the data (with some padding)
    true_ages = val_time[val_index]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.hist(10**inverse_cdf_samples, histtype='step', bins=20, label='Inverse CDF Samples', color='Blue')
    ax.hist(10**true_ages.flatten(), histtype='step', bins=20, label=f'Sim {val_index}', color='Red')
    ax.set_xlabel('Age (Gyr)', fontsize=20)
    ax.set_yscale('log')
    ax.legend(fontsize=20)

    ax = fig.add_subplot(1, 2, 2)
    sort_index_inverse_cdf = np.argsort(inverse_cdf_samples)
    sorted_inverse_cdf = inverse_cdf_samples[sort_index_inverse_cdf]
    sorted_true_ages = true_ages.flatten()[np.argsort(true_ages.flatten())]
    # ax.plot(10**sorted_true_ages, 10**cdf_fn(sorted_true_ages), label='True CDF', color='Red')
    # ax.plot(10**sorted_inverse_cdf, 10**cdf_fn(sorted_inverse_cdf), label='Inverse CDF Samples', color='Blue')
    ax.scatter(10**sorted_true_ages, 10**sorted_inverse_cdf, color='Red', s=10)
    ax.plot([10**sorted_true_ages.min(), 10**sorted_true_ages.max()], [10**sorted_true_ages.min(), 10**sorted_true_ages.max()], 'k--', lw=2)
    ax.set_xlabel('True', fontsize=20)
    ax.set_ylabel('Predicted', fontsize=20)
    ax.legend(fontsize=20)

    fig.savefig(f'./plots_flowmatching/inverse_cdf_samples_index_{val_index}.pdf')
    plt.show()