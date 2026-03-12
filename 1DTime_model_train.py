# %%
import os
import numpy as np
import matplotlib.pyplot as plt


from autocvd import autocvd
autocvd(num_gpus=1)

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

import bayesflow as bf
import keras
from utils import train_test_split_IC_and_times

MODE = 'evaluate'  # 'train' or 'evaluate'
SEED = 0
N_EPOCHS = 1000
BATCH_SIZE = 60_000
CHECKPOINT_DIR = './checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DATA_DIR = "/export/scratch/lbranca/Amanda_emulator/parsed_rotevol/StellarEv_emulator/preprocessing_output"
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

workflow = bf.BasicWorkflow(
        adapter=adapter,
        inference_network=bf.networks.FlowMatching(),
        standardize=['inference_variables', 'inference_conditions'],
        checkpoint_filepath=CHECKPOINT_DIR,
        checkpoint_name=f"model_noOT_{N_EPOCHS}_{BATCH_SIZE}.keras",
    )



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
    workflow.approximator.save(os.path.join(CHECKPOINT_DIR, f'model_noOT_{N_EPOCHS}_{BATCH_SIZE}_final.keras'))

if MODE == 'evaluate':
    workflow.approximator = keras.models.load_model(os.path.join(CHECKPOINT_DIR, f'model_noOT_{N_EPOCHS}_{BATCH_SIZE}_final.keras'))

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


# %%
# loss_plot = bf.diagnostics.plots.loss(history,)

# %%
# df_val = pd.DataFrame(val_data)
# conditions = ['Mstar', 'FeH', 'PMMA', 'PMMB', 'PMMM'] #, 'Xcen', 'logTeff','Patm', 'tau_cz', 'luminosity', 'Prot_mid', 'Bcoronal_mid', 'dMdt_mid']
# grouped = df_val.groupby(conditions)['Age'].apply(np.array).reset_index()
# val_data_grouped = {
#     'Mstar': grouped['Mstar'].values,
#     'FeH': grouped['FeH'].values,
#     'PMMA': grouped['PMMA'].values,
#     'PMMB': grouped['PMMB'].values,
#     'PMMM': grouped['PMMM'].values,
#     'Age': grouped['Age'].values,  # array of arrays
# }


# %%
import keras
workflow.approximator = keras.models.load_model('./models/model_500_2028.keras')
val_index = 3886
# val_index = 0
conditions_i = {k: df_val[k][val_index].reshape(1, 1) for k in conditions}
sample = workflow.sample(
                        num_samples = len(df_val['Age'].iloc[val_index]),
                        conditions  = conditions_i)
print('N samples:', len(df_val['Age'].iloc[val_index]))
print('Conditions used:', conditions_i)


# %%
if val_index == 0:
    color = 'blue'
else:
    color = 'red'
    val_index_label = 1
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1)
ax.hist(10**sample['Age'].flatten(), histtype='step',  label=f'Samples', color='Green')
ax.hist(df_val['Age'][val_index].flatten(), histtype='step', label=f'Sim {val_index_label}', color=color)
ax.set_xlabel('Age (Gyr)', fontsize=20)
ax.set_yscale('log')
ax.legend(fontsize=20)

# %%
# ...existing code...

# We have to do the inverse
n_samples = len(df_val['Age'][val_index])
x = np.log10(df_val['Age'][val_index]).reshape(-1, 1)
# Get conditions for the first validation sample and concatenate
conditions_concat = np.array([[
    df_val['Mstar'][val_index],
    df_val['FeH'][val_index],
    df_val['PMMA'][val_index],
    df_val['PMMB'][val_index],
    df_val['PMMM'][val_index]
]], dtype=np.float32)

# Broadcast to (n_samples, 5)
conditions_broadcast = np.tile(conditions_concat, (n_samples, 1))

print('x shape:', x.shape)
print('Conditions shape:', conditions_broadcast.shape)

z = workflow.inference_network._forward(
    x, 
    conditions=conditions_broadcast,
)

# %%
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 2, 1)
ax.hist(z.flatten(), label='z from x')
ax.set_xlabel('z')

from scipy.stats import norm

u = norm.cdf(z)
ax = fig.add_subplot(1, 2, 2)
ax.hist(u.flatten(), label='u from z')
ax.set_xlabel('u')

# %%
from scipy.stats import norm

# Sort x and get the sorting indices
sort_idx = np.argsort(x.flatten())
x_sorted = x[sort_idx]
u_sorted = norm.cdf(np.array(z))[sort_idx]

# Check for flips: where u decreases while x increases
u_diff = np.diff(u_sorted.flatten())
flip_mask = u_diff < 0  # True where u decreases (flip)

n_flips = np.sum(flip_mask)
print(f"Total consecutive pairs: {len(u_diff)}")
print(f"Number of flips (x_i < x_j but u_i > u_j): {n_flips}")
print(f"Flip percentage: {100 * n_flips / len(u_diff):.2f}%")

# Show the flipped pairs
if n_flips > 0:
    flip_indices = np.where(flip_mask)[0]
    print("\nExample flips (sorted by x):")
    for idx in flip_indices[:10]:  # show first 10
        print(f"  x[{idx}]={x_sorted[idx, 0]:.4f} -> u={u_sorted[idx, 0]:.4f}  |  "
              f"x[{idx+1}]={x_sorted[idx+1, 0]:.4f} -> u={u_sorted[idx+1, 0]:.4f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(x_sorted.flatten(), u_sorted.flatten(), s=5, alpha=0.5)
axes[0].set_xlabel('x (log10 Age, sorted)')
axes[0].set_ylabel('u')
axes[0].set_title('x vs u (should be monotonically increasing)')

axes[1].plot(u_diff, '.', markersize=3)
axes[1].axhline(0, color='r', linestyle='--')
axes[1].set_xlabel('Index (sorted by x)')
axes[1].set_ylabel('Δu')
axes[1].set_title(f'Consecutive u differences ({n_flips} flips)')
plt.tight_layout()

# %%
from scipy.stats import norm
from collections import defaultdict
import tqdm as tqdm

# Group validation simulations by number of Age timesteps
length_groups = defaultdict(list)
for val_index in range(len(df_val)):
    n = len(df_val['Age'][val_index])
    if n < 2:
        continue
    length_groups[n].append(val_index)

print(f"Unique lengths: {len(length_groups)}")
print(f"Length distribution: {sorted((k, len(v)) for k, v in length_groups.items())}")

def single_forward(x, conditions):
    return workflow.inference_network._forward(x, conditions=conditions)

# Max number of simulations per sub-batch (tune based on GPU memory)
MAX_BATCH_SIZE = 54

total_pairs = 0
total_flips = 0
flip_percentages = []

for length, indices in tqdm.tqdm(length_groups.items(), desc="Processing groups"):
    # Split indices into sub-batches for large groups
    for batch_start in range(0, len(indices), MAX_BATCH_SIZE):
        batch_indices = indices[batch_start:batch_start + MAX_BATCH_SIZE]

        x_list = []
        cond_list = []
        for i in batch_indices:
            x_i = np.log10(df_val['Age'][i]).reshape(-1, 1).astype(np.float32)
            cond_i = np.tile(np.array([[
                df_val['Mstar'][i],
                df_val['FeH'][i],
                df_val['PMMA'][i],
                df_val['PMMB'][i],
                df_val['PMMM'][i]
            ]], dtype=np.float32), (length, 1))
            x_list.append(x_i)
            cond_list.append(cond_i)

        # Single forward pass for the sub-batch
        x_all = np.concatenate(x_list, axis=0)       # (sub_batch * length, 1)
        cond_all = np.concatenate(cond_list, axis=0)  # (sub_batch * length, 5)

        z_all = single_forward(x_all, cond_all)
        z_all = np.array(z_all)

        # Split back and compute flips per simulation
        for j in range(len(batch_indices)):
            start = j * length
            end = (j + 1) * length
            x_j = x_list[j]
            z_j = z_all[start:end]

            sort_idx = np.argsort(x_j.flatten())
            u_sorted = norm.cdf(z_j)[sort_idx]
            u_diff = np.diff(u_sorted.flatten())
            n_flips = np.sum(u_diff < 0)
            n_pairs = len(u_diff)

            total_pairs += n_pairs
            total_flips += n_flips
            flip_percentages.append(100 * n_flips / n_pairs if n_pairs > 0 else 0)

flip_percentages = np.array(flip_percentages)

print(f"\n=== Flip Summary over {len(flip_percentages)} validation simulations ===")
print(f"Total consecutive pairs: {total_pairs}")
print(f"Total flips: {total_flips}")
print(f"Overall flip percentage: {100 * total_flips / total_pairs:.2f}%")
print(f"Per-simulation flip %: mean={flip_percentages.mean():.2f}%, "
      f"median={np.median(flip_percentages):.2f}%, "
      f"max={flip_percentages.max():.2f}%")
print(f"Simulations with zero flips: {np.sum(flip_percentages == 0)} / {len(flip_percentages)}")

# Histogram of per-simulation flip percentages
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(flip_percentages, bins=30, edgecolor='k')
axes[0].set_xlabel('Flip percentage per simulation')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of flip % across validation set')

axes[1].hist(flip_percentages[flip_percentages > 0], bins=30, edgecolor='k')
axes[1].set_xlabel('Flip percentage per simulation')
axes[1].set_ylabel('Count')
axes[1].set_title('Distribution of flip % (excluding zero-flip sims)')
plt.tight_layout()

# %%
np.savez(f"./flip_percentages_{n_epochs}_{batch_size}.npz", flip_percentages=flip_percentages, )

# %%



