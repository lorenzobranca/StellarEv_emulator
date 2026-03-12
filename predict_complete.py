import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from scipy.stats import gaussian_kde

from autocvd import autocvd
# autocvd(num_gpus=0)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

import bayesflow as bf
import keras
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

from utils import train_test_split_IC_and_times, train_test_split_unaligned
from arch_grid_split_don import GridSplitBranchDeepONet
from train_grid_split_don import train


DATA_DIR = "/export/scratch/lbranca/Amanda_emulator/parsed_rotevol/StellarEv_emulator/preprocessing_output"
CKPT_DIR = os.path.abspath("checkpoints_grid_split_don/deeponet_params")
PLOTS_DIR = "plots_complete"

SEED = 0

#{'learning_rate': 0.0011755636775666748, 'latent_dim': 516, 'num_layers': 5, 'activation': 'gelu', 'use_curve_bias': True}

DEFAULT_MODEL_CFG = dict(
    latent_dim=516,
    num_layers=5,
    activation_name="gelu",
    use_curve_bias=True,
)

DEFAULT_TRAIN_CFG = dict(
    lr=0.0011755636775666748,
    num_epochs=500,
    batch_size=256,
    l2_reg=0.0,
)


# =========================
# Simple scalers (fit on train, apply to test)
# =========================
def _safe_std(x, eps=1e-8):
    return jnp.std(x) + eps

def fit_standard_scaler(x):
    m = jnp.mean(x)
    s = _safe_std(x)
    return m, s

def apply_standard_scaler(x, m, s):
    return (x - m) / s

def fit_log10_scaler(x, eps=1e-30):
    lx = jnp.log10(jnp.clip(x, a_min=eps))
    m, s = fit_standard_scaler(lx)
    return m, s, eps

def apply_log10_scaler(x, m, s, eps=1e-30):
    lx = jnp.log10(jnp.clip(x, a_min=eps))
    return (lx - m) / s

def scale_outputs_train_test(y_train, y_test, apply_log = False):
    y_train = jnp.array(y_train, dtype=jnp.float32)
    y_test  = jnp.array(y_test, dtype=jnp.float32)

    # Column 0: linear
    m0, s0 = fit_standard_scaler(y_train[:, :, 0])
    y_train = y_train.at[:, :, 0].set(apply_standard_scaler(y_train[:, :, 0], m0, s0))
    y_test  = y_test.at[:, :, 0].set(apply_standard_scaler(y_test[:, :, 0],  m0, s0))

    # Columns 1..end: log10
    for i in range(1, y_train.shape[-1]):
        if apply_log:
            mi, si, eps = fit_log10_scaler(y_train[:, :, i])
            y_train = y_train.at[:, :, i].set(apply_log10_scaler(y_train[:, :, i], mi, si, eps))
            y_test  = y_test.at[:, :, i].set(apply_log10_scaler(y_test[:, :, i],  mi, si, eps))
        else:
            mi, si  = fit_standard_scaler(y_train[:,:,i])
            y_train = y_train.at[:,:,i].set(apply_standard_scaler(y_train[:,:,i], mi, si))
            y_test  = y_test.at[:,:,i].set(apply_standard_scaler(y_test[:,:,i], mi, si))

    return y_train, y_test


# =========================
# Load + split + scale data
# =========================
IC = np.load(os.path.join(DATA_DIR, "initial_conditions.npy"))
output = np.load(os.path.join(DATA_DIR, "output.npy"))

IC = jnp.array(IC, dtype=jnp.float32)
output = jnp.array(output, dtype=jnp.float32)

# ---- Fake time grid: only for plotting / keeping the split logic unchanged ----
N_eval = int(output.shape[1])
t_grid_1d = jnp.linspace(0.0, 1.0, N_eval, dtype=jnp.float32)               # (N_eval,)
time_fake = jnp.broadcast_to(t_grid_1d[None, :], (IC.shape[0], N_eval))     # (B, N_eval)

# We still use the same splitter so train/test stay aligned
IC_train, IC_test, output_train, output_test, _, _ = train_test_split_unaligned(
    IC, output, time_fake, test_ratio=0.1, seed=SEED
)

IC_train = jnp.array(IC_train, dtype=jnp.float32)
IC_test  = jnp.array(IC_test, dtype=jnp.float32)

output_train, output_test = scale_outputs_train_test(output_train, output_test)

OUTPUT_DIM = int(output_train.shape[-1])

#Import Times 
time = np.load(os.path.join(DATA_DIR, "time.npy"))
inference_conditions = ['Mstar', 'FeH', 'PMMA', 'PMMB', 'PMMM']
train_IC, val_IC, train_time, val_time = train_test_split_IC_and_times(IC, time, test_ratio=0.1, seed=SEED)

# =======================
# Bayesflow model loading
# =======================
inference_conditions = ['Mstar', 'FeH', 'PMMA', 'PMMB', 'PMMM']
#Bayesflow workflow setup
N_EPOCHS = 1000
BATCH_SIZE = 60_000
CHECKPOINT_DIR = './checkpoints'
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

workflow.approximator = keras.models.load_model(
    os.path.join(CHECKPOINT_DIR, f'model_noOT_{N_EPOCHS}_{BATCH_SIZE}_final.keras'))

# =========================
# Model builder
# =========================
def get_activation(name: str):
    activation_map = {
        "tanh": jax.nn.tanh,
        "relu": jax.nn.relu,
        "gelu": jax.nn.gelu,
        "silu": jax.nn.silu,
    }
    return activation_map[name]

def build_model(latent_dim: int, num_layers: int, activation_name: str, use_curve_bias: bool):
    activation_fn = get_activation(activation_name)

    branch_layers = tuple([latent_dim for _ in range(num_layers)])

    model = GridSplitBranchDeepONet(
        branch_layers=branch_layers,
        latent_dim=int(latent_dim),
        output_dim=int(OUTPUT_DIM),
        n_eval=int(N_eval),
        activation=activation_fn,
        use_curve_bias=use_curve_bias,
    )
    return model

model = build_model(
        latent_dim=DEFAULT_MODEL_CFG["latent_dim"],
        num_layers=DEFAULT_MODEL_CFG["num_layers"],
        activation_name=DEFAULT_MODEL_CFG["activation_name"],
        use_curve_bias=DEFAULT_MODEL_CFG["use_curve_bias"],
    )

# Dummy state with the right structure
dummy_state = train(
    model,
    IC_train,
    output_train,
    ic_test=IC_test,
    y_test=output_test,
    num_epochs=0,
    batch_size=DEFAULT_TRAIN_CFG["batch_size"],
    lr=DEFAULT_TRAIN_CFG["lr"],
    l2_reg=DEFAULT_TRAIN_CFG["l2_reg"],
    seed=SEED,
)

restored_state = checkpoints.restore_checkpoint(
    ckpt_dir=CKPT_DIR,
    target=dummy_state,
)

y_pred = restored_state.apply_fn(restored_state.params, IC_test)
test_mse = np.mean((np.array(y_pred) - np.array(output_test)) ** 2)
print("Test MSE:", float(test_mse))

os.makedirs(PLOTS_DIR, exist_ok=True)

titles = [
    r"log($T_{\mathrm{eff}}$) - Effective Surface Temperature",
    r"$P_{\mathrm{rot}}$ (days) - Rotation Period",
    r"$B_{\mathrm{coronal}}/B_\odot$ - Coronal Magnetic Field Strength",
    r"$P_{\mathrm{atm}}$ - Photospheric Pressure (cgs)",
    r"$\tau_{\mathrm{cz}}$ - Convective Turnover Time (s)",
    r"$\dot{M}$ - Mass Loss Rate",
    r"Luminosity - Predicted Stellar Luminosity",
]

num_examples = 5
rng = np.random.default_rng(12)
idxs = rng.integers(low=0, high=output_test.shape[0], size=num_examples)

#======================
# Bayesflow inverse CDF
#======================
# def kde_func(x, max_length_new_time):
#     """
#     This function is used to create the times using inverse CDF sampling
#     """
#     kde = gaussian_kde(x)
#     cdf_0 = np.array([kde.integrate_box_1d(-np.inf, xi) for xi in x])
#     u = np.linspace(0, 1, max_length_new_time)
#     new_t = np.interp(u, cdf_0, x)
#     return new_t
# ...existing code...

def kde_func(x, max_length_new_time, n_grid=2000):
    """
    Inverse CDF sampling using a KDE fitted to the bayesflow samples.
    
    1. Fit a Gaussian KDE to the raw samples `x`.
    2. Evaluate the KDE on a fine uniform grid spanning the sample range.
    3. Build the CDF via cumulative trapezoid integration.
    4. Invert the CDF with interpolation (quantile function).
    5. Evaluate the quantile function on a uniform [0,1] grid.
    """
    kde = gaussian_kde(x)

    # Fine uniform grid over the sample range (with a small margin)
    margin = 0.05 * (x.max() - x.min())
    grid = np.linspace(x.min() - margin, x.max() + margin, n_grid)

    # Evaluate PDF on the grid
    pdf_values = kde.evaluate(grid)

    # Build CDF via cumulative trapezoid
    cdf_values = np.zeros_like(pdf_values)
    cdf_values[1:] = cumulative_trapezoid(pdf_values, grid)
    cdf_values /= cdf_values[-1]  # normalise to [0, 1]

    # Ensure strict monotonicity for inversion
    unique_mask = np.diff(cdf_values, prepend=-1) > 0
    interp_kind = 'cubic' if np.sum(unique_mask) >= 4 else 'linear'

    # Inverse CDF (quantile function):  u -> age
    quantile_fn = interp1d(
        cdf_values[unique_mask], grid[unique_mask],
        kind=interp_kind, bounds_error=False,
        fill_value=(grid[0], grid[-1]),
    )

    # Draw new time points uniformly in [0, 1] and map through quantile fn
    u = np.linspace(0, 1, max_length_new_time)
    new_t = quantile_fn(u)
    return new_t

# ...existing code...


INVERSE_CDF_MODE = 'bayesflow'  # 'bayesflow' or 'KDE'
validation_set = {}
for i, k in enumerate(inference_conditions):
    validation_set[k] = val_IC[:, i].reshape(-1, 1)
time_from_interpolation = np.zeros((len(idxs),3999))
if INVERSE_CDF_MODE == 'bayesflow':
    fig = plt.figure(figsize=(15, 5))
    for j, val_index in enumerate(idxs):
        conditions_index  = {k: validation_set[k][:, 0][val_index].reshape(1, 1) for k in inference_conditions}
        observable_index = {}
        observable_index['Age'] = val_time[val_index].reshape(1, -1)
        sampled_ages = workflow.sample(
                                num_samples = len(observable_index['Age'].flatten()),
                                conditions  = conditions_index)

        # sampled_ages['Age'] has shape (1, num_samples, 1) — extract and flatten to (num_samples, 1)
        sampled_ages_array = sampled_ages['Age'].reshape(-1, 1)
        ax = fig.add_subplot(1, len(idxs), j+1)
        ax.hist(sampled_ages_array.flatten(), bins=50, alpha=0.7,label='sampled array')
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

        #attach
        time_from_interpolation[j, :] = inverse_cdf_samples
        ax.hist(time_from_interpolation[j, :].flatten(), bins=50, alpha=0.7,label='inverse cdf')
        ax.legend()
    fig.savefig(f'{PLOTS_DIR}/{INVERSE_CDF_MODE}_distribution_time.png')


elif INVERSE_CDF_MODE == 'KDE':
    fig = plt.figure(figsize=(15, 5))
    for j, val_index in enumerate(idxs):
        conditions_index  = {k: validation_set[k][:, 0][val_index].reshape(1, 1) for k in inference_conditions}
        observable_index = {}
        observable_index['Age'] = val_time[val_index].reshape(1, -1)
        sampled_ages = workflow.sample(
                                num_samples = len(observable_index['Age'].flatten()),
                                conditions  = conditions_index)

        # sampled_ages['Age'] has shape (1, num_samples, 1) — extract and flatten to (num_samples, 1)
        sampled_ages_array = sampled_ages['Age'].reshape(-1, 1)
        ax = fig.add_subplot(1, len(idxs), j+1)
        ax.hist(sampled_ages_array.flatten(), bins=50, alpha=0.7,label='sampled array')
        n_samples = len(sampled_ages_array)
        time_from_interpolation[j, :] = kde_func(sampled_ages_array.flatten(), max_length_new_time=3999)
        ax.hist(time_from_interpolation[j, :].flatten(), bins=50, alpha=0.7,label='inverse cdf')
        ax.legend()
        
    fig.savefig(f'{PLOTS_DIR}/{INVERSE_CDF_MODE}_distribution_time.png')

        



colors = plt.cm.tab10.colors[:num_examples]
t_plot = np.array(t_grid_1d)

fig, axes = plt.subplots(1, OUTPUT_DIM, figsize=(25, 4), sharex=True)

for i in range(OUTPUT_DIM):
    ax = axes[i] if OUTPUT_DIM > 1 else axes
    for j, idx in enumerate(idxs):
        color = colors[j]
        y_true = np.array(output_test[idx, :, i])
        y_model = np.array(y_pred[idx, :, i])

        # same color for truth and prediction, different linestyle
        ax.plot(val_time[idx, :], y_true, color=color, linestyle="-")
        # ax.plot(val_time[idx, :], y_model, color=color, linestyle="--")
        ax.plot(time_from_interpolation[j, :], y_model, color=color, linestyle="--")
        ax.set_xlabel('Age (Gyr)')

    ax.set_title(titles[i] if i < len(titles) else f"Output {i}", fontsize=10)
    ax.grid(True)
    if i == 0:
        ax.set_ylabel("Scaled value")
    if i == max(0, OUTPUT_DIM // 2):
        ax.set_xlabel("Fixed grid (0..1)")

fig.suptitle("Grid Split-Branch DeepONet vs Reference", fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

outpath = os.path.join(PLOTS_DIR, f"test_predictions_comparison_{INVERSE_CDF_MODE}.png")
plt.savefig(outpath, dpi=300)
plt.close()
print(f"Figure saved: {outpath}")