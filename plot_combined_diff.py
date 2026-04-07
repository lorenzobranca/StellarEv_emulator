import numpy as np
import matplotlib.pyplot as plt
import os
import jax.numpy as jnp 
from utils import train_test_split_unaligned
SEED = 0



def _safe_std(x, eps=1e-8):
    return jnp.std(x) + eps

def fit_standard_scaler(x):
    m = jnp.mean(x)
    s = _safe_std(x)
    return m, s

def apply_standard_scaler(x, m, s):
    return (x - m) / s

def apply_standard_scaler_inverse(x_scaled, m, s):
    return x_scaled * s + m

def fit_log10_scaler(x, eps=1e-30):
    lx = jnp.log10(jnp.clip(x, a_min=eps))
    m, s = fit_standard_scaler(lx)
    return m, s, eps

def apply_log10_scaler(x, m, s, eps=1e-30):
    lx = jnp.log10(jnp.clip(x, a_min=eps))
    return (lx - m) / s

def apply_log10_scaler_inverse(x_scaled, m, s):
    lx = x_scaled * s + m
    return 10 ** lx

def scale_outputs_train_test(y_train, y_test, ytrain_from_disk, apply_log = False):
    y_train = jnp.array(y_train, dtype=jnp.float32)
    y_test  = jnp.array(y_test, dtype=jnp.float32)

    # Column 0: linear
    m0, s0 = fit_standard_scaler(ytrain_from_disk[:, :, 0])
    y_train = y_train.at[:, :, 0].set(apply_standard_scaler_inverse(y_train[:, :, 0], m0, s0))
    y_test  = y_test.at[:, :, 0].set(apply_standard_scaler_inverse(y_test[:, :, 0],  m0, s0))

    # Columns 1..end: log10
    for i in range(1, y_train.shape[-1]):
        if apply_log:
            mi, si, eps = fit_log10_scaler(ytrain_from_disk[:, :, i])
            y_train = y_train.at[:, :, i].set(apply_log10_scaler_inverse(y_train[:, :, i], mi, si, ))
            y_test  = y_test.at[:, :, i].set(apply_log10_scaler_inverse(y_test[:, :, i],  mi, si, ))
        else:
            mi, si  = fit_standard_scaler(ytrain_from_disk[:,:,i])
            y_train = y_train.at[:,:,i].set(apply_standard_scaler_inverse(y_train[:,:,i], mi, si))
            y_test  = y_test.at[:,:,i].set(apply_standard_scaler_inverse(y_test[:,:,i], mi, si))

    return y_train, y_test

DATA_DIR  = "/export/data/vgiusepp/StellarEv_emulator/preprocessing_new_log15/"
CKPT_DIR  = os.path.abspath("checkpoints_new/deeponet_params_new_log15_output/")
PLOTS_DIR = "./plots_new_log15_output/"
IC = np.load(os.path.join(DATA_DIR, "initial_conditions.npy"))
output = np.load(os.path.join(DATA_DIR, "output.npy"))

IC = jnp.array(IC, dtype=jnp.float32)
output = jnp.array(output, dtype=jnp.float32)

# ---- Fake time grid: only for plotting / keeping the split logic unchanged ----
N_eval = int(output.shape[1])
t_grid_1d = jnp.linspace(0.0, 1.0, N_eval, dtype=jnp.float32)               # (N_eval,)
time_fake = jnp.broadcast_to(t_grid_1d[None, :], (IC.shape[0], N_eval))     # (B, N_eval)

# We still use the same splitter so train/test stay aligned
IC_train, IC_test, ytrain_from_disk, ytest_from_disk, _, _ = train_test_split_unaligned(
    IC, output, time_fake, test_ratio=0.1, seed=SEED
)

# =========================
PLOTS_DIR_TIME = "./plots_new_log15_time_diff/"
PLOTS_DIR_OUTPUT =  "./plots_new_log15_output/"
PLOTS_DIR = "./plots_new_log15_diff_combined/"
os.makedirs(PLOTS_DIR, exist_ok=True)


time_true = np.load(os.path.join(PLOTS_DIR_TIME, "predictions.npz"))['y_true']
time_true = np.cumsum(time_true, axis=1)
output_true = np.load(os.path.join(PLOTS_DIR_OUTPUT, "predictions.npz"))['y_true']
time_pred = np.load(os.path.join(PLOTS_DIR_TIME, "predictions.npz"))['y_pred']
time_pred = np.cumsum(time_pred, axis=1)
output_pred = np.load(os.path.join(PLOTS_DIR_OUTPUT, "predictions.npz"))['y_pred']

time_pred = np.where(time_pred < 1, np.log10(time_pred), time_pred)
time_true = np.where(time_true < 1, np.log10(time_true), time_true)

output_true, output_pred = scale_outputs_train_test(output_true, output_pred, ytrain_from_disk=ytrain_from_disk, apply_log=False)

OUTPUT_DIM = output_true.shape[-1]
num_examples = 5
rng = np.random.default_rng(22)
idxs = rng.integers(low=0, high=output_pred.shape[0], size=num_examples)
print(f"Loaded predictions: time_true {time_true.shape}, output_true {output_true.shape}, time_pred {time_pred.shape}, output_pred {output_pred.shape}")
print(f"Example idxs: {idxs}")
colors = plt.cm.tab10.colors[:len(idxs)]
titles = [
        r"log($T_{\mathrm{eff}}$) - Effective Surface Temperature",
        r"$log(P_{\mathrm{rot}}$) (days) - Rotation Period",
        r"$log(B_{\mathrm{coronal}}/B_\odot$) - Coronal Magnetic Field Strength",
        r"$log(P_{\mathrm{atm}}$) - Photospheric Pressure (cgs)",
        r"$log(\tau_{\mathrm{cz}}$) - Convective Turnover Time (s)",
        r"$log(\dot{M}$) - Mass Loss Rate",
        r"log(Luminosity) - Predicted Stellar Luminosity",
    ]

fig, axes = plt.subplots(1, OUTPUT_DIM, figsize=(25, 4), sharex=True)

for i in range(OUTPUT_DIM):
    ax = axes[i] if OUTPUT_DIM > 1 else axes
    for j, idx in enumerate(idxs):
        color = colors[j]
        y_true = np.array(output_true[idx, :, i])
        y_model = np.array(output_pred[idx, :, i])

        # same color for truth and prediction, different linestyle
        ax.plot(time_true[idx, :, 0], y_true, color=color, linestyle="-")
        ax.plot(time_pred[idx, :, 0], y_model, color=color, linestyle="--")

    ax.set_title(titles[i] if i < len(titles) else f"Output {i}", fontsize=10)
    # ax.set_xscale('log', )
    ax.grid(True)
    if i == 0:
        ax.set_ylabel("")
    if i == max(0, OUTPUT_DIM // 2):
        ax.set_xlabel("Age [Gyr]")

fig.suptitle("Grid Split-Branch DeepONet vs Reference", fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

outpath = os.path.join(PLOTS_DIR, "test_predictions_comparison.png")
plt.savefig(outpath, dpi=300)
plt.close()
print(f"Figure saved: {outpath}")

