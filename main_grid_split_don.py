import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from autocvd import autocvd
autocvd(num_gpus=1)

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

from arch_grid_split_don import GridSplitBranchDeepONet
from utils import train_test_split_unaligned
from train_grid_split_don import train
from optuna.storages import JournalStorage, JournalFileStorage


# =========================
# CONFIG
# =========================
mode = "predict"  # 'train' | 'predict' | 'optuna'

DATA_DIR  = "/export/data/vgiusepp/StellarEv_emulator/preprocessing_new_log15/"
CKPT_DIR  = os.path.abspath("checkpoints_new/deeponet_params_new_log15_output/")
PLOTS_DIR = "./plots/plots_new_log15_output/"

SEED = 0

#{'learning_rate': 0.0011755636775666748, 'latent_dim': 516, 'num_layers': 5, 'activation': 'gelu', 'use_curve_bias': True}

DEFAULT_MODEL_CFG = dict(
    latent_dim=599,
    num_layers=5,
    activation_name="gelu",
    use_curve_bias=False,
)

DEFAULT_TRAIN_CFG = dict(
    lr=0.0014726203776727874,
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

# =========================
# TRAIN MODE
# =========================
if mode == "train":
    model = build_model(
        latent_dim=DEFAULT_MODEL_CFG["latent_dim"],
        num_layers=DEFAULT_MODEL_CFG["num_layers"],
        activation_name=DEFAULT_MODEL_CFG["activation_name"],
        use_curve_bias=DEFAULT_MODEL_CFG["use_curve_bias"],
    )

    trained_state = train(
        model,
        IC_train,
        output_train,
        ic_test=IC_test,
        y_test=output_test,
        num_epochs=DEFAULT_TRAIN_CFG["num_epochs"],
        batch_size=DEFAULT_TRAIN_CFG["batch_size"],
        lr=DEFAULT_TRAIN_CFG["lr"],
        l2_reg=DEFAULT_TRAIN_CFG["l2_reg"],
        seed=SEED,
    )

    os.makedirs(CKPT_DIR, exist_ok=True)
    checkpoints.save_checkpoint(
        ckpt_dir=CKPT_DIR,
        target=trained_state,
        step=0,
        overwrite=True,
    )
    print(f"Saved checkpoint to: {CKPT_DIR}")

# =========================
# PREDICT MODE
# =========================
elif mode == "predict":
    import matplotlib.pyplot as plt

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

    num_examples = 10
    rng = np.random.default_rng(22)
    idxs = rng.integers(low=0, high=output_test.shape[0], size=num_examples)

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
            ax.plot(t_plot, y_true, color=color, linestyle="-")
            ax.plot(t_plot, y_model, color=color, linestyle="--")

        ax.set_title(titles[i] if i < len(titles) else f"Output {i}", fontsize=10)
        ax.grid(True)
        if i == 0:
            ax.set_ylabel("Scaled value")
        if i == max(0, OUTPUT_DIM // 2):
            ax.set_xlabel("Fixed grid (0..1)")

    fig.suptitle("Grid Split-Branch DeepONet vs Reference", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    outpath = os.path.join(PLOTS_DIR, "test_predictions_comparison.png")
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Figure saved: {outpath}")
    np.savez(os.path.join(PLOTS_DIR, "predictions.npz"), y_pred=np.array(y_pred), y_true=np.array(output_test))

# =========================
# OPTUNA MODE
# =========================
elif mode == "optuna":
    import optuna

    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
        latent_dim = trial.suggest_int("latent_dim", 128, 768)
        num_layers = trial.suggest_int("num_layers", 2, 5)
        activation_name = trial.suggest_categorical("activation", ["tanh", "relu", "gelu", "silu"])
        use_curve_bias = trial.suggest_categorical("use_curve_bias", [True, False])

        model = build_model(
            latent_dim=latent_dim,
            num_layers=num_layers,
            activation_name=activation_name,
            use_curve_bias=use_curve_bias,
        )

        try:
            trained_state = train(
                model,
                IC_train,
                output_train,
                ic_test=IC_test,
                y_test=output_test,
                num_epochs=300,
                batch_size=DEFAULT_TRAIN_CFG["batch_size"],
                lr=learning_rate,
                l2_reg=DEFAULT_TRAIN_CFG["l2_reg"],
                seed=SEED,
            )

            y_pred = trained_state.apply_fn(trained_state.params, IC_test)
            test_mse = np.mean((np.array(y_pred) - np.array(output_test)) ** 2)
            return float(test_mse)

        except Exception as e:
            print(f"[trial failed] {e}")
            return float("inf")
        
    study_name = 'study_deeponet'  # Unique identifier of the study.
    storage_name = JournalStorage(JournalFileStorage("./optuna_new_log15_output.log"))
    study = optuna.create_study(direction="minimize",study_name=study_name, storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=180)

    print("Best trial:")
    print("  Value (MSE):", study.best_value)
    print("  Params:", study.best_params)

else:
    raise ValueError(f"Unknown mode: {mode}")

