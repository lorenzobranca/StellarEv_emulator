import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

from arch_LSTM_DON_v2 import DON_LSTM
from utils import train_test_split_unaligned
from train_v2 import train

# =========================
# CONFIG
# =========================
mode = "train"  # 'train' | 'predict' | 'optuna'

DATA_DIR = "./preprocessing_output"
# use a NEW checkpoint folder because the architecture changed
CKPT_DIR = os.path.abspath("checkpoints_kde_new_DON_LSTM_v2/deeponet_params")
PLOTS_DIR = "plots_DON_LSTM_v2"

SEED = 0

DEFAULT_MODEL_CFG = dict(
    latent_dim=512,
    num_layers=4,
    refiner_hidden=256,
    activation_name="tanh",
    use_lstm_branch=False,
)

DEFAULT_TRAIN_CFG = dict(
    lr=7e-4,
    num_epochs=800,
    batch_size=32,
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
            mi, si  = fit_standard_scaler(y_train[:,:,:])
            y_train = y_train.at[:,:,i].set(apply_standard_scaler(y_train[:,:,i], mi, si))
            y_test  = y_test.at[:,:,i].set(apply_standard_scaler(y_test[:,:,i], mi, si))

    return y_train, y_test

# =========================
# Load + split + scale data
# =========================
IC = jnp.load(os.path.join(DATA_DIR, "initial_conditions.npy"))
output = jnp.load(os.path.join(DATA_DIR, "output.npy"))

IC = jnp.array(IC, dtype=jnp.float32)
output = jnp.array(output, dtype=jnp.float32)

# ---- Fake time grid: length must match label sequence length ----
N_eval = int(output.shape[1])
t_grid_1d = jnp.linspace(0.0, 1.0, N_eval, dtype=jnp.float32)  # (N_eval,)
time_fake = jnp.broadcast_to(t_grid_1d[None, :], (IC.shape[0], N_eval))  # (B, N_eval)

IC_train, IC_test, output_train, output_test, time_train, time_test = train_test_split_unaligned(
    IC, output, time_fake, test_ratio=0.1, seed=SEED
)

IC_train = jnp.array(IC_train, dtype=jnp.float32)
IC_test  = jnp.array(IC_test, dtype=jnp.float32)

output_train, output_test = scale_outputs_train_test(output_train, output_test, apply_log = False)

# time is already fake time in [0, 1]; do not log-scale it
time_train = jnp.array(time_train, dtype=jnp.float32)
time_test  = jnp.array(time_test, dtype=jnp.float32)


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

def build_model(
    latent_dim: int,
    num_layers: int,
    refiner_hidden: int,
    activation_name: str,
    use_lstm_branch: bool = False,
):
    activation_fn = get_activation(activation_name)

    trunk_layers = tuple([latent_dim for _ in range(num_layers)])
    
    branch_mlp = tuple([latent_dim for _ in range(max(2, num_layers - 1))])

    model = DON_LSTM(
        latent_dim=int(latent_dim),
        output_dim=int(OUTPUT_DIM),
        trunk_layers=trunk_layers,
        refiner_hidden=int(refiner_hidden),
        use_lstm_branch=use_lstm_branch,
        branch_mlp=branch_mlp,
        activation=activation_fn,
        use_residual=True,
    )
    return model

# =========================
# TRAIN MODE
# =========================
if mode == "train":
    model = build_model(
        latent_dim=DEFAULT_MODEL_CFG["latent_dim"],
        num_layers=DEFAULT_MODEL_CFG["num_layers"],
        refiner_hidden=DEFAULT_MODEL_CFG["refiner_hidden"],
        activation_name=DEFAULT_MODEL_CFG["activation_name"],
        use_lstm_branch=DEFAULT_MODEL_CFG["use_lstm_branch"],
    )

    trained_state = train(
        model,
        IC_train,
        time_train,
        output_train,
        ic_test=IC_test,
        t_test=time_test,
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
        refiner_hidden=DEFAULT_MODEL_CFG["refiner_hidden"],
        activation_name=DEFAULT_MODEL_CFG["activation_name"],
        use_lstm_branch=DEFAULT_MODEL_CFG["use_lstm_branch"],
    )

    # create a dummy state with correct structure
    dummy_state = train(
        model,
        IC_train,
        time_train,
        output_train,
        ic_test=IC_test,
        t_test=time_test,
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

    y_pred = restored_state.apply_fn(restored_state.params, IC_test, time_test)
    test_mse = np.mean((np.array(y_pred) - np.array(output_test)) ** 2)
    print("Test MSE:", test_mse)

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
    rng = np.random.default_rng(123)
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

            # same color for truth/pred, different linestyle
            ax.plot(t_plot, y_true, color=color, linestyle="-")
            ax.plot(t_plot, y_model, color=color, linestyle="--")

        ax.set_title(titles[i] if i < len(titles) else f"Output {i}", fontsize=10)
        ax.grid(True)
        if i == 0:
            ax.set_ylabel("Scaled value")
        if i == max(0, OUTPUT_DIM // 2):
            ax.set_xlabel("Fake time (0..1)")

    fig.suptitle("DON+LSTM vs Reference on Test Curves", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    outpath = os.path.join(PLOTS_DIR, "test_predictions_comparison.png")
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Figure saved: {outpath}")

# =========================
# OPTUNA MODE
# =========================
elif mode == "optuna":
    import optuna

    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
        latent_dim = trial.suggest_int("latent_dim", 128, 768)
        num_layers = trial.suggest_int("num_layers", 2, 5)
        refiner_hidden = trial.suggest_int("refiner_hidden", 64, 512)
        activation_name = trial.suggest_categorical("activation", ["tanh", "relu", "gelu", "silu"])

        model = build_model(
            latent_dim=latent_dim,
            num_layers=num_layers,
            refiner_hidden=refiner_hidden,
            activation_name=activation_name,
            use_lstm_branch=False,
        )

        try:
            trained_state = train(
                model,
                IC_train,
                time_train,
                output_train,
                ic_test=IC_test,
                t_test=time_test,
                y_test=output_test,
                num_epochs=300,
                batch_size=DEFAULT_TRAIN_CFG["batch_size"],
                lr=learning_rate,
                l2_reg=DEFAULT_TRAIN_CFG["l2_reg"],
                seed=SEED,
            )

            y_pred = trained_state.apply_fn(trained_state.params, IC_test, time_test)
            test_mse = np.mean((np.array(y_pred) - np.array(output_test)) ** 2)
            return float(test_mse)

        except Exception as e:
            print(f"[trial failed] {e}")
            return float("inf")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=180)

    print("Best trial:")
    print("  Value (MSE):", study.best_value)
    print("  Params:", study.best_params)

else:
    raise ValueError(f"Unknown mode: {mode}")

