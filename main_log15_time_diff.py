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
from train_grid_split_don_pos_diff import train
from optuna.storages import JournalStorage, JournalFileStorage


# =========================
# CONFIG
# =========================
mode = "predict"  # 'train' | 'predict' | 'optuna'

DATA_DIR  = "/export/data/vgiusepp/StellarEv_emulator/preprocessing_new_log15/"
CKPT_DIR  = os.path.abspath("checkpoints_new/deeponet_params_new_log15_time_diff/")
PLOTS_DIR = "./plots_new_log15_time_diff/"
TIME_DIR  = "./preprocessing_new_log15/"

SEED = 0

#{'learning_rate': 0.0011755636775666748, 'latent_dim': 516, 'num_layers': 5, 'activation': 'gelu', 'use_curve_bias': True}

DEFAULT_MODEL_CFG = dict(
    latent_dim=138,
    num_layers=7,
    output_dim = 1,
    activation_name="relu",
    use_curve_bias=True,
)
mono_weight = 0.0
lambda_logsum = 0.0

# DEFAULT_MODEL_CFG = dict(
#     latent_dim=223,
#     num_layers=5,
#     output_dim = 1,
#     activation_name="tanh",
#     use_curve_bias=False,
# )

DEFAULT_TRAIN_CFG = dict(
    lr=0.0006587227663328755,
    num_epochs=1_000,
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




# =========================
# Load + split + scale data
# =========================
IC = np.load(os.path.join(DATA_DIR, "initial_conditions.npy"))
# output = np.load(os.path.join(DATA_DIR, "output.npy"))
output = np.load(os.path.join(TIME_DIR, "time.npy"))

#we rescale to work in times
output = 1.5**output


#we can take the diff
output_diff = np.diff(output, axis=1, prepend= np.zeros((output.shape[0], 1)))
# output_diff[:, 0] = np.zeros_like(output[:, 0])  # keep the first column as is (or set to a fixed value)
output = output_diff
print('min max output diff:', output_diff.min(), output_diff.max())
# print(output)

# Force output to be 3D: (Batch, N_eval, 1)
if output.ndim == 2:
    output = output[..., np.newaxis]
print(f"Loaded IC shape: {IC.shape}, output shape: {output.shape}")

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

# m, s = fit_standard_scaler(output_train)
# output_train = apply_standard_scaler(output_train, m, s)
# output_test = apply_standard_scaler(output_test, m, s)
# print('min')

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
        mono_idx=-1,
        mono_weight=mono_weight,
        mono_mode="hinge",
        lambda_logsum=lambda_logsum
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
        mono_idx=-1,
        mono_weight=mono_weight,
        mono_mode="hinge",
    )

    restored_state = checkpoints.restore_checkpoint(
        ckpt_dir=CKPT_DIR,
        target=dummy_state,
    )

    y_pred = restored_state.apply_fn(restored_state.params, IC_test)
    y_pred = np.where(y_pred < 0, -y_pred, y_pred)  # ensure non-negativity for monotonic penalty
    test_mse = np.mean((np.array(y_pred) - np.array(output_test)) ** 2)
    print('output test is always positive?', (output_test < 0).sum())
    print("Test MSE:", float(test_mse))

    os.makedirs(PLOTS_DIR, exist_ok=True)

    titles = [
        r"Time",
    ]

    num_examples = 10
    rng = np.random.default_rng(10)
    idxs = rng.integers(low=0, high=output_test.shape[0], size=num_examples)

    colors = plt.cm.tab10.colors[:num_examples]
    t_plot = np.array(t_grid_1d)

    # 2 rows: top for the values, bottom for the difference
    fig, axes = plt.subplots(2, max(1, int(OUTPUT_DIM)), figsize=(6 * max(1, int(OUTPUT_DIM)), 6), sharex=True)
    
    # Ensure axes is a 2D array for consistent indexing: (2, OUTPUT_DIM)
    if int(OUTPUT_DIM) == 1:
        axes = np.atleast_2d(axes).T

    for i in range(int(OUTPUT_DIM)):
        ax_main = axes[0, i]
        ax_diff = axes[1, i]
        
        for j, idx in enumerate(idxs):
            color = colors[j]
            y_true = np.array(output_test[idx, :, i])
            t_true = np.array(output_test[idx, :, -1])
            y_model = np.array(y_pred[idx, :, i])
            t_model = np.array(y_pred[idx, :, -1])

            #we need to to the cumsum 
            y_model = np.cumsum(y_model)
            y_true = np.cumsum(y_true)

            # same color for truth and prediction, different linestyle
            ax_main.plot(np.arange(len(y_true)), y_true, color=color, linestyle="-")
            ax_main.plot(np.arange(len(y_model)), y_model, color=color, linestyle="--")
            
            # Difference plot
            diff = y_model - y_true
            ax_diff.plot(np.arange(len(diff)), diff, color=color, linestyle="-", alpha=0.7)

        ax_main.grid(True)
        ax_diff.grid(True)
        
        if i == 0:
            ax_main.set_ylabel("Scaled value")
            ax_diff.set_ylabel("Difference\n(Model - True)")
        
        ax_diff.set_xlabel("Fixed grid (0..1)")

    fig.suptitle("Grid Split-Branch DeepONet vs Reference", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    outpath = os.path.join(PLOTS_DIR, "test_predictions_comparison.png")
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Figure saved: {outpath}")
    np.savez(os.path.join(PLOTS_DIR, "predictions.npz"), y_pred=np.array(y_pred), y_true=np.array(output_test), )

    # --- New plot for monotonicity check ---
    cols = 2
    rows = int(np.ceil(num_examples / cols))
    fig2, axes2 = plt.subplots(rows, cols, figsize=(14, 3 * rows), sharex=True)
    axes2 = axes2.flatten()
    
    for j, idx in enumerate(idxs):
        ax = axes2[j]
        color = colors[j]
        
        # Get the predicted and true scaled values
        y_model = np.array(y_pred[idx, :, 0])
        y_true_val = np.array(output_test[idx, :, 0])


        #need the cumsum
        y_model = np.cumsum(y_model)
        y_true_val = np.cumsum(y_true_val)
        
        # Unscale to actual time
        time_pred = y_model 
        time_true = y_true_val 
        
        # Calculate differences between consecutive points
        time_diffs_pred = np.diff(time_pred, prepend=0)  # prepend 0 to keep same length
        time_diffs_true = np.diff(time_true, prepend=0)
        
        ax.plot(np.arange(len(time_diffs_true)), time_diffs_true, color='black', alpha=1, label='True Δt')
        ax.plot(np.arange(len(time_diffs_pred)), time_diffs_pred, color=color, linestyle='--', alpha=0.9, label='Model Δt')
        
        ax.axhline(0, color='red', linestyle=':', linewidth=2, label='Zero (Monotonicity threshold)')
        ax.set_title(f"Example IDX: {idx}")
        ax.grid(True)
        
        if j == 0:
            ax.legend()
            
    # Hide any unused subplots
    for j in range(num_examples, len(axes2)):
        axes2[j].axis('off')

    fig2.supxlabel("Time step index")
    fig2.supylabel("Δt (Differences between consecutive steps)")
    fig2.suptitle("Time Monotonicity Check", fontsize=16)

    outpath2 = os.path.join(PLOTS_DIR, "time_monotonicity_check.png")
    fig2.tight_layout()
    fig2.savefig(outpath2, dpi=300)
    plt.close(fig2)
    print(f"Monotonicity check figure saved: {outpath2}")

# =========================
# OPTUNA MODE
# =========================
elif mode == "optuna":
    import optuna

    from train_grid_split_don_pos_diff import cumsum_loss

    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
        latent_dim = trial.suggest_int("latent_dim", 512, 1024)
        num_layers = trial.suggest_int("num_layers", 4, 8)
        activation_name = trial.suggest_categorical("activation", ["tanh", "relu", "gelu", "silu"])
        use_curve_bias = trial.suggest_categorical("use_curve_bias", [True, False])
        # mono_weight = trial.suggest_float("mono_weight", 1e-2, 5e-1, log = True)
        mono_weight = 0.0
        lambda_logsum = trial.suggest_float("lambda_logsum", 1e-11, 1e-3, log=True)


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
                num_epochs=1000,
                batch_size=DEFAULT_TRAIN_CFG["batch_size"],
                lr=learning_rate,
                l2_reg=DEFAULT_TRAIN_CFG["l2_reg"],
                seed=SEED,
                mono_idx=-1,
                mono_weight=mono_weight,
                mono_mode="hinge",
                lambda_logsum=lambda_logsum
            )

            y_pred = trained_state.apply_fn(trained_state.params, IC_test)
            test_mse = np.mean((np.array(y_pred) - np.array(output_test)) ** 2) + cumsum_loss(y_pred, output_test, lambda_logsum)
            return float(test_mse)

        except Exception as e:
            print(f"[trial failed] {e}")
            return float("inf")
        
    study_name = 'study_deeponet'  # Unique identifier of the study.
    storage_name = JournalStorage(JournalFileStorage("./optuna_new_log15_time_diff_pronto.log"))
    study = optuna.create_study(direction="minimize",study_name=study_name, storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=300)

    print("Best trial:")
    print("  Value (MSE):", study.best_value)
    print("  Params:", study.best_params)

else:
    raise ValueError(f"Unknown mode: {mode}")

