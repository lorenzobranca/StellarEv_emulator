import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from typing import Optional, Union, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax.training import checkpoints
from scipy.interpolate import interp1d

from arch_grid_split_don import GridSplitBranchDeepONet
from train_grid_split_don_pos_diff import train as train_time
from train_grid_split_don import train as train_output
from params_loader import ZenodoSource, restore_checkpoint_or_zenodo

ZENODO = ZenodoSource(
    record_url="https://zenodo.org/records/19736519",
    asset_name="checkpoints_new.zip",
)

# ============================================================
# USER CONFIG
# ============================================================

SAVE_DIR = "./inference_results_diff/"
os.makedirs(SAVE_DIR, exist_ok=True)

# output model
ckpt_output = "checkpoints_new/deeponet_params_new_log15_output"
state_output = restore_checkpoint_or_zenodo(ckpt_output, dummy_state_output, ZENODO)

# time-diff model
ckpt_time_diff = "checkpoints_new/deeponet_params_new_log15_time_diff"
state_time_diff = restore_checkpoint_or_zenodo(ckpt_time_diff, dummy_state_time_diff, ZENODO)

IC_DIM = 5
N_EVAL = 3999
U_NATIVE = np.linspace(0.0, 1.0, N_EVAL, dtype=np.float64)


OUTPUT_MODE = "scaled"   # "scaled" or "physical"


TIME_MODEL_CFG = dict(
    
    # Defaults below follow main_log15_time_diff.py
    latent_dim=138,
    num_layers=7,
    output_dim=1,
    activation_name="relu",
    use_curve_bias=True,
)

TIME_TRAIN_CFG = dict(
    # Only used to build a dummy TrainState for checkpoint restore.
    lr=0.0006587227663328755,
    batch_size=256,
    seed=0,
    
)

TIME_SCALER = dict(
    mean=-8.41216,
    std=10.202949,
)

TIME_BASE = 1.5

# Δt de-scaling for the time-diff model output
#   mode="none"      : model already outputs physical Δt
#   mode="standard"  : Δt_phys = Δt_scaled * std + mean
#   mode="log_base"  : Δt_phys = base ** (Δt_scaled * std + mean)
TIME_DIFF_SCALER = dict(
    mode="none",
    mean=0.0,
    std=1.0,
    base=10.0,
)

DT_EPS = 1e-20

# ----------------------------
# Output model
# ----------------------------
OUTPUT_CKPT_DIR = os.path.abspath("checkpoints_new/deeponet_params_new_log15_output/")

OUTPUT_MODEL_CFG = dict(
    latent_dim=599,
    num_layers=5,
    output_dim=7,
    activation_name="gelu",
    use_curve_bias=False,
)

OUTPUT_TRAIN_CFG = dict(
    lr=0.0014726203776727874,
    batch_size=256,
    l2_reg=0.0,
    seed=0,
)

OUTPUT_SCALER = dict(
    mean=np.array([
        3.6247501,
        0.5581493,
        1.2172874,
        4.7572474,
        6.819683,
        -12.865734,
        -0.010272743,
    ], dtype=np.float64),
    std=np.array([
        0.082149595,
        1.4462037,
        2.7018602,
        0.5947154,
        0.61604226,
        0.98719424,
        1.1583182,
    ], dtype=np.float64),
)

OUTPUT_NAMES = [
    r"log($T_{\mathrm{eff}}$) - Effective Surface Temperature",
    r"$P_{\mathrm{rot}}$ (days) - Rotation Period",
    r"$B_{\mathrm{coronal}}/B_\odot$ - Coronal Magnetic Field Strength",
    r"$P_{\mathrm{atm}}$ - Photospheric Pressure (cgs)",
    r"$\tau_{\mathrm{cz}}$ - Convective Turnover Time (s)",
    r"$\dot{M}$ - Mass Loss Rate",
    r"Luminosity - Predicted Stellar Luminosity",
]


# ============================================================
# HELPERS
# ============================================================

def get_activation(name: str):
    activation_map = {
        "tanh": jax.nn.tanh,
        "relu": jax.nn.relu,
        "gelu": jax.nn.gelu,
        "silu": jax.nn.silu,
    }
    if name not in activation_map:
        raise ValueError(f"Unsupported activation: {name}")
    return activation_map[name]


def build_model(
    n_eval: int,
    output_dim: int,
    latent_dim: int,
    num_layers: int,
    activation_name: str,
    use_curve_bias: bool,
):
    activation_fn = get_activation(activation_name)
    branch_layers = tuple([latent_dim for _ in range(num_layers)])

    return GridSplitBranchDeepONet(
        branch_layers=branch_layers,
        latent_dim=int(latent_dim),
        output_dim=int(output_dim),
        n_eval=int(n_eval),
        activation=activation_fn,
        use_curve_bias=use_curve_bias,
    )


def ensure_batch_ic(ic: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
    ic = jnp.array(ic, dtype=jnp.float32)
    if ic.ndim == 1:
        ic = ic[None, :]
    if ic.ndim != 2:
        raise ValueError(f"IC must have shape (features,) or (batch, features), got {ic.shape}")
    if ic.shape[1] != IC_DIM:
        raise ValueError(f"Expected IC dimension {IC_DIM}, got {ic.shape[1]}")
    return ic


def invert_standard_scaler(x: np.ndarray, m: np.ndarray, s: np.ndarray) -> np.ndarray:
    return x * s + m


def decode_time_from_scaled(time_scaled: np.ndarray, m: float, s: float, base: float = 1.5) -> np.ndarray:
    return np.asarray(base ** (time_scaled * s + m), dtype=np.float64)


def decode_dt_from_scaled(dt_scaled: np.ndarray) -> np.ndarray:
    """Decode the time-diff model output to physical Δt."""
    dt_scaled = np.asarray(dt_scaled, dtype=np.float64)
    mode = TIME_DIFF_SCALER.get("mode", "none")
    if mode == "none":
        return dt_scaled
    if mode == "standard":
        return dt_scaled * float(TIME_DIFF_SCALER["std"]) + float(TIME_DIFF_SCALER["mean"])
    if mode == "log_base":
        base = float(TIME_DIFF_SCALER.get("base", 10.0))
        return base ** (dt_scaled * float(TIME_DIFF_SCALER["std"]) + float(TIME_DIFF_SCALER["mean"]))
    raise ValueError(f"Unknown TIME_DIFF_SCALER['mode']={mode!r}")


def encode_time_to_scaled(time_physical: np.ndarray, m: float, s: float, base: float = 1.5) -> np.ndarray:
    time_physical = np.asarray(time_physical, dtype=np.float64)
    if np.any(time_physical <= 0):
        raise ValueError("All physical query times must be > 0")
    return (np.log(time_physical) / np.log(base) - m) / s


def normalize_target_time(target_time: Optional[Union[np.ndarray, list]], batch_size: int):
    if target_time is None:
        return None

    target_time = np.asarray(target_time, dtype=np.float64)

    if target_time.ndim == 1:
        return np.broadcast_to(target_time[None, :], (batch_size, target_time.shape[0]))

    if target_time.ndim == 2:
        if target_time.shape[0] != batch_size:
            raise ValueError(
                f"2D target_time must have shape (batch, n_times); got {target_time.shape}, batch={batch_size}"
            )
        return target_time

    raise ValueError("target_time must be None, 1D, or 2D")


def make_interp_sorted(x_src, y_src, kind="linear"):
    x_src = np.asarray(x_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)

    order = np.argsort(x_src)
    x_src = x_src[order]
    y_src = y_src[order]

    unique_mask = np.concatenate([[True], np.diff(x_src) > 0])
    x_src = x_src[unique_mask]
    y_src = y_src[unique_mask]

    if len(x_src) < 2:
        raise ValueError("Need at least two unique source points for interpolation")

    return interp1d(
        x_src,
        y_src,
        axis=0,
        kind=kind,
        bounds_error=False,
        fill_value=(y_src[0], y_src[-1]),
        assume_sorted=True,
    )


def make_interp_on_u(y_src, kind="linear"):
    y_src = np.asarray(y_src, dtype=np.float64)

    return interp1d(
        U_NATIVE,
        y_src,
        axis=0,
        kind=kind,
        bounds_error=False,
        fill_value=(y_src[0], y_src[-1]),
        assume_sorted=True,
    )


# ============================================================
# MODEL WRAPPERS
# ============================================================

class TimeModelPredictor:
    def __init__(self):
        self.model = build_model(
            n_eval=N_EVAL,
            output_dim=TIME_MODEL_CFG["output_dim"],
            latent_dim=TIME_MODEL_CFG["latent_dim"],
            num_layers=TIME_MODEL_CFG["num_layers"],
            activation_name=TIME_MODEL_CFG["activation_name"],
            use_curve_bias=TIME_MODEL_CFG["use_curve_bias"],
        )

        x_dummy = jnp.zeros((1, IC_DIM), dtype=jnp.float32)
        y_dummy = jnp.zeros((1, N_EVAL, TIME_MODEL_CFG["output_dim"]), dtype=jnp.float32)

        # Build a dummy TrainState for checkpoint restore.
        # NOTE: signature follows train_grid_split_don_pos_diff.train
        dummy_state = train_time(
            self.model,
            x_dummy,
            y_dummy,
            ic_test=x_dummy,
            y_test=y_dummy,
            num_epochs=0,
            batch_size=TIME_TRAIN_CFG["batch_size"],
            lr=TIME_TRAIN_CFG["lr"],
            seed=TIME_TRAIN_CFG["seed"],
            mono_idx=-1,
            #mono_weight=TIME_TRAIN_CFG["mono_weight"],
            #lambda_logsum=TIME_TRAIN_CFG["lambda_logsum"],
            #lambda_smooth=TIME_TRAIN_CFG["lambda_smooth"],
        )
        
        self.state = restore_checkpoint_or_zenodo(
            ckpt_dir=TIME_CKPT_DIR,
            target=dummy_state,
            zenodo_url=TIME_ZENODO_URL,
            )
        #self.state = checkpoints.restore_checkpoint(TIME_CKPT_DIR, target=dummy_state)

    # --- Δt prediction (model output)
    def predict_dt_scaled_curve(self, ic_batch: jnp.ndarray) -> np.ndarray:
        pred = self.state.apply_fn(self.state.params, ic_batch)  # (B, N_EVAL, 1)
        return np.asarray(pred[..., 0], dtype=np.float64)

    def predict_dt_physical_curve(self, ic_batch: jnp.ndarray) -> np.ndarray:
        dt_scaled = self.predict_dt_scaled_curve(ic_batch)
        dt_phys = decode_dt_from_scaled(dt_scaled)
        # enforce non-negativity and avoid flat segments
        dt_phys = np.maximum(np.abs(dt_phys), DT_EPS)
        return dt_phys

    # --- Reconstructed time curve
    def predict_physical_curve(self, ic_batch: jnp.ndarray) -> np.ndarray:
        dt_phys = self.predict_dt_physical_curve(ic_batch)
        return np.cumsum(dt_phys, axis=1)

    def predict_scaled_curve(self, ic_batch: jnp.ndarray) -> np.ndarray:
        """Optional: provide the same scaled-time diagnostic as in the original script."""
        t_phys = self.predict_physical_curve(ic_batch)
        return encode_time_to_scaled(
            t_phys,
            TIME_SCALER["mean"],
            TIME_SCALER["std"],
            base=TIME_BASE,
        )


class OutputModelPredictor:
    def __init__(self):
        self.model = build_model(
            n_eval=N_EVAL,
            output_dim=OUTPUT_MODEL_CFG["output_dim"],
            latent_dim=OUTPUT_MODEL_CFG["latent_dim"],
            num_layers=OUTPUT_MODEL_CFG["num_layers"],
            activation_name=OUTPUT_MODEL_CFG["activation_name"],
            use_curve_bias=OUTPUT_MODEL_CFG["use_curve_bias"],
        )

        x_dummy = jnp.zeros((1, IC_DIM), dtype=jnp.float32)
        y_dummy = jnp.zeros((1, N_EVAL, OUTPUT_MODEL_CFG["output_dim"]), dtype=jnp.float32)

        dummy_state = train_output(
            self.model,
            x_dummy,
            y_dummy,
            ic_test=x_dummy,
            y_test=y_dummy,
            num_epochs=0,
            batch_size=OUTPUT_TRAIN_CFG["batch_size"],
            lr=OUTPUT_TRAIN_CFG["lr"],
            l2_reg=OUTPUT_TRAIN_CFG["l2_reg"],
            seed=OUTPUT_TRAIN_CFG["seed"],
        )

        self.state = restore_checkpoint_or_zenodo(
            ckpt_dir=OUTPUT_CKPT_DIR,
            target=dummy_state,
            zenodo_url=OUTPUT_ZENODO_URL,
            )
        #self.state = checkpoints.restore_checkpoint(OUTPUT_CKPT_DIR, target=dummy_state)

    def predict_scaled_curve(self, ic_batch: jnp.ndarray) -> np.ndarray:
        pred = self.state.apply_fn(self.state.params, ic_batch)
        return np.asarray(pred, dtype=np.float64)

    def predict_curve(self, ic_batch: jnp.ndarray, output_mode: str = "scaled") -> np.ndarray:
        scaled = self.predict_scaled_curve(ic_batch)

        if output_mode == "scaled":
            return scaled

        if output_mode == "physical":
            m = OUTPUT_SCALER["mean"][None, None, :]
            s = OUTPUT_SCALER["std"][None, None, :]
            return invert_standard_scaler(scaled, m, s)

        raise ValueError(f"Unknown output_mode: {output_mode}")


# ============================================================
# MAIN PREDICTOR
# ============================================================

class CombinedPredictor:
    def __init__(self, output_mode: str = "scaled"):
        self.time_model = TimeModelPredictor()
        self.output_model = OutputModelPredictor()
        self.output_mode = output_mode

    def predict(
        self,
        ic: Union[np.ndarray, jnp.ndarray],
        target_time: Optional[Union[np.ndarray, list]] = None,
    ) -> Dict[str, Any]:
        ic_batch = ensure_batch_ic(ic)
        batch_size = ic_batch.shape[0]

        # Time model now predicts Δt(u); we reconstruct t(u).
        dt_scaled_native = self.time_model.predict_dt_scaled_curve(ic_batch)
        dt_physical_native = self.time_model.predict_dt_physical_curve(ic_batch)
        time_physical_native = self.time_model.predict_physical_curve(ic_batch)
        # Optional scaled-time diagnostic (kept for compatibility with the old script).
        time_scaled_native = self.time_model.predict_scaled_curve(ic_batch)
        output_native = self.output_model.predict_curve(ic_batch, self.output_mode)

        target_time_batch = normalize_target_time(target_time, batch_size)
        if target_time_batch is None:
            return {
                "u_native": np.broadcast_to(U_NATIVE[None, :], (batch_size, N_EVAL)),
                "dt_scaled_native": dt_scaled_native,
                "dt_physical_native": dt_physical_native,
                "time_scaled_native": time_scaled_native,
                "time_physical_native": time_physical_native,
                "output_native": output_native,
                "requested_time_physical": None,
                "requested_time_scaled": None,
                "query_u": None,
                "query_output": None,
                "query_time_physical": None,
                "output_mode": self.output_mode,
            }

        # For logging/debug only (not used for inversion in the Δt setting)
        requested_time_scaled = encode_time_to_scaled(
            target_time_batch,
            TIME_SCALER["mean"],
            TIME_SCALER["std"],
            base=TIME_BASE,
        )

        query_u_list = []
        query_output_list = []
        query_time_physical_list = []

        for i in range(batch_size):
            # Invert physical time curve: t(u) -> u(t)
            phys_time_to_u = make_interp_sorted(
                x_src=time_physical_native[i],
                y_src=U_NATIVE,
                kind="linear",
            )
            u_query = np.asarray(phys_time_to_u(target_time_batch[i]), dtype=np.float64)

            output_on_u = make_interp_on_u(output_native[i], kind="linear")
            output_query = np.asarray(output_on_u(u_query), dtype=np.float64)

            time_on_u = make_interp_on_u(time_physical_native[i], kind="linear")
            time_query = np.asarray(time_on_u(u_query), dtype=np.float64)

            query_u_list.append(u_query)
            query_output_list.append(output_query)
            query_time_physical_list.append(time_query)

        query_u = np.stack(query_u_list, axis=0)
        query_output = np.stack(query_output_list, axis=0)
        query_time_physical = np.stack(query_time_physical_list, axis=0)

        return {
            "u_native": np.broadcast_to(U_NATIVE[None, :], (batch_size, N_EVAL)),
            "dt_scaled_native": dt_scaled_native,
            "dt_physical_native": dt_physical_native,
            "time_scaled_native": time_scaled_native,
            "time_physical_native": time_physical_native,
            "output_native": output_native,
            "requested_time_physical": target_time_batch,
            "requested_time_scaled": requested_time_scaled,
            "query_u": query_u,
            "query_output": query_output,
            "query_time_physical": query_time_physical,
            "output_mode": self.output_mode,
        }

    def save_results(self, result: Dict[str, Any], filename: str = "predictions.npz"):
        path = os.path.join(SAVE_DIR, filename)
        np.savez(
            path,
            u_native=result["u_native"],
            dt_scaled_native=result.get("dt_scaled_native", None),
            dt_physical_native=result.get("dt_physical_native", None),
            time_scaled_native=result["time_scaled_native"],
            time_physical_native=result["time_physical_native"],
            output_native=result["output_native"],
            requested_time_physical=result["requested_time_physical"],
            requested_time_scaled=result["requested_time_scaled"],
            query_u=result["query_u"],
            query_output=result["query_output"],
            query_time_physical=result["query_time_physical"],
        )
        print(f"Saved results to: {path}")

    def plot_time_vs_u(self, result: Dict[str, Any], sample_idx: int = 0, filename: str = "time_vs_u.png"):
        u = result["u_native"][sample_idx]
        t_scaled = result["time_scaled_native"][sample_idx]
        t_phys = result["time_physical_native"][sample_idx]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(u, t_scaled, linewidth=1)
        axes[0].set_xlabel("u")
        axes[0].set_ylabel("Scaled time")
        axes[0].set_title("Scaled time vs u")
        axes[0].grid(True)

        axes[1].plot(u, t_phys, linewidth=1)
        axes[1].set_xlabel("u")
        axes[1].set_ylabel("Physical time")
        axes[1].set_title("Physical time vs u")
        axes[1].set_yscale("log")
        axes[1].grid(True, which="both", alpha=0.4)

        fig.tight_layout()
        path = os.path.join(SAVE_DIR, filename)
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"Saved plot to: {path}")

    def plot_outputs_vs_u(self, result: Dict[str, Any], sample_idx: int = 0, filename: str = "outputs_vs_u.png"):
        u = result["u_native"][sample_idx]
        y = result["output_native"][sample_idx]

        output_dim = y.shape[1]
        fig, axes = plt.subplots(1, output_dim, figsize=(25, 4), sharex=True)

        if output_dim == 1:
            axes = [axes]

        ylabel = "Scaled value" if result["output_mode"] == "scaled" else "Value"

        for i in range(output_dim):
            ax = axes[i]
            ax.plot(u, y[:, i], linewidth=1)
            ax.set_title(OUTPUT_NAMES[i] if i < len(OUTPUT_NAMES) else f"Output {i}", fontsize=10)
            ax.grid(True)
            if i == 0:
                ax.set_ylabel(ylabel)
            if i == max(0, output_dim // 2):
                ax.set_xlabel("u")

        fig.suptitle("Outputs vs u", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        path = os.path.join(SAVE_DIR, filename)
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"Saved plot to: {path}")

    def plot_outputs_vs_time(self, result: Dict[str, Any], sample_idx: int = 0, filename: str = "outputs_vs_time.png"):
        t = result["time_physical_native"][sample_idx]
        y = result["output_native"][sample_idx]

        output_dim = y.shape[1]
        fig, axes = plt.subplots(1, output_dim, figsize=(25, 4), sharex=False)

        if output_dim == 1:
            axes = [axes]

        ylabel = "Scaled value" if result["output_mode"] == "scaled" else "Value"

        for i in range(output_dim):
            ax = axes[i]
            ax.plot(t, y[:, i], linewidth=1)
            ax.set_xscale("log")
            ax.set_title(OUTPUT_NAMES[i] if i < len(OUTPUT_NAMES) else f"Output {i}", fontsize=10)
            ax.grid(True, which="both", alpha=0.4)
            if i == 0:
                ax.set_ylabel(ylabel)
            if i == max(0, output_dim // 2):
                ax.set_xlabel("Time")

        fig.suptitle("Outputs vs time", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        path = os.path.join(SAVE_DIR, filename)
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"Saved plot to: {path}")

    def plot_query_results(self, result: Dict[str, Any], sample_idx: int = 0, filename: str = "query_results.png"):
        if result["requested_time_physical"] is None:
            return

        t_native = result["time_physical_native"][sample_idx]
        y_native = result["output_native"][sample_idx]
        t_query = result["query_time_physical"][sample_idx]
        y_query = result["query_output"][sample_idx]

        output_dim = y_native.shape[1]
        fig, axes = plt.subplots(1, output_dim, figsize=(25, 4), sharex=False)

        if output_dim == 1:
            axes = [axes]

        ylabel = "Scaled value" if result["output_mode"] == "scaled" else "Value"

        for i in range(output_dim):
            ax = axes[i]
            ax.plot(t_native, y_native[:, i], linewidth=1)
            ax.plot(t_query, y_query[:, i], "o")
            ax.set_xscale("log")
            ax.set_title(OUTPUT_NAMES[i] if i < len(OUTPUT_NAMES) else f"Output {i}", fontsize=10)
            ax.grid(True, which="both", alpha=0.4)
            if i == 0:
                ax.set_ylabel(ylabel)
            if i == max(0, output_dim // 2):
                ax.set_xlabel("Time")

        fig.suptitle("Queried outputs", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        path = os.path.join(SAVE_DIR, filename)
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"Saved plot to: {path}")


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    predictor = CombinedPredictor(output_mode=OUTPUT_MODE)

    ic_single = np.array([1.0, 0.1, 1.5, 2.1, 0.3], dtype=np.float32)

    result_native = predictor.predict(ic_single)
    predictor.save_results(result_native, filename="native_curves.npz")
    predictor.plot_time_vs_u(result_native, sample_idx=0, filename="time_vs_u.png")
    predictor.plot_outputs_vs_u(result_native, sample_idx=0, filename="outputs_vs_u.png")
    predictor.plot_outputs_vs_time(result_native, sample_idx=0, filename="outputs_vs_time.png")

    user_time = np.array([1e-4, 3e-2, 1e-1, 3e-1, 1e0], dtype=np.float64)
    result_query = predictor.predict(ic_single, target_time=user_time)
    predictor.save_results(result_query, filename="queried_outputs.npz")
    predictor.plot_query_results(result_query, sample_idx=0, filename="query_results.png")
