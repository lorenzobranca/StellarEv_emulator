"""
Compare the *combined* emulator (time-model + output-model) against the raw dataset.

- Loads IC / time / output arrays from the same DATA_DIR used by the training scripts.
- Uses CombinedPredictor from make_inferences_log.py to generate *predicted curves*.
- Plots 10 random test trajectories: truth (solid) vs emulator (dashed), using the same color per sample.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"  # or 0.3 to be extra safe
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_malloc_async"
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# A&A-compatible style (consistent with plot_combined_paper.py)
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.linewidth": 0.8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": False,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

from make_inferences_log import CombinedPredictor, OUTPUT_NAMES
from utils import train_test_split_unaligned


# ----------------------------
# CONFIG (adjust if needed)
# ----------------------------
DATA_DIR = "./preprocessing_new_log15"
IC_PATH = os.path.join(DATA_DIR, "initial_conditions.npy")
TIME_PATH = os.path.join(DATA_DIR, "time.npy")
Y_PATH = os.path.join(DATA_DIR, "output.npy")

PLOTS_DIR = "./plots/plots_combined_compare"
SEED = 0
TEST_RATIO = 0.1
N_EXAMPLES = 10  # number of random curves to plot


def _as_np(x):
    return np.asarray(x)


def _maybe_decode_physical_time(t_arr: np.ndarray, time_base: float, time_mean: float, time_std: float) -> np.ndarray:
    """
    Heuristic: if time contains non-positive values, it is likely stored in log/standardised form.
    If already positive, assume it is physical and return as-is.

    Adjust this if your time.npy already stores a transformed time.
    """
    t_arr = _as_np(t_arr).astype(np.float64)
    if np.all(t_arr > 0.0):
        return t_arr

    # If not strictly positive, assume "scaled log" as used by the time model:
    # physical = base ** (scaled * std + mean)
    return time_base ** (t_arr) #* time_std + time_mean -5)


def main():
    # ----------------------------
    # Load data
    # ----------------------------
    IC = np.load(IC_PATH).astype(np.float32)      # (N, 5)
    time = np.load(TIME_PATH).astype(np.float64) # (N, N_eval) or (N, ...)
    
    y = np.load(Y_PATH).astype(np.float32)       # (N, N_eval, 7)

    # Split the same way as training scripts
    IC_train, IC_test, y_train, y_test, t_train, t_test = train_test_split_unaligned(
        IC, y, time, test_ratio=TEST_RATIO, seed=SEED
    )

    # ----------------------------
    # Instantiate combined emulator
    # ----------------------------
    predictor = CombinedPredictor(output_mode="physical")
    print("t_true before: ", t_test.min(), t_test.max())
    # Some datasets store time in log/standardised units. If your time.npy is already physical,
    # this is a no-op.
    try:
        t_test_phys = _maybe_decode_physical_time(
            t_test,
            time_base=float(getattr(__import__("make_inferences_log"), "TIME_BASE")),
            time_mean=float(getattr(__import__("make_inferences_log"), "TIME_SCALER")["mean"]),
            time_std=float(getattr(__import__("make_inferences_log"), "TIME_SCALER")["std"]),
        )
    
    except Exception:
        # If constants are not available or something changes, just assume time is physical.
        t_test_phys = _as_np(t_test).astype(np.float64)
    
    print("t_test_phys after: ", t_test_phys.min(), t_test_phys.max())
    
    # ----------------------------
    # Sample random test trajectories
    # ----------------------------
    rng = np.random.default_rng(SEED + 123)
    idxs = rng.integers(low=0, high=IC_test.shape[0], size=N_EXAMPLES)

    ic_batch = IC_test[idxs]  # (B, 5)
    y_true = y_test[idxs]     # (B, N_eval, 7)
    t_true = t_test_phys[idxs]  # (B, N_eval)

    # Predict full curves on the model's native u-grid (internally)
    out = predictor.predict(ic_batch, target_time=None)
    t_pred = _as_np(out["time_physical_native"])    # (B, N_eval)
    y_pred = _as_np(out["output_native"])           # (B, N_eval, 7)

    # ----------------------------
    # Plot: 7 panels, 10 curves each
    # ----------------------------
    os.makedirs(PLOTS_DIR, exist_ok=True)

    output_dim = y_true.shape[-1]
    titles = [
        r"$\log T_{\mathrm{eff}}$",
        r"$\log P_{\mathrm{rot}}$",
        r"$\log B_{\mathrm{cor}}$",
        r"$\log P_{\mathrm{atm}}$",
        r"$\log \tau_{\mathrm{cz}}$",
        r"$\log \dot{M}$",
        r"$\log L$",
    ]

    # 2x4 grid (last axis blank); 7.09" = 180 mm = A&A two-column text width
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.09, 4.0), sharex=False)
    axes = axes.reshape(-1)

    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, N_EXAMPLES))

    for k in range(output_dim):
        ax = axes[k]
        for j in range(N_EXAMPLES):
            c = colors[j]
            ax.plot(np.log10(t_true[j]), y_true[j, :, k], color=c, linestyle="-", alpha=0.9)
            ax.plot(np.log10(t_pred[j]), y_pred[j, :, k], color=c, linestyle="--", alpha=0.9)

        ax.set_title(titles[k])
        ax.set_xlabel(r"$\log_{10}(t\,/\,\mathrm{Gyr})$")
        if k % ncols == 0:
            ax.set_ylabel("Value")

    # Hide unused panel (axis 7 if output_dim=7); place legend there
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="black", lw=1.4, linestyle="-", label="Truth"),
        Line2D([0], [0], color="black", lw=1.4, linestyle="--", label="Emulator"),
    ]
    for k in range(output_dim, nrows * ncols):
        ax_leg = axes[k]
        ax_leg.axis("off")
        if k == output_dim:
            ax_leg.legend(handles=legend_elems, loc="center", frameon=False)

    fig.tight_layout()

    outpath = os.path.join(PLOTS_DIR, "combined_vs_data_10curves.png")
    fig.savefig(outpath)
    fig.savefig(outpath.replace(".png", ".pdf"))
    plt.close()
    print(f"[saved] {outpath}")

    # ----------------------------
    # HR-diagram-style plot: log(T_eff) vs luminosity
    # ----------------------------
    # Convention from the training/output setup:
    #   output channel 0: log(T_eff)
    #   output channel 6: luminosity L
    #
    # The true and predicted tracks are plotted in the same color for each
    # sampled star. The x-axis is inverted, as customary for HR diagrams.
    if output_dim >= 7:
        teff_idx = 0
        lum_idx = 6

        # 3.46" = 88 mm = A&A single-column width
        fig, ax = plt.subplots(figsize=(3.46, 3.2))
        for j in range(N_EXAMPLES):
            c = colors[j]

            ax.plot(
                y_true[j, :, teff_idx],
                y_true[j, :, lum_idx],
                color=c,
                linestyle="-",
                linewidth=1.4,
                alpha=0.9,
            )
            ax.plot(
                y_pred[j, :, teff_idx],
                y_pred[j, :, lum_idx],
                color=c,
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
            )

            # Mark start/end of the true trajectory to show the evolution direction.
            ax.scatter(
                y_true[j, 0, teff_idx],
                y_true[j, 0, lum_idx],
                color=c,
                marker="o",
                s=22,
                alpha=0.9,
            )
            ax.scatter(
                y_true[j, -1, teff_idx],
                y_true[j, -1, lum_idx],
                color=c,
                marker="s",
                s=22,
                alpha=0.9,
            )

        ax.invert_xaxis()
        ax.set_xlabel(r"$\log(T_{\mathrm{eff}})$")
        ax.set_ylabel(r"$\log(L/L_\odot)$")
        # title omitted: goes in the figure caption for paper submission

        hr_legend_elems = [
            Line2D([0], [0], color="black", lw=1.4, linestyle="-", label="Truth"),
            Line2D([0], [0], color="black", lw=1.4, linestyle="--", label="Emulator"),
            Line2D([0], [0], color="black", marker="o", linestyle="None", label="Start"),
            Line2D([0], [0], color="black", marker="s", linestyle="None", label="End"),
        ]
        ax.legend(handles=hr_legend_elems, frameon=False)
        fig.tight_layout()

        hr_outpath = os.path.join(PLOTS_DIR, "combined_vs_data_hr_diagram.png")
        fig.savefig(hr_outpath)
        fig.savefig(hr_outpath.replace(".png", ".pdf"))
        plt.close(fig)
        print(f"[saved] {hr_outpath}")
    else:
        print("[skip] HR diagram needs at least 7 output channels: channel 0=log(T_eff), channel 6=L")

    # Also save the sampled arrays for reproducibility
    np.savez(
        os.path.join(PLOTS_DIR, "combined_vs_data_samples.npz"),
        idxs=idxs,
        ic=ic_batch,
        t_true=t_true,
        y_true=y_true,
        t_pred=t_pred,
        y_pred=y_pred,
    )
    print(f"[saved] {os.path.join(PLOTS_DIR, 'combined_vs_data_samples.npz')}")


if __name__ == "__main__":
    main()
