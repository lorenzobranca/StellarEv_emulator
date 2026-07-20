
"""
Publication-ready plotting utilities for StellarEv_emulator predictions.

This script extends `plot_combined.py` by:
  1) producing cleaner, paper-ready multi-panel plots (PDF + PNG),
  2) adding an HR-diagram style plot in the log(T_eff)–log(L) plane
     showing true vs predicted tracks.

It assumes the following directory structure (same as your existing script):
  - TIME predictions saved to:  PLOTS_DIR_TIME / "predictions.npz"
  - OUTPUT predictions saved to: PLOTS_DIR_OUTPUT / "predictions.npz"

Each `predictions.npz` is expected to contain:
  - y_true: (N, N_eval, D)
  - y_pred: (N, N_eval, D)
and for the TIME file additionally:
  - m: scalar mean used in standardization
  - s: scalar std used in standardization

Run:
  python plot_combined_paper.py

Adjust the configuration block below (paths, indices, scaling) as needed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

# Where the original scripts stored the predictions
PLOTS_DIR_TIME = Path("./plots_new_log15_time/")
PLOTS_DIR_OUTPUT = Path("./plots_new_log15_output/")

# Output directory for paper-ready plots
OUT_DIR = Path("./plots_new_log15_combined_paper/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Time decoding used in your current code (change if needed)
TIME_BASE = 1.5  # used as: t_years = TIME_BASE ** (scaled*s + m)

# Axes choice for time-series plots
TIME_AXIS = "log10_years"  # options: "log10_years" | "years"

# Which output indices correspond to log(T_eff) and log(L)
# (Set these to match your dataset convention)
LOGTEFF_IDX = 0
LOGL_IDX = 6

# Whether the L output is already log10(L) (set False if it is linear)
L_IS_LOG10 = True

# If your outputs are STANDARDIZED, provide mean/std vectors to unscale.
# If you don't set these, the script will plot in "model space" and label accordingly.
# Example:
#   OUT_MEAN = np.array([...], dtype=np.float64)  # shape (7,)
#   OUT_STD  = np.array([...], dtype=np.float64)  # shape (7,)
OUT_MEAN = None  # type: Optional[np.ndarray]
OUT_STD = None   # type: Optional[np.ndarray]

# For time-series panels: number of example trajectories to show
N_EXAMPLES = 6
RANDOM_SEED = 0

# Plot file formats
SAVE_PDF = True
SAVE_PNG = True
PNG_DPI = 300

# ============================================================
# Paper-like matplotlib style
# ============================================================

mpl.rcParams.update({
    # Fonts
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,

    # Lines
    "lines.linewidth": 1.2,
    "lines.markersize": 3,

    # Figure / save
    "figure.dpi": 150,
    "savefig.dpi": PNG_DPI,
    "savefig.bbox": "tight",

    # Axes cosmetics
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
})


# ============================================================
# Helpers
# ============================================================

def _load_predictions_npz(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _maybe_unscale(y: np.ndarray, mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> np.ndarray:
    if mean is None or std is None:
        return y
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape or mean.shape[0] != y.shape[-1]:
        raise ValueError(
            f"OUT_MEAN/OUT_STD must be 1D arrays of shape (D,) matching y[..., D]. "
            f"Got mean={mean.shape}, std={std.shape}, y_lastdim={y.shape[-1]}"
        )
    return y * std[None, None, :] + mean[None, None, :]


def decode_time_from_scaled(time_scaled: np.ndarray, m: float, s: float, base: float = 1.5) -> np.ndarray:
    """
    Decode standardized time output into physical time (years).

    Expected time_scaled shape: (N, N_eval) or (N, N_eval, 1)
    """
    ts = np.asarray(time_scaled, dtype=np.float64)
    if ts.ndim == 3:
        ts = ts[..., 0]
    # reverse standardization: x_phys = x_scaled * s + m
    # then decode base-exponent: t = base**x_phys
    return np.power(base, ts * s + m)


def time_axis_transform(t_years: np.ndarray, axis: str) -> Tuple[np.ndarray, str]:
    """
    Return transformed time axis + label.
    """
    if axis == "years":
        return t_years, r"$t\ \mathrm{[yr]}$"
    if axis == "log10_years":
        # avoid log of 0; should never happen, but keep safe
        t = np.clip(t_years, 1e-30, None)
        return np.log10(t), r"$\log_{10}(t/\mathrm{yr})$"
    raise ValueError(f"Unknown TIME_AXIS={axis!r}")


def save_figure(fig: plt.Figure, stem: str) -> None:
    if SAVE_PDF:
        fig.savefig(OUT_DIR / f"{stem}.pdf")
    if SAVE_PNG:
        fig.savefig(OUT_DIR / f"{stem}.png")


# ============================================================
# Plotting
# ============================================================

def plot_time_series_panels(
    t_true_years: np.ndarray,
    t_pred_years: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    titles: Optional[list[str]] = None,
) -> None:
    """
    Multi-panel plot: for each output dimension, show y_true(t_true) vs y_pred(t_pred)
    for a small subset of trajectories.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = y_true.shape[0]
    idxs = rng.choice(n, size=min(N_EXAMPLES, n), replace=False)

    # Unscale outputs (if requested)
    y_true_u = _maybe_unscale(y_true, OUT_MEAN, OUT_STD)
    y_pred_u = _maybe_unscale(y_pred, OUT_MEAN, OUT_STD)

    # Time axis transformation
    x_true, xlab = time_axis_transform(t_true_years, TIME_AXIS)
    x_pred, _ = time_axis_transform(t_pred_years, TIME_AXIS)

    d = y_true.shape[-1]
    if titles is None:
        titles = [f"Output {k}" for k in range(d)]
    if len(titles) != d:
        raise ValueError("titles length must match output dimension D")

    # 2x4 grid (fits D=7; last panel reserved for legend)
    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 3.8), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    # Determine y-label: physical vs scaled
    y_is_physical = (OUT_MEAN is not None and OUT_STD is not None)
    ylab = "Value" if y_is_physical else "Scaled value"

    # Plot each output in its panel
    for k in range(d):
        ax = axes[k]
        for ii in idxs:
            # same color for true/pred; use solid vs dashed
            (ln,) = ax.plot(x_true[ii], y_true_u[ii, :, k], alpha=0.9)
            ax.plot(x_pred[ii], y_pred_u[ii, :, k], linestyle="--", alpha=0.9, color=ln.get_color())

        ax.set_title(titles[k])
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

    # Legend in the last (8th) panel
    leg_ax = axes[-1]
    leg_ax.axis("off")
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="k", lw=1.4, linestyle="-", label="True"),
        Line2D([0], [0], color="k", lw=1.4, linestyle="--", label="Predicted"),
    ]
    leg_ax.legend(handles=legend_handles, loc="center", frameon=False)

    save_figure(fig, "time_series_outputs_panels")
    plt.close(fig)


def plot_hr_tracks(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    titlestr: str = r"HR diagram: $\log T_{\mathrm{eff}}$ vs $\log L$ (true vs predicted)",
) -> None:
    """
    Plot true vs predicted tracks in the log(T)-log(L) plane (HR-like diagram).

    We treat output indices LOGTEFF_IDX and LOGL_IDX as log10 quantities by default.
    If L_IS_LOG10=False, we plot log10(L) computed from the output.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = y_true.shape[0]
    idxs = rng.choice(n, size=min(N_EXAMPLES, n), replace=False)

    # Unscale (if requested)
    y_true_u = _maybe_unscale(y_true, OUT_MEAN, OUT_STD)
    y_pred_u = _maybe_unscale(y_pred, OUT_MEAN, OUT_STD)

    logT_true = np.asarray(y_true_u[..., LOGTEFF_IDX], dtype=np.float64)
    logT_pred = np.asarray(y_pred_u[..., LOGTEFF_IDX], dtype=np.float64)

    L_true = np.asarray(y_true_u[..., LOGL_IDX], dtype=np.float64)
    L_pred = np.asarray(y_pred_u[..., LOGL_IDX], dtype=np.float64)

    if L_IS_LOG10:
        logL_true, logL_pred = L_true, L_pred
        ylab = r"$\log_{10}(L)$"
    else:
        logL_true = np.log10(np.clip(L_true, 1e-30, None))
        logL_pred = np.log10(np.clip(L_pred, 1e-30, None))
        ylab = r"$\log_{10}(L)$ (from linear $L$)"

    fig, ax = plt.subplots(figsize=(3.6, 3.2), constrained_layout=True)

    for ii in idxs:
        (ln,) = ax.plot(logT_true[ii], logL_true[ii], alpha=0.9)
        ax.plot(logT_pred[ii], logL_pred[ii], linestyle="--", alpha=0.9, color=ln.get_color())

    ax.set_title(titlestr)
    ax.set_xlabel(r"$\log_{10}(T_{\mathrm{eff}})$")
    ax.set_ylabel(ylab)
    ax.invert_xaxis()  # HR diagram convention: hotter to the left

    # Small legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="k", lw=1.4, linestyle="-", label="True"),
        Line2D([0], [0], color="k", lw=1.4, linestyle="--", label="Predicted"),
    ]
    ax.legend(handles=legend_handles, loc="best", frameon=False)

    save_figure(fig, "hr_diagram_tracks")
    plt.close(fig)


def plot_hr_pointwise_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """
    Optional: pointwise scatter of predicted vs true in the HR plane (all samples/times).
    This is useful to show overall agreement but can be dense.
    """
    y_true_u = _maybe_unscale(y_true, OUT_MEAN, OUT_STD)
    y_pred_u = _maybe_unscale(y_pred, OUT_MEAN, OUT_STD)

    logT_true = np.asarray(y_true_u[..., LOGTEFF_IDX], dtype=np.float64).ravel()
    logT_pred = np.asarray(y_pred_u[..., LOGTEFF_IDX], dtype=np.float64).ravel()

    L_true = np.asarray(y_true_u[..., LOGL_IDX], dtype=np.float64).ravel()
    L_pred = np.asarray(y_pred_u[..., LOGL_IDX], dtype=np.float64).ravel()

    if L_IS_LOG10:
        logL_true, logL_pred = L_true, L_pred
        ylab = r"$\log_{10}(L)$"
    else:
        logL_true = np.log10(np.clip(L_true, 1e-30, None))
        logL_pred = np.log10(np.clip(L_pred, 1e-30, None))
        ylab = r"$\log_{10}(L)$ (from linear $L$)"

    # Downsample for aesthetics/performance if extremely large
    N = logT_true.size
    max_points = 200_000
    if N > max_points:
        rng = np.random.default_rng(RANDOM_SEED)
        sel = rng.choice(N, size=max_points, replace=False)
        logT_true, logT_pred = logT_true[sel], logT_pred[sel]
        logL_true, logL_pred = logL_true[sel], logL_pred[sel]

    fig, ax = plt.subplots(figsize=(3.6, 3.2), constrained_layout=True)
    ax.scatter(logT_true, logL_true, s=2, alpha=0.25, label="True")
    ax.scatter(logT_pred, logL_pred, s=2, alpha=0.25, label="Predicted")

    ax.set_title("HR plane point cloud (downsampled)")
    ax.set_xlabel(r"$\log_{10}(T_{\mathrm{eff}})$")
    ax.set_ylabel(ylab)
    ax.invert_xaxis()
    ax.legend(frameon=False, loc="best", markerscale=3)

    save_figure(fig, "hr_diagram_pointcloud")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    time_npz = _load_predictions_npz(PLOTS_DIR_TIME / "predictions.npz")
    out_npz = _load_predictions_npz(PLOTS_DIR_OUTPUT / "predictions.npz")

    # TIME arrays
    time_true = np.asarray(time_npz["y_true"], dtype=np.float64)
    time_pred = np.asarray(time_npz["y_pred"], dtype=np.float64)
    m = float(np.asarray(time_npz["m"]))
    s = float(np.asarray(time_npz["s"]))

    t_true_years = decode_time_from_scaled(time_true, m=m, s=s, base=TIME_BASE)
    t_pred_years = decode_time_from_scaled(time_pred, m=m, s=s, base=TIME_BASE)

    # OUTPUT arrays
    y_true = np.asarray(out_npz["y_true"], dtype=np.float64)
    y_pred = np.asarray(out_npz["y_pred"], dtype=np.float64)

    # The TIME and OUTPUT predictions come from separate .npz files. They are
    # plotted against each other per-sample and per-eval-point, so they must be
    # aligned in sample count and in the number of eval points along the curve.
    if t_true_years.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"Sample-count mismatch between TIME and OUTPUT predictions: "
            f"time has {t_true_years.shape[0]} samples, output has {y_true.shape[0]}. "
            f"Check that {PLOTS_DIR_TIME/'predictions.npz'} and "
            f"{PLOTS_DIR_OUTPUT/'predictions.npz'} were produced from the same run."
        )
    if t_true_years.shape[1] != y_true.shape[1]:
        raise ValueError(
            f"N_eval mismatch between TIME and OUTPUT predictions: "
            f"time has {t_true_years.shape[1]} eval points, output has {y_true.shape[1]}."
        )

    # Titles for the 7 StellarEv outputs (see CLAUDE.md, indices 0-6)
    titles = [
        r"$\log T_{\mathrm{eff}}$",
        r"$P_{\mathrm{rot}}\ \mathrm{[days]}$",
        r"$B_{\mathrm{cor}}/B_\odot$",
        r"$P_{\mathrm{atm}}$",
        r"$\tau_{\mathrm{cz}}\ \mathrm{[s]}$",
        r"$\dot{M}$",
        r"$L$",
    ][: y_true.shape[-1]]

    plot_time_series_panels(t_true_years, t_pred_years, y_true, y_pred, titles=titles)

    # HR diagram plots
    plot_hr_tracks(y_true, y_pred)

    # Optional additional plot:
    # plot_hr_pointwise_scatter(y_true, y_pred)

    print(f"[ok] Saved plots to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
