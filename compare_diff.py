"""
Compare the *combined* DeepONet emulator (time-diff + output model) against the
reference dataset, by plotting 10 random test trajectories (true vs pred).

This script is designed for the "diff" setup (time model predicts Δt(u)).
It mirrors the "log" comparison workflow, but uses `make_inferences_diff.py`.

Usage (examples):
  python compare_combined_vs_data_diff.py \
      --data_dir ./preprocessing_new_log15 \
      --time_base 1.5 \
      --n_samples 10 \
      --seed 22

Notes
-----
- Expects:
    {data_dir}/initial_conditions.npy   shape (N, 5)
    {data_dir}/output.npy               shape (N, N_eval, 7)   (in your "physical" convention)
    {data_dir}/time.npy                 shape (N, N_eval)      (stored in log_{time_base}; will be exponentiated)
- The emulator will load checkpoints locally if present; otherwise it will try
  to download the checkpoints bundle via the Zenodo settings defined in
  `make_inferences_diff.py`.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"  # or 0.3 to be extra safe
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_malloc_async"


import argparse
import numpy as np
import jax.numpy as jnp
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


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _maybe_patch_ckpt_paths(mid_module) -> None:
    """
    Some local copies of `make_inferences_diff.py` may not define TIME_CKPT_DIR.
    Patch sensible defaults before instantiating the predictor.
    """
    if not hasattr(mid_module, "OUTPUT_CKPT_DIR"):
        mid_module.OUTPUT_CKPT_DIR = os.path.abspath(
            "checkpoints_new/deeponet_params_new_log15_output"
        )

    if not hasattr(mid_module, "TIME_CKPT_DIR"):
        # time-diff checkpoint directory
        mid_module.TIME_CKPT_DIR = os.path.abspath(
            "checkpoints_new/deeponet_params_new_log15_time_diff"
        )


def _interp_u_from_time(t_query: np.ndarray, t_of_u: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
    """
    Invert a monotone time curve t(u) by 1D interpolation to obtain u(t).
    - t_query: (T,) query times (same units as t_of_u)
    - t_of_u:  (N_eval,) monotone increasing time curve
    - u_grid:  (N_eval,) corresponding u grid
    Returns: u_query (T,)
    """
    # Ensure monotonicity (guard against tiny numerical violations)
    t_mono = np.maximum.accumulate(t_of_u)
    # Clamp queries to model range
    t0, t1 = float(t_mono[0]), float(t_mono[-1])
    tq = np.clip(t_query, t0, t1)
    return np.interp(tq, t_mono, u_grid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./preprocessing_new_log15",
                    help="Folder containing initial_conditions.npy, output.npy, time.npy")
    ap.add_argument("--time_base", type=float, default=1.5,
                    help="Base used to store time.npy in log-space (time_phys = base ** time.npy)")
    ap.add_argument("--n_samples", type=int, default=10,
                    help="Number of random trajectories to plot")
    ap.add_argument("--seed", type=int, default=24, help="RNG seed for picking examples")
    ap.add_argument("--out_dir", type=str, default="./plots/plots_compare_diff",
                    help="Output directory for figures")
    ap.add_argument("--time_unit", type=str, default="Gyr",
                    choices=["yr", "Myr", "Gyr"],
                    help="Time unit for plotting on x-axis")
    args = ap.parse_args()

    _ensure_dir(args.out_dir)

    # -----------------------------
    # Load data (reference)
    # -----------------------------
    ic_path = os.path.join(args.data_dir, "initial_conditions.npy")
    y_path  = os.path.join(args.data_dir, "output.npy")
    t_path  = os.path.join(args.data_dir, "time.npy")

    IC = np.load(ic_path).astype(np.float32)         # (N, 5)
    Y  = np.load(y_path).astype(np.float32)          # (N, N_eval, 7)
    t_log = np.load(t_path).astype(np.float64)       # (N, N_eval) in log_base

    t_true = (args.time_base ** t_log).astype(np.float64)  # (N, N_eval) physical

    N, N_eval, out_dim = Y.shape
    assert IC.shape[0] == N and t_true.shape[0] == N, "IC/Y/time must have matching first dimension"
    assert t_true.shape[1] == N_eval, "time.npy length must match output curve length"

    # Convert plotting units
    if args.time_unit == "yr":
        t_plot_scale = 1e9
        t_label = "Time [yr]"
    elif args.time_unit == "Myr":
        t_plot_scale = 1e3
        t_label = "Time [Myr]"
    else:
        t_plot_scale = 1.0
        t_label = "Time [Gyr]"

    # -----------------------------
    # Load combined predictor (diff)
    # -----------------------------
    import make_inferences_diff as mid
    _maybe_patch_ckpt_paths(mid)

    predictor = mid.CombinedPredictor(output_mode="physical")

    # -----------------------------
    # Pick random examples
    # -----------------------------
    rng = np.random.default_rng(args.seed)
    idxs = rng.integers(low=0, high=N, size=args.n_samples)

    ic_batch = jnp.array(IC[idxs], dtype=jnp.float32)  # (B, 5)

    # Query times for `predict`: choose mid-point time for each sample to ensure in-range
    t_query = np.array([t_true[i, N_eval // 2] for i in idxs], dtype=np.float64)

    # Run predictor once for the batch, get the native curves too
    res = predictor.predict(ic_batch, target_time=t_query)

    # Extract native curves (B, N_eval, ...)
    t_pred_native = np.asarray(res["time_physical_native"], dtype=np.float64)    # (B, N_eval)
    print("min max time pred: ", t_pred_native.min(), t_pred_native.max())
    y_pred_native = np.asarray(res["output_native"], dtype=np.float64) # (B, N_eval, 7)

    u_grid = np.linspace(0.0, 1.0, N_eval, dtype=np.float64)

    # Build y_pred evaluated at the *true* physical times for each sample
    y_pred_on_true = np.empty_like(Y[idxs], dtype=np.float64)
    for b in range(args.n_samples):
        u_star = _interp_u_from_time(t_true[idxs[b]], t_pred_native[b], u_grid)  # (N_eval,)
        for k in range(out_dim):
            y_pred_on_true[b, :, k] = np.interp(u_star, u_grid, y_pred_native[b, :, k])

    # -----------------------------
    # Plot: 7 panels, each shows 10 curves (true vs pred, same color)
    # -----------------------------
    titles = [
        r"$\log T_{\mathrm{eff}}$",
        r"$\log P_{\mathrm{rot}}$",
        r"$\log B_{\mathrm{cor}}$",
        r"$\log P_{\mathrm{atm}}$",
        r"$\log \tau_{\mathrm{cz}}$",
        r"$\log \dot{M}$",
        r"$\log L$",
    ]

    # 2x4 grid (last panel blank); 7.09" = 180 mm = A&A two-column text width
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.09, 4.0), sharex=False)
    axes = axes.reshape(-1)

    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, args.n_samples))
    for k in range(out_dim):
        ax = axes[k]
        for b in range(args.n_samples):
            c = colors[b]
            tt = t_true[idxs[b]] * t_plot_scale
            ax.plot(tt, Y[idxs[b], :, k], color=c, ls="-", alpha=0.95)
            ax.plot(tt, y_pred_on_true[b, :, k], color=c, ls="--", alpha=0.95)
        ax.set_title(titles[k] if k < len(titles) else f"Output {k}")
        if k == 0:
            ax.set_ylabel("Value")
        ax.set_xlabel(t_label)

    # Hide unused panels; place legend in the first empty panel
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="black", lw=1.4, linestyle="-", label="Truth"),
        Line2D([0], [0], color="black", lw=1.4, linestyle="--", label="Emulator"),
    ]
    for k in range(out_dim, nrows * ncols):
        ax_leg = axes[k]
        ax_leg.axis("off")
        if k == out_dim:
            ax_leg.legend(handles=legend_elems, loc="center", frameon=False)

    fig.tight_layout()

    out_png = os.path.join(args.out_dir, "compare_diff_true_vs_pred.png")
    fig.savefig(out_png)
    fig.savefig(out_png.replace(".png", ".pdf"))
    plt.close(fig)

    # -----------------------------
    # Optional: also save a time-map sanity plot (t_true(u) vs t_pred(u))
    # -----------------------------
    # 3.46" = 88 mm = A&A single-column width
    fig, ax = plt.subplots(figsize=(3.46, 3.0))
    for b in range(args.n_samples):
        c = colors[b]
        ax.plot(u_grid, t_true[idxs[b]] * t_plot_scale, color=c, ls="-", alpha=0.95)
        ax.plot(u_grid, t_pred_native[b] * t_plot_scale, color=c, ls="--", alpha=0.95)
    ax.set_xlabel(r"Pseudo-time $u$")
    ax.set_ylabel(t_label)
    # title omitted: goes in the figure caption for paper submission
    out_png2 = os.path.join(args.out_dir, "compare_diff_time_map.png")
    fig.tight_layout()
    fig.savefig(out_png2)
    fig.savefig(out_png2.replace(".png", ".pdf"))
    plt.close(fig)

    # -----------------------------
    # Plot: HR diagram / evolutionary tracks
    # -----------------------------
    # Output convention used here:
    #   channel 0 = log(T_eff)
    #   channel 6 = L
    # If your luminosity channel is already log(L), the y-axis label below is still correct
    # up to your data convention; otherwise it shows the physical luminosity value.
    if out_dim >= 7:
        # 3.46" = 88 mm = A&A single-column width
        fig, ax = plt.subplots(figsize=(3.46, 3.2))

        for b in range(args.n_samples):
            c = colors[b]
            ax.plot(Y[idxs[b], :, 0], Y[idxs[b], :, 6], color=c, ls="-", alpha=0.95)
            ax.plot(y_pred_on_true[b, :, 0], y_pred_on_true[b, :, 6], color=c, ls="--", alpha=0.95)
            ax.scatter(Y[idxs[b], 0, 0], Y[idxs[b], 0, 6], color=c, s=14, marker="o", alpha=0.9)
            ax.scatter(Y[idxs[b], -1, 0], Y[idxs[b], -1, 6], color=c, s=16, marker="s", alpha=0.9)

        ax.invert_xaxis()
        ax.set_xlabel(r"$\log(T_{\mathrm{eff}})$")
        ax.set_ylabel(r"$\log(L/L_\odot)$")
        # title omitted: goes in the figure caption for paper submission

        hr_legend_elems = [
            Line2D([0], [0], color="black", lw=1.4, linestyle="-", label="Truth"),
            Line2D([0], [0], color="black", lw=1.4, linestyle="--", label="Emulator"),
            Line2D([0], [0], color="black", marker="o", lw=0, label="Start"),
            Line2D([0], [0], color="black", marker="s", lw=0, label="End"),
        ]
        ax.legend(handles=hr_legend_elems, frameon=False, loc="best")

        fig.tight_layout()
        out_png3 = os.path.join(args.out_dir, "compare_diff_hr_diagram.png")
        fig.savefig(out_png3)
        fig.savefig(out_png3.replace(".png", ".pdf"))
        plt.close(fig)
    else:
        out_png3 = None
        print("[warn] Skipping HR diagram: expected at least 7 output channels "
              "(channel 0 = log(T_eff), channel 6 = L).")


    saved_paths = [out_png, out_png2]
    if out_png3 is not None:
        saved_paths.append(out_png3)
    print("[ok] Saved:\n  " + "\n  ".join(saved_paths))


if __name__ == "__main__":
    main()
