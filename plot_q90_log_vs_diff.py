"""
Single figure comparing the age-binned q90 error of the two emulator variants
(log time model vs. diff time model), one panel per output channel.

Reads the CSV tables written by compare_log.py / compare_diff.py
(q90_per_age_bin_log.csv, q90_per_age_bin_diff.csv). No GPU needed.

Run:
  python plot_q90_log_vs_diff.py
"""

import argparse
import csv
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# A&A-compatible style (same as compare_log.py / compare_diff.py)
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "lines.linewidth": 1.2,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "axes.linewidth": 0.8,
    "axes.grid": False,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

TITLES = [
    r"$\log T_{\mathrm{eff}}$", r"$\log P_{\mathrm{rot}}$", r"$\log B_{\mathrm{cor}}$",
    r"$\log P_{\mathrm{atm}}$", r"$\log \tau_{\mathrm{cz}}$", r"$\log \dot{M}$", r"$\log L$",
]

# Variant colours: the two ends of the channel palette used elsewhere in the paper.
COLOR_LOG = plt.cm.RdYlBu_r(0.0)
COLOR_DIFF = plt.cm.RdYlBu_r(1.0)


def read_q90_csv(path):
    """Return (edges (n_bins+1,), table (n_bins, D), counts (n_bins,))."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    lo = np.array([float(r[0]) for r in body])
    hi = np.array([float(r[1]) for r in body])
    counts = np.array([int(r[2]) for r in body])
    table = np.array([[float(v) for v in r[3:]] for r in body])
    edges = np.concatenate([lo, hi[-1:]])
    return edges, table, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_csv", default="./plots/plots_combined_compare/q90_per_age_bin_log.csv")
    ap.add_argument("--diff_csv", default="./plots/plots_compare_diff/q90_per_age_bin_diff.csv")
    ap.add_argument("--out_dir", default="./plots/plots_compare_log_vs_diff")
    args = ap.parse_args()

    edges_l, tab_l, _ = read_q90_csv(args.log_csv)
    edges_d, tab_d, _ = read_q90_csv(args.diff_csv)
    assert np.allclose(edges_l, edges_d), "age bins differ between the two CSVs"
    assert tab_l.shape == tab_d.shape
    edges = edges_l
    d = tab_l.shape[1]

    # 2x4 grid, last panel holds the legend; 7.09" = 180 mm = A&A two-column width
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.09, 4.0), sharex=True, sharey=True)
    axes = axes.reshape(-1)

    for k in range(d):
        ax = axes[k]
        ax.stairs(tab_l[:, k], edges, baseline=None, color=COLOR_LOG, linestyle="-", linewidth=1.4)
        ax.stairs(tab_d[:, k], edges, baseline=None, color=COLOR_DIFF, linestyle="--", linewidth=1.4)
        ax.axvline(1.0, color="0.75", linewidth=0.6, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(TITLES[k] if k < len(TITLES) else f"Output {k}")
        if k // ncols == nrows - 1 or k + ncols >= d:
            ax.set_xlabel("Age [Gyr]")
        if k % ncols == 0:
            ax.set_ylabel(r"$q_{90}\,|y_{\mathrm{pred}} - y_{\mathrm{true}}|$ [dex]")

    for k in range(d, nrows * ncols):
        axes[k].axis("off")
    handles = [
        Line2D([0], [0], color=COLOR_LOG, linestyle="-", lw=1.4, label="log time model"),
        Line2D([0], [0], color=COLOR_DIFF, linestyle="--", lw=1.4, label=r"$\Delta t$ time model"),
    ]
    axes[d].legend(handles=handles, loc="center", frameon=False)

    fig.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "q90_vs_age_log_vs_diff.png")
    fig.savefig(out)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"[saved] {out}")

    # Print which variant is better per age bin (geometric mean over channels)
    ratio = np.exp(np.nanmean(np.log(tab_d / tab_l), axis=1))
    print("\nq90(diff)/q90(log), geometric mean over channels:")
    for j in range(len(ratio)):
        print(f"  [{edges[j]:.3g}, {edges[j+1]:.3g}) Gyr: {ratio[j]:.2f}  "
              f"({'diff better' if ratio[j] < 1 else 'log better'})")


if __name__ == "__main__":
    main()
