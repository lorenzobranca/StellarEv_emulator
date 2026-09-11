"""
Single-panel figure: q90 of |error| pooled over ALL output channels, as a function
of true age, for the log time model vs. the Δt time model.

The age axis is logarithmic up to 1 Gyr and linear above (single axis, custom scale).
Reads the per-point error dumps written by compare_log.py / compare_diff.py
(errors_log.npz, errors_diff.npz). No GPU needed.

Run:
  python plot_q90_pooled_log_vs_diff.py
"""

import argparse
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.scale import FuncScale
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

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
    "ytick.minor.visible": True,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "axes.linewidth": 0.8,
    "axes.grid": False,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

COLOR_LOG = plt.cm.RdYlBu_r(0.0)
COLOR_DIFF = plt.cm.RdYlBu_r(1.0)

T_BREAK = 1.0    # Gyr: log axis below, linear above
T_MAX = 14.0     # Gyr
LOG_MIN = 1e-6   # Gyr
# How many "decades" of axis width the linear part (1..T_MAX Gyr) occupies.
LIN_WIDTH_DECADES = 3.0


def _fwd(t):
    t = np.asarray(t, dtype=np.float64)
    out = np.empty_like(t)
    lo = t <= T_BREAK
    with np.errstate(divide="ignore", invalid="ignore"):
        out[lo] = np.log10(t[lo] / T_BREAK)
    out[~lo] = (t[~lo] - T_BREAK) / (T_MAX - T_BREAK) * LIN_WIDTH_DECADES
    return out


def _inv(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    lo = x <= 0
    out[lo] = T_BREAK * 10 ** x[lo]
    out[~lo] = T_BREAK + x[~lo] / LIN_WIDTH_DECADES * (T_MAX - T_BREAK)
    return out


def pooled_q90_per_bin(npz_path, edges, q=0.90):
    """q-quantile of |error| pooled over all channels and all valid grid points, per age bin."""
    d = np.load(npz_path)
    e = d["abs_err"].astype(np.float64)          # (N, N_eval, D)
    t = d["t_true"].astype(np.float64)           # (N, N_eval)
    v = d["valid"]                               # (N, N_eval)
    nd = e.shape[-1]
    e = e[v].reshape(-1)                         # pool channels
    t = np.repeat(t[v], nd)
    idx = np.digitize(t, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    counts = np.zeros(len(edges) - 1, dtype=int)
    for j in range(len(edges) - 1):
        m = idx == j
        counts[j] = m.sum()
        if counts[j]:
            out[j] = np.quantile(e[m], q)
    return out, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_npz", default="./plots/plots_combined_compare/errors_log.npz")
    ap.add_argument("--diff_npz", default="./plots/plots_compare_diff/errors_diff.npz")
    ap.add_argument("--out_dir", default="./plots/plots_compare_log_vs_diff")
    ap.add_argument("--q", type=float, default=0.90)
    args = ap.parse_args()

    # Half-decade bins below 1 Gyr, 1 Gyr bins above.
    edges = np.concatenate([10 ** np.arange(np.log10(LOG_MIN), 0.0, 0.5),
                            np.arange(T_BREAK, T_MAX + 1e-9, 1.0)])

    q_log, n_log = pooled_q90_per_bin(args.log_npz, edges, args.q)
    q_diff, n_diff = pooled_q90_per_bin(args.diff_npz, edges, args.q)
    assert np.array_equal(n_log, n_diff), "the two error dumps do not share the same test grid"

    # 3.46" = 88 mm = A&A single-column width
    fig, ax = plt.subplots(figsize=(3.46, 2.9))
    ax.stairs(q_log, edges, baseline=None, color=COLOR_LOG, linestyle="-", linewidth=1.4)
    ax.stairs(q_diff, edges, baseline=None, color=COLOR_DIFF, linestyle="--", linewidth=1.4)
    ax.axvline(T_BREAK, color="0.75", linewidth=0.6, zorder=0)

    ax.set_xscale(FuncScale(ax.xaxis, (_fwd, _inv)))
    ax.set_xlim(LOG_MIN, T_MAX)
    major = [1e-6, 1e-4, 1e-2, 1.0, 5.0, 10.0]
    minor = list(10 ** np.arange(-6.0, 0.0, 1.0)) + list(np.arange(2.0, 14.0, 1.0))
    ax.xaxis.set_major_locator(FixedLocator(major))
    ax.xaxis.set_minor_locator(FixedLocator(minor))

    def _fmt(t, _pos):
        if t < 1:
            return rf"$10^{{{int(round(np.log10(t)))}}}$"
        return f"{t:g}"
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))

    ax.set_yscale("log")
    ax.set_xlabel("Age [Gyr]  (log below 1 Gyr, linear above)")
    ax.set_ylabel(rf"$q_{{{int(round(args.q*100))}}}\,|y_{{\mathrm{{pred}}}} - y_{{\mathrm{{true}}}}|$ [dex], all channels")
    handles = [
        Line2D([0], [0], color=COLOR_LOG, linestyle="-", lw=1.4, label="log time model"),
        Line2D([0], [0], color=COLOR_DIFF, linestyle="--", lw=1.4, label=r"time increment model"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    fig.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "q90_pooled_vs_age_log_vs_diff.png")
    fig.savefig(out)
    fig.savefig(out.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"[saved] {out}")

    print(f"\npooled q{int(round(args.q*100))} |error| (dex), all channels, non-padded points:")
    print(f"{'age bin [Gyr]':>20s} {'n_pts':>9s} {'log':>10s} {'diff':>10s} {'diff/log':>9s}")
    for j in range(len(edges) - 1):
        print(f"[{edges[j]:.3g}, {edges[j+1]:.3g}) ".rjust(21)
              + f"{n_log[j]:9d} {q_log[j]:10.3e} {q_diff[j]:10.3e} {q_diff[j]/q_log[j]:9.2f}")


if __name__ == "__main__":
    main()
