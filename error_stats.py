"""
Shared error statistics and plots for compare_log.py / compare_diff.py.

All errors are absolute differences between emulator and reference outputs,
evaluated at the *true* physical ages of the test-set grid points. Since all
output channels are log10 quantities, the errors are in dex.
"""

import csv
import os
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

# Age bins (Gyr) for the per-age q90 table. The dataset spans ~1e-6 to 13.8 Gyr.
AGE_BIN_EDGES_GYR = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 5.0, 14.0])


def _labels(titles: Optional[Sequence[str]], d: int):
    titles = list(titles) if titles is not None else []
    return [titles[i] if i < len(titles) else f"Output {i}" for i in range(d)]


def strictly_increasing_mask(t_true: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    True where the true age strictly increases w.r.t. the previous grid point.
    The preprocessed tracks are padded with repeated ages at both ends (about 12%
    of all grid points, mostly at the final age), where the outputs still vary by
    up to several dex. On those points the age->output mapping is degenerate, so an
    error "at the true age" is ill-defined and they are excluded from the age-binned
    statistics. The first grid point of each track is kept.
    """
    t = np.asarray(t_true, dtype=np.float64)
    inc = np.diff(t, axis=1) > eps
    return np.concatenate([np.ones((t.shape[0], 1), dtype=bool), inc], axis=1)


def save_errors(path: str, abs_err: np.ndarray, t_true: np.ndarray, valid: np.ndarray) -> None:
    """Save per-grid-point absolute errors, true ages and the padding mask for later pooling."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path,
                        abs_err=np.asarray(abs_err, dtype=np.float32),
                        t_true=np.asarray(t_true, dtype=np.float32),
                        valid=np.asarray(valid, dtype=bool))
    print(f"[saved] {path}")


def q90_per_age_bin(abs_err: np.ndarray, t_true: np.ndarray,
                    edges: np.ndarray = AGE_BIN_EDGES_GYR, q: float = 0.90,
                    valid: Optional[np.ndarray] = None):
    """
    abs_err: (N, N_eval, D) absolute errors
    t_true:  (N, N_eval) true physical age (Gyr) of each grid point
    valid:   optional (N, N_eval) bool mask; only True points enter the statistics
    Returns (table, counts): table (n_bins, D) q-quantile per bin, NaN if empty;
    counts (n_bins,) number of grid points per bin.
    """
    d = abs_err.shape[-1]
    e = abs_err.reshape(-1, d)
    t = np.asarray(t_true, dtype=np.float64).reshape(-1)
    if valid is not None:
        v = np.asarray(valid, dtype=bool).reshape(-1)
        e, t = e[v], t[v]
    idx = np.digitize(t, edges) - 1  # bin j covers [edges[j], edges[j+1])
    n_bins = len(edges) - 1
    table = np.full((n_bins, d), np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for j in range(n_bins):
        m = idx == j
        counts[j] = int(m.sum())
        if counts[j] > 0:
            table[j] = np.quantile(e[m], q, axis=0)
    return table, counts


def print_q90_table(table, counts, edges, titles, q: float = 0.90) -> None:
    d = table.shape[1]
    lbls = _labels(titles, d)
    print(f"\n{q:.2f}-quantile |error| per age bin (dex):")
    header = f"{'age bin [Gyr]':>22s} {'n_pts':>9s} " + " ".join(f"{i:>9d}" for i in range(d))
    print(header)
    for j in range(len(counts)):
        row = f"[{edges[j]:.3g}, {edges[j+1]:.3g})"
        vals = " ".join(f"{table[j, i]:9.3e}" if np.isfinite(table[j, i]) else f"{'--':>9s}" for i in range(d))
        print(f"{row:>22s} {counts[j]:9d} {vals}")
    print("columns: " + ", ".join(f"{i}={lbls[i]}" for i in range(d)))


def save_q90_table_csv(path: str, table, counts, edges, titles) -> None:
    d = table.shape[1]
    lbls = _labels(titles, d)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["age_min_Gyr", "age_max_Gyr", "n_points"] + lbls)
        for j in range(len(counts)):
            w.writerow([f"{edges[j]:.6g}", f"{edges[j+1]:.6g}", counts[j]]
                       + [f"{table[j, i]:.6e}" for i in range(d)])
    print(f"[saved] {path}")


def plot_error_histogram(err_flat: np.ndarray, q90: np.ndarray, titles, path: str,
                         nbins: int = 80) -> None:
    """
    Histogram of |error| per channel on log-spaced bins, shared across channels.
    y = fraction of test grid points per bin (bins are equal in log10|error|).
    """
    d = err_flat.shape[1]
    lbls = _labels(titles, d)
    pos = err_flat[err_flat > 0]
    lo, hi = float(pos.min()), float(pos.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), nbins + 1)

    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, d))
    fig, ax = plt.subplots(figsize=(3.46, 3.2))
    for i in range(d):
        e = err_flat[:, i]
        e = e[e > 0]
        if e.size == 0:
            continue
        ax.hist(e, bins=bins, weights=np.full(e.size, 1.0 / e.size),
                histtype="step", linewidth=1.5, color=colors[i],
                label=fr"{lbls[i]}  (q90={q90[i]:.1e})")
        ax.axvline(q90[i], color=colors[i], linestyle="--", linewidth=0.9, alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|y_{\mathrm{pred}} - y_{\mathrm{true}}|$ [dex]")
    ax.set_ylabel("Fraction of test points per bin")
    ax.legend(fontsize=6.5, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(os.path.splitext(path)[0] + ".pdf")
    plt.close(fig)


def plot_q90_vs_age(table, edges, titles, path: str, q: float = 0.90) -> None:
    """Step plot of the per-age-bin q-quantile error for each channel."""
    d = table.shape[1]
    lbls = _labels(titles, d)
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, d))
    fig, ax = plt.subplots(figsize=(3.46, 3.2))
    for i in range(d):
        ax.stairs(table[:, i], edges, baseline=None, color=colors[i], linewidth=1.5, label=lbls[i])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Age [Gyr]")
    ax.set_ylabel(fr"$q_{{{int(round(q*100))}}}\,|y_{{\mathrm{{pred}}}} - y_{{\mathrm{{true}}}}|$ [dex]")
    ax.legend(fontsize=6.5, frameon=False, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(os.path.splitext(path)[0] + ".pdf")
    plt.close(fig)
