"""Appendix histograms of mitral-cell angle at fixed memory ages.

Layout: 3 rows (network) x 5 cols (lag in {0, 50, 200, 400, 800}).
Pooled over `seed × memory` from the Fig 2 baseline condition (K=5, CL=1.0,
r_input=0.9) using the available seed directories. Each panel shows a
50-bin histogram in [0, 95] degrees with a
solid vertical line at the pooled mean and a dashed vertical line at the
pooled median.

Purpose: reconcile percentile-stepped and mean/std-smooth forgetting-curve
appearances by showing the underlying mixture-like distribution at large lag.
"""
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

from plot_metric_utils import load_pooled_summary, seed_sibling_dirs, angle_deg_from_cos

mpl.rcParams.update({
    "axes.labelsize": 24, "axes.titlesize": 22,
    "xtick.labelsize": 16, "ytick.labelsize": 16,
    "legend.fontsize": 16, "lines.linewidth": 2.4,
    "figure.titlesize": 26,
})

ROOT = Path("results")
OUT_DIR = ROOT / "v2_n1300_tmp_figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR = Path("paper_v3/figures")

NETS = ["ng1_v2", "random_k_v2", "topk_noinit_v2"]
NET_LABEL = {"ng1_v2": "Neurogenesis", "random_k_v2": "Random allocation", "topk_noinit_v2": "Input-based allocation"}
ROW_LABEL = {"ng1_v2": "Neurogen.\ncount", "random_k_v2": "Random\ncount", "topk_noinit_v2": "Input-based\ncount"}
NET_COLOR = {"ng1_v2": "#0072B2", "random_k_v2": "#D55E00", "topk_noinit_v2": "#009E73"}

LAGS = [0, 50, 200, 400, 800]
N_BINS = 50
X_RANGE = (0, 95)


def angle_at_lag_pooled(net):
    """Return angle samples at index 0 (lag=-1 baseline excluded) for the Fig 2 condition,
    pooled across available seeds and across memories. Shape after slicing: (T, n_pool)."""
    d = load_pooled_summary(*[sd / net for sd in seed_sibling_dirs(ROOT / "v2_n1300_fig2")])
    if d is None or "cos_angles" not in d:
        return None
    ang = angle_deg_from_cos(d["cos_angles"])  # (n_pool_seeds, T, n_test)
    # T axis index 0 = lag=-1 baseline; index i+1 = lag=i.
    return ang


def main():
    fig, axes = plt.subplots(len(NETS), len(LAGS),
                             figsize=(3.0 * len(LAGS), 3.0 * len(NETS)),
                             sharex=True, sharey=False)

    for r, net in enumerate(NETS):
        ang = angle_at_lag_pooled(net)
        if ang is None:
            print(f"[skip] {net}: no data")
            for c in range(len(LAGS)):
                axes[r, c].set_visible(False)
            continue
        T = ang.shape[1]
        for c, lag in enumerate(LAGS):
            ax = axes[r, c]
            t_index = lag + 1  # index 0 is baseline; lag=i is index i+1
            if t_index >= T:
                ax.set_visible(False)
                continue
            samples = ang[:, t_index, :].ravel()
            ax.hist(samples, bins=N_BINS, range=X_RANGE,
                    color=NET_COLOR[net], alpha=0.85, edgecolor="white", linewidth=0.4)
            mean_v = float(np.mean(samples))
            median_v = float(np.median(samples))
            ax.axvline(mean_v, color="k", linestyle="-", lw=1.2)
            ax.axvline(median_v, color="k", linestyle="--", lw=1.2)
            ax.set_xlim(*X_RANGE)
            ax.text(0.97, 0.93, f"n={samples.size}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=14)
            if r == 0:
                ax.set_title(f"lag = {lag}")
            if c == 0:
                ax.set_ylabel(ROW_LABEL[net])
            if r == len(NETS) - 1:
                ax.set_xlabel("Angle (°)")

    fig.suptitle("Angle distributions at fixed memory ages")
    fig.tight_layout(rect=[0, 0, 1, 0.94], pad=0.8, w_pad=0.55, h_pad=1.05)

    pdf_path = OUT_DIR / "fig_angle_hist_supp.pdf"
    png_path = OUT_DIR / "fig_angle_hist_supp.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

    target = PAPER_FIG_DIR / "fig_angle_hist_supp.pdf"
    shutil.copy(pdf_path, target)
    print(f"Copied to: {target}")


if __name__ == "__main__":
    main()
