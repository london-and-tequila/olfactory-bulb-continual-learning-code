"""
Figure 4 (final version):
  Panel a — Forgetting curves of neurogenesis network as a function of CL
  Panel b — Forgetting curves of random-k network as a function of CL
  Panel c — Forgetting curves of topk-noinit (optional, added when data present)
  Panel d — AUC[max(log SNR_std, 0)] vs CL for all available network types

Data source: results/v2_n1300_fig4/<network>/cl<CL>/summary.npz (fallback: cos_angles.npy).

Convention: saved arrays have shape (n_seeds, T, n_test) with index 0 = lag=-1
(pre-train snapshot). Plot x = np.arange(T) - 1; SNR AUC is computed over lag >= 0.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from pathlib import Path

mpl.rcParams.update({
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "lines.linewidth": 2.0,
})

CL_VALUES  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
PLOT_CL    = CL_VALUES
LEFT_LAG = -40
BAND_ALPHA = 0.06
LEGEND_FONTSIZE = 16

ROOT = Path("results/v2_n1300_fig4")
OUT_DIR = Path("paper_v3/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAPACITY = 2000 // 5  # = 400

NETWORK_TYPES = ["ng1_v2", "random_k_v2", "topk_noinit_v2"]
NETWORK_LABEL = {
    "ng1_v2": "Neurogenesis",
    "random_k_v2": "Random allocation",
    "topk_noinit_v2": "Input-based allocation",
}
NET_COLOR = {"ng1_v2": "#0072B2", "random_k_v2": "#D55E00", "topk_noinit_v2": "#009E73"}
NET_MARKER = {"ng1_v2": "o-", "random_k_v2": "s--", "topk_noinit_v2": "^:"}


def extend_baseline_left(lag, phi, std_pool=None, left_lag=LEFT_LAG):
    """Show a flat pre-training baseline from left_lag through lag=-1."""
    baseline_lag = np.arange(left_lag, 0)
    plot_lag = np.concatenate([baseline_lag, lag[1:]])
    plot_phi = np.concatenate([np.full(len(baseline_lag), phi[0]), phi[1:]])
    if std_pool is None:
        return plot_lag, plot_phi
    plot_std = np.concatenate([np.full(len(baseline_lag), std_pool[0]), std_pool[1:]])
    return plot_lag, plot_phi, plot_std


from plot_metric_utils import (load_pooled_summary, seed_sibling_dirs, stats_pool,
                               snr_std_curve, logsnr, auc_positive_logsnr,
                               angle_deg_from_cos)

# Pool the base seed directory with available _seedN siblings.
SEED_DIRS = seed_sibling_dirs(ROOT)
print(f"pooled seed dirs: {[str(d) for d in SEED_DIRS]}")


def load_network(net: str) -> dict:
    """Return {cl: cos_angles (n_seeds_pool, T, n_test)} for available CL values."""
    data = {}
    for cl in CL_VALUES:
        d = load_pooled_summary(*[sd / net / f"cl{cl:.1f}" for sd in SEED_DIRS])
        if d is not None and "cos_angles" in d:
            data[cl] = d["cos_angles"]
            continue
        # Fall back to legacy single-seed cos_angles.npy.
        legacy = ROOT / net / f"cl{cl:.1f}" / "cos_angles.npy"
        if legacy.exists():
            data[cl] = np.load(legacy)
    if data:
        print(f"  {net}: loaded {len(data)}/{len(CL_VALUES)} CL values")
    return data


def per_test_angles(cl_cos: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(cl_cos, -1.0, 1.0)) * 180.0 / np.pi


loaded = {net: load_network(net) for net in NETWORK_TYPES}
for r in ("ng1_v2", "random_k_v2"):
    if not loaded[r]:
        raise FileNotFoundError(f"No v2 data for required network '{r}' under {ROOT}")

any_cl = next(cl for cl in CL_VALUES if cl in loaded["ng1_v2"])
T = loaded["ng1_v2"][any_cl].shape[1]
lag_axis = np.arange(T) - 1

cmap = plt.cm.viridis
n_cl = len(PLOT_CL)
colors = {cl: cmap(i / max(n_cl - 1, 1)) for i, cl in enumerate(PLOT_CL)}

has_topk = bool(loaded["topk_noinit_v2"])
n_sweep = 3 if has_topk else 2

n_panels = n_sweep + 1
fig, all_axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels + 0.4, 5.2))
sweep_axes = list(all_axes[:n_sweep])
nauc_ax = all_axes[n_sweep]


def plot_net_panel(ax, net: str, title: str, legend_loc: str = "upper right",
                   show_band: bool = False):
    data = loaded[net]
    legend_handles = []
    legend_labels = []
    for cl in PLOT_CL:
        if cl not in data:
            continue
        ang = per_test_angles(data[cl])                              # (n_seeds, T, n_test)
        # Transpose before reshape to preserve the lag axis when n_seeds > 1.
        ang_pool = np.transpose(ang, (1, 0, 2)).reshape(ang.shape[1], -1)
        phi = ang_pool.mean(axis=1)                                  # (T,)
        std_pool = ang_pool.std(axis=1)                              # (T,)
        plot_lag, plot_phi, plot_std = extend_baseline_left(lag_axis, phi, std_pool)
        label = f"CL={cl}"
        line, = ax.plot(plot_lag, plot_phi, label=label, color=colors[cl])
        if label not in legend_labels:
            legend_handles.append(line)
            legend_labels.append(label)
        if show_band:
            band_lo = np.maximum(plot_phi - plot_std, 0.0)
            band_hi = plot_phi + plot_std
            ax.fill_between(plot_lag, band_lo, band_hi, alpha=BAND_ALPHA,
                            color=colors[cl], linewidth=0)
    ax.axvline(CAPACITY, color="gray", linestyle="--", lw=0.9)
    ax.axvline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlim(LEFT_LAG - 0.5, T - 2)
    ax.set_xlabel("Lag (# tasks after learning)")
    ax.set_ylabel("Angle between representations (°)")
    ax.set_ylim(10, 92)
    ax.set_title(title)
    return legend_handles, legend_labels


# All forgetting panels show pooled seed × memory bands, with angle axes capped near 90°.
sweep_handles, sweep_labels = plot_net_panel(
    sweep_axes[0], "ng1_v2", "(a) Neurogenesis — forgetting",
    legend_loc="lower left", show_band=True
)
plot_net_panel(sweep_axes[1], "random_k_v2", "(b) Random allocation — forgetting",
               show_band=True)
if has_topk:
    plot_net_panel(sweep_axes[2], "topk_noinit_v2", "(c) Input-based allocation — forgetting",
                   show_band=True)
    nauc_title = r"(d) AUC of $\max(\log\,\mathrm{SNR}_\mathrm{std}, 0)$ vs CL"
else:
    nauc_title = r"(c) AUC of $\max(\log\,\mathrm{SNR}_\mathrm{std}, 0)$ vs CL"

fig.subplots_adjust(left=0.05, right=0.97, top=0.82, bottom=0.17, wspace=0.35)

cl_arr = np.array(CL_VALUES)
for net in NETWORK_TYPES:
    if not loaded[net]:
        continue
    aucs = []
    for cl in CL_VALUES:
        if cl not in loaded[net]:
            aucs.append(np.nan)
            continue
        ang = angle_deg_from_cos(loaded[net][cl])  # (n_pool_seeds, T, n_test)
        s = stats_pool(ang)
        aucs.append(auc_positive_logsnr(logsnr(snr_std_curve(s))))
    nauc_ax.plot(cl_arr, np.array(aucs), NET_MARKER[net], color=NET_COLOR[net],
                 lw=2, ms=6, label=NETWORK_LABEL[net])
nauc_ax.set_xlabel("Coding level (CL)")
nauc_ax.set_ylabel(r"AUC of $\max(\log\,\mathrm{SNR}_\mathrm{std}, 0)$")
nauc_ax.set_title(nauc_title)
nauc_ax.set_xlim(0.05, 1.05)
net_handles, net_labels = nauc_ax.get_legend_handles_labels()
if sweep_handles:
    fig.legend(
        sweep_handles, sweep_labels, frameon=False,
        loc="upper center", bbox_to_anchor=(0.39, 1.05),
        ncol=5, fontsize=LEGEND_FONTSIZE,
    )
if net_handles:
    nauc_ax.legend(net_handles, net_labels, frameon=False, loc="upper left",
                   fontsize=11)

pdf_path = OUT_DIR / "fig_coding_sparsity.pdf"
png_path = OUT_DIR / "fig_coding_sparsity.png"
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {pdf_path}")
print(f"Saved: {png_path}")

# Summary table
header = f"{'CL':>6}  " + "  ".join(f"{NETWORK_LABEL[n]+ ' logSNR AUC':>22}" for n in NETWORK_TYPES if loaded[n])
print(f"\n{header}")
for cl in CL_VALUES:
    row = [f"{cl:>6.1f}"]
    for net in NETWORK_TYPES:
        if not loaded[net]:
            continue
        if cl not in loaded[net]:
            row.append(f"{'n/a':>22}")
            continue
        ang = angle_deg_from_cos(loaded[net][cl])
        s = stats_pool(ang)
        val = auc_positive_logsnr(logsnr(snr_std_curve(s)))
        row.append(f"{val:>22.4f}")
    print("  ".join(row))
