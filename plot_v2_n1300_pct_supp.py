"""Supplementary robustness figure for std vs percentile summaries in Fig 3/4/5.

Layout: 3 rows (K sweep / CL sweep / r_input sweep) x 4 cols.
  col 1: angle median + 20-80 percentile band per sweep value
         (descriptive visualization only; not used as an AUC metric)
  col 2: pooled log(SNR_std)(t) curve, same definition as main-text AUC
  col 3: pooled log(SNR_pct)(t) curve, percentile-noise counterpart
  col 4: AUC[max(log SNR_pct, 0)] vs sweep variable, no error bars

All summaries pooled over `seed × memory` samples from the available seed
directories.
"""
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_metric_utils import (
    load_pooled_summary, seed_sibling_dirs, angle_deg_from_cos, stats_pool,
    snr_std_curve, snr_pct_curve, logsnr, auc_positive_logsnr,
)

mpl.rcParams.update({
    "axes.labelsize": 26, "axes.titlesize": 22,
    "xtick.labelsize": 18, "ytick.labelsize": 18,
    "legend.fontsize": 13, "lines.linewidth": 2.8,
    "figure.titlesize": 28,
})

ROOT = Path("results")
OUT_DIR = ROOT / "v2_n1300_tmp_figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR = Path("paper_v3/figures")

NETS = ["ng1_v2", "random_k_v2", "topk_noinit_v2"]
NET_LABEL = {"ng1_v2": "Neurogenesis", "random_k_v2": "Random allocation", "topk_noinit_v2": "Input-based allocation"}
NET_COLOR = {"ng1_v2": "#0072B2", "random_k_v2": "#D55E00", "topk_noinit_v2": "#009E73"}
NET_MARKER = {"ng1_v2": "o-", "random_k_v2": "s--", "topk_noinit_v2": "^:"}

K_VALUES = [1, 2, 5, 8, 10, 15, 20]
CL_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
CORR_VALUES = [0.7, 0.8, 0.9, 0.95, 0.99]


def corr_subdir(c):
    return f"r{c:.2f}".replace(".", "p")


def _seed_dirs(stem):
    return seed_sibling_dirs(ROOT / stem)


def _n_pooled_seeds(stem):
    return len(_seed_dirs(stem))


def load_sweep(stem, net, sub_fn, sweep_values):
    """For one (sweep_root, net), return dict {sweep_val: stats_pool} for available values."""
    out = {}
    for v in sweep_values:
        d = load_pooled_summary(*[sd / net / sub_fn(v) for sd in _seed_dirs(stem)])
        if d is None or "cos_angles" not in d:
            continue
        ang = angle_deg_from_cos(d["cos_angles"])  # (n_pool, T, n_test)
        out[v] = (stats_pool(ang), ang.shape[1])
    return out


def panel_angle_pct(ax, loaded_net, sweep_values, sweep_label_fn, color_for, T):
    """col 1: angle median + 20-80 percentile band per sweep value."""
    lag = np.arange(T) - 1
    for v in sweep_values:
        if v not in loaded_net:
            continue
        s, _ = loaded_net[v]
        # Pre-train baseline -> median_B; per-lag median plotted from lag=0.
        full_med = np.concatenate([[s["median_B"]], s["median"]])
        full_p20 = np.concatenate([[s["p20_B"]], s["p20"]])
        full_p80 = np.concatenate([[s["p80_B"]], s["p80"]])
        ax.plot(lag, full_med, label=sweep_label_fn(v), color=color_for(v))
        ax.fill_between(lag, full_p20, full_p80, alpha=0.10,
                        color=color_for(v), linewidth=0)
    ax.axvline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlim(-1, T - 1)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Angle (°)")
    ax.set_ylim(0, 92)


def panel_logsnr(ax, loaded_net, sweep_values, sweep_label_fn, color_for, T, *, snr_fn, ylabel):
    """Pooled log(SNR)(t) curve, no band, y-floor at -2."""
    lag_pos = np.arange(T - 1)  # lag = 0 .. T-2
    for v in sweep_values:
        if v not in loaded_net:
            continue
        s, _ = loaded_net[v]
        ax.plot(lag_pos, logsnr(snr_fn(s)), label=sweep_label_fn(v), color=color_for(v))
    ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
    ax.set_xlim(0, T - 2)
    ax.set_ylim(bottom=-2)
    ax.set_xlabel("Lag")
    ax.set_ylabel(ylabel)


def panel_auc_pct(ax, loaded_by_net, sweep_values, sweep_xlabel):
    """col 3: AUC[max(log SNR_pct, 0)] vs sweep variable; no error bars."""
    x = np.array(sweep_values)
    for net in NETS:
        if not loaded_by_net.get(net):
            continue
        aucs = []
        for v in sweep_values:
            if v not in loaded_by_net[net]:
                aucs.append(np.nan)
                continue
            s, _ = loaded_by_net[net][v]
            aucs.append(auc_positive_logsnr(logsnr(snr_pct_curve(s))))
        ax.plot(x, np.array(aucs), NET_MARKER[net], color=NET_COLOR[net],
                lw=2.8, ms=8, label=NET_LABEL[net])
    ax.set_xlabel(sweep_xlabel)
    ax.set_ylabel(r"$\mathrm{AUC}_\mathrm{pct}$")


def sweep_legend_handles(sweep_values, loaded_net, label_fn, color_for):
    return [
        Line2D([0], [0], color=color_for(v), lw=2.8, label=label_fn(v))
        for v in sweep_values
        if v in loaded_net
    ]


def network_legend_handles(loaded_by_net):
    return [
        Line2D([0], [0], color=NET_COLOR[net], marker=NET_MARKER[net][0],
               linestyle=NET_MARKER[net][1:], lw=2.8, ms=8, label=NET_LABEL[net])
        for net in NETS
        if loaded_by_net.get(net)
    ]


def main():
    cmap = plt.cm.viridis
    color_K = {k: cmap(i / max(len(K_VALUES) - 1, 1)) for i, k in enumerate(K_VALUES)}
    color_CL = {cl: cmap(i / max(len(CL_VALUES) - 1, 1)) for i, cl in enumerate(CL_VALUES)}
    color_R = {r: cmap(i / max(len(CORR_VALUES) - 1, 1)) for i, r in enumerate(CORR_VALUES)}

    # Headline-curve net for columns 1--3 of each row: Neurogenesis.
    # Column 4 includes all networks.
    fig3 = {net: load_sweep("v2_n1300_fig3", net, lambda k: f"k{k}", K_VALUES)
            for net in NETS}
    fig4 = {net: load_sweep("v2_n1300_fig4", net, lambda cl: f"cl{cl:.1f}", CL_VALUES)
            for net in NETS}
    fig5 = {net: load_sweep("v2_n1300_fig5", net, corr_subdir, CORR_VALUES)
            for net in NETS}

    def first_T(loaded_by_net):
        for net in NETS:
            for v, (_, T) in (loaded_by_net.get(net) or {}).items():
                return T
        return None

    T3, T4, T5 = first_T(fig3), first_T(fig4), first_T(fig5)
    assert all(T is not None for T in [T3, T4, T5]), "Missing data for at least one sweep."

    fig, axes = plt.subplots(3, 4, figsize=(20, 12.8))

    # Row 1: K sweep
    panel_angle_pct(axes[0, 0], fig3["ng1_v2"], K_VALUES, lambda k: f"K={k}",
                    lambda k: color_K[k], T3)
    axes[0, 0].set_title("(a1) K angle")
    panel_logsnr(axes[0, 1], fig3["ng1_v2"], K_VALUES, lambda k: f"K={k}",
                 lambda k: color_K[k], T3, snr_fn=snr_std_curve,
                 ylabel=r"$\log\,\mathrm{SNR}_\mathrm{std}(t)$")
    axes[0, 1].set_title(r"(a2) K $\log\,\mathrm{SNR}_\mathrm{std}$")
    panel_logsnr(axes[0, 2], fig3["ng1_v2"], K_VALUES, lambda k: f"K={k}",
                 lambda k: color_K[k], T3, snr_fn=snr_pct_curve,
                 ylabel=r"$\log\,\mathrm{SNR}_\mathrm{pct}(t)$")
    axes[0, 2].set_title(r"(a3) K $\log\,\mathrm{SNR}_\mathrm{pct}$")
    panel_auc_pct(axes[0, 3], fig3, K_VALUES, "K")
    axes[0, 3].set_title(r"(a4) AUC$_\mathrm{pct}$")

    # Row 2: CL sweep
    panel_angle_pct(axes[1, 0], fig4["ng1_v2"], CL_VALUES, lambda cl: f"CL={cl}",
                    lambda cl: color_CL[cl], T4)
    axes[1, 0].set_title("(b1) CL angle")
    panel_logsnr(axes[1, 1], fig4["ng1_v2"], CL_VALUES, lambda cl: f"CL={cl}",
                 lambda cl: color_CL[cl], T4, snr_fn=snr_std_curve,
                 ylabel=r"$\log\,\mathrm{SNR}_\mathrm{std}(t)$")
    axes[1, 1].set_title(r"(b2) CL $\log\,\mathrm{SNR}_\mathrm{std}$")
    panel_logsnr(axes[1, 2], fig4["ng1_v2"], CL_VALUES, lambda cl: f"CL={cl}",
                 lambda cl: color_CL[cl], T4, snr_fn=snr_pct_curve,
                 ylabel=r"$\log\,\mathrm{SNR}_\mathrm{pct}(t)$")
    axes[1, 2].set_title(r"(b3) CL $\log\,\mathrm{SNR}_\mathrm{pct}$")
    panel_auc_pct(axes[1, 3], fig4, CL_VALUES, "Coding level")
    axes[1, 3].set_title(r"(b4) AUC$_\mathrm{pct}$")

    # Row 3: r_input sweep
    panel_angle_pct(axes[2, 0], fig5["ng1_v2"], CORR_VALUES,
                    lambda r: rf"$r_\mathrm{{input}}={r}$",
                    lambda r: color_R[r], T5)
    axes[2, 0].set_title(r"(c1) $r_\mathrm{input}$ angle")
    panel_logsnr(axes[2, 1], fig5["ng1_v2"], CORR_VALUES,
                 lambda r: rf"$r_\mathrm{{input}}={r}$",
                 lambda r: color_R[r], T5, snr_fn=snr_std_curve,
                 ylabel=r"$\log\,\mathrm{SNR}_\mathrm{std}(t)$")
    axes[2, 1].set_title(r"(c2) $r_\mathrm{input}$ $\log\,\mathrm{SNR}_\mathrm{std}$")
    panel_logsnr(axes[2, 2], fig5["ng1_v2"], CORR_VALUES,
                 lambda r: rf"$r_\mathrm{{input}}={r}$",
                 lambda r: color_R[r], T5, snr_fn=snr_pct_curve,
                 ylabel=r"$\log\,\mathrm{SNR}_\mathrm{pct}(t)$")
    axes[2, 2].set_title(r"(c3) $r_\mathrm{input}$ $\log\,\mathrm{SNR}_\mathrm{pct}$")
    panel_auc_pct(axes[2, 3], fig5, CORR_VALUES, r"$r_\mathrm{input}$")
    axes[2, 3].set_title(r"(c4) AUC$_\mathrm{pct}$")

    fig.suptitle("Robustness to percentile-based summary statistics")
    fig.tight_layout(rect=[0.055, 0.035, 0.835, 0.95], pad=0.9, w_pad=1.5, h_pad=1.3)

    legend_x = 0.85
    fig.legend(
        handles=sweep_legend_handles(
            K_VALUES, fig3["ng1_v2"], lambda k: f"K={k}", lambda k: color_K[k]
        ),
        title="K sweep", loc="center left", bbox_to_anchor=(legend_x, 0.79),
        ncol=2, frameon=False, fontsize=12, title_fontsize=13,
    )
    fig.legend(
        handles=sweep_legend_handles(
            CL_VALUES, fig4["ng1_v2"], lambda cl: f"CL={cl}", lambda cl: color_CL[cl]
        ),
        title="CL sweep", loc="center left", bbox_to_anchor=(legend_x, 0.49),
        ncol=2, frameon=False, fontsize=12, title_fontsize=13,
    )
    fig.legend(
        handles=sweep_legend_handles(
            CORR_VALUES, fig5["ng1_v2"],
            lambda r: rf"$r_\mathrm{{input}}={r}$", lambda r: color_R[r]
        ),
        title=r"$r_\mathrm{input}$ sweep", loc="center left",
        bbox_to_anchor=(legend_x, 0.245), ncol=1, frameon=False,
        fontsize=12, title_fontsize=13,
    )
    fig.legend(
        handles=network_legend_handles(fig3),
        title="AUC networks", loc="center left", bbox_to_anchor=(legend_x, 0.075),
        ncol=1, frameon=False, fontsize=12, title_fontsize=13,
    )

    pdf_path = OUT_DIR / "fig_pct_supp.pdf"
    png_path = OUT_DIR / "fig_pct_supp.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

    # Copy PDF into the paper figure dir.
    target = PAPER_FIG_DIR / "fig_pct_supp.pdf"
    shutil.copy(pdf_path, target)
    print(f"Copied to: {target}")


if __name__ == "__main__":
    main()
