"""
Generate supplementary 5-metric forgetting figures for K / CL / ρ sweeps.

Mirrors the 1×5 panel layout of make_fig2_full.py: (a) angle, (b) M1-M2 L2,
(c) M1-M2 Pearson, (d) cos sim to random, (e) chord-approx relative L2.
Each supp figure plots one (param, value) configuration with the three
network types (neurogenesis1, random_k, topk_noinit) overlaid.

Data dumps already exist at:
  results/v2_n1300_fig3{,_seed1,_seed2,_seed3}/<net>/k<K>/summary.npz       (K-sweep)
  results/v2_n1300_fig4{,_seed1,_seed2,_seed3}/<net>/cl<CL>/summary.npz     (CL-sweep)
  results/v2_n1300_fig5{,_seed1,_seed2,_seed3}/<net>/r<rho_p>/summary.npz   (ρ-sweep)

Output: paper_v3/figures/figS_<param><value>.{pdf,png}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_metric_utils import load_pooled_summary, seed_sibling_dirs

mpl.rcParams.update({
    "axes.labelsize": 24,
    "axes.titlesize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 12,
    "lines.linewidth": 2.6,
})

KEYS = ["angle_deg", "m1m2_l2_dists", "m1m2_corr", "rand_cos_sim"]

NETS = ("neurogenesis1", "random_k", "topk_noinit")
NET_LABEL = {
    "neurogenesis1": "Neurogenesis",
    "random_k": "Random allocation",
    "topk_noinit": "Input-based allocation",
}
NET_COLOR = {
    "neurogenesis1": "#0072B2",
    "random_k": "#D55E00",
    "topk_noinit": "#009E73",
}

PAPER = Path("paper_v3/figures")
RESULTS = Path("results")

SOURCE_ROOT = {
    "K": RESULTS / "v2_n1300_fig3",
    "CL": RESULTS / "v2_n1300_fig4",
    "rho": RESULTS / "v2_n1300_fig5",
}

SOURCE_NET_DIR = {
    "neurogenesis1": "ng1_v2",
    "random_k": "random_k_v2",
    "topk_noinit": "topk_noinit_v2",
}


# ---- directory + filename helpers -------------------------------------------------

def _k_dir(K: int) -> str:
    return f"k{K}"


def _cl_dir(cl: float) -> str:
    return f"cl{cl:.1f}"


def _corr_dir(rho: float) -> str:
    # mirrors run_fig5_corr_sweep._corr_dir
    return f"r{rho:.2f}".replace(".", "p")


def _output_stem(param: str, value) -> str:
    if param == "K":
        return f"figS_K{int(value)}"
    if param == "CL":
        return f"figS_CL{value:.1f}".replace(".", "p")
    if param == "rho":
        return f"figS_rho{value:.2f}".replace(".", "p")
    raise ValueError(param)


def _data_subdir(param: str, value) -> str:
    if param == "K":
        return _k_dir(int(value))
    if param == "CL":
        return _cl_dir(value)
    if param == "rho":
        return _corr_dir(value)
    raise ValueError(param)


# ---- numerics ---------------------------------------------------------------------

def mean_band(arr: np.ndarray):
    """arr: (n_seeds, T, n_last); return mean ± pooled seed-by-memory std."""
    flat = np.transpose(arr, (1, 0, 2)).reshape(arr.shape[1], -1)
    m = flat.mean(axis=1)
    sigma = flat.std(axis=1)
    return m, m - sigma, m + sigma


def derive_l2_relative(rand_cos_sim: np.ndarray) -> np.ndarray:
    """Chord approximation: ||m - m0|| / ||m0|| ≈ sqrt(2 (1 - cos))."""
    return np.sqrt(np.clip(2.0 * (1.0 - rand_cos_sim), 0.0, None))


def load_summary(param: str, value, net: str):
    dirs = [
        seed_dir / SOURCE_NET_DIR[net] / _data_subdir(param, value)
        for seed_dir in seed_sibling_dirs(SOURCE_ROOT[param])
    ]
    loaded = load_pooled_summary(*dirs)
    if loaded is None:
        return None
    d = {k: loaded[k] for k in KEYS if k in loaded}
    if "rand_cos_sim" in d:
        d["rand_l2_relative"] = derive_l2_relative(d["rand_cos_sim"])
    return d


# ---- rendering --------------------------------------------------------------------

PANELS = [
    ("angle_deg",        "(a) Memory angle",            "Angle (°)",         {"ylim": (10, 100)},  False),
    ("m1m2_l2_dists",    "(b) L2 distance (m1 vs m2)",  "L2 distance",       {},                   False),
    ("m1m2_corr",        "(c) Pearson corr (m1 vs m2)", "Pearson r",         {},                   False),
    ("rand_cos_sim",     "(d) Rand pattern cos sim",    "Rand cos sim",      {"ylim": (0.9, 1.01)}, False),
    ("rand_l2_relative", "(e) Rand pattern L2 relative change",
        "Rand L2 rel. change", {"ylim": (-0.01, 0.5)}, False),
]

# Short metric labels used in grid panel titles (full names from PANELS are used in 1×5 mode).
COL_LETTERS = ["a", "b", "c", "d", "e"]
METRIC_SHORT = {
    "angle_deg":        "Angle",
    "m1m2_l2_dists":    "L2",
    "m1m2_corr":        "Pearson corr",
    "rand_cos_sim":     "Rand cos",
    "rand_l2_relative": "Rand L2",
}


def render_5panel(net_data: dict, *, capacity, out_pdf: Path, out_png: Path):
    """Render a 1x5 figure overlaying available networks.

    net_data: {net_name: dict-of-arrays or None}
    capacity: int or None. If None, omit the capacity axvline.
    """
    any_net = next((d for d in net_data.values() if d is not None), None)
    if any_net is None:
        raise RuntimeError(f"All networks missing for {out_pdf.stem}")
    T = any_net["angle_deg"].shape[1]
    lag = np.arange(T) - 1

    fig, axes = plt.subplots(1, 5, figsize=(18.5, 5.0))
    for i, (key, title, ylabel, opts, skip_neg1) in enumerate(PANELS):
        ax = axes[i]
        x = lag[1:] if skip_neg1 else lag
        sl = slice(1, None) if skip_neg1 else slice(None)

        for net in NETS:
            d = net_data.get(net)
            if d is None or key not in d:
                continue
            m, lo, hi = mean_band(d[key])
            ax.plot(x, m[sl], color=NET_COLOR[net], lw=2.6, label=NET_LABEL[net])
            ax.fill_between(x, lo[sl], hi[sl], alpha=0.30, color=NET_COLOR[net], linewidth=0)

        if capacity is not None:
            ax.axvline(capacity, color="gray", ls="--", lw=1.5,
                       label=f"Capacity G/K={capacity}")
        ax.axvline(0, color="k", lw=1.0, alpha=0.4)
        ax.set_xlim(-1, T - 1)
        ax.set_title(title)
        ax.set_xlabel("Lag")
        ax.set_ylabel(ylabel)
        if "ylim" in opts:
            ax.set_ylim(*opts["ylim"])

    legend_handles = [
        Line2D([0], [0], color=NET_COLOR[net], lw=2.6, label=NET_LABEL[net])
        for net in NETS
        if net_data.get(net) is not None
    ]
    if capacity is not None:
        legend_handles.append(
            Line2D([0], [0], color="gray", ls="--", lw=1.5, label="Capacity G/K")
        )

    fig.tight_layout(rect=[0, 0.13, 1, 1], pad=0.8, w_pad=0.8)
    if legend_handles:
        fig.legend(
            handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.025),
            ncol=len(legend_handles), frameon=False, fontsize=14,
        )
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_grid(rows, *, out_pdf: Path, out_png: Path, angle_ylim=(10, 100)):
    """Render an n_rows × 5 grid; each row is one (param-value) configuration.

    rows: list of (row_label, capacity, net_data_dict).
        row_label: short string shown on the leftmost panel (e.g., "K=10").
        capacity: int or None — vertical capacity line position.
        net_data_dict: {net_name: dict-of-arrays or None}.
    """
    n_rows = len(rows)
    any_d = next(
        (d for _, _, nd in rows for d in nd.values() if d is not None), None
    )
    if any_d is None:
        raise RuntimeError(f"All networks missing across all rows for {out_pdf.stem}")
    T = any_d["angle_deg"].shape[1]
    lag = np.arange(T) - 1

    fig, axes = plt.subplots(n_rows, 5, figsize=(18.5, 4.05 * n_rows), squeeze=False)
    for r_idx, (label, capacity, net_data) in enumerate(rows):
        is_top = r_idx == 0
        is_bot = r_idx == n_rows - 1
        row_num = r_idx + 1  # 1-indexed
        for i, (key, _full_title, ylabel, opts, skip_neg1) in enumerate(PANELS):
            ax = axes[r_idx, i]
            col_letter = COL_LETTERS[i]
            panel_id = f"({col_letter}{row_num})"
            x = lag[1:] if skip_neg1 else lag
            sl = slice(1, None) if skip_neg1 else slice(None)

            for net in NETS:
                d = net_data.get(net)
                if d is None or key not in d:
                    continue
                m, lo, hi = mean_band(d[key])
                ax.plot(x, m[sl], color=NET_COLOR[net], lw=2.6, label=NET_LABEL[net])
                ax.fill_between(x, lo[sl], hi[sl], alpha=0.30,
                                color=NET_COLOR[net], linewidth=0)

            if capacity is not None:
                ax.axvline(capacity, color="gray", ls="--", lw=1.5,
                           label=f"Capacity G/K={capacity}")
            ax.axvline(0, color="k", lw=1.0, alpha=0.4)
            ax.set_xlim(-1, T - 1)
            # Two-line title: panel id + metric name on line 1, row indicator on line 2
            ax.set_title(f"{panel_id} {METRIC_SHORT[key]}\n{label}")
            if is_bot:
                ax.set_xlabel("Lag")
            else:
                ax.tick_params(labelbottom=False)
            ax.set_ylabel(ylabel)
            if "ylim" in opts:
                ax.set_ylim(*opts["ylim"])
            if key == "angle_deg":
                ax.set_ylim(*angle_ylim)

    legend_handles = [
        Line2D([0], [0], color=NET_COLOR[net], lw=2.6, label=NET_LABEL[net])
        for net in NETS
    ]
    if any(capacity is not None for _, capacity, _ in rows):
        legend_handles.append(
            Line2D([0], [0], color="gray", ls="--", lw=1.5, label="Capacity G/K")
        )

    fig.tight_layout(rect=[0, 0.08, 1, 1], pad=0.8, w_pad=0.7, h_pad=1.0)
    fig.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=len(legend_handles), frameon=False, fontsize=15,
    )
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- configurations ---------------------------------------------------------------

CONFIGS = [
    # (sweep, value, capacity_for_axvline)
    # K-sweep: capacity = G/K = 2000/K, omit when out of the plotted lag window.
    ("K", 1, None),     # G/K=2000
    ("K", 2, 1000),     # G/K=1000, at the right edge of the n_train=1300 window
    ("K", 10, 200),
    ("K", 15, 2000 // 15),
    # CL-sweep: K=5, capacity=400
    ("CL", 0.1, 400),
    ("CL", 0.3, 400),
    ("CL", 0.5, 400),
    ("CL", 0.7, 400),
    # ρ-sweep: K=5, capacity=400
    ("rho", 0.70, 400),
    ("rho", 0.80, 400),
    ("rho", 0.95, 400),
    ("rho", 0.99, 400),
]


def _row_label(param: str, value) -> str:
    if param == "K":
        return f"K={int(value)}"
    if param == "CL":
        return f"CL={value:.1f}"
    if param == "rho":
        return rf"$\rho$={value:.2f}"
    raise ValueError(param)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweeps",
        default="K,CL,rho",
        help="Comma-separated subset of {K, CL, rho} to render. Default: all 12.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Render one merged 4×5 grid per sweep instead of separate 1×5 figures.",
    )
    args = parser.parse_args()
    requested = {s.strip() for s in args.sweeps.split(",") if s.strip()}
    bad = requested - {"K", "CL", "rho"}
    if bad:
        raise SystemExit(f"Unknown sweep(s): {sorted(bad)}; choose from K, CL, rho")

    if args.merge:
        for sweep in ["K", "CL", "rho"]:
            if sweep not in requested:
                continue
            rows = []
            missing_any = []
            for param, value, capacity in CONFIGS:
                if param != sweep:
                    continue
                net_data = {net: load_summary(param, value, net) for net in NETS}
                missing = [n for n, d in net_data.items() if d is None]
                if missing:
                    missing_any.append((value, missing))
                    continue
                rows.append((_row_label(param, value), capacity, net_data))
            if missing_any:
                for value, miss in missing_any:
                    print(f"[skip-row] {sweep}={value}: missing {miss}")
            if not rows:
                print(f"[skip] {sweep}: no complete rows")
                continue
            stem = f"figS_{sweep}_grid"
            # rho values >= 0.95 collapse to ~4°; widen angle ylim to keep them visible.
            angle_ylim = (0, 100) if sweep == "rho" else (10, 100)
            render_grid(
                rows,
                out_pdf=PAPER / f"{stem}.pdf",
                out_png=PAPER / f"{stem}.png",
                angle_ylim=angle_ylim,
            )
            print(f"[ok]   {stem}.pdf  rows={len(rows)}")
        return

    rendered, skipped = [], []
    for param, value, capacity in CONFIGS:
        if param not in requested:
            continue
        stem = _output_stem(param, value)
        net_data = {net: load_summary(param, value, net) for net in NETS}
        missing = [net for net, d in net_data.items() if d is None]
        if missing:
            print(f"[skip] {stem}: missing summary.npz for {missing}")
            skipped.append((stem, missing))
            continue
        out_pdf = PAPER / f"{stem}.pdf"
        out_png = PAPER / f"{stem}.png"
        render_5panel(net_data, capacity=capacity, out_pdf=out_pdf, out_png=out_png)
        print(f"[ok]   {stem}.pdf  capacity={capacity}")
        rendered.append(stem)

    print(f"\nRendered {len(rendered)} / requested {sum(1 for c in CONFIGS if c[0] in requested)}.")
    if skipped:
        print("Skipped:")
        for stem, miss in skipped:
            print(f"  - {stem}: {miss}")


if __name__ == "__main__":
    main()
