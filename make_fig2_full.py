"""
Generate Fig2 with all 5 recorder metrics in a 2x5 layout:
  (a) Cosine angle  (b) L2 distance  (c) Pearson corr
  (d) Rand cos sim  (e) Rand L2 change

The top row shows the full long-lag trajectories. The bottom row zooms the
same metrics over lag=-10..20 to make the before/after task-learning transition
visible without covering the main curves. The saved data contain one
pre-learning snapshot at lag=-1, so lags -40..-2 repeat that same baseline for
visual context.

Loads three allocation strategies (v2 = upstream-faithful init: Bernoulli mask,
sqrt(n_mitral/6) granule_thres; lr_B=0.3, n_train=1300) from
  results/v2_n1300_fig2/{ng1_v2, random_k_v2, topk_noinit_v2}/summary.npz
and matching seed sibling directories when present.

Convention for x-axis: saved arrays have shape (n_seeds, T, n_test) where
index 0 along axis 1 = lag=-1 (pre-train snapshot). We plot x = np.arange(T) - 1.
rand_cos_sim[0] and rand_l2_change[0] are mathematical identities
(1.0 and 0.0), so those panels plot from lag=0 onward (i.e. skip index 0).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

from plot_metric_utils import seed_sibling_dirs

mpl.rcParams.update({
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 9.5,
    "lines.linewidth": 1.35,
})

BASE  = Path("results/v2_n1300_fig2")
SEED_DIRS = seed_sibling_dirs(BASE)
PAPER = Path("paper_v3/figures")
PAPER.mkdir(parents=True, exist_ok=True)
KEYS = ["angle_deg", "m1m2_l2_dists", "m1m2_corr", "rand_cos_sim", "rand_l2_change"]
BAND_ALPHA = 0.12
PRE_LAG_MIN = -40
ZOOM_LAG_MIN = -10
ZOOM_LAG_MAX = 20
ZOOM_XTICKS = [-10, 0, 10, 20]


def load_summary(subdir: str):
    parts = []
    for seed_dir in SEED_DIRS:
        npz_path = seed_dir / subdir / "summary.npz"
        if not npz_path.exists():
            print(f"  [missing] {subdir}: {npz_path}")
            continue
        z = np.load(npz_path)
        parts.append({k: z[k] for k in KEYS if k in z.files})
    if not parts:
        return None
    d = {}
    for k in KEYS:
        if all(k in part for part in parts):
            d[k] = np.concatenate([part[k] for part in parts], axis=0)
    print(f"  loaded {subdir}: angle_deg shape {d['angle_deg'].shape} from {len(parts)} seed dirs")
    return d


ng1 = load_summary("ng1_v2")
rk  = load_summary("random_k_v2")
tk  = load_summary("topk_noinit_v2")

if ng1 is None or rk is None:
    raise FileNotFoundError(
        "Required data missing; need ng1_v2 and random_k_v2 under "
        f"{SEED_DIRS}"
    )

T = ng1["angle_deg"].shape[1]
lag = np.arange(T) - 1   # index 0 → lag=-1, index 1 → lag=0
pre_lags = np.arange(PRE_LAG_MIN, -1)
plot_lag = np.concatenate([pre_lags, lag])
LAG_NEG1_IDX = int(np.where(plot_lag == -1)[0][0])
LAG0_IDX = int(np.where(plot_lag == 0)[0][0])
capacity = 2000 // 5     # default K=5


def mean_band(arr):
    """arr: (n_seeds, T, n_last); returns mean ± std over pooled seed×memory."""
    flat = np.transpose(arr, (1, 0, 2)).reshape(arr.shape[1], -1)
    m = flat.mean(axis=1)
    sigma = flat.std(axis=1)
    return m, m - sigma, m + sigma


def extend_prebaseline(y):
    """Repeat the single stored lag=-1 baseline back to PRE_LAG_MIN."""
    return np.concatenate([np.repeat(y[:1], len(pre_lags)), y])


def draw_band(ax, x, lo, hi, color, key):
    lo_plot = lo
    hi_plot = hi
    if key == "angle_deg":
        lo_plot = np.maximum(lo_plot, 0.0)
    ax.fill_between(x, lo_plot, hi_plot, alpha=BAND_ALPHA, color=color, linewidth=0)


def early_ylim(series, key):
    if key == "rand_cos_sim":
        return 0.99995, 1.00005
    if key == "rand_l2_relative":
        return 0.0, 0.0002
    mask = (plot_lag >= ZOOM_LAG_MIN) & (plot_lag <= ZOOM_LAG_MAX)
    vals = np.concatenate([m[mask] for m, *_ in series])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    lo = float(vals.min())
    hi = float(vals.max())
    pad = 0.1 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.02
    lo -= pad
    hi += pad
    if key == "angle_deg":
        lo = max(0.0, lo)
        hi = min(100.0, hi)
    return lo, hi


def derive_l2_relative(d):
    """Approximate ||rand_ms - rand_eval_m0|| / ||rand_eval_m0||
    using cos_sim under the assumption that ||rand_ms|| ≈ ||rand_eval_m0||
    (true at lag=-1 by construction; approximately true thereafter while
    pretrain-anchored statistics are stable).

    chord on hypersphere: rel ≈ sqrt(2 * (1 - cos_sim))
    """
    cos = d["rand_cos_sim"]
    return np.sqrt(np.clip(2.0 * (1.0 - cos), 0.0, None))


C_NG1 = "#0072B2"
C_RK  = "#D55E00"
C_TK  = "#009E73"

fig, axes_grid = plt.subplots(
    2,
    5,
    figsize=(10.2, 4.05),
    gridspec_kw={"height_ratios": [1.7, 0.82]},
)
main_axes = axes_grid[0]
zoom_axes = axes_grid[1]

# Inject a derived 'rand_l2_relative' field per network (cos_sim → chord approximation)
for d in (ng1, rk, tk):
    if d is not None and "rand_cos_sim" in d:
        d["rand_l2_relative"] = derive_l2_relative(d)

panels = [
    ("angle_deg",      "(a) Memory angle",            "Angle (°)",         {"ylim": (10, 92)}, False),
    ("m1m2_l2_dists",  "(b) Pair L2",                 "L2 distance",       {},                  False),
    ("m1m2_corr",      "(c) Pair corr",               "Correlation",       {},                  False),
    ("rand_cos_sim",   "(d) Rand cos",                "Cosine similarity", {"ylim": (0.9, 1.01)}, False),
    ("rand_l2_relative", "(e) Rand rel. L2",
        r"$\|m - m_0\|_2 / \|m_0\|_2$", {"ylim": (-0.01, 0.5)}, False),
]

early_series_by_key = {}

for i, (key, title, ylabel, opts, skip_lag_neg1) in enumerate(panels):
    ax = main_axes[i]
    x = plot_lag

    m_ng_raw, lo_ng_raw, hi_ng_raw = mean_band(ng1[key])
    m_rk_raw, lo_rk_raw, hi_rk_raw = mean_band(rk[key])
    m_ng, lo_ng, hi_ng = map(extend_prebaseline, (m_ng_raw, lo_ng_raw, hi_ng_raw))
    m_rk, lo_rk, hi_rk = map(extend_prebaseline, (m_rk_raw, lo_rk_raw, hi_rk_raw))
    early_series = [(m_ng, C_NG1, "Neurogenesis"), (m_rk, C_RK, "Random allocation")]
    ax.plot(x, m_ng, color=C_NG1, lw=1.45, label="Neurogenesis")
    draw_band(ax, x, lo_ng, hi_ng, C_NG1, key)
    ax.plot(x, m_rk, color=C_RK, lw=1.45, label="Random allocation")
    draw_band(ax, x, lo_rk, hi_rk, C_RK, key)

    if tk is not None and key in tk:
        m_tk_raw, lo_tk_raw, hi_tk_raw = mean_band(tk[key])
        m_tk, lo_tk, hi_tk = map(extend_prebaseline, (m_tk_raw, lo_tk_raw, hi_tk_raw))
        early_series.append((m_tk, C_TK, "Input-based allocation"))
        ax.plot(x, m_tk, color=C_TK, lw=1.45, label="Input-based allocation")
        draw_band(ax, x, lo_tk, hi_tk, C_TK, key)
    early_series_by_key[key] = early_series

    ax.axvline(capacity, color="gray", ls="--", lw=0.8, label=f"Capacity G/K={capacity}")
    ax.axvline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlim(PRE_LAG_MIN, T - 1)
    ax.set_xticks([PRE_LAG_MIN, capacity, T - 1])
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel(ylabel)
    if "ylim" in opts:
        ax.set_ylim(*opts["ylim"])

zoom_titles = [
    "(f) Early angle",
    "(g) Early L2",
    "(h) Early corr",
    "(i) Early rand cos",
    "(j) Early rand L2",
]
zoom_ylabels = ["Angle (°)", "L2", "Corr.", "Cos.", r"Rel. L2 ($10^{-4}$)"]
mask = (plot_lag >= ZOOM_LAG_MIN) & (plot_lag <= ZOOM_LAG_MAX)
for i, (key, *_rest) in enumerate(panels):
    ax = zoom_axes[i]
    series = early_series_by_key[key]
    for m, color, _label in series:
        ax.plot(plot_lag[mask], m[mask], color=color, lw=1.45)
        ax.scatter(
            [-1, 0],
            [m[LAG_NEG1_IDX], m[LAG0_IDX]],
            s=22,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )
    ax.axvline(0, color="k", lw=0.7, alpha=0.45)
    ax.set_xlim(ZOOM_LAG_MIN - 0.4, ZOOM_LAG_MAX + 0.6)
    ylim = early_ylim(series, key)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xticks(ZOOM_XTICKS)
    ax.set_xticklabels(["-10", "0", "10", "20"], rotation=30, ha="right")
    ax.set_title(zoom_titles[i], fontsize=9.5)
    ax.set_xlabel("Early lag", fontsize=8.8)
    ax.set_ylabel(zoom_ylabels[i], fontsize=8.8)
    ax.tick_params(axis="both", labelsize=7.5, length=2.2, pad=1.2)
    if key == "rand_cos_sim":
        ax.set_yticks([0.99995, 1.0, 1.00005])
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    elif key == "rand_l2_relative":
        ax.set_yticks([0.0, 0.0001, 0.0002])
        ax.set_yticklabels(["0", "1", "2"])
    else:
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.locator_params(axis="y", nbins=4)

handles, labels = main_axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02),
           ncol=4, frameon=False, fontsize=9.5)

fig.subplots_adjust(left=0.052, right=0.995, top=0.9, bottom=0.19,
                    wspace=0.47, hspace=0.7)
fig.savefig(PAPER / "fig2.pdf", bbox_inches="tight")
fig.savefig(PAPER / "fig2.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Fig2 saved.")

# Summary (index 1 = lag=0; index -1 = lag=T-2)
def at(d, key, idx):
    if d is None or key not in d:
        return float("nan")
    return float(d[key][:, idx, :].mean())

print("\nKey values (mean over pooled seed×memory samples):")
print(f"  ng1  angle  lag=-1: {at(ng1, 'angle_deg', 0):.1f}°   lag=0: {at(ng1, 'angle_deg', 1):.1f}°   lag={T-2}: {at(ng1, 'angle_deg', -1):.1f}°")
print(f"  rk   angle  lag=-1: {at(rk,  'angle_deg', 0):.1f}°   lag=0: {at(rk,  'angle_deg', 1):.1f}°   lag={T-2}: {at(rk,  'angle_deg', -1):.1f}°")
if tk is not None:
    print(f"  topk angle  lag=-1: {at(tk, 'angle_deg', 0):.1f}°   lag=0: {at(tk, 'angle_deg', 1):.1f}°   lag={T-2}: {at(tk, 'angle_deg', -1):.1f}°")
