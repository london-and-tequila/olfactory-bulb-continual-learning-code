"""
Figure 3: K (granule cells per task) sweep across three allocation strategies.

Runs continual-learning experiments across K ∈ {1, 2, 5, 8, 10, 15, 20} for each
--network_type (neurogenesis1 / random_k / topk_noinit), and plots the 3-panel
figure reading all three networks' summaries from results/v2_fig3/<network>/k<K>/.

Per-K output: results/v2_fig3/<network>/k<K>/summary.npz containing
cos_angles, angle_deg, m1m2_l2_dists, m1m2_corr, rand_cos_sim, rand_l2_change.
Plus a legacy cos_angles.npy (written last).

Usage:
  python3 run_fig3_k_sweep.py --network_type neurogenesis1 --n_seeds 3
  python3 run_fig3_k_sweep.py --network_type random_k       --n_seeds 3
  python3 run_fig3_k_sweep.py --network_type topk_noinit    --n_seeds 3
  python3 run_fig3_k_sweep.py --plot_only   # reads all three networks
"""

import argparse
import gc
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from run_neurogenesis_multi_pair import run_single_seed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

K_VALUES = [1, 2, 5, 8, 10, 15, 20]
PLOT_K = [1, 2, 5, 8, 10, 15, 20]
LEFT_LAG = -40
BAND_ALPHA = 0.06
LEGEND_FONTSIZE = 16

# Index 0 of each saved array = lag=-1 (pre-train snapshot); index i+1 = lag=i.
# Plot convention: m1m2_* metrics include index 0; rand_cos_sim / rand_l2_change
# are mathematical identities at index 0, so they are plotted from index 1 onward.
METRIC_KEYS = ["cos_angles", "m1m2_l2_dists", "m1m2_corr", "rand_cos_sim", "rand_l2_change"]
NETWORK_TYPES = ["ng1_v2", "random_k_v2", "topk_noinit_v2"]
NETWORK_LABEL = {
    "ng1_v2": "Neurogenesis",
    "random_k_v2": "Random allocation",
    "topk_noinit_v2": "Input-based allocation",
}


def extend_baseline_left(lag, phi, std_pool=None, left_lag=LEFT_LAG):
    """Show a flat pre-training baseline from left_lag through lag=-1."""
    baseline_lag = np.arange(left_lag, 0)
    plot_lag = np.concatenate([baseline_lag, lag[1:]])
    plot_phi = np.concatenate([np.full(len(baseline_lag), phi[0]), phi[1:]])
    if std_pool is None:
        return plot_lag, plot_phi
    plot_std = np.concatenate([np.full(len(baseline_lag), std_pool[0]), std_pool[1:]])
    return plot_lag, plot_phi, plot_std

BASE_ARGS = dict(
    n_mitral=400,
    n_granule=2000,
    granule_nonlinear="piecewise_linear",
    mitral_nonlinear="relu",
    tau=5.0,
    lr_F=0.1,
    lr_B=0.1,
    lr_th_g=0.01,
    decay_granule_thres=0.9,
    th_g_hi_ratio=0.95,
    n_pretrain_pairs=500,
    n_train_pairs=800,
    n_test_pairs=300,
    n_epochs_per_pair=300,
    n_steps_to_steady=100,
    correlation=0.9,
    exp_name="",
    save_plots=False,
)


def _free_gpu():
    gc.collect()
    try:
        jax.clear_caches()
    except AttributeError:
        pass


def _expected_T() -> int:
    return BASE_ARGS["n_train_pairs"] - BASE_ARGS["n_test_pairs"] + 1


def _save_summary(k_dir: Path, per_seed: dict):
    """Stack per-seed arrays and save summary.npz + legacy cos_angles.npy (in that order)."""
    out = {}
    for k, vals in per_seed.items():
        if vals:
            out[k] = np.stack(vals, axis=0)
    if "cos_angles" in out:
        out["angle_deg"] = np.arccos(np.clip(out["cos_angles"], -1.0, 1.0)) * 180.0 / np.pi
    npz_path = k_dir / "summary.npz"
    np.savez(npz_path, **out)
    np.save(k_dir / "cos_angles.npy", out["cos_angles"])
    return out, npz_path


def _load_summary(k_dir: Path) -> dict:
    npz_path = k_dir / "summary.npz"
    return dict(np.load(npz_path))


def run_sweep(coding_level: float, n_seeds: int, network_type: str, out_dir: Path,
              resume: bool, seed_start: int = 0) -> dict:
    all_data = {}
    expected_T = _expected_T()
    for k in K_VALUES:
        k_dir = out_dir / f"k{k}"
        k_dir.mkdir(parents=True, exist_ok=True)
        npz_path = k_dir / "summary.npz"

        if resume and npz_path.exists():
            data = _load_summary(k_dir)
            cos_shape = data["cos_angles"].shape
            if cos_shape[1] != expected_T:
                raise ValueError(
                    f"K={k}: existing summary.npz has cos_angles.shape[1]={cos_shape[1]} "
                    f"but expected {expected_T} (n_train - n_test + 1). "
                    f"Delete {k_dir} to recompute."
                )
            all_data[k] = data
            print(f"K={k}: SKIPPED (loaded {npz_path} cos_angles.shape={cos_shape})")
            continue

        print(f"\n{'='*60}")
        print(f"K={k}  network={network_type}")
        print(f"{'='*60}")

        args_dict = dict(BASE_ARGS)
        args_dict["coding_level"] = coding_level
        args_dict["n_granule_per_task"] = k
        args_dict["n_seeds"] = n_seeds
        args_dict["seed"] = seed_start
        args_dict["output_dir"] = str(k_dir)
        args_dict["network_type"] = network_type
        args = SimpleNamespace(**args_dict)

        per_seed = {key: [] for key in METRIC_KEYS}
        for s in range(n_seeds):
            actual_seed = seed_start + s
            print(f"  seed {s+1}/{n_seeds} (seed={actual_seed})")
            t0 = time.time()
            res = run_single_seed(args, actual_seed)
            elapsed = time.time() - t0
            print(f"  done in {elapsed:.1f}s")
            for key in METRIC_KEYS:
                if key in res:
                    per_seed[key].append(res[key])
            del res
            _free_gpu()

        data, npz_path = _save_summary(k_dir, per_seed)
        all_data[k] = data
        print(f"K={k} saved → {npz_path}  cos_angles.shape={data['cos_angles'].shape}")

    return all_data


def per_test_angles(cos_angles: np.ndarray) -> np.ndarray:
    """(n_seeds, T, n_test) → angle_deg in degrees."""
    return np.arccos(np.clip(cos_angles, -1.0, 1.0)) * 180.0 / np.pi


def plot_fig3(root_dir: Path):
    """Load all three networks from root_dir/<network>/k<K>/summary.npz and plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import BoundaryNorm
    from matplotlib.cm import ScalarMappable

    mpl.rcParams.update({
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "lines.linewidth": 2.0,
    })

    # Pool the base seed directory with available _seedN siblings.
    from plot_metric_utils import load_pooled_summary, seed_sibling_dirs
    seed_dirs = seed_sibling_dirs(root_dir)
    print(f"  pooled seed dirs: {[str(d) for d in seed_dirs]}")

    loaded = {}
    for net in NETWORK_TYPES:
        loaded[net] = {}
        for k in K_VALUES:
            d = load_pooled_summary(*[sd / net / f"k{k}" for sd in seed_dirs])
            if d is None:
                # Fall back to legacy single-seed cos_angles.npy.
                legacy = root_dir / net / f"k{k}" / "cos_angles.npy"
                if legacy.exists():
                    loaded[net][k] = {"cos_angles": np.load(legacy)}
                    print(f"  [legacy] {net} K={k}: loaded {legacy}")
                    continue
                print(f"  [missing] {net} K={k}")
                continue
            loaded[net][k] = d

    # Require at least ng1_v2 + random_k_v2; topk_noinit_v2 optional
    required = {"ng1_v2", "random_k_v2"}
    for r in required:
        if not loaded.get(r):
            raise FileNotFoundError(f"No data for required network '{r}' under {root_dir}")

    # Shape check (all K must agree on T)
    any_k = next(k for k in K_VALUES if k in loaded["ng1_v2"])
    T = loaded["ng1_v2"][any_k]["cos_angles"].shape[1]
    # x-axis: index 0 → lag=-1, index i+1 → lag=i
    lag = np.arange(T) - 1

    # Sequential colormap for K, indexed by position in PLOT_K (so visually 1→7 looks ordered)
    cmap = plt.cm.viridis
    n_k = len(PLOT_K)
    colors = {k: cmap(i / max(n_k - 1, 1)) for i, k in enumerate(PLOT_K)}
    net_marker = {"ng1_v2": "o-", "random_k_v2": "s--", "topk_noinit_v2": "^:"}
    net_color = {"ng1_v2": "#0072B2", "random_k_v2": "#D55E00", "topk_noinit_v2": "#009E73"}

    has_topk = bool(loaded.get("topk_noinit_v2"))
    n_sweep = 3 if has_topk else 2

    n_panels = n_sweep + 1
    fig, all_axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels + 0.4, 5.0))
    sweep_axes = list(all_axes[:n_sweep])
    nauc_ax = all_axes[n_sweep]

    def plot_panel(ax, net: str, title: str, legend_loc: str = "upper right",
                   show_band: bool = False):
        legend_handles = []
        legend_labels = []
        for k in PLOT_K:
            if k not in loaded[net]:
                continue
            cos_a = loaded[net][k]["cos_angles"]
            ang = per_test_angles(cos_a)
            # Pooled mean/std across seeds × memories. Transpose first so
            # reshape preserves the lag axis (otherwise reshape interleaves
            # adjacent lags from seed 0 when n_seeds > 1).
            ang_pool = np.transpose(ang, (1, 0, 2)).reshape(ang.shape[1], -1)
            phi = ang_pool.mean(axis=1)                                 # (T,)
            std_pool = ang_pool.std(axis=1)                             # (T,)
            capacity = BASE_ARGS["n_granule"] // k
            plot_lag, plot_phi, plot_std = extend_baseline_left(lag, phi, std_pool)
            label = f"K={k}"
            line, = ax.plot(plot_lag, plot_phi, label=label, color=colors[k])
            if label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)
            if show_band:
                band_lo = np.maximum(plot_phi - plot_std, 0.0)
                band_hi = plot_phi + plot_std
                ax.fill_between(plot_lag, band_lo, band_hi, alpha=BAND_ALPHA,
                                color=colors[k], linewidth=0)
            if capacity <= T - 1:
                ax.axvline(capacity, color=colors[k], linestyle="--", lw=0.8, alpha=0.6)
        ax.axvline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_xlim(LEFT_LAG - 0.5, T - 2)
        ax.set_xlabel("Lag (# tasks after learning)")
        ax.set_ylabel("Angle between representations (°)")
        ax.set_ylim(10, 92)
        ax.set_title(title)
        return legend_handles, legend_labels

    # All forgetting panels show pooled seed × memory bands, with angle axes capped near 90°.
    sweep_handles, sweep_labels = plot_panel(
        sweep_axes[0], "ng1_v2", "(a) Neurogenesis — forgetting",
        legend_loc="lower left", show_band=True
    )
    plot_panel(sweep_axes[1], "random_k_v2", "(b) Random allocation — forgetting",
               show_band=True)
    if has_topk:
        plot_panel(sweep_axes[2], "topk_noinit_v2", "(c) Input-based allocation — forgetting",
                   show_band=True)
        nauc_title = r"(d) AUC of $\max(\log\,\mathrm{SNR}_\mathrm{std}, 0)$ vs K"
    else:
        nauc_title = r"(c) AUC of $\max(\log\,\mathrm{SNR}_\mathrm{std}, 0)$ vs K"

    fig.subplots_adjust(left=0.05, right=0.97, top=0.82, bottom=0.17, wspace=0.35)

    # ── AUC[max(log SNR_std, 0)] vs K panel ──────────────────────────────────
    # Pooled per (network, K) over seed × memory; no error bars.
    from plot_metric_utils import (stats_pool, snr_std_curve, logsnr,
                                   auc_positive_logsnr, angle_deg_from_cos)
    k_arr = np.array(K_VALUES)
    for net in NETWORK_TYPES:
        if not loaded.get(net):
            continue
        aucs = []
        for k in K_VALUES:
            if k not in loaded[net]:
                aucs.append(np.nan)
                continue
            cos_a = loaded[net][k]["cos_angles"]
            angle = angle_deg_from_cos(cos_a)              # (n_pool_seeds, T, n_last)
            s = stats_pool(angle)
            aucs.append(auc_positive_logsnr(logsnr(snr_std_curve(s))))
        nauc_ax.plot(k_arr, np.array(aucs), net_marker[net], color=net_color[net],
                     lw=2, ms=6, label=NETWORK_LABEL[net])
    nauc_ax.set_xlabel("K (granule cells per task)")
    nauc_ax.set_ylabel(r"AUC of $\max(\log\,\mathrm{SNR}_\mathrm{std}, 0)$")
    nauc_ax.set_title(nauc_title)
    nauc_ax.set_xlim(0, 22)
    net_handles, net_labels = nauc_ax.get_legend_handles_labels()
    if sweep_handles:
        fig.legend(
            sweep_handles, sweep_labels, frameon=False,
            loc="upper center", bbox_to_anchor=(0.39, 1.05),
            ncol=7, fontsize=LEGEND_FONTSIZE,
        )
    if net_handles:
        nauc_ax.legend(net_handles, net_labels, frameon=False, loc="upper right",
                       fontsize=11)

    pdf_path = Path("paper_v3/figures/fig_k_sweep.pdf")
    png_path = Path("paper_v3/figures/fig_k_sweep.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Figure 3 K sweep")
    parser.add_argument("--coding_level", type=float, default=1.0)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--seed_start", type=int, default=0,
                        help="First seed to use; iterates seed_start..seed_start+n_seeds-1")
    parser.add_argument("--output_root", type=str, default="results/v2_n1300_fig3",
                        help="Root dir; final output goes to <output_root>/<network_type>/")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override final dir (skips <output_root>/<network_type>)")
    parser.add_argument("--network_type", type=str, default="ng1_v2",
                        choices=NETWORK_TYPES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot_only", action="store_true")
    args_main = parser.parse_args()

    root = Path(args_main.output_root)
    if args_main.output_dir is not None:
        out_dir = Path(args_main.output_dir)
    else:
        out_dir = root / args_main.network_type

    print(f"JAX devices: {jax.devices()}")
    print(f"JAX version: {jax.__version__}")
    print(f"K sweep: {K_VALUES}  CL={args_main.coding_level}  n_seeds={args_main.n_seeds}  "
          f"seed_start={args_main.seed_start}  network={args_main.network_type}")
    print(f"output_dir: {out_dir}")

    if not args_main.plot_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        if args_main.network_type in ("topk_noinit_v2", "ng1_v2", "random_k_v2"):
            BASE_ARGS["lr_B"] = 0.3
            BASE_ARGS["n_train_pairs"] = 1300
            print(f"  [v2] BASE_ARGS overridden: lr_B=0.3, n_train_pairs=1300")
        run_sweep(args_main.coding_level, args_main.n_seeds, args_main.network_type, out_dir,
                  args_main.resume, seed_start=args_main.seed_start)

    plot_fig3(root)
    print("Done.")


if __name__ == "__main__":
    main()
