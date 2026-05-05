"""
Figure 4: Coding level (CL) sweep across three allocation strategies.

Runs continual-learning experiments across CL ∈ {0.1, …, 1.0} × seeds {0, 1, 2}
for one --network_type (neurogenesis1 / random_k / topk_noinit). Plotting combines
all three networks by reading results/v2_fig4/<network>/cl<CL>/summary.npz.

Per-CL output: results/v2_fig4/<network>/cl<CL>/summary.npz with
cos_angles, angle_deg, m1m2_l2_dists, m1m2_corr, rand_cos_sim, rand_l2_change.

Index 0 of each array = lag=-1 (pre-train); index i+1 = lag=i. rand_cos_sim[0]=1.0 and
rand_l2_change[0]=0.0 are mathematical identities at index 0; plot those metrics from index 1.

Usage:
  python3 run_fig4_cl_sweep.py --network_type neurogenesis1 --adaptive_lr
  python3 run_fig4_cl_sweep.py --network_type random_k       --adaptive_lr
  python3 run_fig4_cl_sweep.py --network_type topk_noinit
  python3 run_fig4_cl_sweep.py --plot_only
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

CL_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEEDS = [0, 1, 2]

PLOT_CL = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]   # representative CL values

METRIC_KEYS = ["cos_angles", "m1m2_l2_dists", "m1m2_corr", "rand_cos_sim", "rand_l2_change"]
NETWORK_TYPES = ["neurogenesis1", "random_k", "topk_noinit"]
NETWORK_LABEL = {
    "neurogenesis1": "Neurogenesis",
    "random_k": "Random allocation",
    "topk_noinit": "Hebbian",
}

BASE_ARGS = dict(
    n_mitral=400,
    n_granule=2000,
    n_granule_per_task=5,
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
    n_seeds=len(SEEDS),
    seed=0,
    output_dir="results/v2_fig4",
    exp_name="",
    save_plots=False,
)

TOTAL_RUNS = len(CL_VALUES) * len(SEEDS)
LR_ADAPTIVE = False


def _expected_T() -> int:
    return BASE_ARGS["n_train_pairs"] - BASE_ARGS["n_test_pairs"] + 1


def make_args(coding_level: float) -> SimpleNamespace:
    d = dict(BASE_ARGS)
    d["coding_level"] = coding_level
    if LR_ADAPTIVE:
        d["adaptive_lr"] = True
        d["lr_F_max"] = 2.0
        d["lr_B_max"] = 2.0
    return SimpleNamespace(**d)


def _vram_str() -> str:
    try:
        stats = jax.devices()[0].memory_stats()
        used_mb = stats.get("bytes_in_use", 0) / 1024 ** 2
        limit_mb = stats.get("bytes_limit", 0) / 1024 ** 2
        if limit_mb > 0:
            return f"VRAM {used_mb:.0f}/{limit_mb:.0f} MB ({100*used_mb/limit_mb:.1f}%)"
        return f"VRAM {used_mb:.0f} MB"
    except Exception:
        return "VRAM N/A"


def _free_gpu():
    gc.collect()
    try:
        jax.clear_caches()
    except AttributeError:
        pass


def _save_summary(cl_dir: Path, per_seed: dict):
    out = {}
    for k, vals in per_seed.items():
        if vals:
            out[k] = np.stack(vals, axis=0)
    if "cos_angles" in out:
        out["angle_deg"] = np.arccos(np.clip(out["cos_angles"], -1.0, 1.0)) * 180.0 / np.pi
    npz_path = cl_dir / "summary.npz"
    np.savez(npz_path, **out)
    np.save(cl_dir / "cos_angles.npy", out["cos_angles"])
    return out, npz_path


def _load_summary(cl_dir: Path) -> dict:
    return dict(np.load(cl_dir / "summary.npz"))


def run_sweep(network_type: str, out_dir: Path, resume: bool) -> dict:
    all_data: dict = {}
    run_idx = 0
    sweep_start = time.time()
    elapsed_per_run = []
    expected_T = _expected_T()

    for cl_idx, cl in enumerate(CL_VALUES):
        cl_dir = out_dir / f"cl{cl:.1f}"
        cl_dir.mkdir(parents=True, exist_ok=True)
        npz_path = cl_dir / "summary.npz"

        if resume and npz_path.exists():
            data = _load_summary(cl_dir)
            cos_shape = data["cos_angles"].shape
            if cos_shape[1] != expected_T:
                raise ValueError(
                    f"CL={cl:.1f}: existing summary.npz has cos_angles.shape[1]={cos_shape[1]} "
                    f"but expected {expected_T} (n_train - n_test + 1). Delete {cl_dir} to recompute."
                )
            all_data[cl] = data
            run_idx += len(SEEDS)
            print(f"[{cl_idx+1}/{len(CL_VALUES)}] CL={cl:.1f}  SKIPPED (loaded {npz_path})")
            continue

        print(f"\n{'='*64}")
        print(f"[{cl_idx+1}/{len(CL_VALUES)}] CL={cl:.1f}  network={network_type}  "
              f"total progress: {run_idx}/{TOTAL_RUNS}  |  {_vram_str()}")
        print(f"{'='*64}")

        args = make_args(cl)
        args.network_type = network_type
        per_seed = {k: [] for k in METRIC_KEYS}

        for s_idx, seed in enumerate(SEEDS):
            run_idx += 1
            if elapsed_per_run:
                avg = sum(elapsed_per_run) / len(elapsed_per_run)
                remaining = (TOTAL_RUNS - run_idx + 1) * avg
                eta = f"ETA ~{remaining/60:.1f} min"
            else:
                eta = "ETA unknown"

            print(f"\n  ── seed {s_idx+1}/{len(SEEDS)} (seed={seed})  "
                  f"run {run_idx}/{TOTAL_RUNS}  {eta}  |  {_vram_str()}")

            t0 = time.time()
            res = run_single_seed(args, seed)
            elapsed = time.time() - t0
            elapsed_per_run.append(elapsed)

            for k in METRIC_KEYS:
                if k in res:
                    per_seed[k].append(res[k])
            del res
            _free_gpu()
            print(f"  ── seed {seed} done in {elapsed:.1f}s  |  {_vram_str()}")

        data, npz_path = _save_summary(cl_dir, per_seed)
        all_data[cl] = data
        print(f"\n  CL={cl:.1f} saved → {npz_path}  cos_angles.shape={data['cos_angles'].shape}")

    total_elapsed = time.time() - sweep_start
    print(f"\nSweep complete in {total_elapsed/60:.1f} min.")
    return all_data


def per_test_angles(cos_angles: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(cos_angles, -1.0, 1.0)) * 180.0 / np.pi


def compute_forgetting_curve(cos_angles: np.ndarray) -> np.ndarray:
    mean_cos = cos_angles.mean(axis=2)
    return np.arccos(np.clip(mean_cos, -1.0, 1.0)) * 180.0 / np.pi


def compute_nAUC_vs_ref(phi: np.ndarray, phi_ref: np.ndarray) -> float:
    """phi, phi_ref shape: (T,).  Uses lag >= 0 only (skip index 0 = lag=-1)."""
    phi_p = phi[1:]
    ref_p = phi_ref[1:]
    denom = np.sum(ref_p - ref_p[0])
    return float("nan") if abs(denom) < 1e-8 else float(np.sum(phi_p - phi_p[0]) / denom)


def plot_fig4(root_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loaded = {}
    for net in NETWORK_TYPES:
        loaded[net] = {}
        for cl in CL_VALUES:
            npz_path = root_dir / net / f"cl{cl:.1f}" / "summary.npz"
            if npz_path.exists():
                loaded[net][cl] = dict(np.load(npz_path))
            else:
                print(f"  [missing] {net} CL={cl:.1f}: {npz_path}")

    for r in ("neurogenesis1", "random_k"):
        if not loaded.get(r):
            raise FileNotFoundError(f"No data for required network '{r}' under {root_dir}")

    capacity = BASE_ARGS["n_granule"] // BASE_ARGS["n_granule_per_task"]
    any_cl = next(cl for cl in CL_VALUES if cl in loaded["neurogenesis1"])
    T = loaded["neurogenesis1"][any_cl]["cos_angles"].shape[1]
    lag_axis = np.arange(T) - 1

    cmap = plt.cm.viridis
    colors = {cl: cmap(i / max(len(PLOT_CL) - 1, 1)) for i, cl in enumerate(PLOT_CL)}
    net_color = {"neurogenesis1": "#0072B2", "random_k": "#D55E00", "topk_noinit": "#009E73"}
    net_marker = {"neurogenesis1": "o-", "random_k": "s--", "topk_noinit": "^:"}

    has_topk = bool(loaded.get("topk_noinit"))
    n_cols = 4 if has_topk else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4))

    def plot_net_panel(ax, net: str, title: str):
        for cl in PLOT_CL:
            if cl not in loaded[net]:
                continue
            cos_a = loaded[net][cl]["cos_angles"]
            phi_per_seed = compute_forgetting_curve(cos_a)      # (n_seeds, T)
            phi_mean = phi_per_seed.mean(axis=0)
            lo = np.percentile(phi_per_seed, 20, axis=0)
            hi = np.percentile(phi_per_seed, 80, axis=0)
            ax.plot(lag_axis, phi_mean, label=f"CL={cl}", color=colors[cl])
            ax.fill_between(lag_axis, lo, hi, alpha=0.15, color=colors[cl])
        ax.axvline(capacity, color="gray", linestyle="--", linewidth=0.8,
                   label=f"Capacity G/K={capacity}")
        ax.axvline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_xlim(-1, T - 1)
        ax.set_xlabel("Lag (tasks after learning)")
        ax.set_ylabel("Angle (degrees)")
        ax.set_ylim(10, 100)
        ax.set_title(title)
        ax.legend(fontsize=11, loc="lower left")

    plot_net_panel(axes[0], "neurogenesis1", "(a) Neurogenesis — forgetting")
    plot_net_panel(axes[1], "random_k", "(b) Random allocation — forgetting")
    if has_topk:
        plot_net_panel(axes[2], "topk_noinit", "(c) Hebbian — forgetting")
        nauc_ax = axes[3]
        nauc_title = "(d) Normalized AUC vs CL"
    else:
        nauc_ax = axes[2]
        nauc_title = "(c) Normalized AUC vs CL"

    # nAUC panel — reference curve = neurogenesis1 CL=1.0 per seed
    ref_net = "neurogenesis1"
    ref_cl = 1.0
    if ref_cl not in loaded[ref_net]:
        raise FileNotFoundError(f"Missing reference data for nAUC: {ref_net} CL={ref_cl}")
    phi_ref_per_seed = compute_forgetting_curve(loaded[ref_net][ref_cl]["cos_angles"])
    phi_ref_mean = phi_ref_per_seed.mean(axis=0)

    cl_arr = np.array(CL_VALUES)
    for net in NETWORK_TYPES:
        if not loaded.get(net):
            continue
        means, stds = [], []
        for cl in CL_VALUES:
            if cl not in loaded[net]:
                means.append(np.nan)
                stds.append(0.0)
                continue
            phi_per_seed = compute_forgetting_curve(loaded[net][cl]["cos_angles"])
            vals = np.array([compute_nAUC_vs_ref(phi_per_seed[s], phi_ref_mean)
                             for s in range(phi_per_seed.shape[0])])
            means.append(np.nanmean(vals))
            stds.append(np.nanstd(vals) / np.sqrt(max(vals.size, 1)))
        nauc_ax.errorbar(cl_arr, np.array(means), yerr=np.array(stds),
                         fmt=net_marker[net], color=net_color[net], lw=2, ms=6, capsize=3,
                         label=NETWORK_LABEL[net])
    nauc_ax.set_xlabel("Coding level (CL)")
    nauc_ax.set_ylabel("nAUC (vs NG CL=1.0 reference)")
    nauc_ax.set_title(nauc_title)
    nauc_ax.set_xlim(0, 1.05)
    nauc_ax.legend(fontsize=12)

    fig.tight_layout()
    pdf_path = Path("paper_v3/figures/fig4.pdf")
    png_path = Path("paper_v3/figures/fig4.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 4 saved: {pdf_path},  {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Figure 4 coding-level sweep")
    parser.add_argument("--output_root", type=str, default="results/v2_fig4")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override final dir (skips <output_root>/<network_type>)")
    parser.add_argument("--network_type", type=str, default="neurogenesis1",
                        choices=NETWORK_TYPES + ["topk_noinit_v2", "ng1_v2", "random_k_v2"])
    parser.add_argument("--resume", action="store_true",
                        help="Skip CL values whose summary.npz already exists (shape-validated)")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip experiments; load existing summaries and plot")
    parser.add_argument("--adaptive_lr", action="store_true",
                        help="Scale lr_F by 1/c^3.5 and lr_B by 1/c^2 (capped at 2.0)")
    parser.add_argument("--cl_values", type=float, nargs="+", default=None,
                        help="Override default CL sweep (space-separated floats)")
    parser.add_argument("--n_seeds", type=int, default=3,
                        help="Number of seeds.")
    parser.add_argument("--seed_start", type=int, default=0,
                        help="First seed to use; iterates seed_start..seed_start+n_seeds-1")
    args_main = parser.parse_args()

    global LR_ADAPTIVE, CL_VALUES, SEEDS, TOTAL_RUNS
    SEEDS = list(range(args_main.seed_start, args_main.seed_start + args_main.n_seeds))
    BASE_ARGS["n_seeds"] = len(SEEDS)
    TOTAL_RUNS = len(CL_VALUES) * len(SEEDS)
    BASE_ARGS["network_type"] = args_main.network_type
    if args_main.network_type in ("topk_noinit_v2", "ng1_v2", "random_k_v2"):
        BASE_ARGS["lr_B"] = 0.3
        BASE_ARGS["n_train_pairs"] = 1300
        print(f"  [v2] BASE_ARGS overridden: lr_B=0.3, n_train_pairs=1300")
    if args_main.adaptive_lr:
        LR_ADAPTIVE = True
        print("adaptive_lr enabled: lr_F ∝ 1/c^3.5, lr_B ∝ 1/c^2 (max=2.0)")
    if args_main.cl_values is not None:
        CL_VALUES = sorted(args_main.cl_values)

    root = Path(args_main.output_root)
    if args_main.output_dir is not None:
        out_dir = Path(args_main.output_dir)
    else:
        out_dir = root / args_main.network_type

    print(f"JAX devices: {jax.devices()}")
    print(f"JAX version: {jax.__version__}")
    print(f"Total runs planned: {len(CL_VALUES)} × {len(SEEDS)} = {len(CL_VALUES) * len(SEEDS)}")
    print(f"SEEDS: {SEEDS}")
    print(f"output_dir: {out_dir}")

    if not args_main.plot_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        run_sweep(args_main.network_type, out_dir, resume=args_main.resume)

    if args_main.network_type in ("topk_noinit_v2", "ng1_v2", "random_k_v2"):
        print("Done (v2 sweep — plotting skipped).")
    else:
        plot_fig4(root)
        print("Done.")


if __name__ == "__main__":
    main()
