"""
Figure 2: Baseline neurogenesis continual learning experiment.

Runs multi-pair experiment at a single CL (default 1.0) with n_seeds=3.
Generates a single-panel forgetting curve.

NOTE (legacy): The canonical paper fig2 (5 panels: angle, L2, Pearson,
rand-cos, rand-L2) is produced by `make_fig2_full.py`. This script writes
to the same `paper_v3/figures/fig2.pdf` path and will overwrite it — do
not run this script before the paper compile unless you re-run
`make_fig2_full.py` afterwards.

Usage:
  python3 run_fig2_baseline.py [--coding_level 1.0] [--n_seeds 3]
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
    exp_name="",
    save_plots=False,
)


def _free_gpu():
    gc.collect()
    try:
        jax.clear_caches()
    except AttributeError:
        pass


METRIC_KEYS = ["cos_angles", "m1m2_l2_dists", "m1m2_corr", "rand_cos_sim", "rand_l2_change"]


def run_fig2(coding_level: float, n_seeds: int, out_dir: Path, seed_start: int = 0):
    """Run n_seeds × single-CL experiment and save summary.npz + legacy cos_angles.npy.

    Seeds iterated: seed_start, seed_start+1, ..., seed_start+n_seeds-1.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "summary.npz"
    legacy_path = out_dir / "cos_angles.npy"

    args_dict = dict(BASE_ARGS)
    args_dict["coding_level"] = coding_level
    args_dict["n_seeds"] = n_seeds
    args_dict["seed"] = seed_start
    args_dict["output_dir"] = str(out_dir)
    args = SimpleNamespace(**args_dict)

    per_seed = {k: [] for k in METRIC_KEYS}
    for s in range(n_seeds):
        seed = seed_start + s
        print(f"\n--- Seed {s+1}/{n_seeds} (seed={seed}) ---")
        t0 = time.time()
        res = run_single_seed(args, seed)
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.1f}s")
        for k in METRIC_KEYS:
            if k in res:
                per_seed[k].append(res[k])
        del res
        _free_gpu()

    out = {}
    for k, vals in per_seed.items():
        if vals:
            out[k] = np.stack(vals, axis=0)
    out["angle_deg"] = np.arccos(np.clip(out["cos_angles"], -1.0, 1.0)) * 180.0 / np.pi

    np.savez(npz_path, **out)
    print(f"\nSaved: {npz_path}  cos_angles.shape={out['cos_angles'].shape}")
    np.save(legacy_path, out["cos_angles"])  # kept for legacy readers
    return out["cos_angles"]


def plot_fig2(cl_cos: np.ndarray, out_dir: Path, coding_level: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_seeds, T, n_test = cl_cos.shape
    # Per-test angles (n_seeds, T, n_test) — kept for SEM bands across test patterns
    per_test_ang = np.arccos(np.clip(cl_cos, -1.0, 1.0)) * 180.0 / np.pi
    # Per-seed mean curves (n_seeds, T) — kept for plotting individual seed traces
    mean_cos = cl_cos.mean(axis=2)
    angle_deg = np.arccos(np.clip(mean_cos, -1.0, 1.0)) * 180.0 / np.pi

    grand_mean = per_test_ang.mean(axis=(0, 2))                                   # (T,)
    sem = (per_test_ang.std(axis=2) / np.sqrt(n_test)).mean(axis=0)               # (T,) across-test SEM
    # x-axis: index 0 = lag=-1 (pre-training baseline), index i+1 = lag=i
    lag = np.arange(T) - 1
    capacity = BASE_ARGS["n_granule"] // BASE_ARGS["n_granule_per_task"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(lag, grand_mean - sem, grand_mean + sem, alpha=0.30,
                    color="steelblue", linewidth=0,
                    label=f"±SEM (n={n_test} test patterns)")
    for s in range(n_seeds):
        ax.plot(lag, angle_deg[s], lw=0.7, alpha=0.4)
    ax.plot(lag, grand_mean, lw=2, color="steelblue",
            label=f"Mean (n={n_seeds} seeds)")
    ax.axvline(capacity, color="gray", linestyle="--", lw=0.9,
               label=f"Capacity G/K={capacity}")
    ax.set_xlabel("Lag (tasks after learning)")
    ax.set_ylabel("Memory angle (degrees)")
    ax.set_title(f"Fig 2: Neurogenesis memory  CL={coding_level}")
    ax.legend(fontsize=12)
    fig.tight_layout()

    network_type = BASE_ARGS.get("network_type", "neurogenesis1")
    png_path = out_dir / "fig2.png"
    fig.savefig(png_path, dpi=150)
    print(f"Saved: {png_path}")
    # Only the canonical neurogenesis1 run writes to paper_v3/figures/fig2.pdf.
    # Other network_types (incl. topk_noinit_v2) keep their PNG under their own
    # results dir and leave the paper figure intact.
    if network_type == "neurogenesis1":
        pdf_path = Path("paper_v3/figures/fig2.pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved: {pdf_path}")
    plt.close(fig)

    print(f"Grand mean at lag=-1:  {grand_mean[0]:.2f}°  (pre-train baseline)")
    print(f"Grand mean at lag=0:   {grand_mean[1]:.2f}°")
    print(f"Grand mean at lag={T-2}: {grand_mean[-1]:.2f}°")
    print(f"Peak (lag≥0):          {grand_mean[1:].max():.2f}° at lag={grand_mean[1:].argmax()}")


def main():
    parser = argparse.ArgumentParser(description="Figure 2 baseline experiment")
    parser.add_argument("--coding_level", type=float, default=1.0)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--seed_start", type=int, default=0,
                        help="First seed to use; iterates seed_start..seed_start+n_seeds-1")
    # Default now writes under results/v2_fig2/<network>/, keeping legacy results/fig2* untouched.
    parser.add_argument("--output_root", type=str, default="results/v2_fig2",
                        help="Root dir; final output goes to <output_root>/<network_type>/")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override final dir (skips <output_root>/<network_type> layout)")
    parser.add_argument("--network_type", type=str, default="neurogenesis1",
                        choices=["neurogenesis1", "random_k", "topk_noinit",
                                 "topk_noinit_v2", "ng1_v2", "random_k_v2"])
    parser.add_argument("--plot_only", action="store_true")
    args_main = parser.parse_args()

    if args_main.output_dir is not None:
        out_dir = Path(args_main.output_dir)
    else:
        out_dir = Path(args_main.output_root) / args_main.network_type

    print(f"JAX devices: {jax.devices()}")
    print(f"JAX version: {jax.__version__}")
    print(f"CL={args_main.coding_level}, n_seeds={args_main.n_seeds}, "
          f"seed_start={args_main.seed_start}, network={args_main.network_type}")
    print(f"output_dir: {out_dir}")

    BASE_ARGS["network_type"] = args_main.network_type
    if args_main.network_type in ("topk_noinit_v2", "ng1_v2", "random_k_v2"):
        BASE_ARGS["lr_B"] = 0.3
        BASE_ARGS["n_train_pairs"] = 1300
        print(f"  [v2] BASE_ARGS overridden: lr_B=0.3, n_train_pairs=1300")

    if args_main.plot_only:
        npz_path = out_dir / "summary.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            cl_cos = data["cos_angles"]
            print(f"Loaded: {npz_path}  shape={cl_cos.shape}")
        else:
            p = out_dir / "cos_angles.npy"
            cl_cos = np.load(p)
            print(f"Loaded (legacy): {p}  shape={cl_cos.shape}")
    else:
        cl_cos = run_fig2(args_main.coding_level, args_main.n_seeds, out_dir,
                          seed_start=args_main.seed_start)

    plot_fig2(cl_cos, out_dir, args_main.coding_level)
    print("Done.")


if __name__ == "__main__":
    main()
