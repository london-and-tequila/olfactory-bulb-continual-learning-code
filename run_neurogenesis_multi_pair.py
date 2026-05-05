"""
Neurogenesis continual learning experiment script.
Converted from experimental_notebooks/multi_pair_test.ipynb.
"""

import argparse
import os
import sys
import pickle
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from lib.input_gen import Uniform_Correlated
from lib.network import (
    NeurogenesisHyperConfig,
    NeurogenesisHyperDynConfig,
    Neurogenesis1,
    Neurogenesis_randomKSelection,
    Neurogenesis_topKSelection_noInit,
    Neurogenesis_topKSelection_noInit_v2,
    Neurogenesis1_v2,
    Neurogenesis_randomKSelection_v2,
)
from lib.driver import Driver1


def parse_args():
    parser = argparse.ArgumentParser(description="Neurogenesis multi-pair continual learning experiment")

    # Network structure
    parser.add_argument("--n_mitral", type=int, default=400)
    parser.add_argument("--n_granule", type=int, default=2000)
    parser.add_argument("--n_granule_per_task", type=int, default=5, help="K: granule cells allocated per task")
    parser.add_argument("--granule_nonlinear", type=str, default="piecewise_linear")
    parser.add_argument("--mitral_nonlinear", type=str, default="relu")

    # Dynamics (--tau sets both tau_mitral and tau_granule)
    parser.add_argument("--tau", type=float, default=5.0)
    parser.add_argument("--lr_F", type=float, default=0.1, help="Forward weight learning rate")
    parser.add_argument("--lr_B", type=float, default=0.1, help="Backward weight learning rate")
    parser.add_argument("--lr_th_g", type=float, default=0.01, help="Granule threshold learning rate")
    parser.add_argument("--decay_granule_thres", type=float, default=0.9)
    parser.add_argument("--th_g_hi_ratio", type=float, default=0.95)

    # Training scale
    parser.add_argument("--n_pretrain_pairs", type=int, default=500)
    parser.add_argument("--n_train_pairs", type=int, default=800)
    parser.add_argument("--n_test_pairs", type=int, default=300)
    parser.add_argument("--n_epochs_per_pair", type=int, default=300)
    parser.add_argument("--n_steps_to_steady", type=int, default=100)

    # Stimulus
    parser.add_argument("--correlation", type=float, default=0.9, help="Input pair correlation (pairCorrelation)")
    parser.add_argument("--coding_level", type=float, default=1.0,
                        help="Mitral coding level c in (0,1]. Default=1.0 (dense). "
                             "Sets mitral_thres=1-c and scales F_norm by 1/c^2.")

    # Network type
    parser.add_argument("--network_type", type=str, default="neurogenesis1",
                        choices=["neurogenesis1", "random_k", "topk_noinit", "topk_noinit_v2",
                                 "ng1_v2", "random_k_v2"],
                        help="Granule cell allocation. *_v2 variants use the paper-default "
                             "(Bernoulli mask init, sqrt(n_mitral/6) th_g, (m_bar-m) Hebbian, β_B decay).")

    # Experiment management
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="Base seed; actual seeds = seed, seed+1, ...")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--exp_name", type=str, default="", help="Subdirectory name; auto-generated if empty")
    parser.add_argument("--save_plots", action="store_true")

    # Adaptive learning rate
    parser.add_argument("--adaptive_lr", action="store_true",
                        help="Scale lr_F/lr_B to compensate sparse input")
    parser.add_argument("--lr_F_max", type=float, default=2.0,
                        help="Max effective lr_F when adaptive_lr is on")
    parser.add_argument("--lr_B_max", type=float, default=2.0,
                        help="Max effective lr_B when adaptive_lr is on")

    return parser.parse_args()


def make_exp_id(args):
    net = getattr(args, "network_type", "neurogenesis1")
    net_tag = f"_{net}" if net != "neurogenesis1" else ""
    return (
        f"m{args.n_mitral}_g{args.n_granule}_K{args.n_granule_per_task}"
        f"_tau{args.tau}_corr{args.correlation}_cl{args.coding_level}{net_tag}"
    )


def run_single_seed(args, seed: int):
    n_mitral = args.n_mitral
    n_granule = args.n_granule

    c = args.coding_level
    net_type = getattr(args, "network_type", "neurogenesis1")
    if net_type == "topk_noinit":
        # run_topk_no_theta.py recipe: max_dyn_range_ratio=2.0, denominator=2/3 → 60 at c=1
        max_dyn_range_ratio = 2.0
        F_norm = max_dyn_range_ratio / (1 - args.th_g_hi_ratio) / (2 / 3) / (c ** 2)
    elif net_type in ("topk_noinit_v2", "ng1_v2", "random_k_v2"):
        # Notebook recipe (verbatim): max_dyn_range_ratio=1.5, denominator=sqrt(n_mitral/6).
        max_dyn_range_ratio = 1.5
        F_norm = max_dyn_range_ratio / (1 - args.th_g_hi_ratio) / jnp.sqrt(n_mitral / 6) / (c ** 2)
    else:
        F_norm = 1.0 / (1 - args.th_g_hi_ratio) / (c ** 2)
    print(f"  F_norm = {F_norm:.4f}  (network_type={getattr(args, 'network_type', 'neurogenesis1')}, coding_level={c}, th_g_hi_ratio={args.th_g_hi_ratio})")

    # 自适应学习率：补偿稀疏输入下的 Hebbian 更新衰减
    # F 矩阵（超球面角速度）：Δθ ∝ lr_F * c^{3.5}，补偿 1/c^{3.5}
    # B 矩阵（元素级更新）：ΔB ∝ c²，精确补偿 1/c²
    lr_F_effective = args.lr_F
    lr_B_effective = args.lr_B
    if getattr(args, "adaptive_lr", False):
        lr_F_max = getattr(args, "lr_F_max", 2.0)
        lr_B_max = getattr(args, "lr_B_max", 2.0)
        lr_F_effective = min(lr_F_max, args.lr_F / (c ** 3.5))
        lr_B_effective = min(lr_B_max, args.lr_B / (c ** 2.0))
        print(f"  adaptive_lr: lr_F={lr_F_effective:.4f} (raw={args.lr_F/c**3.5:.4f}), "
              f"lr_B={lr_B_effective:.4f} (raw={args.lr_B/c**2:.4f})")

    hyper = NeurogenesisHyperConfig(
        n_mitral=n_mitral,
        n_granule=n_granule,
        n_granule_per_task=args.n_granule_per_task,
        mitral_nonlinear_func=args.mitral_nonlinear,
        granule_nonlinear_func=args.granule_nonlinear,
        n_steps_to_steady=args.n_steps_to_steady,
        n_epochs_per_pair=args.n_epochs_per_pair,
    )

    # decay_rate_B: ng1/rk learnt with B reset each epoch (β_B=0); topk_noinit's
    # notebook recipe uses β_B=0.9 (B accumulates with 10% decay) — without this,
    # topk inhibition never builds up and learning fails (lag=0 stays ~14° instead of ~90°).
    decay_rate_B = 0.9 if net_type in ("topk_noinit", "topk_noinit_v2", "ng1_v2", "random_k_v2") else 0.0
    hyperD = NeurogenesisHyperDynConfig(
        learning_rate_B=lr_B_effective,
        learning_rate_F=lr_F_effective,
        learning_rate_th_g=args.lr_th_g,
        learning_rate_th_m=0.0,
        decay_rate_B=decay_rate_B,
        decay_rate_F=0.0,
        decay_mitral_thres=0.0,
        decay_granule_thres=args.decay_granule_thres,
        th_g_hi_ratio=args.th_g_hi_ratio,
        mitral_self_excitation=0.0,
        tau_mitral=args.tau,
        tau_granule=args.tau,
        granule_activation_scaling=1.0,
        F_norm=float(F_norm),
    )

    if net_type == "random_k":
        network = Neurogenesis_randomKSelection(hyper)
    elif net_type == "topk_noinit":
        network = Neurogenesis_topKSelection_noInit(hyper)
    elif net_type == "topk_noinit_v2":
        network = Neurogenesis_topKSelection_noInit_v2(hyper)
    elif net_type == "ng1_v2":
        network = Neurogenesis1_v2(hyper)
    elif net_type == "random_k_v2":
        network = Neurogenesis_randomKSelection_v2(hyper)
    else:
        network = Neurogenesis1(hyper)

    # Wrap generate() to use the desired correlation (Driver1.run() doesn't pass it)
    xs_gen = Uniform_Correlated(nE=n_mitral)
    xs_gen.generate = partial(xs_gen._generate, pairCorrelation=args.correlation)

    # RNG split path matches Driver1.run's internal init: PRNGKey → (root, driver_rng) →
    # (driver_rng_inner, init_rng, random_x_rng). Keeping this exact path means
    # initial F is bitwise identical to what Driver would generate from the same root,
    # which is critical for topk_noinit where top-K selection is F-driven.
    root_rng = jax.random.PRNGKey(seed * 1000 + 123123123)
    _, driver_rng = jax.random.split(root_rng)
    driver_rng_inner, init_rng, random_x_rng = jax.random.split(driver_rng, 3)
    random_xs1, _ = xs_gen.generate(random_x_rng, 0)
    initial_state = network.init_state(init_rng, hyperD, random_xs1)
    if args.coding_level < 1.0:
        initial_state = initial_state.replace(
            mitral_thres=jnp.ones(n_mitral) * (1.0 - args.coding_level)
        )
        print(f"  mitral_thres set to {1.0 - args.coding_level:.3f} for coding_level={args.coding_level}")

    # Use driver_rng_inner (the leftover from the 3-way split) so Driver continues
    # from where its own internal init would have left off, matching notebook RNG path.
    driver = Driver1(
        network,
        xs_gen,
        driver_rng_inner,
        hyperD,
        initial_state=initial_state,
        n_test_pairs=args.n_test_pairs,
    )

    print(f"  [seed={seed}] Starting: pretrain={args.n_pretrain_pairs}, train={args.n_train_pairs}")
    driver.run(
        n_pretrain_pairs=args.n_pretrain_pairs,
        n_train_pairs=args.n_train_pairs,
        n_random_pairs=0,
    )
    print(f"  [seed={seed}] Done. cos_angles shape: {driver.record['m1m2_cos_angles'].shape}")

    # Extract only CPU numpy arrays; do NOT keep driver.state (GPU JAX arrays)
    # to avoid VRAM accumulation across sequential runs.
    result = {
        "cos_angles": np.array(driver.record["m1m2_cos_angles"]),
        "seed": seed,
        "args": vars(args),
    }
    for key in ["m1m2_l2_dists", "m1m2_corr", "rand_cos_sim", "rand_l2_change"]:
        if key in driver.record:
            result[key] = np.array(driver.record[key])
    return result


def save_results(results_per_seed, out_dir: Path, exp_id: str, save_plots: bool):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save per-seed results
    for res in results_per_seed:
        seed = res["seed"]
        fpath = out_dir / f"{exp_id}_seed{seed}.pkl"
        with open(fpath, "wb") as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved: {fpath}")

    # Save summary (all seeds' arrays stacked); shape[1] = n_train - n_test + 1 (lag=-1 at index 0)
    all_angles = np.stack([res["cos_angles"] for res in results_per_seed], axis=0)
    summary = {
        "cos_angles": all_angles,          # (n_seeds, n_train-n_test+1, n_test)
        "angle_deg": np.arccos(np.clip(all_angles, -1, 1)) * 180 / np.pi,
        "args": results_per_seed[0]["args"],
    }
    for key in ["m1m2_l2_dists", "m1m2_corr", "rand_cos_sim", "rand_l2_change"]:
        if key in results_per_seed[0]:
            summary[key] = np.stack([res[key] for res in results_per_seed], axis=0)

    # Write summary.npz FIRST (plot scripts depend on it), then summary.pkl (full payload incl. args dict)
    npz_path = out_dir / "summary.npz"
    npz_arrays = {k: v for k, v in summary.items() if k != "args"}
    np.savez(npz_path, **npz_arrays)
    print(f"  Summary npz saved: {npz_path}")

    summary_path = out_dir / "summary.pkl"
    with open(summary_path, "wb") as f:
        pickle.dump(summary, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Summary pkl saved: {summary_path}")

    if save_plots:
        _save_plot(summary, out_dir, exp_id)


def _save_plot(summary, out_dir: Path, exp_id: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        angle_deg = summary["angle_deg"]  # (n_seeds, T, n_test)
        # Average over seeds and test pairs
        angle_flat = angle_deg.reshape(angle_deg.shape[0], angle_deg.shape[1], -1)
        all_flat = angle_flat.reshape(-1, angle_flat.shape[2])  # (n_seeds*T, n_test)
        median = np.median(angle_deg, axis=(0, 2))    # (T,)
        upper = np.percentile(angle_deg, 80, axis=(0, 2))
        lower = np.percentile(angle_deg, 20, axis=(0, 2))

        fig, ax = plt.subplots(figsize=(8, 4))
        xs = np.arange(len(median))
        ax.fill_between(xs, lower, upper, alpha=0.3)
        ax.plot(xs, median)
        ax.set_xlabel("Task index (lag-0 aligned)")
        ax.set_ylabel("Angle (degrees)")
        ax.set_title(f"Neurogenesis continual learning\n{exp_id}")
        fig.tight_layout()
        plot_path = out_dir / "learning_curve.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {plot_path}")
    except Exception as e:
        print(f"  Warning: plotting failed ({e})")


def main():
    args = parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"JAX version: {jax.__version__}")

    exp_id = make_exp_id(args)
    exp_name = args.exp_name if args.exp_name else exp_id
    out_dir = Path(args.output_dir) / exp_name

    print(f"\nExperiment: {exp_id}")
    print(f"Output dir: {out_dir}")
    print(f"Seeds: {args.n_seeds} (base seed: {args.seed})")

    results_per_seed = []
    for i in range(args.n_seeds):
        seed = args.seed + i
        print(f"\n--- Seed {i+1}/{args.n_seeds} (seed={seed}) ---")
        res = run_single_seed(args, seed)
        results_per_seed.append(res)

    print(f"\nSaving results to {out_dir} ...")
    save_results(results_per_seed, out_dir, exp_id, args.save_plots)
    print("Done.")


if __name__ == "__main__":
    main()
