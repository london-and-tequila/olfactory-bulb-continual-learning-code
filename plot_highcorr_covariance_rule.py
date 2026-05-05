"""Plot the high-correlation covariance-rule control.

The covariance-control input is a single-seed sequential-allocation run from
angle_vs_lag_different_combs:
  2_2_1_correlation_0.99.npy

For visual reference, the main-rule failure curve is pooled from the four
Fig. 4 r_input=0.99 sequential-allocation runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


C_NG = "#0072B2"
C_REF = "0.45"
BAND_ALPHA = 0.14
REF_BAND_ALPHA = 0.10


mpl.rcParams.update(
    {
        "axes.labelsize": 10.5,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "lines.linewidth": 1.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_covariance_control(data_dir: Path) -> np.ndarray:
    path = data_dir / "2_2_1_correlation_0.99.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.load(path)
    if arr.shape != (1000, 300):
        raise ValueError(f"{path} has shape {arr.shape}, expected (1000, 300)")
    if np.isnan(arr).any():
        raise ValueError(f"{path} contains NaNs")
    return arr


def load_main_rule_reference(paths: list[Path]) -> np.ndarray:
    pools = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path)
        if "angle_deg" in data:
            arr = data["angle_deg"]
        elif "cos_angles" in data:
            arr = np.degrees(np.arccos(np.clip(data["cos_angles"], -1.0, 1.0)))
        else:
            raise KeyError(f"{path} lacks angle_deg and cos_angles")
        if arr.shape != (1, 1001, 300):
            raise ValueError(f"{path} has angle shape {arr.shape}, expected (1, 1001, 300)")
        pools.append(arr[:, 1:, :].reshape(1000, 300))
    return np.concatenate(pools, axis=1)


def summarize(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.nanmean(arr, axis=1),
        np.nanpercentile(arr, 20, axis=1),
        np.nanpercentile(arr, 80, axis=1),
    )


def format_axis(ax, zoom: bool) -> None:
    ax.set_ylim(0, 92)
    ax.grid(True, color="#d0d0d0", alpha=0.28, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Lag (# tasks after learning)")
    if zoom:
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
    else:
        ax.set_xlim(0, 999)
        ax.set_xticks([0, 400, 800, 999])
        ax.axvline(400, color="0.45", linestyle="--", linewidth=0.9)


def draw_panel(ax, x: np.ndarray, cov_arr: np.ndarray, ref_arr: np.ndarray, zoom: bool) -> None:
    cov_mean, cov_p20, cov_p80 = summarize(cov_arr)
    ref_mean, ref_p20, ref_p80 = summarize(ref_arr)

    ax.plot(
        x,
        ref_mean,
        color=C_REF,
        linestyle="--",
        label="Main uncentered rule (4 seeds)",
    )
    ax.fill_between(x, ref_p20, ref_p80, color=C_REF, alpha=REF_BAND_ALPHA, linewidth=0)
    ax.plot(
        x,
        cov_mean,
        color=C_NG,
        label="Covariance-style rule (single seed)",
    )
    ax.fill_between(x, cov_p20, cov_p80, color=C_NG, alpha=BAND_ALPHA, linewidth=0)
    format_axis(ax, zoom=zoom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/learning_rule_controls"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/learning_rule_combs_bopt1"),
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("paper_v3/figures"),
    )
    args = parser.parse_args()

    ref_paths = [
        Path("results/v2_n1300_fig5/ng1_v2/r0p99/summary.npz"),
        Path("results/v2_n1300_fig5_seed1/ng1_v2/r0p99/summary.npz"),
        Path("results/v2_n1300_fig5_seed2/ng1_v2/r0p99/summary.npz"),
        Path("results/v2_n1300_fig5_seed3/ng1_v2/r0p99/summary.npz"),
    ]

    cov = load_covariance_control(args.data_dir)
    ref = load_main_rule_reference(ref_paths)
    x = np.arange(1000)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.45),
        sharey=True,
        gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.16},
    )
    axes[0].set_title("Full lag")
    axes[1].set_title("Early-lag zoom")

    draw_panel(axes[0], x, cov, ref, zoom=False)
    draw_panel(axes[1], x, cov, ref, zoom=True)
    axes[0].set_ylabel("Angle (deg)")
    axes[1].tick_params(labelleft=False)

    for idx, ax in enumerate(axes):
        ax.text(
            0.02,
            0.95,
            f"({chr(ord('a') + idx)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.2),
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.52, 1.02),
    )
    fig.text(
        0.52,
        0.86,
        r"$r_\mathrm{input}=0.99$; sequential allocation; covariance-style $\Delta F$ uses $(m_k-\bar{m})$",
        ha="center",
        va="center",
        fontsize=10.0,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.19, top=0.74)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.paper_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        args.out_dir / "highcorr_covariance_r0p99_angle.pdf",
        args.out_dir / "highcorr_covariance_r0p99_angle.png",
        args.paper_dir / "fig_highcorr_covariance_r0p99.pdf",
        args.paper_dir / "fig_highcorr_covariance_r0p99.png",
    ]
    for out in outputs:
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"saved {out}")

    for name, arr in [
        ("covariance-style rule", cov),
        ("main uncentered rule", ref),
    ]:
        mean = np.nanmean(arr, axis=1)
        print(
            f"{name:24s} lag0={mean[0]:.3f} "
            f"lag50={mean[50]:.3f} lag400={mean[400]:.3f} lag999={mean[999]:.3f}"
        )


if __name__ == "__main__":
    main()
