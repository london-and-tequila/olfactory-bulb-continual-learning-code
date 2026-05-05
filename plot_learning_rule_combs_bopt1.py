"""Plot B-rule learning-rule controls as one compact supplementary figure.

Inputs are .npy arrays from angle_vs_lag_different_combs.zip:
  1_1_1.npy: uncentered F, no-m-factor B, sequential/NG
  1_1_2.npy: uncentered F, no-m-factor B, input-based allocation
  2_1_1.npy: covariance-style F, no-m-factor B, sequential/NG
  2_1_2.npy: covariance-style F, no-m-factor B, input-based allocation

Each array has shape (lag, tested_pattern) = (1000, 300). Bands are 20--80
percentiles across tested patterns at each lag.
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
C_TOPK = "#009E73"
BAND_ALPHA = 0.14


mpl.rcParams.update(
    {
        "axes.labelsize": 10.5,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 10,
        "lines.linewidth": 1.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_array(data_dir: Path, name: str) -> np.ndarray:
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.load(path)
    if arr.shape != (1000, 300):
        raise ValueError(f"{name} has shape {arr.shape}, expected (1000, 300)")
    return arr


def summarize(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.nanmean(arr, axis=1),
        np.nanpercentile(arr, 20, axis=1),
        np.nanpercentile(arr, 80, axis=1),
    )


def draw_curve(ax, x, arr, color, label=None):
    mean, p20, p80 = summarize(arr)
    ax.plot(x, mean, color=color, label=label)
    ax.fill_between(x, p20, p80, color=color, alpha=BAND_ALPHA, linewidth=0)
    return mean


def format_axis(ax, zoom=False):
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

    data = {
        "f1_ng": load_array(args.data_dir, "1_1_1.npy"),
        "f1_topk": load_array(args.data_dir, "1_1_2.npy"),
        "f2_ng": load_array(args.data_dir, "2_1_1.npy"),
        "f2_topk": load_array(args.data_dir, "2_1_2.npy"),
    }

    x = np.arange(1000)
    b_rule = (
        r"$\Delta B \propto g_1(\bar{m}-m_1)+g_2(\bar{m}-m_2),"
        r"\ \beta_B=0.9$"
    )
    rows = [
        (r"$\Delta F \propto g_1m_1+g_2m_2$", data["f1_ng"], data["f1_topk"]),
        (
            r"$\Delta F \propto g_1(m_1-\bar{m})+g_2(m_2-\bar{m}),\ F>0$",
            data["f2_ng"],
            data["f2_topk"],
        ),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 3.95),
        sharey=True,
        gridspec_kw={"width_ratios": [1.55, 1.0], "hspace": 0.36, "wspace": 0.16},
    )

    axes[0, 0].set_title("Full lag")
    axes[0, 1].set_title("Early-lag zoom")

    for row_idx, (row_label, ng, topk) in enumerate(rows):
        for col_idx, zoom in enumerate([False, True]):
            ax = axes[row_idx, col_idx]
            draw_curve(ax, x, ng, C_NG, "Sequential allocation" if row_idx == 0 and col_idx == 0 else None)
            draw_curve(ax, x, topk, C_TOPK, "Input-based allocation" if row_idx == 0 and col_idx == 0 else None)
            format_axis(ax, zoom=zoom)
            if row_idx == 0:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
            if col_idx == 0:
                ax.set_ylabel("Angle (deg)")
            else:
                ax.tick_params(labelleft=False)
            panel = chr(ord("a") + 2 * row_idx + col_idx)
            ax.text(
                0.02,
                0.95,
                f"({panel})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10.5,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.2),
            )
            if col_idx == 0:
                ax.text(
                    0.13,
                    0.95,
                    row_label,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.7,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.2),
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.52, 0.995),
    )
    fig.text(0.52, 0.905, b_rule, ha="center", va="center", fontsize=10.5)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.13, top=0.80)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.paper_dir.mkdir(parents=True, exist_ok=True)

    for out in [
        args.out_dir / "learning_rules_bopt1_combined.pdf",
        args.out_dir / "learning_rules_bopt1_combined.png",
        args.paper_dir / "fig_learning_rules_bopt1_combined.pdf",
        args.paper_dir / "fig_learning_rules_bopt1_combined.png",
    ]:
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"saved {out}")

    for name, arr in [
        ("uncentered F NG", data["f1_ng"]),
        ("uncentered F input-based allocation", data["f1_topk"]),
        ("centered F NG", data["f2_ng"]),
        ("centered F input-based allocation", data["f2_topk"]),
    ]:
        mean = np.nanmean(arr, axis=1)
        print(
            f"{name:12s} lag0={mean[0]:.3f} "
            f"lag50={mean[50]:.3f} lag999={mean[-1]:.3f}"
        )


if __name__ == "__main__":
    main()
