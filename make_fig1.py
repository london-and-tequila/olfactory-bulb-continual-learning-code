"""Reproduce paper fig1 (4-panel conceptual schematic) from scratch.

Output: paper_v3/figures/fig1.pdf + fig1.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox

OUT = Path(__file__).parent / "paper_v3" / "figures"

# ---- Style constants ----
COL_MC = "#F08000"          # MC outline (orange)
COL_RGC = "#2CA02C"         # rGC outline (green)
COL_ABGC = "#7BAFDF"        # abGC fill (soft blue)
COL_RED = "#D62728"         # high-plasticity red & task-relevant
COL_GRAY_LINE = "#BDBDBD"   # MC-GC light gray connections
COL_PANEL_BG = "#D9D9D9"    # gray panel background in d

# Task palette: hue-distinct from cell-type palette (purple→magenta→pink→gold)
COL_TASK1 = "#3B1F6B"       # deep purple
COL_TASK2 = "#6E3B9C"       # mid purple
COL_TASK3 = "#A064C2"       # light purple
COL_TASKN = "#5A4A7A"       # muted purple-gray (avoid clashing with MC orange)

# Typography hierarchy
FS_PANEL_TITLE = 14
FS_INPLACE     = 12
FS_LEGEND      = 11
FS_TASK_BOX    = 11
FS_PANEL_LABEL = 18

# Stroke
LW_NEURON  = 1.6
LW_ARROW   = 1.6
LW_PLAST   = 2.6
MUT_ARROW  = 14


# ===========================================================================
# Panel a / b — 3D unit-sphere representation cartoon
# ===========================================================================

def _normalize_rows(vecs):
    vecs = np.asarray(vecs, dtype=float)
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    return vecs / np.maximum(norms, 1e-12)


def _unit_vec(azimuth_deg, elevation_deg):
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    return np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])


def _orthogonal_unit(vec):
    vec = _normalize_rows(np.asarray(vec))[()]
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(vec, ref)) > 0.85:
        ref = np.array([0.0, 1.0, 0.0])
    tangent = ref - np.dot(ref, vec) * vec
    return _normalize_rows(tangent)[()]


def make_unit_gray_vecs(n, rng):
    """n unit vectors radiating from the origin to the sphere surface."""
    vecs = rng.normal(size=(n, 3))
    vecs[:, 2] *= 0.65
    return _normalize_rows(vecs)


def make_red_pair(separation_deg):
    """Two unit vectors with a controlled angular separation."""
    center = _unit_vec(55, 18)
    tangent = _orthogonal_unit(center)
    half_angle = np.deg2rad(separation_deg / 2)
    return _normalize_rows([
        np.cos(half_angle) * center - np.sin(half_angle) * tangent,
        np.cos(half_angle) * center + np.sin(half_angle) * tangent,
    ])


def _draw_unit_sphere(ax):
    u = np.linspace(0, 2 * np.pi, 32)
    v = np.linspace(0, np.pi, 16)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(
        x, y, z,
        rstride=1, cstride=1,
        color="#F0F0F0", alpha=0.18,
        linewidth=0, shade=False, zorder=1,
    )
    ax.plot_wireframe(
        x, y, z,
        rstride=4, cstride=4,
        color="#A8A8A8", alpha=0.42,
        linewidth=0.55, zorder=2,
    )


def _draw_unit_sphere_vectors(ax, gray_vecs, red_vecs, title):
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_box_aspect((1, 1, 1), zoom=1.35)
    ax.set_proj_type("ortho")
    ax.view_init(elev=20, azim=-55)
    ax.set_axis_off()
    ax.set_facecolor("white")

    _draw_unit_sphere(ax)

    # Task-irrelevant directions, preserved across panels.
    for v in gray_vecs:
        ax.quiver(
            0, 0, 0, v[0], v[1], v[2],
            length=0.96, normalize=False,
            color="#7E7E7E", linewidth=1.25,
            arrow_length_ratio=0.12, alpha=0.78,
        )

    # Task-relevant directions, emphasized.
    for v in red_vecs:
        ax.quiver(
            0, 0, 0, v[0], v[1], v[2],
            length=1.0, normalize=False,
            color=COL_RED, linewidth=3.0,
            arrow_length_ratio=0.15,
        )

    ax.scatter([0], [0], [0], s=10, color="#555555", depthshade=False)
    ax.set_title(title, fontsize=FS_PANEL_TITLE, pad=0)


def panel_a(ax, gray_vecs):
    # Before learning: 2 reds tightly clustered (~20° apart)
    _draw_unit_sphere_vectors(
        ax, gray_vecs, make_red_pair(20), "Before task learning"
    )
    ax.text2D(0.75, 0.52, "small separation",
              transform=ax.transAxes,
              ha="left", va="center",
              fontsize=FS_INPLACE + 1, color="#444444", style="italic")


def panel_b(ax, gray_vecs):
    # After learning: 2 reds spread (~90° apart) — task-relevant subspace
    # has expanded.
    _draw_unit_sphere_vectors(
        ax, gray_vecs, make_red_pair(90), "After task learning"
    )
    ax.text2D(0.73, 0.52, "larger separation",
              transform=ax.transAxes,
              ha="left", va="center",
              fontsize=FS_INPLACE + 1, color="#444444", style="italic")


def panel_arrow(ax):
    """Transition arrow + label between panel a (top) and b (bottom)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.annotate(
        "", xy=(0.58, 0.02), xytext=(0.58, 0.98),
        arrowprops=dict(arrowstyle="-|>", color="#333333",
                        lw=2.0, mutation_scale=MUT_ARROW + 4),
    )


def panel_replg(ax):
    """Shared compact side legend for panel a/b."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box = FancyBboxPatch(
        (0.06, 0.05), 0.88, 0.90,
        boxstyle="round,pad=0.02",
        linewidth=0.8, edgecolor="#666666", facecolor="white",
    )
    ax.add_patch(box)

    # Red task-relevant (top)
    ax.annotate(
        "", xy=(0.72, 0.78), xytext=(0.28, 0.78),
        arrowprops=dict(arrowstyle="-|>", color=COL_RED,
                        lw=2.4, mutation_scale=MUT_ARROW),
    )
    ax.text(0.50, 0.60, "task-relevant",
            ha="center", va="center", fontsize=FS_LEGEND - 1)

    # Gray task-irrelevant (bottom)
    ax.annotate(
        "", xy=(0.72, 0.37), xytext=(0.28, 0.37),
        arrowprops=dict(arrowstyle="-|>", color="#7E7E7E",
                        lw=1.4, mutation_scale=MUT_ARROW),
    )
    ax.text(0.50, 0.19, "task-irrelevant",
            ha="center", va="center", fontsize=FS_LEGEND - 1)


# ===========================================================================
# Panel c — circuit schematic
# ===========================================================================

def panel_c(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    mc_x = 4.2
    gc_x = 6.8
    mc_ys = np.linspace(2.0, 7.5, 6)            # 6 MCs
    gc_ys = np.linspace(1.6, 8.1, 8)            # 8 GCs

    abgc_indices = {3, 4}                       # 2 adjacent abGCs (cohort)
    plast_pairs = [
        (3, 0), (3, 2), (3, 4),                 # lower abGC -> MCs
        (4, 1), (4, 5),                         # upper abGC -> MCs
    ]

    # MC -> GC sampled gray connectivity (deterministic stride; ~33%)
    for mi, ymc in enumerate(mc_ys):
        for gi, ygc in enumerate(gc_ys):
            if (mi + gi) % 3 == 0:
                ax.plot([mc_x, gc_x], [ymc, ygc],
                        color=COL_GRAY_LINE, lw=0.5,
                        alpha=0.45, zorder=1)

    # Red high-plasticity overlay: glow first, then main line
    for gi, mi in plast_pairs:
        ax.plot([mc_x, gc_x], [mc_ys[mi], gc_ys[gi]],
                color=COL_RED, lw=5.6, alpha=0.18, zorder=2)
    for gi, mi in plast_pairs:
        ax.plot([mc_x, gc_x], [mc_ys[mi], gc_ys[gi]],
                color=COL_RED, lw=LW_PLAST, alpha=1.0, zorder=3)

    neuron_r = 0.32

    # Recruited cohort outline tightly around the two adjacent abGCs
    abgc_ys_sorted = sorted(gc_ys[i] for i in abgc_indices)
    cohort_top = max(abgc_ys_sorted) + 0.45
    cohort_bot = min(abgc_ys_sorted) - 0.45
    ax.add_patch(FancyBboxPatch(
        (gc_x - 0.48, cohort_bot),
        0.96, cohort_top - cohort_bot,
        boxstyle="round,pad=0.04,rounding_size=0.28",
        linewidth=0.9, edgecolor="#A6C7E6", facecolor="none",
        linestyle=(0, (3, 2)), zorder=2,
    ))

    # MCs (orange empty circles)
    for y in mc_ys:
        ax.add_patch(mpatches.Circle(
            (mc_x, y), neuron_r,
            facecolor="white", edgecolor=COL_MC,
            linewidth=LW_NEURON + 0.2, zorder=4,
        ))

    # GCs
    for i, y in enumerate(gc_ys):
        if i in abgc_indices:
            ax.add_patch(mpatches.Circle(
                (gc_x, y), neuron_r,
                facecolor=COL_ABGC, edgecolor="black",
                linewidth=LW_NEURON - 0.4, zorder=4,
            ))
        else:
            ax.add_patch(mpatches.Circle(
                (gc_x, y), neuron_r,
                facecolor="white", edgecolor=COL_RGC,
                linewidth=LW_NEURON + 0.2, zorder=4,
            ))

    # In-place "Recruited cohort" label to the right of the bracket
    ax.text(gc_x + 0.72, (cohort_top + cohort_bot) / 2,
            "recruited\ncohort",
            ha="left", va="center",
            fontsize=FS_INPLACE - 1, color="#6F8FB0", style="italic")

    # In-place label for resident GC (replaces external legend)
    rgc_indices = [i for i in range(len(gc_ys)) if i not in abgc_indices]
    top_rgc_y = max(gc_ys[i] for i in rgc_indices)
    ax.text(gc_x + 0.55, top_rgc_y, "resident GC",
            ha="left", va="center",
            fontsize=FS_INPLACE - 1, color=COL_RGC, style="italic")

    # Glomeruli input arrows (left of MC column)
    for y in mc_ys:
        ax.annotate(
            "", xy=(mc_x - 0.42, y), xytext=(mc_x - 1.5, y),
            arrowprops=dict(arrowstyle="-|>", color="black",
                            lw=LW_ARROW, mutation_scale=MUT_ARROW),
        )
    ax.text(mc_x - 1.65, np.mean(mc_ys), "Glomeruli\nInput",
            ha="right", va="center", fontsize=FS_INPLACE)

    # Output arrow from top MC
    top_mc_y = mc_ys[-1]
    ax.annotate(
        "", xy=(mc_x, top_mc_y + 1.4),
        xytext=(mc_x, top_mc_y + 0.35),
        arrowprops=dict(arrowstyle="-|>", color="black",
                        lw=LW_ARROW + 0.4, mutation_scale=MUT_ARROW + 4),
    )
    ax.text(mc_x, top_mc_y + 1.6, "Output",
            ha="center", va="bottom", fontsize=FS_INPLACE)

    # Inhibit (GC -> MC) arrow
    ax.annotate(
        "", xy=(mc_x + 0.55, top_mc_y + 0.55),
        xytext=(gc_x - 0.45, top_mc_y + 0.55),
        arrowprops=dict(arrowstyle="-|>", color="#777777",
                        lw=LW_ARROW, mutation_scale=MUT_ARROW),
    )
    ax.text((mc_x + gc_x) / 2, top_mc_y + 1.0, "inhibit",
            ha="center", va="bottom",
            fontsize=FS_INPLACE, color="#555555")

    # Excite (MC -> GC) arrow
    excite_y = mc_ys[0] - 0.2
    ax.annotate(
        "", xy=(gc_x - 0.45, excite_y),
        xytext=(mc_x + 0.55, excite_y),
        arrowprops=dict(arrowstyle="-|>", color="#777777",
                        lw=LW_ARROW, mutation_scale=MUT_ARROW),
    )
    ax.text((mc_x + gc_x) / 2, excite_y - 0.55, "excite",
            ha="center", va="top",
            fontsize=FS_INPLACE, color="#555555")

    # Bottom column labels (short, dark gray)
    ax.text(mc_x, 0.7, "MC", ha="center", va="top",
            fontsize=FS_PANEL_TITLE, fontweight="bold", color="#333333")
    ax.text(gc_x, 0.7, "GC", ha="center", va="top",
            fontsize=FS_PANEL_TITLE, fontweight="bold", color="#333333")



# ===========================================================================
# Panel d — sequential learning + cumulative evaluation
# ===========================================================================

def _task_box(ax, x, y, w, h, color, label, fontsize=14, alpha=1.0):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.10",
        linewidth=0.8, edgecolor="black", facecolor=color, alpha=alpha,
    )
    ax.add_patch(rect)
    # White text for dark task colors (T1 purple, T2 magenta, T3 pink);
    # dark text for the gold TN.
    text_color = "white"
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight="bold")


def _draw_dots(ax, cx, cy, n=3, dr=0.08, gap=0.30, orientation="horizontal"):
    """Render a row/column of round dots in place of Unicode ellipses."""
    for i in range(n):
        offset = (i - (n - 1) / 2) * gap
        if orientation == "horizontal":
            x, y = cx + offset, cy
        else:
            x, y = cx, cy + offset
        ax.add_patch(mpatches.Circle((x, y), dr, color="#666666",
                                     zorder=5))


def panel_d(ax):
    ax.set_xlim(-1.5, 12)
    ax.set_ylim(0, 6.3)
    ax.axis("off")

    # 5 columns aligned between training and evaluation rows.
    col_centers = np.array([2.6, 4.7, 6.8, 8.9, 11.0])

    # ---- Top training-sequence row ----
    top_y, top_h = 5.0, 1.0

    slot_w, slot_h = 1.6, 0.62
    slot_y = top_y + (top_h - slot_h) / 2

    task_specs = [
        (COL_TASK1, "Task 1"),
        (COL_TASK2, "Task 2"),
        (COL_TASK3, "Task 3"),
        (None, None),
        (COL_TASKN, "Task N"),
    ]

    for (color, label), cx in zip(task_specs, col_centers):
        if color is None:
            _draw_dots(ax, cx, slot_y + slot_h / 2,
                       n=3, dr=0.05, gap=0.18,
                       orientation="horizontal")
        else:
            _task_box(ax, cx - slot_w / 2, slot_y, slot_w, slot_h,
                      color, label, fontsize=FS_TASK_BOX + 1)

    # arrows between consecutive training slots
    for i in range(len(col_centers) - 1):
        x_from = col_centers[i] + slot_w / 2
        x_to = col_centers[i + 1] - slot_w / 2
        ay = slot_y + slot_h / 2
        ax.annotate(
            "", xy=(x_to, ay), xytext=(x_from, ay),
            arrowprops=dict(arrowstyle="-|>", color="black",
                            lw=LW_ARROW - 0.2, mutation_scale=MUT_ARROW - 2),
        )

    ax.text(-0.4, top_y + top_h / 2, "Training\nsequence",
            ha="center", va="center",
            fontsize=FS_PANEL_TITLE, fontweight="bold")

    # ---- Down chevron arrows (After training) ----
    chevron_bottom = 4.0
    arrow_xs = [col_centers[i] for i in [0, 1, 2, 4]]
    for ax_x in arrow_xs:
        ax.annotate(
            "", xy=(ax_x, chevron_bottom),
            xytext=(ax_x, top_y - 0.05),
            arrowprops=dict(arrowstyle="-|>", color="#444444",
                            lw=LW_ARROW + 0.2, mutation_scale=MUT_ARROW + 2),
        )
    ax.text(-0.4, (top_y + chevron_bottom) / 2 - 0.15,
            "After training", ha="center", va="center",
            fontsize=FS_PANEL_TITLE, fontweight="bold")

    # ---- Bottom evaluation row (faint column guides instead of panel bg) ----
    bot_y, bot_h = 0.3, 3.5

    # Faint column guide lines (very subtle, behind everything)
    for col_idx, cx in enumerate(col_centers):
        if col_idx == 3:
            continue  # skip the ellipsis column
        ax.plot([cx, cx], [bot_y + 0.15, bot_y + bot_h - 0.55],
                color="#E8E8E8", lw=0.8, zorder=0)

    # Column headers (just below panel top)
    header_y = bot_y + bot_h - 0.28
    headers = ["After T1", "After T2", "After T3", None, "After TN"]
    for hd, cx in zip(headers, col_centers):
        if hd is None:
            _draw_dots(ax, cx, header_y, n=3, dr=0.04, gap=0.16,
                       orientation="horizontal")
        else:
            ax.text(cx, header_y, hd, ha="center", va="center",
                    fontsize=FS_INPLACE, color="#222222",
                    fontweight="bold")

    # Stacked task boxes per column (compact, abbreviated, faded fill)
    cell_w, cell_h = 0.95, 0.50
    gap = 0.08
    # First cell's bottom y, leaving room above for header.
    y_top = bot_y + bot_h - 0.95
    EVAL_ALPHA = 0.75

    def stack_in_column(col_idx, items):
        """items: list of (color, label) or ('ellipsis', None)."""
        cx = col_centers[col_idx]
        x = cx - cell_w / 2
        for j, item in enumerate(items):
            y = y_top - j * (cell_h + gap)
            color, label = item
            if color == "ellipsis":
                _draw_dots(ax, cx, y + cell_h / 2,
                           n=3, dr=0.05, gap=0.18,
                           orientation="vertical")
            else:
                _task_box(ax, x, y, cell_w, cell_h, color, label,
                          fontsize=FS_TASK_BOX, alpha=EVAL_ALPHA)

    # Column 1: T1
    stack_in_column(0, [(COL_TASK1, "T1")])
    # Column 2: T1, T2
    stack_in_column(1, [(COL_TASK1, "T1"),
                        (COL_TASK2, "T2")])
    # Column 3: T1, T2, T3
    stack_in_column(2, [(COL_TASK1, "T1"),
                        (COL_TASK2, "T2"),
                        (COL_TASK3, "T3")])
    # Column 4: horizontal dots between T3 and TN columns
    cx = col_centers[3]
    _draw_dots(ax, cx, bot_y + bot_h / 2 - 0.1,
               n=3, dr=0.06, gap=0.20,
               orientation="horizontal")
    # Column 5: T1, T2, T3, ⋮, TN
    stack_in_column(4, [(COL_TASK1, "T1"),
                        (COL_TASK2, "T2"),
                        (COL_TASK3, "T3"),
                        ("ellipsis", None),
                        (COL_TASKN, "TN")])

    ax.text(-0.4, bot_y + bot_h - 0.6, "Evaluation\non tasks",
            ha="center", va="center",
            fontsize=FS_PANEL_TITLE, fontweight="bold")


# ===========================================================================
# Main
# ===========================================================================

def main():
    fig = plt.figure(figsize=(13.5, 6.0))

    # 3-column layout, packed tighter horizontally; (c) and (d) full-height
    ax_a     = fig.add_axes([0.018, 0.565, 0.225, 0.405],
                            projection="3d")
    ax_arrow = fig.add_axes([0.055, 0.490, 0.150, 0.115])
    ax_b     = fig.add_axes([0.018, 0.055, 0.225, 0.405],
                            projection="3d")
    ax_replg = fig.add_axes([0.200, 0.380, 0.085, 0.250])

    ax_c     = fig.add_axes([0.295, 0.08, 0.315, 0.86])

    ax_d     = fig.add_axes([0.630, 0.08, 0.365, 0.86])

    rng = np.random.default_rng(2)
    gray_vecs = make_unit_gray_vecs(8, rng)

    panel_a(ax_a, gray_vecs)
    panel_b(ax_b, gray_vecs)
    panel_arrow(ax_arrow)
    panel_replg(ax_replg)
    panel_c(ax_c)
    panel_d(ax_d)

    # Panel labels — top-left of each panel, consistent offset from axes
    label_positions = {
        "a": (0.008, 0.955),
        "b": (0.008, 0.475),
        "c": (0.278, 0.955),
        "d": (0.613, 0.955),
    }
    for lab, (x, y) in label_positions.items():
        fig.text(x, y, lab, fontsize=FS_PANEL_LABEL, fontweight="bold")

    OUT.mkdir(parents=True, exist_ok=True)
    fig_w, fig_h = fig.get_size_inches()
    crop_bottom = 0.30
    savefig_kwargs = dict(
        bbox_inches=Bbox.from_bounds(0, crop_bottom, fig_w, fig_h - crop_bottom)
    )
    fig.savefig(OUT / "fig1.pdf", **savefig_kwargs)
    fig.savefig(OUT / "fig1.png", dpi=150, **savefig_kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
