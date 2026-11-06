import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Arc
from shapely.geometry import Polygon

from tessellations import build_tessellation


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_NAME = "angular_alignment_sensitivity"


# ============================================================
# GENERAL SETTINGS
# ============================================================

TAUS = [
    "square",
    "triangle",
    "hex",
]

TAU_LABELS = {
    "square": "Square",
    "triangle": "Triangular",
    "hex": "Hexagonal",
}

# Same nominal cell area for the three tessellations.
# Used only for the methodological illustration.
AC_DEMO = 0.60


# ============================================================
# DEMONSTRATION DOMAIN
# ============================================================

def build_demo_domain():
    """
    Fixed concave continuous domain used only for the
    methodological illustration.

    The physical domain remains unchanged in all panels.
    """
    points = np.array([
        [-4.8,  2.9],
        [-1.8,  2.3],
        [ 2.6,  1.8],
        [ 4.8,  0.5],
        [ 2.4, -0.8],
        [ 4.0, -3.1],
        [ 0.4, -2.2],
        [-4.3, -3.5],
        [-2.2, -0.5],
    ])

    return Polygon(points)


# ============================================================
# ROTATIONAL SYMMETRY
# ============================================================

def symmetry_period(tau):
    """
    Fundamental rotational period T_tau of the complete lattice.
    """
    if tau == "square":
        return 90.0

    if tau in ("triangle", "hex"):
        return 60.0

    raise ValueError(f"Unknown tessellation: {tau}")


def experimental_angles(tau):
    """
    Four uniformly spaced non-redundant orientations in the
    fundamental rotational interval [0, T_tau).

    T_tau itself is excluded because it is symmetry-equivalent
    to theta = 0.
    """
    T = symmetry_period(tau)

    return [
        0.0,
        T / 4.0,
        T / 2.0,
        3.0 * T / 4.0,
    ]


# ============================================================
# DRAWING UTILITIES
# ============================================================

def draw_tessellation(
    ax,
    tessellation,
    alpha=1.0,
    cell_linewidth=0.60,
):
    """
    Draw complete retained cells.

    Cell centroids are intentionally omitted to reduce visual
    clutter at publication size.
    """
    for cell in tessellation.cells.values():

        poly = cell.full_polygon
        x, y = poly.exterior.xy

        ax.plot(
            x,
            y,
            color="0.68",
            linewidth=cell_linewidth,
            alpha=0.85 * alpha,
            zorder=1,
        )


def draw_domain(
    ax,
    omega,
    alpha=1.0,
):
    """
    Draw the fixed continuous search-domain boundary.
    """
    x, y = omega.exterior.xy

    ax.plot(
        x,
        y,
        color="black",
        linewidth=1.55,
        alpha=alpha,
        zorder=20,
    )


def draw_anchor(
    ax,
    anchor,
    alpha=1.0,
):
    """
    Common domain-centroid rotation anchor.
    """
    ax.scatter(
        anchor[0],
        anchor[1],
        s=27,
        color="black",
        marker="o",
        alpha=alpha,
        zorder=70,
    )


def draw_reference_ray(
    ax,
    anchor,
    length=1.15,
    alpha=1.0,
):
    """
    Canonical angular reference direction.
    """
    anchor = np.asarray(anchor, dtype=float)

    end = anchor + np.array([
        length,
        0.0,
    ])

    ax.plot(
        [anchor[0], end[0]],
        [anchor[1], end[1]],
        linestyle="--",
        linewidth=1.0,
        color="0.35",
        alpha=0.8 * alpha,
        zorder=50,
    )


def draw_rotated_ray(
    ax,
    anchor,
    theta_deg,
    length=1.15,
    alpha=1.0,
):
    """
    Rotated angular reference direction.
    """
    if np.isclose(theta_deg, 0.0):
        return

    theta = math.radians(theta_deg)

    anchor = np.asarray(
        anchor,
        dtype=float,
    )

    end = (
        anchor
        +
        length
        * np.array([
            math.cos(theta),
            math.sin(theta),
        ])
    )

    ax.plot(
        [anchor[0], end[0]],
        [anchor[1], end[1]],
        linewidth=1.35,
        color="tab:orange",
        alpha=0.85 * alpha,
        zorder=51,
    )


def draw_rotation_arc(
    ax,
    anchor,
    theta_deg,
    radius=0.82,
    alpha=1.0,
):
    """
    Angular arc between the canonical orientation and theta_deg.
    """
    if np.isclose(theta_deg, 0.0):
        return

    anchor = np.asarray(
        anchor,
        dtype=float,
    )

    arc = Arc(
        tuple(anchor),
        width=2.0 * radius,
        height=2.0 * radius,
        angle=0.0,
        theta1=0.0,
        theta2=theta_deg,
        linewidth=1.35,
        color="tab:orange",
        alpha=0.85 * alpha,
        zorder=60,
    )

    ax.add_patch(arc)

    # Arrowhead at end of the arc
    theta_end = math.radians(theta_deg)

    end = (
        anchor
        +
        radius
        * np.array([
            math.cos(theta_end),
            math.sin(theta_end),
        ])
    )

    tangent = np.array([
        -math.sin(theta_end),
        math.cos(theta_end),
    ])

    ax.annotate(
        "",
        xy=end,
        xytext=end - 0.16 * tangent,
        arrowprops=dict(
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.35,
            color="tab:orange",
            alpha=0.85 * alpha,
        ),
        zorder=61,
    )


def draw_angle_label(
    ax,
    theta_deg,
    equivalent=False,
):
    """
    Compact angle annotation.
    """
    if equivalent:

        text = (
            rf"$\theta={theta_deg:g}^\circ"
            r"\equiv 0^\circ$"
        )

    else:

        text = (
            rf"$\theta={theta_deg:g}^\circ$"
        )

    ax.text(
        0.03,
        0.03,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=dict(
            boxstyle="round,pad=0.16",
            facecolor="white",
            edgecolor="0.78",
            linewidth=0.55,
            alpha=0.92,
        ),
        zorder=100,
    )


def configure_axis(
    ax,
    omega,
    margin=0.65,
):
    """
    Common panel formatting.
    """
    minx, miny, maxx, maxy = omega.bounds

    ax.set_xlim(
        minx - margin,
        maxx + margin,
    )

    ax.set_ylim(
        miny - margin,
        maxy + margin,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("0.45")


# ============================================================
# MAIN FIGURE
# ============================================================

def plot_angular_alignment_sensitivity():

    omega = build_demo_domain()

    # Common rotation anchor = centroid of continuous domain
    anchor = np.array([
        omega.centroid.x,
        omega.centroid.y,
    ])

    column_titles = [
        "Canonical",
        r"$T_\tau/4$",
        r"$T_\tau/2$",
        r"$3T_\tau/4$",
        "Symmetry-equivalent",
    ]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=5,
        figsize=(14.6, 7.6),
    )

    for row, tau in enumerate(TAUS):

        T = symmetry_period(tau)

        # Four configurations actually used in the experiment
        angles = experimental_angles(tau)

        # Add T only for visualization of rotational equivalence
        display_angles = angles + [T]

        for col, theta_current in enumerate(display_angles):

            ax = axes[row, col]

            endpoint_equivalent = (
                col == len(display_angles) - 1
            )

            # Fade the symmetry-equivalent endpoint because
            # it is not an additional experimental condition.
            panel_alpha = (
                0.48
                if endpoint_equivalent
                else 1.0
            )

            tessellation = build_tessellation(
                omega=omega,
                tau=tau,
                nominal_cell_area=AC_DEMO,
                anchor=anchor,
                phase=(0.0, 0.0),
                orientation_deg=theta_current,
            )

            draw_tessellation(
                ax,
                tessellation,
                alpha=panel_alpha,
            )

            draw_domain(
                ax,
                omega,
                alpha=panel_alpha,
            )

            draw_anchor(
                ax,
                anchor,
                alpha=panel_alpha,
            )

            draw_reference_ray(
                ax,
                anchor,
                alpha=panel_alpha,
            )

            if not np.isclose(theta_current, 0.0):

                draw_rotated_ray(
                    ax,
                    anchor,
                    theta_current,
                    alpha=panel_alpha,
                )

                draw_rotation_arc(
                    ax,
                    anchor,
                    theta_current,
                    alpha=panel_alpha,
                )

            draw_angle_label(
                ax,
                theta_current,
                equivalent=endpoint_equivalent,
            )

            if row == 0:

                ax.set_title(
                    column_titles[col],
                    fontsize=10.5,
                    pad=7,
                )

            if col == 0:

                ax.set_ylabel(
                    TAU_LABELS[tau],
                    fontsize=10.5,
                    labelpad=9,
                )

            configure_axis(
                ax,
                omega,
            )

    # No global title: explanation belongs in the paper caption.

    fig.tight_layout(
        pad=0.8,
        w_pad=0.7,
        h_pad=0.8,
    )

    pdf_path = (
        OUTPUT_DIR
        /
        f"{OUTPUT_NAME}.pdf"
    )

    png_path = (
        OUTPUT_DIR
        /
        f"{OUTPUT_NAME}.png"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    plot_angular_alignment_sensitivity()