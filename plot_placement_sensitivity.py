import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon

from tessellations import build_tessellation


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_NAME = "lattice_placement_sensitivity_basis"


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

# Same nominal full-cell area for all tessellations
AC_DEMO = 0.60


# ============================================================
# NORMALIZED PHASE CONFIGURATIONS
# ============================================================

PHASE_SET = [
    (0.00, 0.00),
    (0.50, 0.00),
    (0.00, 0.50),
    (0.50, 0.50),
]


# ============================================================
# DEMONSTRATION DOMAIN
# ============================================================

def build_demo_domain():
    """
    Fixed continuous concave domain.

    The domain, nominal cell area, and lattice orientation
    remain unchanged. Only the phase of the periodic lattice
    is modified.
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
# PERIODICITY VECTORS
# ============================================================

def lattice_basis(tau, side):
    """
    Return two independent periodicity vectors of the canonical
    lattice.

    The normalized phase is defined as

        phi_tau(u,v)
        = u a1_tau + v a2_tau.

    The vectors are expressed in Cartesian coordinates, while
    (u,v) are coordinates in the tessellation-specific periodic
    basis.
    """

    s = float(side)

    if tau == "square":

        a1 = np.array([
            s,
            0.0,
        ])

        a2 = np.array([
            0.0,
            s,
        ])

    elif tau == "triangle":

        h = math.sqrt(3.0) * s / 2.0

        a1 = np.array([
            s,
            0.0,
        ])

        a2 = np.array([
            s / 2.0,
            h,
        ])

    elif tau == "hex":

        h = math.sqrt(3.0) * s

        a1 = np.array([
            1.5 * s,
            h / 2.0,
        ])

        a2 = np.array([
            0.0,
            h,
        ])

    else:

        raise ValueError(
            f"Unknown tessellation: {tau}"
        )

    return a1, a2


def phase_from_uv(
    tau,
    side,
    u,
    v,
):
    """
    Convert normalized lattice-phase coordinates (u,v)
    into the corresponding physical translation vector.

        phi_tau(u,v)
        = u a1_tau + v a2_tau
    """

    a1, a2 = lattice_basis(
        tau,
        side,
    )

    phase = (
        u * a1
        +
        v * a2
    )

    return phase


# ============================================================
# MAIN-DOMAIN DRAWING
# ============================================================

def draw_tessellation(
    ax,
    tessellation,
    cell_linewidth=0.60,
):
    """
    Draw complete retained cells.
    """

    for cell in tessellation.cells.values():

        poly = cell.full_polygon
        x, y = poly.exterior.xy

        ax.plot(
            x,
            y,
            color="0.68",
            linewidth=cell_linewidth,
            alpha=0.85,
            zorder=1,
        )


def draw_domain(
    ax,
    omega,
):
    """
    Draw the fixed continuous search-domain boundary.
    """

    x, y = omega.exterior.xy

    ax.plot(
        x,
        y,
        color="black",
        linewidth=1.35,
        zorder=20,
    )


def configure_axis(
    ax,
    omega,
    margin=0.65,
):
    """
    Common formatting for the main panels.
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
# PHASE LABEL
# ============================================================

def draw_phase_label(
    ax,
    u,
    v,
):
    """
    Display normalized phase coordinates inside each panel.
    """

    text = (
        rf"$(u,v)=({u:.1f},{v:.1f})$"
    )

    ax.text(
        0.035,
        0.055,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.8,
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="white",
            edgecolor="0.78",
            linewidth=0.50,
            alpha=0.92,
        ),
        zorder=100,
    )


# ============================================================
# LOCAL LATTICE-BASIS INSET
# ============================================================

def add_basis_inset(
    ax,
    a1,
    a2,
    u,
    v,
):
    """
    Add a small local diagram showing:

        a1_tau
        a2_tau
        phi_tau(u,v)

    The inset is schematic.

    Basis-vector directions and relative geometry are preserved.
    The phase vector is slightly enlarged for readability only;
    this visual scaling does not modify the physical translation
    applied to the lattice.
    """

    # --------------------------------------------------------
    # Create inset
    # --------------------------------------------------------

    inset = ax.inset_axes([
        0.66,
        0.64,
        0.27,
        0.27,
    ])

    inset.set_facecolor("white")

    # --------------------------------------------------------
    # Normalize basis for drawing
    # --------------------------------------------------------

    a1 = np.asarray(
        a1,
        dtype=float,
    )

    a2 = np.asarray(
        a2,
        dtype=float,
    )

    # Common scale preserves relative angular geometry
    max_norm = max(
        np.linalg.norm(a1),
        np.linalg.norm(a2),
    )

    a1_vis = a1 / max_norm
    a2_vis = a2 / max_norm

    # Phase in same normalized basis
    phi_vis = (
        u * a1_vis
        +
        v * a2_vis
    )

    origin = np.array([
        0.0,
        0.0,
    ])

    # --------------------------------------------------------
    # Basis vector a1
    # --------------------------------------------------------

    inset.annotate(
        "",
        xy=a1_vis,
        xytext=origin,
        arrowprops=dict(
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.0,
            color="0.35",
        ),
        zorder=10,
    )

    inset.text(
        a1_vis[0] * 1.05,
        a1_vis[1] * 1.05,
        r"$\mathbf{a}_{1,\tau}$",
        fontsize=7.2,
        color="0.25",
        ha="center",
        va="center",
    )

    # --------------------------------------------------------
    # Basis vector a2
    # --------------------------------------------------------

    inset.annotate(
        "",
        xy=a2_vis,
        xytext=origin,
        arrowprops=dict(
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.0,
            color="0.35",
        ),
        zorder=10,
    )

    inset.text(
        a2_vis[0] * 1.05,
        a2_vis[1] * 1.05,
        r"$\mathbf{a}_{2,\tau}$",
        fontsize=7.2,
        color="0.25",
        ha="center",
        va="center",
    )

    # --------------------------------------------------------
    # Resulting phase vector
    # --------------------------------------------------------

    phi_display_scale = 1.35

    if np.linalg.norm(phi_vis) > 1e-12:

        phi_display = (
            phi_display_scale
            * phi_vis
        )

        inset.annotate(
            "",
            xy=phi_display,
            xytext=origin,
            arrowprops=dict(
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.5,
                color="tab:blue",
            ),
            zorder=20,
        )

        inset.text(
            phi_display[0] * 1.05,
            phi_display[1] * 1.05,
            r"$\boldsymbol{\phi}_{\tau}$",
            fontsize=7.5,
            color="tab:blue",
            ha="center",
            va="center",
        )

    else:

        inset.text(
            0.05,
            0.05,
            r"$\boldsymbol{\phi}_{\tau}=\mathbf{0}$",
            fontsize=7.0,
            color="tab:blue",
            ha="left",
            va="bottom",
        )

    # --------------------------------------------------------
    # Automatic limits
    # --------------------------------------------------------

    vectors_to_check = [
        a1_vis,
        a2_vis,
        phi_vis * phi_display_scale,
    ]

    xs = [0.0]
    ys = [0.0]

    for vec in vectors_to_check:

        xs.append(vec[0])
        ys.append(vec[1])

    xmin = min(xs) - 0.25
    xmax = max(xs) + 0.35

    ymin = min(ys) - 0.25
    ymax = max(ys) + 0.35

    inset.set_xlim(
        xmin,
        xmax,
    )

    inset.set_ylim(
        ymin,
        ymax,
    )

    inset.set_aspect(
        "equal",
        adjustable="box",
    )

    inset.set_xticks([])
    inset.set_yticks([])

    for spine in inset.spines.values():

        spine.set_linewidth(0.5)
        spine.set_color("0.75")


# ============================================================
# MAIN FIGURE
# ============================================================

def plot_lattice_placement_sensitivity():

    omega = build_demo_domain()

    # --------------------------------------------------------
    # Common reference anchor
    #
    # Used internally to construct the canonical lattice.
    # It is intentionally not drawn because phase translation
    # represents a global lattice displacement rather than a
    # vector originating from the domain centroid.
    # --------------------------------------------------------

    anchor = np.array([
        omega.centroid.x,
        omega.centroid.y,
    ])

    column_titles = [
        r"Canonical phase: $(u,v)=(0,0)$",
        r"$(u,v)=(0.5,0)$",
        r"$(u,v)=(0,0.5)$",
        r"$(u,v)=(0.5,0.5)$",
    ]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(12.3, 7.7),
    )

    # ========================================================
    # LOOP OVER TESSELLATIONS
    # ========================================================

    for row, tau in enumerate(TAUS):

        # ----------------------------------------------------
        # Canonical tessellation
        #
        # Used to recover the tessellation-specific side length
        # associated with the common nominal cell area.
        # ----------------------------------------------------

        canonical = build_tessellation(
            omega=omega,
            tau=tau,
            nominal_cell_area=AC_DEMO,
            anchor=anchor,
            phase=(0.0, 0.0),
            orientation_deg=0.0,
        )

        side = canonical.side_length

        a1, a2 = lattice_basis(
            tau,
            side,
        )

        # ====================================================
        # LOOP OVER PHASE CONFIGURATIONS
        # ====================================================

        for col, (u, v) in enumerate(PHASE_SET):

            ax = axes[row, col]

            # ------------------------------------------------
            # Normalized phase -> physical translation
            # ------------------------------------------------

            phase = phase_from_uv(
                tau=tau,
                side=side,
                u=u,
                v=v,
            )

            # ------------------------------------------------
            # Build translated lattice
            # ------------------------------------------------

            tessellation = build_tessellation(
                omega=omega,
                tau=tau,
                nominal_cell_area=AC_DEMO,
                anchor=anchor,
                phase=phase,
                orientation_deg=0.0,
            )

            # ------------------------------------------------
            # Draw spatial configuration
            # ------------------------------------------------

            draw_tessellation(
                ax,
                tessellation,
            )

            draw_domain(
                ax,
                omega,
            )

            # ------------------------------------------------
            # Phase-coordinate annotation
            # ------------------------------------------------

            draw_phase_label(
                ax,
                u,
                v,
            )

            # ------------------------------------------------
            # Local periodic-basis inset
            # ------------------------------------------------

            add_basis_inset(
                ax,
                a1,
                a2,
                u,
                v,
            )

            # ------------------------------------------------
            # Column titles
            # ------------------------------------------------

            if row == 0:

                ax.set_title(
                    column_titles[col],
                    fontsize=10,
                    pad=7,
                )

            # ------------------------------------------------
            # Row labels
            # ------------------------------------------------

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

    # ========================================================
    # LAYOUT
    # ========================================================

    # No global title; explanation belongs in the caption.
    fig.tight_layout(
        pad=0.8,
        w_pad=0.7,
        h_pad=0.8,
    )

    # ========================================================
    # SAVE
    # ========================================================

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
    plot_lattice_placement_sensitivity()