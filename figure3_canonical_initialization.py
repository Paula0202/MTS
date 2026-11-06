
from pathlib import Path
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.affinity import scale, translate
from shapely.geometry import Point, Polygon


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path(".")
OUTPUT_PDF = OUTPUT_DIR / "figure3_canonical_initialization.pdf"
OUTPUT_PNG = OUTPUT_DIR / "figure3_canonical_initialization.png"

# Continuous-domain area used in the paper.
A0 = 100.0

# Reference resolution used only to make the tessellation visible.
# Change it if you prefer a coarser/finer appearance.
N_REF = 100

# Replace these vertices by the exact vertices used in the original
# Fig. 3 if you want to reproduce exactly the same concave domain.
#
# The polygon below is only a convenient concave example with a
# similar visual character.
DOMAIN_VERTICES = np.array([
    [1.0, 0.0],
    [ 3.0,  2.0],
    [ 0.0, 5.0],
    [6.0, 3.0],
], dtype=float)


# ============================================================
# DOMAIN
# ============================================================

def make_domain(vertices, target_area=A0):
    """Create, center, and scale a polygon to the requested area."""
    poly = Polygon(vertices)

    if not poly.is_valid:
        poly = poly.buffer(0)

    # Move centroid to the common physical origin x0 = (0, 0).
    c = poly.centroid
    poly = translate(poly, xoff=-c.x, yoff=-c.y)

    # Scale uniformly so A(Omega) = target_area.
    factor = math.sqrt(target_area / poly.area)
    poly = scale(poly, xfact=factor, yfact=factor, origin=(0.0, 0.0))

    return poly


# ============================================================
# TESSELLATION GENERATORS
# ============================================================

def _plot_bounds(domain, margin=2.5):
    minx, miny, maxx, maxy = domain.bounds
    return minx - margin, miny - margin, maxx + margin, maxy + margin


def make_square_cells(domain, nominal_area):
    """
    Square tessellation anchored so that one cell centroid is exactly x0=(0,0).
    """
    side = math.sqrt(nominal_area)
    minx, miny, maxx, maxy = _plot_bounds(domain, 2.0 * side)

    i_min = math.floor(minx / side) - 1
    i_max = math.ceil(maxx / side) + 1
    j_min = math.floor(miny / side) - 1
    j_max = math.ceil(maxy / side) + 1

    cells = []

    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            cx = i * side
            cy = j * side

            cell = Polygon([
                (cx - side / 2, cy - side / 2),
                (cx + side / 2, cy - side / 2),
                (cx + side / 2, cy + side / 2),
                (cx - side / 2, cy + side / 2),
            ])

            if domain.contains(Point(cx, cy)):
                cells.append(cell)

    return cells


def make_triangular_cells(domain, nominal_area):
    """
    Equilateral-triangular tessellation anchored so that one retained
    triangle has centroid exactly at x0=(0,0).

    The regular triangular lattice is built by splitting each
    parallelogram of the basis into two equilateral triangles.
    """
    side = math.sqrt(4.0 * nominal_area / math.sqrt(3.0))
    height = math.sqrt(3.0) * side / 2.0

    a1 = np.array([side, 0.0])
    a2 = np.array([side / 2.0, height])

    # Triangle (0, a1, a2) has centroid:
    anchor_shift = -(a1 + a2) / 3.0

    minx, miny, maxx, maxy = _plot_bounds(domain, 3.0 * side)
    extent = max(
        abs(minx), abs(maxx), abs(miny), abs(maxy)
    ) + 4.0 * side

    n = int(math.ceil(extent / min(side / 2.0, height))) + 5
    cells = []

    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            p00 = i * a1 + j * a2 + anchor_shift
            p10 = (i + 1) * a1 + j * a2 + anchor_shift
            p01 = i * a1 + (j + 1) * a2 + anchor_shift
            p11 = (i + 1) * a1 + (j + 1) * a2 + anchor_shift

            triangles = (
                Polygon([p00, p10, p01]),
                Polygon([p10, p11, p01]),
            )

            for tri in triangles:
                c = tri.centroid
                if domain.contains(c):
                    cells.append(tri)

    return cells


def make_hexagonal_cells(domain, nominal_area):
    """
    Regular flat-top hexagonal tessellation anchored so that the
    q=r=0 hexagon has centroid exactly at x0=(0,0).
    """
    side = math.sqrt(
        2.0 * nominal_area / (3.0 * math.sqrt(3.0))
    )

    minx, miny, maxx, maxy = _plot_bounds(domain, 3.0 * side)
    extent = max(
        abs(minx), abs(maxx), abs(miny), abs(maxy)
    ) + 4.0 * side

    # Axial-coordinate centre spacing for flat-top hexagons.
    n = int(math.ceil(extent / (1.5 * side))) + 6
    cells = []

    angles = np.deg2rad(np.arange(0.0, 360.0, 60.0))

    for q in range(-n, n + 1):
        for r in range(-n, n + 1):
            cx = 1.5 * side * q
            cy = math.sqrt(3.0) * side * (r + q / 2.0)

            if not domain.contains(Point(cx, cy)):
                continue

            vertices = np.column_stack([
                cx + side * np.cos(angles),
                cy + side * np.sin(angles),
            ])

            cells.append(Polygon(vertices))

    return cells


# ============================================================
# PLOTTING
# ============================================================

def find_initial_cell(cells, x0=(0.0, 0.0), tol=1e-10):
    """Find the cell whose centroid coincides with the common x0."""
    x0 = np.asarray(x0, dtype=float)

    distances = [
        np.linalg.norm(
            np.array([cell.centroid.x, cell.centroid.y]) - x0
        )
        for cell in cells
    ]

    idx = int(np.argmin(distances))

    if distances[idx] > tol:
        raise RuntimeError(
            "No cell centroid coincides with x0. "
            f"Nearest centroid is at distance {distances[idx]:.3e}."
        )

    return idx


def draw_polygon(ax, polygon, **kwargs):
    xy = np.asarray(polygon.exterior.coords)
    patch = MplPolygon(xy, closed=True, **kwargs)
    ax.add_patch(patch)


def plot_tessellation(ax, domain, cells, title):
    initial_idx = find_initial_cell(cells)

    # Draw tessellation cells.
    for idx, cell in enumerate(cells):
        draw_polygon(
            ax,
            cell,
            facecolor="none",
            edgecolor="0.72",
            linewidth=0.55 if idx != initial_idx else 1.5,
            zorder=1,
        )

    # Draw retained centroids.
    centroids = np.array([
        [cell.centroid.x, cell.centroid.y]
        for cell in cells
    ])

    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=8,
        color="0.55",
        zorder=2,
    )

    # Continuous-domain boundary.
    boundary = np.asarray(domain.exterior.coords)
    ax.plot(
        boundary[:, 0],
        boundary[:, 1],
        color="black",
        linewidth=2.0,
        zorder=4,
    )

    # Common physical initial position and tessellation-specific
    # initial-state centroid coincide exactly.
    x0 = np.array([0.0, 0.0])

    ax.scatter(
        [x0[0]],
        [x0[1]],
        marker="o",
        s=72,
        facecolors="none",
        edgecolors="tab:blue",
        linewidths=1.8,
        zorder=6,
    )

    ax.scatter(
        [x0[0]],
        [x0[1]],
        marker="x",
        s=55,
        color="tab:red",
        linewidths=1.8,
        zorder=7,
    )

    ax.annotate(
        r"$\mathbf{x}_0$",
        xy=x0,
        xytext=(8, 10),
        textcoords="offset points",
        color="tab:red",
        fontsize=11,
        fontweight="bold",
    )

    ax.annotate(
        r"$\mathbf{q}_{\tau}(s_{0,\tau})$",
        xy=x0,
        xytext=(8, -18),
        textcoords="offset points",
        color="tab:blue",
        fontsize=10,
        fontweight="bold",
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    domain = make_domain(DOMAIN_VERTICES, target_area=A0)

    nominal_area = A0 / N_REF

    tessellations = [
        (
            "Square",
            make_square_cells(domain, nominal_area),
        ),
        (
            "Triangular",
            make_triangular_cells(domain, nominal_area),
        ),
        (
            "Hexagonal",
            make_hexagonal_cells(domain, nominal_area),
        ),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.5, 4.4),
        constrained_layout=True,
    )

    fig.suptitle(
        r"Common UAV initialization in the canonical tessellation configuration",
        fontsize=15,
        fontweight="bold",
    )

    for ax, (title, cells) in zip(axes, tessellations):
        plot_tessellation(
            ax=ax,
            domain=domain,
            cells=cells,
            title=title,
        )

    # Common statement for the three panels.
    fig.text(
        0.5,
        0.01,
        r"$\mathbf{q}_{\tau}(s_{0,\tau})=\mathbf{x}_0,"
        r"\qquad \forall\,\tau\in\mathcal{T}$",
        ha="center",
        va="bottom",
        fontsize=12,
    )

    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print(f"Saved: {OUTPUT_PDF.resolve()}")
    print(f"Saved: {OUTPUT_PNG.resolve()}")


if __name__ == "__main__":
    main()
