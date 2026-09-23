import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import networkx as nx

from shapely.geometry import Polygon, Point
from shapely.geometry.base import BaseGeometry
from shapely.affinity import translate, rotate
from shapely import set_precision, union_all


# ============================================================
# NUMERICAL TOLERANCES
# ============================================================

AREA_REL_TOL = 1e-9
DIST_REL_TOL = 1e-9
ABS_TOL = 1e-12


def area_tolerance(nominal_cell_area):
    return max(
        ABS_TOL,
        AREA_REL_TOL * nominal_cell_area
    )


def distance_tolerance(nominal_cell_area):
    return max(
        ABS_TOL,
        DIST_REL_TOL * math.sqrt(nominal_cell_area)
    )


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Cell:
    id: int

    # Logical lattice index
    lattice_index: tuple

    # Complete regular cell h_{tau,i}
    full_polygon: Polygon

    # Real centroid q_{tau,i}
    centroid: np.ndarray

    # Valid support C_{tau,i} = h_{tau,i} ∩ Omega
    valid_support: BaseGeometry

    # Full cell area
    full_area: float

    # Valid area A(C_{tau,i})
    valid_area: float

    # Boundary-cell indicator
    is_boundary: bool

    # Only used for triangular cells
    orientation: Optional[str] = None


@dataclass
class Tessellation:
    tau: str

    nominal_cell_area: float
    side_length: float

    cells: dict
    index_to_id: dict

    # Navigation graph G_tau
    graph: nx.Graph

    # Lattice parameters
    anchor: np.ndarray
    phase: np.ndarray
    orientation_deg: float


# ============================================================
# DOMAIN VALIDATION
# ============================================================

def validate_domain(omega):
    if omega.is_empty:
        raise ValueError(
            "Search domain is empty."
        )

    if not omega.is_valid:
        raise ValueError(
            "Search domain is invalid."
        )

    if omega.area <= 0:
        raise ValueError(
            "Search domain must have positive area."
        )


# ============================================================
# BASIC CELL GEOMETRIES
# ============================================================

def square_polygon(cx, cy, side):
    r = side / 2.0

    return Polygon([
        (cx - r, cy - r),
        (cx + r, cy - r),
        (cx + r, cy + r),
        (cx - r, cy + r)
    ])


def hex_polygon_flat(cx, cy, side):
    h = math.sqrt(3.0) * side / 2.0

    return Polygon([
        (cx - side / 2.0, cy + h),
        (cx + side / 2.0, cy + h),
        (cx + side,       cy),
        (cx + side / 2.0, cy - h),
        (cx - side / 2.0, cy - h),
        (cx - side,       cy)
    ])


# ============================================================
# COMMON GEOMETRIC UTILITIES
# ============================================================

def domain_radius(omega, anchor):
    minx, miny, maxx, maxy = omega.bounds

    corners = np.array([
        [minx, miny],
        [minx, maxy],
        [maxx, miny],
        [maxx, maxy]
    ])

    return np.max(
        np.linalg.norm(
            corners - anchor,
            axis=1
        )
    )


def transform_cell(
    polygon,
    anchor,
    phase,
    orientation_deg
):
    """
    Applies lattice phase first and lattice rotation second.
    """

    transformed = translate(
        polygon,
        xoff=phase[0],
        yoff=phase[1]
    )

    if orientation_deg != 0.0:
        transformed = rotate(
            transformed,
            orientation_deg,
            origin=tuple(anchor),
            use_radians=False
        )

    return transformed


def create_retained_cell(
    cell_id,
    lattice_index,
    polygon,
    omega,
    nominal_cell_area,
    orientation=None
):
    """
    Creates a retained Cell after the centroid-retention rule
    has already been checked.
    """

    centroid_geom = polygon.centroid

    centroid = np.array([
        centroid_geom.x,
        centroid_geom.y
    ])

    valid_support = polygon.intersection(
        omega
    )

    full_area = polygon.area
    valid_area = valid_support.area

    tol = area_tolerance(
        nominal_cell_area
    )

    is_boundary = (
        valid_area
        <
        nominal_cell_area - tol
    )

    return Cell(
        id=cell_id,
        lattice_index=lattice_index,
        full_polygon=polygon,
        centroid=centroid,
        valid_support=valid_support,
        full_area=full_area,
        valid_area=valid_area,
        is_boundary=is_boundary,
        orientation=orientation
    )


# ============================================================
# SQUARE TESSELLATION
# ============================================================

def build_square_tessellation(
    omega,
    nominal_cell_area,
    anchor=None,
    phase=(0.0, 0.0),
    orientation_deg=0.0
):
    validate_domain(omega)

    side = math.sqrt(
        nominal_cell_area
    )

    if anchor is None:
        anchor = np.array([
            omega.centroid.x,
            omega.centroid.y
        ])
    else:
        anchor = np.asarray(
            anchor,
            dtype=float
        )

    phase = np.asarray(
        phase,
        dtype=float
    )

    R = domain_radius(
        omega,
        anchor
    )

    margin = (
        2.0 * side
        +
        np.linalg.norm(phase)
    )

    n = (
        math.ceil(
            (R + margin) / side
        )
        + 2
    )

    cells = {}
    index_to_id = {}

    cell_id = 0

    for i in range(-n, n + 1):
        for j in range(-n, n + 1):

            cx = (
                anchor[0]
                + j * side
            )

            cy = (
                anchor[1]
                + i * side
            )

            candidate = square_polygon(
                cx,
                cy,
                side
            )

            candidate = transform_cell(
                candidate,
                anchor,
                phase,
                orientation_deg
            )

            centroid = candidate.centroid

            # Strict centroid-retention rule
            if not omega.contains(
                centroid
            ):
                continue

            cell = create_retained_cell(
                cell_id=cell_id,
                lattice_index=(i, j),
                polygon=candidate,
                omega=omega,
                nominal_cell_area=nominal_cell_area
            )

            cells[cell_id] = cell
            index_to_id[(i, j)] = cell_id

            cell_id += 1

    graph = nx.Graph()

    graph.add_nodes_from(
        cells.keys()
    )

    square_neighbours = [
        (-1, 0),
        (+1, 0),
        (0, -1),
        (0, +1)
    ]

    for (i, j), cid in index_to_id.items():

        for di, dj in square_neighbours:

            neighbour_index = (
                i + di,
                j + dj
            )

            if neighbour_index not in index_to_id:
                continue

            nid = index_to_id[
                neighbour_index
            ]

            if cid >= nid:
                continue

            qi = cells[cid].centroid
            qj = cells[nid].centroid

            weight = np.linalg.norm(
                qi - qj
            )

            graph.add_edge(
                cid,
                nid,
                weight=weight
            )

    return Tessellation(
        tau="square",
        nominal_cell_area=nominal_cell_area,
        side_length=side,
        cells=cells,
        index_to_id=index_to_id,
        graph=graph,
        anchor=anchor,
        phase=phase,
        orientation_deg=orientation_deg
    )


# ============================================================
# HEXAGONAL TESSELLATION
# ============================================================

def build_hex_tessellation(
    omega,
    nominal_cell_area,
    anchor=None,
    phase=(0.0, 0.0),
    orientation_deg=0.0
):
    validate_domain(omega)

    side = math.sqrt(
        2.0
        * nominal_cell_area
        /
        (
            3.0
            * math.sqrt(3.0)
        )
    )

    h_hex = (
        math.sqrt(3.0)
        * side
    )

    if anchor is None:
        anchor = np.array([
            omega.centroid.x,
            omega.centroid.y
        ])
    else:
        anchor = np.asarray(
            anchor,
            dtype=float
        )

    phase = np.asarray(
        phase,
        dtype=float
    )

    R = domain_radius(
        omega,
        anchor
    )

    margin = (
        2.0 * side
        +
        np.linalg.norm(phase)
    )

    n_j = (
        math.ceil(
            (R + margin)
            /
            (1.5 * side)
        )
        + 3
    )

    n_i = (
        math.ceil(
            (R + margin)
            /
            h_hex
        )
        + 3
    )

    cells = {}
    index_to_id = {}

    cell_id = 0

    for i in range(-n_i, n_i + 1):
        for j in range(-n_j, n_j + 1):

            cx = (
                anchor[0]
                +
                1.5 * j * side
            )

            delta_j = (
                h_hex / 2.0
                if j % 2 != 0
                else 0.0
            )

            cy = (
                anchor[1]
                +
                i * h_hex
                +
                delta_j
            )

            candidate = hex_polygon_flat(
                cx,
                cy,
                side
            )

            candidate = transform_cell(
                candidate,
                anchor,
                phase,
                orientation_deg
            )

            centroid = candidate.centroid

            if not omega.contains(
                centroid
            ):
                continue

            cell = create_retained_cell(
                cell_id=cell_id,
                lattice_index=(i, j),
                polygon=candidate,
                omega=omega,
                nominal_cell_area=nominal_cell_area
            )

            cells[cell_id] = cell
            index_to_id[(i, j)] = cell_id

            cell_id += 1

    graph = nx.Graph()

    graph.add_nodes_from(
        cells.keys()
    )

    for (i, j), cid in index_to_id.items():

        if j % 2 == 0:

            neighbours = [
                (i - 1, j),
                (i + 1, j),

                (i - 1, j - 1),
                (i,     j - 1),

                (i - 1, j + 1),
                (i,     j + 1)
            ]

        else:

            neighbours = [
                (i - 1, j),
                (i + 1, j),

                (i,     j - 1),
                (i + 1, j - 1),

                (i,     j + 1),
                (i + 1, j + 1)
            ]

        for neighbour_index in neighbours:

            if neighbour_index not in index_to_id:
                continue

            nid = index_to_id[
                neighbour_index
            ]

            if cid >= nid:
                continue

            qi = cells[cid].centroid
            qj = cells[nid].centroid

            weight = np.linalg.norm(
                qi - qj
            )

            graph.add_edge(
                cid,
                nid,
                weight=weight
            )

    return Tessellation(
        tau="hex",
        nominal_cell_area=nominal_cell_area,
        side_length=side,
        cells=cells,
        index_to_id=index_to_id,
        graph=graph,
        anchor=anchor,
        phase=phase,
        orientation_deg=orientation_deg
    )


# ============================================================
# TRIANGULAR TESSELLATION
# ============================================================

def build_triangle_tessellation(
    omega,
    nominal_cell_area,
    anchor=None,
    phase=(0.0, 0.0),
    orientation_deg=0.0
):
    validate_domain(omega)

    side = math.sqrt(
        4.0
        * nominal_cell_area
        /
        math.sqrt(3.0)
    )

    h_tri = (
        math.sqrt(3.0)
        * side
        / 2.0
    )

    if anchor is None:
        anchor = np.array([
            omega.centroid.x,
            omega.centroid.y
        ])
    else:
        anchor = np.asarray(
            anchor,
            dtype=float
        )

    phase = np.asarray(
        phase,
        dtype=float
    )

    # Up triangle (0, 0) centroid = anchor
    origin = np.array([
        anchor[0] - side / 2.0,
        anchor[1] - h_tri / 3.0
    ])

    e1 = np.array([
        side,
        0.0
    ])

    e2 = np.array([
        side / 2.0,
        h_tri
    ])

    R = domain_radius(
        omega,
        anchor
    )

    margin = (
        2.0 * side
        +
        np.linalg.norm(phase)
    )

    n_i = (
        math.ceil(
            (R + margin)
            /
            h_tri
        )
        + 4
    )

    n_j = (
        math.ceil(
            (R + margin)
            /
            side
        )
        + n_i
        + 4
    )

    cells = {}
    index_to_id = {}

    cell_id = 0

    for i in range(-n_i, n_i + 1):
        for j in range(-n_j, n_j + 1):

            p00 = (
                origin
                + j * e1
                + i * e2
            )

            p10 = (
                p00 + e1
            )

            p01 = (
                p00 + e2
            )

            p11 = (
                p00
                + e1
                + e2
            )

            # -------------------------------------------------
            # UPWARD TRIANGLE
            # -------------------------------------------------

            tri_up = Polygon([
                tuple(p00),
                tuple(p10),
                tuple(p01)
            ])

            tri_up = transform_cell(
                tri_up,
                anchor,
                phase,
                orientation_deg
            )

            if omega.contains(
                tri_up.centroid
            ):

                key = (
                    i,
                    j,
                    0
                )

                cell = create_retained_cell(
                    cell_id=cell_id,
                    lattice_index=key,
                    polygon=tri_up,
                    omega=omega,
                    nominal_cell_area=nominal_cell_area,
                    orientation="up"
                )

                cells[cell_id] = cell
                index_to_id[key] = cell_id

                cell_id += 1

            # -------------------------------------------------
            # DOWNWARD TRIANGLE
            # -------------------------------------------------

            tri_down = Polygon([
                tuple(p10),
                tuple(p11),
                tuple(p01)
            ])

            tri_down = transform_cell(
                tri_down,
                anchor,
                phase,
                orientation_deg
            )

            if omega.contains(
                tri_down.centroid
            ):

                key = (
                    i,
                    j,
                    1
                )

                cell = create_retained_cell(
                    cell_id=cell_id,
                    lattice_index=key,
                    polygon=tri_down,
                    omega=omega,
                    nominal_cell_area=nominal_cell_area,
                    orientation="down"
                )

                cells[cell_id] = cell
                index_to_id[key] = cell_id

                cell_id += 1

    graph = nx.Graph()

    graph.add_nodes_from(
        cells.keys()
    )

    for key, cid in index_to_id.items():

        i, j, orientation_index = (
            key
        )

        if orientation_index == 0:

            neighbours = [
                (i - 1, j,     1),
                (i,     j,     1),
                (i,     j - 1, 1)
            ]

        else:

            neighbours = [
                (i,     j,     0),
                (i,     j + 1, 0),
                (i + 1, j,     0)
            ]

        for neighbour_index in neighbours:

            if neighbour_index not in index_to_id:
                continue

            nid = index_to_id[
                neighbour_index
            ]

            if cid >= nid:
                continue

            qi = cells[cid].centroid
            qj = cells[nid].centroid

            weight = np.linalg.norm(
                qi - qj
            )

            graph.add_edge(
                cid,
                nid,
                weight=weight
            )

    return Tessellation(
        tau="triangle",
        nominal_cell_area=nominal_cell_area,
        side_length=side,
        cells=cells,
        index_to_id=index_to_id,
        graph=graph,
        anchor=anchor,
        phase=phase,
        orientation_deg=orientation_deg
    )


# ============================================================
# GENERAL CONSTRUCTOR
# ============================================================

def build_tessellation(
    omega,
    tau,
    nominal_cell_area,
    anchor=None,
    phase=(0.0, 0.0),
    orientation_deg=0.0
):
    if tau == "square":

        return build_square_tessellation(
            omega=omega,
            nominal_cell_area=nominal_cell_area,
            anchor=anchor,
            phase=phase,
            orientation_deg=orientation_deg
        )

    elif tau == "triangle":

        return build_triangle_tessellation(
            omega=omega,
            nominal_cell_area=nominal_cell_area,
            anchor=anchor,
            phase=phase,
            orientation_deg=orientation_deg
        )

    elif tau == "hex":

        return build_hex_tessellation(
            omega=omega,
            nominal_cell_area=nominal_cell_area,
            anchor=anchor,
            phase=phase,
            orientation_deg=orientation_deg
        )

    raise ValueError(
        f"Unknown tessellation: {tau}"
    )


# ============================================================
# REPRESENTED REGION
# ============================================================

def represented_region(
    tessellation
):
    """
    Complete represented region:

        Omega_hat_tau = union_i h_{tau,i}

    Uses full retained cells, not clipped supports.

    Precision normalization avoids numerical microgaps.
    """

    if len(tessellation.cells) == 0:
        return Polygon()

    grid_size = max(
        1e-12,
        1e-9 * tessellation.side_length
    )

    polygons = [
        set_precision(
            cell.full_polygon,
            grid_size=grid_size
        )
        for cell
        in tessellation.cells.values()
    ]

    return union_all(
        polygons
    )


# ============================================================
# INITIAL STATE
# ============================================================

def initial_state_from_position(
    tessellation,
    x0
):
    """
    Returns the retained cell whose centroid is nearest to x0.

    Numerical ties are resolved using the lowest cell ID.
    """

    x0 = np.asarray(
        x0,
        dtype=float
    )

    if len(tessellation.cells) == 0:
        raise ValueError(
            "Tessellation contains no retained cells."
        )

    distances = {
        cid:
            np.linalg.norm(
                cell.centroid - x0
            )
        for cid, cell
        in tessellation.cells.items()
    }

    d_min = min(
        distances.values()
    )

    tol = distance_tolerance(
        tessellation.nominal_cell_area
    )

    tied_ids = [
        cid
        for cid, d
        in distances.items()
        if abs(d - d_min) <= tol
    ]

    return min(
        tied_ids
    )


# ============================================================
# BASIC TESSELLATION VALIDATION
# ============================================================

def validate_tessellation(
    omega,
    tessellation
):
    Ac = (
        tessellation.nominal_cell_area
    )

    area_tol = area_tolerance(
        Ac
    )

    # 1. All retained centroids must lie strictly inside Omega.
    assert all(
        omega.contains(
            Point(cell.centroid)
        )
        for cell
        in tessellation.cells.values()
    )

    # 2. Full retained cells must have nominal area Ac.
    assert all(
        abs(
            cell.full_area - Ac
        )
        <= area_tol
        for cell
        in tessellation.cells.values()
    )

    # 3. Valid support area cannot exceed full cell area.
    assert all(
        cell.valid_area
        <= Ac + area_tol
        for cell
        in tessellation.cells.values()
    )

    # 4. Boundary flag must agree with the mathematical definition.
    assert all(
        cell.is_boundary
        ==
        (
            cell.valid_area
            < Ac - area_tol
        )
        for cell
        in tessellation.cells.values()
    )

    # 5. Graph nodes must correspond exactly to retained cells.
    assert (
        set(tessellation.graph.nodes)
        ==
        set(tessellation.cells.keys())
    )


# ============================================================
# EXPECTED NEIGHBOUR VALIDATION
# ============================================================

def rotate_vector(
    vector,
    angle_deg
):
    theta = math.radians(
        angle_deg
    )

    rotation_matrix = np.array([
        [
            math.cos(theta),
            -math.sin(theta)
        ],
        [
            math.sin(theta),
            math.cos(theta)
        ]
    ])

    return (
        rotation_matrix
        @ np.asarray(
            vector,
            dtype=float
        )
    )


def theoretical_neighbour_offsets(
    tessellation,
    cell
):
    l = tessellation.side_length

    # --------------------------------------------------------
    # SQUARE
    # --------------------------------------------------------

    if tessellation.tau == "square":

        offsets = [
            np.array([+l, 0.0]),
            np.array([-l, 0.0]),
            np.array([0.0, +l]),
            np.array([0.0, -l])
        ]

    # --------------------------------------------------------
    # HEXAGONAL
    # --------------------------------------------------------

    elif tessellation.tau == "hex":

        h = (
            math.sqrt(3.0)
            * l
        )

        offsets = [
            np.array([0.0, +h]),
            np.array([0.0, -h]),

            np.array([
                +1.5 * l,
                +h / 2.0
            ]),

            np.array([
                +1.5 * l,
                -h / 2.0
            ]),

            np.array([
                -1.5 * l,
                +h / 2.0
            ]),

            np.array([
                -1.5 * l,
                -h / 2.0
            ])
        ]

    # --------------------------------------------------------
    # TRIANGULAR
    # --------------------------------------------------------

    elif tessellation.tau == "triangle":

        h = (
            math.sqrt(3.0)
            * l
            / 2.0
        )

        d = (
            l
            /
            math.sqrt(3.0)
        )

        if cell.orientation == "up":

            offsets = [
                np.array([
                    0.0,
                    -d
                ]),

                np.array([
                    +l / 2.0,
                    +h / 3.0
                ]),

                np.array([
                    -l / 2.0,
                    +h / 3.0
                ])
            ]

        elif cell.orientation == "down":

            offsets = [
                np.array([
                    0.0,
                    +d
                ]),

                np.array([
                    +l / 2.0,
                    -h / 3.0
                ]),

                np.array([
                    -l / 2.0,
                    -h / 3.0
                ])
            ]

        else:

            raise ValueError(
                "Triangular cell without orientation."
            )

    else:

        raise ValueError(
            f"Unknown tessellation: "
            f"{tessellation.tau}"
        )

    return [
        rotate_vector(
            offset,
            tessellation.orientation_deg
        )
        for offset in offsets
    ]


def validate_expected_degree(
    omega,
    tessellation
):
    """
    Checks that each retained cell has exactly the number of
    graph neighbours implied by the infinite lattice and the
    centroid-retention rule.
    """

    bad_cells = []

    for cid, cell in (
        tessellation.cells.items()
    ):

        expected_degree = 0

        offsets = (
            theoretical_neighbour_offsets(
                tessellation,
                cell
            )
        )

        for offset in offsets:

            q_neighbour = (
                cell.centroid
                +
                offset
            )

            if omega.contains(
                Point(
                    q_neighbour[0],
                    q_neighbour[1]
                )
            ):
                expected_degree += 1

        actual_degree = (
            tessellation.graph.degree[
                cid
            ]
        )

        if (
            actual_degree
            != expected_degree
        ):

            bad_cells.append({
                "id":
                    cid,

                "actual_degree":
                    actual_degree,

                "expected_degree":
                    expected_degree,

                "centroid":
                    cell.centroid
            })

    if bad_cells:

        raise AssertionError(
            f"{tessellation.tau}: "
            f"{len(bad_cells)} cells have "
            f"an incorrect graph degree. "
            f"First cases: {bad_cells[:5]}"
        )

    return bad_cells