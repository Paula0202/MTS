from pathlib import Path
import math
import os
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import networkx as nx

from shapely import contains_xy, union_all
from shapely.geometry import Point
from shapely.strtree import STRtree

from domains import make_normalized_domains

from priors import (
    make_single_hotspot_prior,
    make_multimodal_prior,
)

from tessellations import (
    build_tessellation,
    initial_state_from_position,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = (
    BASE_DIR
    / "results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RAW_OUTPUT = (
    OUTPUT_DIR
    / "search_experiments.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "search_experiments_summary.csv"
)

ORIENTATION_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "search_experiments_orientation_summary.csv"
)


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

A0 = 100.0
L0 = np.sqrt(A0)

DOMAIN_ORDER = [
    "Square",
    "Circle",
    "Triangle",
    "Convex",
    "Concave",
]

PRIOR_ORDER = [
    "single_hotspot",
    "multimodal",
]

NREF_ORDER = [
    100,
    300,
    500,
    800,
]

TAU_ORDER = [
    "square",
    "triangle",
    "hex",
]

STRATEGY_ORDER = [
    "lawnmower",
    "greedy",
]

N_MC = 1000

BASE_SEED = 12345


# ============================================================
# COMMON LAWNMOWER CONFIGURATION
# ============================================================

# The systematic baseline is evaluated over a uniform set of physical
# sweep orientations. Angles are line orientations, so 180 degrees is
# equivalent to 0 degrees and is intentionally excluded.
LAWNMOWER_ANGLES_DEG = tuple(
    float(angle)
    for angle in range(0, 180, 15)
)

# Common physical strip spacing:
#
#     Delta_sweep = sqrt(A(Omega) / N_ref)
#
# The same value is used for square, triangular, and hexagonal
# tessellations at a given domain and nominal resolution.

# Tiny common phase offset used only to avoid exact half-band ties.
# It is independent of the tessellation and negligible relative to the
# physical strip spacing.
LAWNMOWER_PHASE_EPS_FRACTION = 1e-9


# ============================================================
# PARALLEL EXECUTION
# ============================================================

# Parallelization is performed over independent base conditions
# (domain, prior, N_ref). Each worker keeps all 12 lawnmower angles,
# the three tessellations, and the same N_MC physical targets together.
# This preserves the paired experimental design.
MAX_WORKERS = min(
    6,
    max(
        1,
        (os.cpu_count() or 2) - 1,
    ),
)

# ============================================================
# SEARCH HORIZON
# ============================================================

# Provisional generous horizon.
#
# Before the final paper results, inspect the censoring rate
# and freeze K_max explicitly.
#
# Importantly, K_max is the SAME for the three tessellations
# at a given N_ref.

K_MAX_BY_NREF = {
    100: 1000,
    300: 3000,
    500: 5000,
    800: 8000,
}


# ============================================================
# UAV SPEED
# ============================================================

# If a physical UAV speed is later fixed, set it here.
# Otherwise T_det is not reported.

UAV_SPEED = None


# ============================================================
# NUMERICAL INTEGRATION
# ============================================================

PRIOR_QUADRATURE_SIDE = 20

EPS = 1e-15


# ============================================================
# PRIOR DEFINITION
# ============================================================

def prior_spec(
    omega,
    prior_name,
):
    """
    Return the Gaussian parameters used to construct the
    continuous prior.

    These parameters are also used to sample physical target
    locations directly from the continuous prior conditioned
    on the common support.
    """

    center = np.array([
        omega.centroid.x,
        omega.centroid.y,
    ])

    if prior_name == "single_hotspot":

        means = np.array([
            center
            + np.array([
                0.15 * L0,
                -0.10 * L0,
            ])
        ])

        sigmas = np.array([
            0.15 * L0
        ])

        weights = np.array([
            1.0
        ])

        return (
            means,
            sigmas,
            weights,
        )

    if prior_name == "multimodal":

        means = np.array([
            center
            + np.array([
                -0.20 * L0,
                -0.12 * L0,
            ]),

            center
            + np.array([
                +0.18 * L0,
                +0.16 * L0,
            ]),
        ])

        sigmas = np.array([
            0.12 * L0,
            0.12 * L0,
        ])

        weights = np.array([
            0.6,
            0.4,
        ])

        return (
            means,
            sigmas,
            weights,
        )

    raise ValueError(
        f"Unknown prior: {prior_name}"
    )


def make_prior(
    omega,
    prior_name,
    normalization_grid=400,
):
    """
    Construct the same continuous prior used in the
    probabilistic representation experiments.
    """

    means, sigmas, weights = (
        prior_spec(
            omega,
            prior_name,
        )
    )

    if prior_name == "single_hotspot":

        return make_single_hotspot_prior(
            omega=omega,
            mean=means[0],
            sigma=sigmas[0],
            n_grid=normalization_grid,
        )

    if prior_name == "multimodal":

        return make_multimodal_prior(
            omega=omega,
            means=[
                m
                for m in means
            ],
            sigmas=[
                float(s)
                for s in sigmas
            ],
            weights=[
                float(w)
                for w in weights
            ],
            n_grid=normalization_grid,
        )

    raise ValueError(
        f"Unknown prior: {prior_name}"
    )


# ============================================================
# GENERIC PRIOR EVALUATION
# ============================================================

def evaluate_prior(
    p0,
    points,
):
    """
    Evaluate p0 at an array of shape (N, 2).

    Several common function signatures are supported so that
    this script remains compatible with the existing priors.py.
    """

    points = np.asarray(
        points,
        dtype=float,
    )

    if points.ndim == 1:
        points = points.reshape(
            1,
            2,
        )

    x = points[:, 0]
    y = points[:, 1]

    # --------------------------------------------------------
    # p0(x, y)
    # --------------------------------------------------------

    try:

        values = p0(
            x,
            y,
        )

        values = np.asarray(
            values,
            dtype=float,
        )

        if values.size == len(points):

            return values.reshape(-1)

    except Exception:
        pass

    # --------------------------------------------------------
    # p0(points)
    # --------------------------------------------------------

    try:

        values = p0(
            points
        )

        values = np.asarray(
            values,
            dtype=float,
        )

        if values.size == len(points):

            return values.reshape(-1)

    except Exception:
        pass

    # --------------------------------------------------------
    # p0.pdf(...)
    # --------------------------------------------------------

    if hasattr(
        p0,
        "pdf",
    ):

        try:

            values = p0.pdf(
                points
            )

            values = np.asarray(
                values,
                dtype=float,
            )

            if values.size == len(points):

                return values.reshape(-1)

        except Exception:
            pass

    # --------------------------------------------------------
    # Scalar fallback
    # --------------------------------------------------------

    values = []

    for point in points:

        success = False

        try:

            value = p0(
                point
            )

            values.append(
                float(value)
            )

            success = True

        except Exception:
            pass

        if success:
            continue

        try:

            value = p0(
                point[0],
                point[1],
            )

            values.append(
                float(value)
            )

            success = True

        except Exception:
            pass

        if not success:

            raise TypeError(
                "Could not determine how to evaluate p0. "
                "Adapt evaluate_prior() to the signature "
                "used in priors.py."
            )

    return np.asarray(
        values,
        dtype=float,
    )


# ============================================================
# TESSELLATION CONSTRUCTION
# ============================================================

def build_canonical_tessellation(
    omega,
    tau,
    N_ref,
):
    """
    Canonical search configuration:
        phase = (0,0)
        orientation = 0 degrees
        anchor = domain centroid
    """

    x0 = np.array([
        omega.centroid.x,
        omega.centroid.y,
    ])

    A_c = (
        omega.area
        / N_ref
    )

    return build_tessellation(
        omega=omega,
        tau=tau,
        nominal_cell_area=A_c,
        anchor=x0,
        phase=(0.0, 0.0),
        orientation_deg=0.0,
    )


# ============================================================
# REPRESENTED SUPPORT
# ============================================================

def represented_support(
    tessellation,
):
    """
    C_tau = union of valid cell supports.
    """

    geometries = [
        cell.valid_support
        for cell
        in tessellation.cells.values()
        if not cell.valid_support.is_empty
    ]

    if not geometries:

        raise ValueError(
            "Tessellation has no represented support."
        )

    return union_all(
        geometries
    )


def common_support(
    tessellations,
):
    """
    C_cap = C_square ∩ C_triangle ∩ C_hex.
    """

    supports = {
        tau:
            represented_support(
                tess
            )
        for tau, tess
        in tessellations.items()
    }

    C_cap = (
        supports["square"]
        .intersection(
            supports["triangle"]
        )
        .intersection(
            supports["hex"]
        )
    )

    if C_cap.is_empty:

        raise ValueError(
            "Common represented support is empty."
        )

    if not C_cap.is_valid:

        C_cap = C_cap.buffer(
            0
        )

    return C_cap


# ============================================================
# TARGET SAMPLING
# ============================================================

def sample_targets_from_common_prior(
    omega,
    prior_name,
    C_cap,
    n_samples,
    seed,
):
    """
    Sample x* ~ p0 conditioned on C_cap.

    Because p0 is a Gaussian or Gaussian mixture conditioned
    on Omega, sampling from the corresponding unconditioned
    Gaussian mixture and rejecting samples outside C_cap
    yields exactly the desired density proportional to

        p0(x) I_Ccap(x).

    The same sampled physical targets are subsequently reused
    across all tessellations and strategies.
    """

    means, sigmas, weights = (
        prior_spec(
            omega,
            prior_name,
        )
    )

    rng = np.random.default_rng(
        seed
    )

    samples = []

    n_components = len(
        weights
    )

    batch_size = max(
        2000,
        4 * n_samples,
    )

    while len(samples) < n_samples:

        components = rng.choice(
            n_components,
            size=batch_size,
            p=weights,
        )

        points = np.empty(
            (
                batch_size,
                2,
            ),
            dtype=float,
        )

        for component in range(
            n_components
        ):

            mask = (
                components
                == component
            )

            count = int(
                np.sum(
                    mask
                )
            )

            if count == 0:
                continue

            points[mask] = rng.normal(
                loc=means[component],
                scale=sigmas[component],
                size=(
                    count,
                    2,
                ),
            )

        inside = contains_xy(
            C_cap,
            points[:, 0],
            points[:, 1],
        )

        accepted = (
            points[
                inside
            ]
        )

        samples.extend(
            accepted.tolist()
        )

    samples = np.asarray(
        samples[:n_samples],
        dtype=float,
    )

    return samples


# ============================================================
# PRIOR MASS IN ONE GEOMETRY
# ============================================================

def integrate_prior_over_geometry(
    p0,
    geometry,
    n_side=PRIOR_QUADRATURE_SIDE,
):
    """
    Deterministic midpoint-grid approximation of

        integral_geometry p0(x) dx.

    Only relative masses are required, since the complete set
    of cell masses is normalized afterwards.
    """

    if (
        geometry.is_empty
        or geometry.area <= 0.0
    ):
        return 0.0

    minx, miny, maxx, maxy = (
        geometry.bounds
    )

    width = (
        maxx - minx
    )

    height = (
        maxy - miny
    )

    if (
        width <= 0.0
        or height <= 0.0
    ):
        return 0.0

    dx = (
        width
        / n_side
    )

    dy = (
        height
        / n_side
    )

    xs = (
        minx
        + (
            np.arange(
                n_side
            )
            + 0.5
        )
        * dx
    )

    ys = (
        miny
        + (
            np.arange(
                n_side
            )
            + 0.5
        )
        * dy
    )

    X, Y = np.meshgrid(
        xs,
        ys,
    )

    x_flat = X.ravel()
    y_flat = Y.ravel()

    mask = contains_xy(
        geometry,
        x_flat,
        y_flat,
    )

    if np.any(
        mask
    ):

        points = np.column_stack([
            x_flat[mask],
            y_flat[mask],
        ])

        density = evaluate_prior(
            p0,
            points,
        )

        integral = (
            float(
                np.sum(
                    density
                )
            )
            * dx
            * dy
        )

    else:

        integral = 0.0

    # --------------------------------------------------------
    # Fallback for tiny geometries that may be missed by the
    # midpoint grid.
    # --------------------------------------------------------

    if (
        integral <= 0.0
        and geometry.area > 0.0
    ):

        point = (
            geometry
            .representative_point()
        )

        density = evaluate_prior(
            p0,
            np.array([[
                point.x,
                point.y,
            ]]),
        )[0]

        integral = (
            float(density)
            * geometry.area
        )

    return max(
        0.0,
        integral,
    )


# ============================================================
# DISCRETE COMMON PRIOR
# ============================================================

def discretize_common_prior(
    tessellation,
    p0,
    C_cap,
):
    """
    Compute

        B_0,tau^cap(h_i)
        = integral_{C_i ∩ C_cap} p0(x) dx

    followed by normalization.

    The normalization constant of p0^cap cancels, so it is not
    necessary to compute it separately.
    """

    cell_ids = sorted(
        tessellation.cells.keys()
    )

    max_id = max(
        cell_ids
    )

    belief = np.zeros(
        max_id + 1,
        dtype=float,
    )

    for cid in cell_ids:

        cell = (
            tessellation
            .cells[cid]
        )

        geometry = (
            cell.valid_support
            .intersection(
                C_cap
            )
        )

        belief[cid] = (
            integrate_prior_over_geometry(
                p0=p0,
                geometry=geometry,
            )
        )

    total = float(
        np.sum(
            belief
        )
    )

    if total <= 0.0:

        raise ValueError(
            "Discrete common prior has zero mass."
        )

    belief /= total

    return belief


# ============================================================
# TARGET -> CELL MAPPING
# ============================================================

class TargetCellMapper:

    def __init__(
        self,
        tessellation,
    ):

        self.tessellation = (
            tessellation
        )

        self.cell_ids = sorted(
            tessellation.cells.keys()
        )

        self.geometries = [
            tessellation
            .cells[cid]
            .valid_support
            for cid
            in self.cell_ids
        ]

        self.tree = STRtree(
            self.geometries
        )

    def map_point(
        self,
        xy,
    ):
        """
        Return the retained cell whose valid support contains
        the physical target location.
        """

        point = Point(
            float(xy[0]),
            float(xy[1]),
        )

        candidates = (
            self.tree.query(
                point
            )
        )

        matches = []

        for index in np.atleast_1d(
            candidates
        ):

            index = int(
                index
            )

            geometry = (
                self.geometries[
                    index
                ]
            )

            if geometry.covers(
                point
            ):

                matches.append(
                    self.cell_ids[
                        index
                    ]
                )

        if not matches:

            # Numerical fallback.
            for cid in self.cell_ids:

                if (
                    self.tessellation
                    .cells[cid]
                    .valid_support
                    .covers(point)
                ):

                    matches.append(
                        cid
                    )

        if not matches:

            raise ValueError(
                "Target point could not be mapped "
                "to a retained cell."
            )

        # Boundary ties have probability zero for a continuous
        # target distribution, but are resolved deterministically.
        return min(
            matches
        )


# ============================================================
# ROUTE RECORDER
# ============================================================

class RouteRecorder:

    def __init__(
        self,
        tessellation,
        B0,
        s0,
        K_max,
    ):

        self.tessellation = (
            tessellation
        )

        self.B0 = B0

        self.K_max = K_max

        self.states = [
            s0
        ]

        self.cum_distance = [
            0.0
        ]

        self.cum_unique = [
            1
        ]

        self.cum_prior_mass = [
            float(
                B0[s0]
            )
        ]

        self.cum_revisits = [
            0
        ]

        self.visited = {
            s0
        }

    @property
    def current(
        self,
    ):

        return self.states[-1]

    @property
    def K(
        self,
    ):

        return (
            len(self.states)
            - 1
        )

    def can_step(
        self,
    ):

        return (
            self.K
            < self.K_max
        )

    def append(
        self,
        next_state,
    ):

        if not self.can_step():

            return False

        current = (
            self.current
        )

        edge_data = (
            self.tessellation
            .graph[current][next_state]
        )

        distance = float(
            edge_data[
                "weight"
            ]
        )

        revisit = (
            next_state
            in self.visited
        )

        if revisit:

            n_unique = (
                self.cum_unique[-1]
            )

            mass = (
                self.cum_prior_mass[-1]
            )

            revisits = (
                self.cum_revisits[-1]
                + 1
            )

        else:

            self.visited.add(
                next_state
            )

            n_unique = (
                self.cum_unique[-1]
                + 1
            )

            mass = (
                self.cum_prior_mass[-1]
                + float(
                    self.B0[
                        next_state
                    ]
                )
            )

            revisits = (
                self.cum_revisits[-1]
            )

        self.states.append(
            next_state
        )

        self.cum_distance.append(
            self.cum_distance[-1]
            + distance
        )

        self.cum_unique.append(
            n_unique
        )

        self.cum_prior_mass.append(
            min(
                1.0,
                mass,
            )
        )

        self.cum_revisits.append(
            revisits
        )

        return True

    def first_visit_indices(
        self,
    ):

        first = {}

        for k, state in enumerate(
            self.states
        ):

            if state not in first:

                first[state] = k

        return first

# ============================================================
# COMMON PHYSICAL-BAND LAWNMOWER SPECIFICATION
# ============================================================

def build_common_lawnmower_spec(
    omega,
    N_ref,
    x0,
    angle_deg,
):
    """
    Build one tessellation-independent lawnmower specification.

    For a given continuous domain, nominal resolution, and physical
    orientation, the same strip coordinate system is reused unchanged
    for square, triangular, and hexagonal representations.

    The physical spacing is

        Delta_sweep = sqrt(A(Omega) / N_ref).

    The sweep is defined by two orthonormal physical axes:

        u : direction along a strip,
        v : transverse direction between strips.

    The common band centers are

        v = v_phase + k * Delta_sweep,

    where v_phase is anchored at the common physical initial position
    x0. Cell assignment is performed later using centroid coordinates
    only; lattice indices and cell shape are never used to define the
    systematic ordering.

    The four symmetry-equivalent serpentine conventions are resolved
    from continuous geometry only, by selecting the convention whose
    first domain-bounding corner is closest to x0. No graph, prior, or
    target information enters this choice.
    """

    x0 = np.asarray(
        x0,
        dtype=float,
    )

    nominal_cell_area = (
        float(omega.area)
        / float(N_ref)
    )

    spacing = math.sqrt(
        nominal_cell_area
    )

    angle_deg = (
        float(angle_deg)
        % 180.0
    )

    theta = math.radians(
        angle_deg
    )

    u = np.array([
        math.cos(theta),
        math.sin(theta),
    ])

    v = np.array([
        -math.sin(theta),
        math.cos(theta),
    ])

    # Continuous-domain projection bounds. Using the convex hull keeps
    # this definition independent of the tessellation while remaining
    # valid for concave search domains.
    hull = omega.convex_hull

    hull_points = np.asarray(
        hull.exterior.coords,
        dtype=float,
    )[:-1]

    u_values = (
        hull_points
        @ u
    )

    v_values = (
        hull_points
        @ v
    )

    u_min = float(
        np.min(u_values)
    )

    u_max = float(
        np.max(u_values)
    )

    v_min = float(
        np.min(v_values)
    )

    v_max = float(
        np.max(v_values)
    )

    u0 = float(
        x0 @ u
    )

    v0 = float(
        x0 @ v
    )

    # Negligible common offset to avoid exact half-band ties.
    v_phase = (
        v0
        + LAWNMOWER_PHASE_EPS_FRACTION
        * spacing
    )

    candidates = []

    for reverse_strips in [
        False,
        True,
    ]:

        for reverse_first_strip in [
            False,
            True,
        ]:

            first_v = (
                v_max
                if reverse_strips
                else v_min
            )

            first_u = (
                u_max
                if reverse_first_strip
                else u_min
            )

            entry_distance = math.hypot(
                first_u - u0,
                first_v - v0,
            )

            candidates.append(
                {
                    "entry_distance_continuous":
                        float(entry_distance),

                    "reverse_strips":
                        reverse_strips,

                    "reverse_first_strip":
                        reverse_first_strip,
                }
            )

    candidates.sort(
        key=lambda candidate: (
            candidate[
                "entry_distance_continuous"
            ],
            candidate[
                "reverse_strips"
            ],
            candidate[
                "reverse_first_strip"
            ],
        )
    )

    selected = candidates[0]

    return {
        "angle_deg":
            angle_deg,

        "spacing":
            float(spacing),

        "nominal_cell_area":
            float(nominal_cell_area),

        "u":
            u,

        "v":
            v,

        "u_min":
            u_min,

        "u_max":
            u_max,

        "v_min":
            v_min,

        "v_max":
            v_max,

        "v_phase":
            float(v_phase),

        "entry_distance_continuous":
            selected[
                "entry_distance_continuous"
            ],

        "reverse_strips":
            selected[
                "reverse_strips"
            ],

        "reverse_first_strip":
            selected[
                "reverse_first_strip"
            ],
    }


# ============================================================
# COMMON PHYSICAL BANDS -> DISCRETE STATE ORDER
# ============================================================

def common_lawnmower_order(
    tessellation,
    node_ids,
    lawnmower_spec,
):
    """
    Map the same physical strip partition onto one tessellation.

    Every reachable retained cell is assigned to the nearest common
    physical band using ONLY its physical centroid q_i:

        k_i = round((q_i . v - v_phase) / Delta_sweep).

    States assigned to the same band are ordered by their physical
    coordinate q_i . u. Consecutive bands alternate direction to form
    a serpentine systematic sweep.

    Important fairness property
    ---------------------------
    This function never uses lattice_index, cell shape, graph degree,
    prior mass, or target position to construct the desired order.
    Square, triangular, and hexagonal grids therefore receive exactly
    the same physical band definition and ordering rule.
    """

    node_ids = set(
        node_ids
    )

    u = (
        lawnmower_spec[
            "u"
        ]
    )

    v = (
        lawnmower_spec[
            "v"
        ]
    )

    spacing = float(
        lawnmower_spec[
            "spacing"
        ]
    )

    v_phase = float(
        lawnmower_spec[
            "v_phase"
        ]
    )

    strips = {}
    strip_by_state = {}

    for cid in node_ids:

        centroid = np.asarray(
            tessellation
            .cells[cid]
            .centroid,
            dtype=float,
        )

        along_strip = float(
            centroid @ u
        )

        transverse = float(
            centroid @ v
        )

        normalized_band = (
            (transverse - v_phase)
            / spacing
        )

        # np.rint gives a deterministic nearest-band assignment. The
        # tiny common phase offset above prevents systematic exact ties.
        strip_key = int(
            np.rint(
                normalized_band
            )
        )

        strips.setdefault(
            strip_key,
            [],
        )

        strips[
            strip_key
        ].append(
            (
                along_strip,
                cid,
            )
        )

        strip_by_state[
            cid
        ] = strip_key

    strip_keys = sorted(
        strips.keys()
    )

    if lawnmower_spec[
        "reverse_strips"
    ]:
        strip_keys.reverse()

    order = []

    for strip_number, strip_key in enumerate(
        strip_keys
    ):

        entries = (
            strips[
                strip_key
            ]
        )

        entries.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

        reverse_this_strip = (
            lawnmower_spec[
                "reverse_first_strip"
            ]
            if strip_number % 2 == 0
            else not lawnmower_spec[
                "reverse_first_strip"
            ]
        )

        if reverse_this_strip:
            entries.reverse()

        order.extend(
            cid
            for _, cid
            in entries
        )

    if len(order) != len(node_ids):
        raise RuntimeError(
            "Common physical-band lawnmower did not assign "
            "every reachable state exactly once."
        )

    if len(set(order)) != len(order):
        raise RuntimeError(
            "Common physical-band lawnmower assigned at least "
            "one reachable state more than once."
        )

    return (
        order,
        strip_by_state,
        len(strip_keys),
    )


# ============================================================
# ANALYSE INTENDED LAWNMOWER ORDER
# ============================================================

def analyse_lawnmower_order(
    tessellation,
    order,
    strip_by_state,
):
    """
    Quantify how the common physical-band systematic ordering maps to
    one navigation graph.

    Consecutive desired states are classified as directly adjacent or
    as requiring a weighted shortest-path connector. Non-adjacent pairs
    are separated into within-band and between-band transitions.
    """

    G = (
        tessellation.graph
    )

    n_pairs = max(
        0,
        len(order) - 1,
    )

    n_direct = 0
    n_nonadjacent = 0

    n_interstrip = 0
    n_intrastrip = 0

    n_nonadjacent_interstrip = 0
    n_nonadjacent_intrastrip = 0

    for state_a, state_b in zip(
        order[:-1],
        order[1:],
    ):

        strip_a = (
            strip_by_state[
                state_a
            ]
        )

        strip_b = (
            strip_by_state[
                state_b
            ]
        )

        same_strip = (
            strip_a
            == strip_b
        )

        if same_strip:
            n_intrastrip += 1
        else:
            n_interstrip += 1

        if G.has_edge(
            state_a,
            state_b,
        ):
            n_direct += 1

        else:
            n_nonadjacent += 1

            if same_strip:
                n_nonadjacent_intrastrip += 1
            else:
                n_nonadjacent_interstrip += 1

    nonadjacent_fraction = (
        n_nonadjacent
        / n_pairs
        if n_pairs > 0
        else 0.0
    )

    return {
        "order_pairs":
            n_pairs,

        "direct_pairs":
            n_direct,

        "nonadjacent_pairs":
            n_nonadjacent,

        "nonadjacent_fraction":
            nonadjacent_fraction,

        "interstrip_pairs":
            n_interstrip,

        "intrastrip_pairs":
            n_intrastrip,

        "nonadjacent_interstrip":
            n_nonadjacent_interstrip,

        "nonadjacent_intrastrip":
            n_nonadjacent_intrastrip,
    }


# ============================================================
# COMMON PHYSICAL-BAND LAWNMOWER ROUTE
# ============================================================

def build_lawnmower_route(
    tessellation,
    B0,
    s0,
    K_max,
    lawnmower_spec,
):
    """
    Execute one common physical-band lawnmower on G_tau.

    The desired systematic state order is defined without consulting
    graph structure. If two consecutive desired states are adjacent,
    the UAV executes that edge directly. Otherwise it uses a WEIGHTED
    shortest path with graph-edge weight equal to physical centroid
    distance.

    All intermediate states are sensed normally. The prescribed sweep
    is not shortened merely because a connector happened to sense a
    future desired state: if the systematic order later returns there,
    the motion is retained and counted as a physical revisit.
    """

    G = (
        tessellation.graph
    )

    reachable = set(
        nx.node_connected_component(
            G,
            s0,
        )
    )

    recorder = RouteRecorder(
        tessellation=tessellation,
        B0=B0,
        s0=s0,
        K_max=K_max,
    )

    (
        desired_order,
        strip_by_state,
        n_nonempty_strips,
    ) = common_lawnmower_order(
        tessellation=tessellation,
        node_ids=reachable,
        lawnmower_spec=(
            lawnmower_spec
        ),
    )

    order_diagnostics = (
        analyse_lawnmower_order(
            tessellation=tessellation,
            order=desired_order,
            strip_by_state=(
                strip_by_state
            ),
        )
    )

    actual_goal_moves = 0
    direct_goal_moves = 0
    shortest_path_connections = 0

    entry_connectors = 0
    interstrip_connectors = 0
    intrastrip_connectors = 0

    connector_steps = 0
    connector_extra_steps = 0
    connector_revisit_steps = 0

    sweep_started = False

    for goal in desired_order:

        if not recorder.can_step():
            break

        current = (
            recorder.current
        )

        if goal == current:
            sweep_started = True
            continue

        goal_strip = (
            strip_by_state[
                goal
            ]
        )

        actual_goal_moves += 1

        if G.has_edge(
            current,
            goal,
        ):

            direct_goal_moves += 1

            path = [
                current,
                goal,
            ]

        else:

            shortest_path_connections += 1

            # Weighted physical shortest path. Edge weights are the
            # Euclidean centroid-to-centroid distances already stored
            # in tessellation.graph.
            path = nx.shortest_path(
                G,
                source=current,
                target=goal,
                weight="weight",
            )

            if not sweep_started:
                entry_connectors += 1

            else:

                current_strip = (
                    strip_by_state[
                        current
                    ]
                )

                if (
                    goal_strip
                    != current_strip
                ):
                    interstrip_connectors += 1
                else:
                    intrastrip_connectors += 1

            n_steps = (
                len(path)
                - 1
            )

            connector_steps += (
                n_steps
            )

            connector_extra_steps += max(
                0,
                n_steps - 1,
            )

            temporary_visited = set(
                recorder.visited
            )

            for state in path[1:]:

                if state in temporary_visited:
                    connector_revisit_steps += 1

                temporary_visited.add(
                    state
                )

        for state in path[1:]:

            if not recorder.append(
                state
            ):
                break

        sweep_started = True

        if not recorder.can_step():
            break

    termination = (
        "kmax"
        if recorder.K >= K_max
        else "component_exhausted"
    )

    shortest_path_fraction = (
        shortest_path_connections
        / actual_goal_moves
        if actual_goal_moves > 0
        else 0.0
    )

    recorder.lawnmower_diagnostics = {
        "angle_deg":
            lawnmower_spec[
                "angle_deg"
            ],

        "spacing":
            lawnmower_spec[
                "spacing"
            ],

        "entry_distance_continuous":
            lawnmower_spec[
                "entry_distance_continuous"
            ],

        "reverse_strips":
            lawnmower_spec[
                "reverse_strips"
            ],

        "reverse_first_strip":
            lawnmower_spec[
                "reverse_first_strip"
            ],

        "n_nonempty_strips":
            n_nonempty_strips,

        **order_diagnostics,

        "actual_goal_moves":
            actual_goal_moves,

        "direct_goal_moves":
            direct_goal_moves,

        "shortest_path_connections":
            shortest_path_connections,

        "shortest_path_fraction":
            shortest_path_fraction,

        "entry_connectors":
            entry_connectors,

        "interstrip_connectors":
            interstrip_connectors,

        "intrastrip_connectors":
            intrastrip_connectors,

        "connector_steps":
            connector_steps,

        "connector_extra_steps":
            connector_extra_steps,

        "connector_revisit_steps":
            connector_revisit_steps,
    }

    return (
        recorder,
        reachable,
        termination,
    )


# ============================================================
# IDEAL BELIEF UPDATE
# ============================================================

def ideal_non_detection_update(
    belief,
    state,
):
    """
    Ideal cell-local sensor:
    after a non-detection, the visited state's probability is
    eliminated and the remaining belief is renormalized.
    """

    belief[
        state
    ] = 0.0

    remaining = float(
        np.sum(
            belief
        )
    )

    if remaining > EPS:

        belief /= remaining

    return belief


# ============================================================
# DETERMINISTIC ARGMAX
# ============================================================

def deterministic_argmax(
    candidates,
    belief,
):
    """
    Highest-belief state with deterministic tie breaking
    through the lowest cell ID.
    """

    candidates = sorted(
        candidates
    )

    values = np.array([
        belief[cid]
        for cid
        in candidates
    ])

    best_value = float(
        np.max(
            values
        )
    )

    tied = [
        cid
        for cid
        in candidates
        if abs(
            belief[cid]
            - best_value
        )
        <= EPS
    ]

    return min(
        tied
    )


# ============================================================
# PROBABILITY-GUIDED GREEDY ROUTE
# ============================================================

def build_greedy_route(
    tessellation,
    B0,
    s0,
    K_max,
):
    """
    Build the full no-detection route for the probability-
    guided greedy strategy.

    Local rule:
        select the unvisited neighbour with highest belief.

    Fallback:
        if all neighbours have already been visited, select the
        highest-belief unvisited reachable state and follow a
        weighted shortest path to it.

    Every intermediate state is processed as a normal sensing
    state.
    """

    G = (
        tessellation.graph
    )

    reachable = set(
        nx.node_connected_component(
            G,
            s0,
        )
    )

    recorder = RouteRecorder(
        tessellation=tessellation,
        B0=B0,
        s0=s0,
        K_max=K_max,
    )

    belief = B0.copy()

    # Initial state has been sensed without detection in the
    # hypothetical no-detection route.
    ideal_non_detection_update(
        belief,
        s0,
    )

    while recorder.can_step():

        current = (
            recorder.current
        )

        unvisited_neighbours = [
            neighbour
            for neighbour
            in G.neighbors(
                current
            )
            if neighbour
            not in recorder.visited
        ]

        # ----------------------------------------------------
        # Local greedy move
        # ----------------------------------------------------

        if unvisited_neighbours:

            goal = deterministic_argmax(
                unvisited_neighbours,
                belief,
            )

            path = [
                current,
                goal,
            ]

        # ----------------------------------------------------
        # Global fallback
        # ----------------------------------------------------

        else:

            unvisited_reachable = [
                state
                for state
                in reachable
                if state
                not in recorder.visited
            ]

            if not unvisited_reachable:
                break

            goal = deterministic_argmax(
                unvisited_reachable,
                belief,
            )

            path = nx.shortest_path(
                G,
                source=current,
                target=goal,
                weight="weight",
            )

        # ----------------------------------------------------
        # Execute path
        # ----------------------------------------------------

        moved = False

        for state in path[1:]:

            if not recorder.append(
                state
            ):

                break

            moved = True

            ideal_non_detection_update(
                belief,
                state,
            )

        if not moved:
            break

    termination = (
        "kmax"
        if recorder.K >= K_max
        else "component_exhausted"
    )

    return (
        recorder,
        reachable,
        termination,
    )


# ============================================================
# TARGET EVALUATION ON A PRECOMPUTED ROUTE
# ============================================================

def evaluate_target_on_route(
    tessellation,
    recorder,
    reachable,
    route_termination,
    target_cell,
    first_visit=None,
):
    """
    Evaluate one physical target realization on a precomputed
    no-detection route.

    first_visit can be supplied once per route to avoid rebuilding the
    same state -> first-visit-index dictionary for every Monte Carlo
    target realization.
    """

    if first_visit is None:
        first_visit = (
            recorder
            .first_visit_indices()
        )

    success = (
        target_cell
        in first_visit
    )

    if success:

        k = int(
            first_visit[
                target_cell
            ]
        )

        K_det = k

        L_det = float(
            recorder
            .cum_distance[k]
        )

        if UAV_SPEED is None:
            T_det = np.nan
        else:
            T_det = (
                L_det
                / UAV_SPEED
            )

        n_unique = int(
            recorder
            .cum_unique[k]
        )

        visited_fraction = (
            n_unique
            / len(
                tessellation.cells
            )
        )

        prior_mass = float(
            recorder
            .cum_prior_mass[k]
        )

        n_revisits = int(
            recorder
            .cum_revisits[k]
        )

        revisit_fraction = (
            n_revisits
            / K_det
            if K_det > 0
            else 0.0
        )

        termination_reason = (
            "detected"
        )

    else:

        K_det = np.nan
        L_det = np.nan
        T_det = np.nan

        n_unique = int(
            recorder
            .cum_unique[-1]
        )

        visited_fraction = (
            n_unique
            / len(
                tessellation.cells
            )
        )

        prior_mass = float(
            recorder
            .cum_prior_mass[-1]
        )

        n_revisits = int(
            recorder
            .cum_revisits[-1]
        )

        K_end = (
            recorder.K
        )

        revisit_fraction = (
            n_revisits
            / K_end
            if K_end > 0
            else 0.0
        )

        if target_cell not in reachable:
            termination_reason = (
                "unreachable_component"
            )

        elif (
            route_termination
            == "kmax"
        ):
            termination_reason = (
                "kmax"
            )

        else:
            termination_reason = (
                "not_reached"
            )

    return {
        "success":
            success,

        "termination_reason":
            termination_reason,

        "K_det":
            K_det,

        "L_det":
            L_det,

        "T_det":
            T_det,

        "n_unique_visited":
            n_unique,

        "visited_fraction":
            visited_fraction,

        "cumulative_prior_mass":
            prior_mass,

        "n_revisits":
            n_revisits,

        "revisit_fraction":
            revisit_fraction,

        "K_end":
            recorder.K,

        "L_end":
            float(
                recorder
                .cum_distance[-1]
            ),
    }


# ============================================================
# CONDITION SEED
# ============================================================

def condition_seed(
    domain_index,
    prior_index,
    resolution_index,
):
    """
    Stable seed independent of Python's randomized hash().

    The seed intentionally does NOT depend on tessellation, search
    strategy, or lawnmower angle. Therefore the same N_MC physical
    targets are reused across all paired comparisons in one base
    condition.
    """

    return (
        BASE_SEED
        + 10000
        * domain_index
        + 1000
        * prior_index
        + 100
        * resolution_index
    )


# ============================================================
# BASE-CONDITION TASKS
# ============================================================

def build_base_condition_tasks():

    tasks = []
    condition_id = 0

    for domain_index, domain_name in enumerate(
        DOMAIN_ORDER
    ):

        for prior_index, prior_name in enumerate(
            PRIOR_ORDER
        ):

            for resolution_index, N_ref in enumerate(
                NREF_ORDER
            ):

                condition_id += 1

                tasks.append(
                    (
                        condition_id,
                        domain_index,
                        domain_name,
                        prior_index,
                        prior_name,
                        resolution_index,
                        N_ref,
                    )
                )

    return tasks


# ============================================================
# APPEND ONE PRECOMPUTED ROUTE TO RAW ROWS
# ============================================================

def append_route_outcomes(
    rows,
    domain_name,
    prior_name,
    N_ref,
    targets,
    tau,
    tessellation,
    s0,
    target_cells,
    K_max,
    C_cap,
    strategy,
    recorder,
    reachable,
    route_termination,
    route_runtime,
    sweep_angle_deg=np.nan,
    lawnmower_spec=None,
):
    """
    Evaluate all paired physical targets on one precomputed route and
    append the resulting raw records.
    """

    first_visit = (
        recorder
        .first_visit_indices()
    )

    lawnmower_diag = getattr(
        recorder,
        "lawnmower_diagnostics",
        {},
    )

    for mc_id in range(
        N_MC
    ):

        target = (
            targets[
                mc_id
            ]
        )

        target_cell = int(
            target_cells[
                mc_id
            ]
        )

        outcome = (
            evaluate_target_on_route(
                tessellation=tessellation,
                recorder=recorder,
                reachable=reachable,
                route_termination=(
                    route_termination
                ),
                target_cell=(
                    target_cell
                ),
                first_visit=(
                    first_visit
                ),
            )
        )

        rows.append(
            {
                "domain":
                    domain_name,

                "prior":
                    prior_name,

                "N_ref":
                    N_ref,

                "mc_id":
                    mc_id,

                "target_x":
                    target[0],

                "target_y":
                    target[1],

                "tau":
                    tau,

                "strategy":
                    strategy,

                # NaN for greedy; one of LAWNMOWER_ANGLES_DEG for
                # the systematic baseline.
                "sweep_angle_deg":
                    sweep_angle_deg,

                "N_tau":
                    len(
                        tessellation.cells
                    ),

                "s0":
                    s0,

                "target_cell":
                    target_cell,

                "K_max":
                    K_max,

                "common_support_area":
                    C_cap.area,

                "route_build_time_s":
                    route_runtime,

                "lawnmower_spacing":
                    (
                        lawnmower_spec[
                            "spacing"
                        ]
                        if lawnmower_spec is not None
                        else np.nan
                    ),

                "lawnmower_reverse_strips":
                    lawnmower_diag.get(
                        "reverse_strips",
                        np.nan,
                    ),

                "lawnmower_reverse_first_strip":
                    lawnmower_diag.get(
                        "reverse_first_strip",
                        np.nan,
                    ),

                "lawnmower_n_strips":
                    lawnmower_diag.get(
                        "n_nonempty_strips",
                        np.nan,
                    ),

                "lawnmower_nonadjacent_fraction":
                    lawnmower_diag.get(
                        "nonadjacent_fraction",
                        np.nan,
                    ),

                "lawnmower_nonadjacent_interstrip":
                    lawnmower_diag.get(
                        "nonadjacent_interstrip",
                        np.nan,
                    ),

                "lawnmower_nonadjacent_intrastrip":
                    lawnmower_diag.get(
                        "nonadjacent_intrastrip",
                        np.nan,
                    ),

                "lawnmower_shortest_path_connections":
                    lawnmower_diag.get(
                        "shortest_path_connections",
                        np.nan,
                    ),

                "lawnmower_shortest_path_fraction":
                    lawnmower_diag.get(
                        "shortest_path_fraction",
                        np.nan,
                    ),

                "lawnmower_entry_connectors":
                    lawnmower_diag.get(
                        "entry_connectors",
                        np.nan,
                    ),

                "lawnmower_interstrip_connectors":
                    lawnmower_diag.get(
                        "interstrip_connectors",
                        np.nan,
                    ),

                "lawnmower_intrastrip_connectors":
                    lawnmower_diag.get(
                        "intrastrip_connectors",
                        np.nan,
                    ),

                "lawnmower_connector_steps":
                    lawnmower_diag.get(
                        "connector_steps",
                        np.nan,
                    ),

                "lawnmower_connector_extra_steps":
                    lawnmower_diag.get(
                        "connector_extra_steps",
                        np.nan,
                    ),

                "lawnmower_connector_revisit_steps":
                    lawnmower_diag.get(
                        "connector_revisit_steps",
                        np.nan,
                    ),

                **outcome,
            }
        )


# ============================================================
# RUN ONE INDEPENDENT BASE CONDITION
# ============================================================

def run_base_condition(
    task,
):
    """
    Worker entry point.

    One task contains exactly one

        (domain, prior, N_ref)

    base condition. The worker generates one set of N_MC paired
    physical targets, builds all three canonical tessellations, runs
    greedy once per tessellation, and runs lawnmower for every common
    physical orientation in LAWNMOWER_ANGLES_DEG.

    Keeping the complete base condition inside one worker preserves the
    paired design and avoids sending Shapely geometries or NetworkX
    graphs between processes.
    """

    (
        condition_id,
        domain_index,
        domain_name,
        prior_index,
        prior_name,
        resolution_index,
        N_ref,
    ) = task

    domains = make_normalized_domains(
        target_area=A0
    )

    omega = (
        domains[
            domain_name
        ]
    )

    x0 = np.array([
        omega.centroid.x,
        omega.centroid.y,
    ])

    p0 = make_prior(
        omega,
        prior_name,
    )

    K_max = (
        K_MAX_BY_NREF[
            N_ref
        ]
    )

    tessellations = {
        tau:
            build_canonical_tessellation(
                omega=omega,
                tau=tau,
                N_ref=N_ref,
            )
        for tau
        in TAU_ORDER
    }

    C_cap = common_support(
        tessellations
    )

    seed = condition_seed(
        domain_index,
        prior_index,
        resolution_index,
    )

    targets = (
        sample_targets_from_common_prior(
            omega=omega,
            prior_name=prior_name,
            C_cap=C_cap,
            n_samples=N_MC,
            seed=seed,
        )
    )

    # The same 12 specifications are reused unchanged across all three
    # tessellations in this base condition.
    lawnmower_specs = {
        angle:
            build_common_lawnmower_spec(
                omega=omega,
                N_ref=N_ref,
                x0=x0,
                angle_deg=angle,
            )
        for angle
        in LAWNMOWER_ANGLES_DEG
    }

    rows = []

    for tau in TAU_ORDER:

        tess = (
            tessellations[
                tau
            ]
        )

        s0 = (
            initial_state_from_position(
                tess,
                x0,
            )
        )

        B0 = (
            discretize_common_prior(
                tessellation=tess,
                p0=p0,
                C_cap=C_cap,
            )
        )

        mapper = (
            TargetCellMapper(
                tess
            )
        )

        target_cells = np.array([
            mapper.map_point(
                target
            )
            for target
            in targets
        ])

        # ----------------------------------------------------
        # Greedy: exactly one route per tessellation.
        # ----------------------------------------------------

        start_time = (
            time.perf_counter()
        )

        (
            recorder,
            reachable,
            route_termination,
        ) = build_greedy_route(
            tessellation=tess,
            B0=B0,
            s0=s0,
            K_max=K_max,
        )

        route_runtime = (
            time.perf_counter()
            - start_time
        )

        append_route_outcomes(
            rows=rows,
            domain_name=domain_name,
            prior_name=prior_name,
            N_ref=N_ref,
            targets=targets,
            tau=tau,
            tessellation=tess,
            s0=s0,
            target_cells=target_cells,
            K_max=K_max,
            C_cap=C_cap,
            strategy="greedy",
            recorder=recorder,
            reachable=reachable,
            route_termination=route_termination,
            route_runtime=route_runtime,
            sweep_angle_deg=np.nan,
            lawnmower_spec=None,
        )

        # ----------------------------------------------------
        # Lawnmower: same 12 physical orientations for every
        # tessellation.
        # ----------------------------------------------------

        for angle in LAWNMOWER_ANGLES_DEG:

            lawnmower_spec = (
                lawnmower_specs[
                    angle
                ]
            )

            start_time = (
                time.perf_counter()
            )

            (
                recorder,
                reachable,
                route_termination,
            ) = build_lawnmower_route(
                tessellation=tess,
                B0=B0,
                s0=s0,
                K_max=K_max,
                lawnmower_spec=(
                    lawnmower_spec
                ),
            )

            route_runtime = (
                time.perf_counter()
                - start_time
            )

            append_route_outcomes(
                rows=rows,
                domain_name=domain_name,
                prior_name=prior_name,
                N_ref=N_ref,
                targets=targets,
                tau=tau,
                tessellation=tess,
                s0=s0,
                target_cells=target_cells,
                K_max=K_max,
                C_cap=C_cap,
                strategy="lawnmower",
                recorder=recorder,
                reachable=reachable,
                route_termination=route_termination,
                route_runtime=route_runtime,
                sweep_angle_deg=angle,
                lawnmower_spec=(
                    lawnmower_spec
                ),
            )

    condition_df = pd.DataFrame(
        rows
    )

    metadata = {
        "condition_id":
            condition_id,

        "domain":
            domain_name,

        "prior":
            prior_name,

        "N_ref":
            N_ref,

        "common_support_area":
            float(C_cap.area),
    }

    return (
        condition_df,
        metadata,
    )


# ============================================================
# PERCENTILE
# ============================================================

def percentile_90(
    series,
):

    values = (
        series
        .dropna()
        .to_numpy()
    )

    if len(values) == 0:
        return np.nan

    return float(
        np.percentile(
            values,
            90,
        )
    )


# ============================================================
# ORIENTATION-SPECIFIC DESCRIPTIVE SUMMARY
# ============================================================

def make_orientation_summary(
    df,
):
    """
    Keep lawnmower orientations separate.

    This file is useful for orientation-sensitivity analysis. Greedy is
    represented by one group with sweep_angle_deg = NaN.
    """

    group_columns = [
        "domain",
        "prior",
        "N_ref",
        "tau",
        "strategy",
        "sweep_angle_deg",
    ]

    metrics = [
        "K_det",
        "L_det",
        "T_det",
        "visited_fraction",
        "cumulative_prior_mass",
        "n_revisits",
        "revisit_fraction",
    ]

    rows = []

    for keys, group in df.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):

        successful = (
            group[
                group[
                    "success"
                ]
            ]
        )

        row = dict(
            zip(
                group_columns,
                keys,
            )
        )

        row[
            "n_runs"
        ] = len(group)

        row[
            "detection_probability"
        ] = float(
            group[
                "success"
            ].mean()
        )

        row[
            "n_kmax"
        ] = int(
            (
                group[
                    "termination_reason"
                ]
                == "kmax"
            ).sum()
        )

        row[
            "n_unreachable"
        ] = int(
            (
                group[
                    "termination_reason"
                ]
                == "unreachable_component"
            ).sum()
        )

        for metric in metrics:

            values = (
                successful[
                    metric
                ]
                .dropna()
            )

            if len(values) == 0:

                row[
                    f"{metric}_mean_success"
                ] = np.nan

                row[
                    f"{metric}_median_success"
                ] = np.nan

                row[
                    f"{metric}_std_success"
                ] = np.nan

                row[
                    f"{metric}_p90_success"
                ] = np.nan

            else:

                row[
                    f"{metric}_mean_success"
                ] = values.mean()

                row[
                    f"{metric}_median_success"
                ] = values.median()

                row[
                    f"{metric}_std_success"
                ] = values.std(
                    ddof=1
                )

                row[
                    f"{metric}_p90_success"
                ] = percentile_90(
                    values
                )

        if (
            group[
                "strategy"
            ].iloc[0]
            == "lawnmower"
        ):

            for diagnostic in [
                "lawnmower_n_strips",
                "lawnmower_nonadjacent_fraction",
                "lawnmower_shortest_path_connections",
                "lawnmower_shortest_path_fraction",
                "lawnmower_connector_extra_steps",
                "lawnmower_connector_revisit_steps",
            ]:

                row[
                    diagnostic
                ] = (
                    group[
                        diagnostic
                    ].iloc[0]
                )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# TARGET-LEVEL ORIENTATION AVERAGING
# ============================================================

def build_target_level_table(
    df,
):
    """
    Collapse the 12 lawnmower orientations at the target level before
    computing the final strategy summary.

    This avoids treating the same physical target under 12 sweep
    orientations as 12 independent Monte Carlo realizations.

    For lawnmower, each metric is first averaged across orientation for
    a fixed (domain, prior, N_ref, tau, mc_id). Greedy already has one
    observation per target and tessellation.
    """

    metrics = [
        "K_det",
        "L_det",
        "T_det",
        "visited_fraction",
        "cumulative_prior_mass",
        "n_revisits",
        "revisit_fraction",
    ]

    key_columns = [
        "domain",
        "prior",
        "N_ref",
        "tau",
        "mc_id",
    ]

    pieces = []

    # --------------------------------------------------------
    # Greedy: one route per target/tessellation.
    # --------------------------------------------------------

    greedy = (
        df[
            df[
                "strategy"
            ]
            == "greedy"
        ]
        .copy()
    )

    if not greedy.empty:

        greedy_target = greedy[
            key_columns
            + [
                "target_x",
                "target_y",
                "success",
                "termination_reason",
            ]
            + metrics
        ].copy()

        greedy_target[
            "strategy"
        ] = "greedy"

        greedy_target[
            "n_orientations"
        ] = 1

        greedy_target[
            "orientation_success_fraction"
        ] = (
            greedy_target[
                "success"
            ].astype(float)
        )

        greedy_target[
            "n_kmax_orientations"
        ] = (
            greedy_target[
                "termination_reason"
            ]
            .eq("kmax")
            .astype(int)
        )

        greedy_target[
            "n_unreachable_orientations"
        ] = (
            greedy_target[
                "termination_reason"
            ]
            .eq("unreachable_component")
            .astype(int)
        )

        for metric in metrics:

            greedy_target[
                f"{metric}_orientation_std"
            ] = np.where(
                greedy_target[
                    metric
                ].notna(),
                0.0,
                np.nan,
            )

        pieces.append(
            greedy_target
        )

    # --------------------------------------------------------
    # Lawnmower: average the repeated orientation factor within
    # each physical Monte Carlo target.
    # --------------------------------------------------------

    lawn = (
        df[
            df[
                "strategy"
            ]
            == "lawnmower"
        ]
        .copy()
    )

    if not lawn.empty:

        grouped = lawn.groupby(
            key_columns,
            sort=False,
        )

        base = grouped.agg(
            target_x=(
                "target_x",
                "first",
            ),
            target_y=(
                "target_y",
                "first",
            ),
            n_orientations=(
                "sweep_angle_deg",
                "nunique",
            ),
            orientation_success_fraction=(
                "success",
                "mean",
            ),
            n_kmax_orientations=(
                "termination_reason",
                lambda values:
                    int(
                        (
                            values
                            == "kmax"
                        ).sum()
                    ),
            ),
            n_unreachable_orientations=(
                "termination_reason",
                lambda values:
                    int(
                        (
                            values
                            == "unreachable_component"
                        ).sum()
                    ),
            ),
        ).reset_index()

        base[
            "strategy"
        ] = "lawnmower"

        base[
            "success"
        ] = (
            base[
                "orientation_success_fraction"
            ]
            > 0.0
        )

        base[
            "termination_reason"
        ] = np.where(
            base[
                "orientation_success_fraction"
            ]
            >= 1.0 - EPS,
            "detected_all_orientations",
            "mixed_or_censored",
        )

        for metric in metrics:

            if not lawn[
                metric
            ].notna().any():

                base[
                    metric
                ] = np.nan

                base[
                    f"{metric}_orientation_std"
                ] = np.nan

                continue

            metric_mean = (
                grouped[
                    metric
                ]
                .mean()
                .rename(
                    metric
                )
                .reset_index()
            )

            metric_std = (
                grouped[
                    metric
                ]
                .std(
                    ddof=0
                )
                .rename(
                    f"{metric}_orientation_std"
                )
                .reset_index()
            )

            base = base.merge(
                metric_mean,
                on=key_columns,
                how="left",
                validate="one_to_one",
            )

            base = base.merge(
                metric_std,
                on=key_columns,
                how="left",
                validate="one_to_one",
            )

        pieces.append(
            base
        )

    return pd.concat(
        pieces,
        ignore_index=True,
    )


# ============================================================
# FINAL DESCRIPTIVE SUMMARY
# ============================================================

def make_descriptive_summary(
    df,
):
    """
    Final strategy-level descriptive summary.

    Lawnmower orientations are averaged within each physical target
    first. Therefore the 12 repeated orientations do not inflate the
    effective Monte Carlo sample size.
    """

    target_df = (
        build_target_level_table(
            df
        )
    )

    group_columns = [
        "domain",
        "prior",
        "N_ref",
        "tau",
        "strategy",
    ]

    metrics = [
        "K_det",
        "L_det",
        "T_det",
        "visited_fraction",
        "cumulative_prior_mass",
        "n_revisits",
        "revisit_fraction",
    ]

    rows = []

    for keys, group in target_df.groupby(
        group_columns,
        sort=False,
    ):

        row = dict(
            zip(
                group_columns,
                keys,
            )
        )

        row[
            "n_targets"
        ] = len(group)

        row[
            "n_orientations"
        ] = int(
            group[
                "n_orientations"
            ].max()
        )

        # Probability over the paired target x orientation design.
        row[
            "detection_probability"
        ] = float(
            group[
                "orientation_success_fraction"
            ].mean()
        )

        row[
            "n_kmax_orientation_runs"
        ] = int(
            group[
                "n_kmax_orientations"
            ].sum()
        )

        row[
            "n_unreachable_orientation_runs"
        ] = int(
            group[
                "n_unreachable_orientations"
            ].sum()
        )

        for metric in metrics:

            values = (
                group[
                    metric
                ]
                .dropna()
            )

            orientation_stds = (
                group[
                    f"{metric}_orientation_std"
                ]
                .dropna()
            )

            if len(values) == 0:

                row[
                    f"{metric}_mean"
                ] = np.nan

                row[
                    f"{metric}_median"
                ] = np.nan

                row[
                    f"{metric}_std"
                ] = np.nan

                row[
                    f"{metric}_p90"
                ] = np.nan

            else:

                row[
                    f"{metric}_mean"
                ] = values.mean()

                row[
                    f"{metric}_median"
                ] = values.median()

                row[
                    f"{metric}_std"
                ] = values.std(
                    ddof=1
                )

                row[
                    f"{metric}_p90"
                ] = percentile_90(
                    values
                )

            row[
                f"{metric}_orientation_std_mean"
            ] = (
                orientation_stds.mean()
                if len(orientation_stds) > 0
                else np.nan
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# CONDITION-LEVEL SANITY CHECK
# ============================================================

def sanity_check_condition(
    df,
    metadata,
):

    expected_per_target = (
        len(TAU_ORDER)
        * (
            1
            + len(
                LAWNMOWER_ANGLES_DEG
            )
        )
    )

    expected_rows = (
        N_MC
        * expected_per_target
    )

    if len(df) != expected_rows:
        raise AssertionError(
            f"Condition {metadata['condition_id']}: "
            f"obtained {len(df)} rows, expected {expected_rows}."
        )

    counts = (
        df.groupby(
            "mc_id"
        )
        .size()
    )

    if not (
        counts
        == expected_per_target
    ).all():
        raise AssertionError(
            "Paired design broken: every physical target must appear "
            f"exactly {expected_per_target} times."
        )

    target_variation = (
        df.groupby(
            "mc_id"
        )[
            [
                "target_x",
                "target_y",
            ]
        ]
        .nunique()
    )

    if (
        target_variation
        > 1
    ).any().any():
        raise AssertionError(
            "Physical targets differ across paired tessellation / "
            "strategy / orientation runs."
        )

    greedy = (
        df[
            df[
                "strategy"
            ]
            == "greedy"
        ]
    )

    lawn = (
        df[
            df[
                "strategy"
            ]
            == "lawnmower"
        ]
    )

    if not greedy[
        "sweep_angle_deg"
    ].isna().all():
        raise AssertionError(
            "Greedy rows must have sweep_angle_deg = NaN."
        )

    expected_angles = set(
        LAWNMOWER_ANGLES_DEG
    )

    obtained_angles = set(
        float(value)
        for value in lawn[
            "sweep_angle_deg"
        ].dropna().unique()
    )

    if obtained_angles != expected_angles:
        raise AssertionError(
            "Lawnmower angle set does not match "
            "LAWNMOWER_ANGLES_DEG."
        )

    lawn_counts = (
        lawn.groupby(
            [
                "mc_id",
                "tau",
            ]
        )[
            "sweep_angle_deg"
        ]
        .nunique()
    )

    if not (
        lawn_counts
        == len(
            LAWNMOWER_ANGLES_DEG
        )
    ).all():
        raise AssertionError(
            "Each target/tessellation pair must contain every "
            "lawnmower orientation exactly once."
        )

    greedy_counts = (
        greedy.groupby(
            "mc_id"
        )
        .size()
    )

    if not (
        greedy_counts
        == len(TAU_ORDER)
    ).all():
        raise AssertionError(
            "Greedy must appear exactly once per tessellation and "
            "physical target."
        )


# ============================================================
# PARALLEL SEARCH EXPERIMENT
# ============================================================

def run_search_experiments_parallel():
    """
    Run all 40 independent base conditions in parallel.

    Workers never write to the shared raw CSV. The main process is the
    only writer, so concurrent file corruption is avoided. Each
    completed condition is appended immediately, providing a natural
    checkpoint without retaining the complete 1.56M-row data set in
    memory.
    """

    tasks = (
        build_base_condition_tasks()
    )

    total_conditions = len(
        tasks
    )

    expected_total_rows = (
        total_conditions
        * N_MC
        * len(TAU_ORDER)
        * (
            1
            + len(
                LAWNMOWER_ANGLES_DEG
            )
        )
    )

    # Start a fresh result file for this experiment version.
    if RAW_OUTPUT.exists():
        RAW_OUTPUT.unlink()

    orientation_summary_parts = []
    descriptive_summary_parts = []

    total_rows_written = 0
    first_write = True
    completed = 0

    detection_accumulator = {}
    termination_counts = {}

    print(
        "\n"
        "========================================"
    )

    print(
        "PARALLEL SEARCH EXPERIMENT"
    )

    print(
        "========================================"
    )

    print(
        f"Base conditions: {total_conditions}"
    )

    print(
        f"Lawnmower angles: {LAWNMOWER_ANGLES_DEG}"
    )

    print(
        f"Workers: {MAX_WORKERS}"
    )

    print(
        f"Expected raw rows: {expected_total_rows:,}"
    )

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
    ) as executor:

        future_to_task = {
            executor.submit(
                run_base_condition,
                task,
            ):
                task
            for task
            in tasks
        }

        for future in as_completed(
            future_to_task
        ):

            task = (
                future_to_task[
                    future
                ]
            )

            try:
                condition_df, metadata = (
                    future.result()
                )

            except Exception as exc:

                for pending in future_to_task:
                    pending.cancel()

                raise RuntimeError(
                    "Base condition failed: "
                    f"domain={task[2]}, "
                    f"prior={task[4]}, "
                    f"N_ref={task[6]}"
                ) from exc

            sanity_check_condition(
                condition_df,
                metadata,
            )

            condition_df.to_csv(
                RAW_OUTPUT,
                mode=(
                    "w"
                    if first_write
                    else "a"
                ),
                header=first_write,
                index=False,
            )

            first_write = False

            total_rows_written += len(
                condition_df
            )

            orientation_summary_parts.append(
                make_orientation_summary(
                    condition_df
                )
            )

            descriptive_summary_parts.append(
                make_descriptive_summary(
                    condition_df
                )
            )

            # Small global sanity aggregates; no need to retain raw rows.
            detection_group = (
                condition_df.groupby(
                    [
                        "tau",
                        "strategy",
                    ]
                )[
                    "success"
                ]
                .agg([
                    "sum",
                    "count",
                ])
            )

            for key, values in detection_group.iterrows():

                previous = detection_accumulator.get(
                    key,
                    [0, 0],
                )

                previous[0] += int(
                    values[
                        "sum"
                    ]
                )

                previous[1] += int(
                    values[
                        "count"
                    ]
                )

                detection_accumulator[
                    key
                ] = previous

            for reason, count in (
                condition_df[
                    "termination_reason"
                ]
                .value_counts()
                .items()
            ):

                termination_counts[
                    reason
                ] = (
                    termination_counts.get(
                        reason,
                        0,
                    )
                    + int(count)
                )

            completed += 1

            print(
                f"[{completed:02d}/{total_conditions}] "
                f"{metadata['domain']} | "
                f"{metadata['prior']} | "
                f"N_ref={metadata['N_ref']} | "
                f"rows={len(condition_df):,} | "
                f"total={total_rows_written:,}"
            )

    if total_rows_written != expected_total_rows:
        raise AssertionError(
            f"Global row count mismatch: obtained "
            f"{total_rows_written}, expected {expected_total_rows}."
        )

    orientation_summary = pd.concat(
        orientation_summary_parts,
        ignore_index=True,
    )

    orientation_summary = (
        orientation_summary.sort_values(
            [
                "domain",
                "prior",
                "N_ref",
                "tau",
                "strategy",
                "sweep_angle_deg",
            ],
            na_position="first",
        )
        .reset_index(
            drop=True
        )
    )

    expected_orientation_summary_rows = (
        total_conditions
        * len(TAU_ORDER)
        * (
            1
            + len(LAWNMOWER_ANGLES_DEG)
        )
    )

    if len(orientation_summary) != expected_orientation_summary_rows:
        raise AssertionError(
            "Unexpected orientation-summary row count: "
            f"obtained {len(orientation_summary)}, "
            f"expected {expected_orientation_summary_rows}."
        )

    orientation_summary.to_csv(
        ORIENTATION_SUMMARY_OUTPUT,
        index=False,
    )

    summary = pd.concat(
        descriptive_summary_parts,
        ignore_index=True,
    )

    summary = (
        summary.sort_values(
            [
                "domain",
                "prior",
                "N_ref",
                "tau",
                "strategy",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    expected_summary_rows = (
        total_conditions
        * len(TAU_ORDER)
        * len(STRATEGY_ORDER)
    )

    if len(summary) != expected_summary_rows:
        raise AssertionError(
            "Unexpected final-summary row count: "
            f"obtained {len(summary)}, "
            f"expected {expected_summary_rows}."
        )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "GLOBAL SANITY CHECK"
    )

    print(
        "========================================"
    )

    print(
        f"Rows obtained: {total_rows_written:,}"
    )

    print(
        f"Rows expected: {expected_total_rows:,}"
    )

    print(
        "\nDetection probability by tessellation and strategy:"
    )

    for key in sorted(
        detection_accumulator
    ):

        successes, count = (
            detection_accumulator[
                key
            ]
        )

        probability = (
            successes
            / count
            if count > 0
            else np.nan
        )

        print(
            f"  {key}: {probability:.6f}"
        )

    print(
        "\nTermination reasons:"
    )

    for reason in sorted(
        termination_counts
    ):

        print(
            f"  {reason}: "
            f"{termination_counts[reason]:,}"
        )

    return (
        summary,
        orientation_summary,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    (
        summary,
        orientation_summary,
    ) = run_search_experiments_parallel()

    print(
        "\nFinal raw results saved to:"
    )

    print(
        RAW_OUTPUT
    )

    print(
        "\nOrientation-specific summary saved to:"
    )

    print(
        ORIENTATION_SUMMARY_OUTPUT
    )

    print(
        "\nOrientation-averaged descriptive summary saved to:"
    )

    print(
        SUMMARY_OUTPUT
    )

    print(
        "\nDone."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()
