from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely import union_all 
import numpy as np

from scipy.ndimage import maximum_filter
from scipy.optimize import minimize, linear_sum_assignment
from priors import (integrate_density_over_geometry)


def continuous_modes(
    p0,
    geometry,
    max_modes,
    n_grid=300,
    relative_threshold=0.05,
    merge_tolerance=0.05
):
    """
    Numerically detects the dominant modes of a continuous density
    inside a given geometry.

    Parameters
    ----------
    p0 : callable
        Continuous probability density.
    geometry : shapely geometry
        Spatial support over which modes are searched.
    max_modes : int
        Maximum number of modes to retain.
    n_grid : int
        Auxiliary grid resolution used to detect candidate maxima.
    relative_threshold : float
        Candidates below this fraction of the global maximum density
        are discarded.
    merge_tolerance : float
        Minimum separation between distinct modes, expressed relative
        to sqrt(area).

    Returns
    -------
    list of dict
        Each element contains:
            position : np.ndarray([x, y])
            density  : float
    """

    minx, miny, maxx, maxy = geometry.bounds

    dx = (maxx - minx) / n_grid
    dy = (maxy - miny) / n_grid

    xs = minx + (np.arange(n_grid) + 0.5) * dx
    ys = miny + (np.arange(n_grid) + 0.5) * dy

    density_grid = np.full(
        (n_grid, n_grid),
        -np.inf,
        dtype=float
    )

    # --------------------------------------------------------
    # Evaluate density on an auxiliary midpoint grid
    # --------------------------------------------------------

    for ix, x in enumerate(xs):

        for iy, y in enumerate(ys):

            point = Point(
                float(x),
                float(y)
            )

            if not geometry.covers(point):
                continue

            position = np.array(
                [x, y],
                dtype=float
            )

            density_grid[iy, ix] = float(
                p0(position)
            )

    finite_values = density_grid[
        np.isfinite(density_grid)
    ]

    if len(finite_values) == 0:
        return []

    global_max = np.max(
        finite_values
    )

    # --------------------------------------------------------
    # Candidate local maxima
    # --------------------------------------------------------

    local_maximum = (
        density_grid
        ==
        maximum_filter(
            density_grid,
            size=3,
            mode="constant",
            cval=-np.inf
        )
    )

    candidate_mask = (
        local_maximum
        &
        np.isfinite(density_grid)
        &
        (
            density_grid
            >=
            relative_threshold
            *
            global_max
        )
    )

    candidate_indices = np.argwhere(
        candidate_mask
    )

    candidate_indices = sorted(
        candidate_indices,
        key=lambda ij:
            density_grid[
                ij[0],
                ij[1]
            ],
        reverse=True
    )

    # --------------------------------------------------------
    # Continuous refinement of each candidate
    # --------------------------------------------------------

    characteristic_length = np.sqrt(
        geometry.area
    )

    absolute_merge_tolerance = (
        merge_tolerance
        *
        characteristic_length
    )

    modes = []

    def objective(position):

        point = Point(
            float(position[0]),
            float(position[1])
        )

        if not geometry.covers(point):

            return (
                1e6
                +
                1e6
                *
                geometry.distance(point)
            )

        return -float(
            p0(
                np.asarray(
                    position,
                    dtype=float
                )
            )
        )

    for iy, ix in candidate_indices:

        initial_position = np.array(
            [
                xs[ix],
                ys[iy]
            ],
            dtype=float
        )

        result = minimize(
            objective,
            x0=initial_position,
            method="Nelder-Mead",
            options={
                "xatol": 1e-10,
                "fatol": 1e-12,
                "maxiter": 1000
            }
        )

        if result.success:

            refined_position = (
                np.asarray(
                    result.x,
                    dtype=float
                )
            )

        else:

            refined_position = (
                initial_position
            )

        refined_point = Point(
            float(
                refined_position[0]
            ),
            float(
                refined_position[1]
            )
        )

        if not geometry.covers(
            refined_point
        ):

            refined_position = (
                initial_position
            )

        refined_density = float(
            p0(
                refined_position
            )
        )

        duplicate = False

        for mode in modes:

            distance = np.linalg.norm(
                refined_position
                -
                mode["position"]
            )

            if (
                distance
                <
                absolute_merge_tolerance
            ):

                duplicate = True
                break

        if duplicate:
            continue

        modes.append(
            {
                "position":
                    refined_position,

                "density":
                    refined_density
            }
        )

        if (
            len(modes)
            >=
            max_modes
        ):
            break

    modes.sort(
        key=lambda mode:
            mode["density"],
        reverse=True
    )

    return modes

def discrete_modes(
    tessellation,
    belief,
    max_modes,
    relative_threshold=0.05
):
    """
    Detects graph-local maxima of the piecewise-constant
    reconstructed discrete density.
    """

    densities = (
        reconstructed_cell_densities(
            tessellation=tessellation,
            belief=belief
        )
    )

    positive_densities = [
        density
        for density
        in densities.values()
        if density > 0.0
    ]

    if len(
        positive_densities
    ) == 0:

        return []

    global_max = max(
        positive_densities
    )

    modes = []

    for (
        cell_id,
        cell
    ) in tessellation.cells.items():

        density_i = densities[
            cell_id
        ]

        if (
            density_i
            <
            relative_threshold
            *
            global_max
        ):
            continue

        neighbors = list(
            tessellation.graph.neighbors(
                cell_id
            )
        )

        is_local_maximum = all(
            density_i
            >=
            densities[neighbor_id]

            for neighbor_id
            in neighbors
        )

        if not is_local_maximum:
            continue

        modes.append(
            {
                "cell_id":
                    cell_id,

                "position":
                    np.asarray(
                        cell.centroid,
                        dtype=float
                    ),

                "density":
                    float(
                        density_i
                    ),

                "mass":
                    float(
                        belief[cell_id]
                    )
            }
        )

    modes.sort(
        key=lambda mode:
            mode["density"],
        reverse=True
    )

    return modes[
        :max_modes
    ]


def match_probability_modes(
    continuous_mode_list,
    discrete_mode_list
):
    """
    Matches continuous and discrete modes by minimizing
    total Euclidean localization error.
    """

    if (
        len(continuous_mode_list) == 0
        or
        len(discrete_mode_list) == 0
    ):
        return []

    continuous_positions = np.vstack(
        [
            mode["position"]
            for mode
            in continuous_mode_list
        ]
    )

    discrete_positions = np.vstack(
        [
            mode["position"]
            for mode
            in discrete_mode_list
        ]
    )

    cost_matrix = np.linalg.norm(
        continuous_positions[:, None, :]
        -
        discrete_positions[None, :, :],
        axis=2
    )

    rows, columns = (
        linear_sum_assignment(
            cost_matrix
        )
    )

    matches = []

    for (
        continuous_index,
        discrete_index
    ) in zip(
        rows,
        columns
    ):

        matches.append(
            {
                "continuous_index":
                    int(
                        continuous_index
                    ),

                "discrete_index":
                    int(
                        discrete_index
                    ),

                "continuous_position":
                    continuous_positions[
                        continuous_index
                    ],

                "discrete_position":
                    discrete_positions[
                        discrete_index
                    ],

                "error":
                    float(
                        cost_matrix[
                            continuous_index,
                            discrete_index
                        ]
                    )
            }
        )

    return matches

def mode_localization_error(
    tessellation,
    p0,
    belief,
    expected_modes,
    n_grid=300
):
    """
    Computes localization error between continuous probability
    modes and graph-local maxima of the discrete reconstruction.
    """

    support = (
        represented_valid_support(
            tessellation
        )
    )

    continuous_mode_list = (
        continuous_modes(
            p0=p0,
            geometry=support,
            max_modes=expected_modes,
            n_grid=n_grid
        )
    )

    discrete_mode_list = (
        discrete_modes(
            tessellation=tessellation,
            belief=belief,
            max_modes=expected_modes
        )
    )

    matches = (
        match_probability_modes(
            continuous_mode_list,
            discrete_mode_list
        )
    )

    errors = np.array(
        [
            match["error"]
            for match
            in matches
        ],
        dtype=float
    )

    characteristic_length = (
        np.sqrt(
            support.area
        )
    )

    if len(errors) == 0:

        mean_error = np.nan
        max_error = np.nan

        normalized_mean_error = (
            np.nan
        )

        normalized_max_error = (
            np.nan
        )

    else:

        mean_error = float(
            np.mean(
                errors
            )
        )

        max_error = float(
            np.max(
                errors
            )
        )

        normalized_mean_error = (
            mean_error
            /
            characteristic_length
        )

        normalized_max_error = (
            max_error
            /
            characteristic_length
        )

    return {
        "mode_error_mean":
            mean_error,

        "mode_error_max":
            max_error,

        "mode_error_mean_norm":
            normalized_mean_error,

        "mode_error_max_norm":
            normalized_max_error,

        "n_continuous_modes":
            len(
                continuous_mode_list
            ),

        "n_discrete_modes":
            len(
                discrete_mode_list
            ),

        "n_matched_modes":
            len(
                matches
            ),

        "n_missing_modes":
            max(
                0,
                len(
                    continuous_mode_list
                )
                -
                len(
                    discrete_mode_list
                )
            ),

        "n_extra_modes":
            max(
                0,
                len(
                    discrete_mode_list
                )
                -
                len(
                    continuous_mode_list
                )
            ),

        "continuous_modes":
            continuous_mode_list,

        "discrete_modes":
            discrete_mode_list,

        "matches":
            matches
    }


def continuous_hpd_region(
    tessellation,
    p0,
    represented_mass,
    gamma=0.8,
    n_grid=240
):
    valid_supports = [
        cell.valid_support
        for cell in tessellation.cells.values()
        if not cell.valid_support.is_empty
    ]

    if len(valid_supports) == 0:
        return [], 0.0, 0.0

    support_union = unary_union(
        valid_supports
    )

    minx, miny, maxx, maxy = (
        support_union.bounds
    )

    dx = (
        maxx - minx
    ) / n_grid

    dy = (
        maxy - miny
    ) / n_grid

    integration_area = (
        dx * dy
    )

    xs = (
        minx
        +
        (
            np.arange(n_grid)
            +
            0.5
        )
        *
        dx
    )

    ys = (
        miny
        +
        (
            np.arange(n_grid)
            +
            0.5
        )
        *
        dy
    )

    samples = []

    for x in xs:

        for y in ys:

            point = Point(
                float(x),
                float(y)
            )

            if not support_union.covers(
                point
            ):
                continue

            position = np.array(
                [
                    x,
                    y
                ]
            )

            density = (
                p0(position)
                /
                represented_mass
            )

            probability_mass = (
                density
                *
                integration_area
            )

            samples.append(
                (
                    density,
                    probability_mass,
                    point
                )
            )

    samples.sort(
        key=lambda item: item[0],
        reverse=True
    )

    selected_points = []
    accumulated_mass = 0.0

    for (
        density,
        probability_mass,
        point
    ) in samples:

        if accumulated_mass >= gamma:
            break

        selected_points.append(
            point
        )

        accumulated_mass += (
            probability_mass
        )

    return (
        selected_points,
        integration_area,
        accumulated_mass
    )

def discrete_hpd_cells(
    tessellation,
    belief,
    gamma=0.8
):
    ranked_cells = []

    for cell_id, cell in tessellation.cells.items():

        area = cell.valid_area

        if area <= 0:
            continue

        probability = belief[cell_id]

        density = probability / area

        ranked_cells.append(
            (
                density,
                probability,
                cell_id,
                cell
            )
        )

    # Sort AFTER all cells have been added
    ranked_cells.sort(
        key=lambda item: item[0],
        reverse=True
    )

    selected_cells = []
    accumulated_mass = 0.0

    for (
        density,
        probability,
        cell_id,
        cell
    ) in ranked_cells:

        if accumulated_mass >= gamma:
            break

        selected_cells.append(cell)

        accumulated_mass += probability

    return (
        selected_cells,
        accumulated_mass
    )

def hpd_iou(
    tessellation,
    p0,
    belief,
    represented_mass,
    gamma=0.8,
    n_grid=240
):
    """
    Computes HPD-region IoU between:

        R_gamma
            continuous HPD region

    and

        R_hat_gamma_tau
            union of highest-density discrete cells.

    IoU is evaluated on the same independent numerical grid
    used to approximate the continuous HPD region.
    """

    # --------------------------------------------------------
    # Continuous HPD
    # --------------------------------------------------------

    (
        continuous_points,
        integration_area,
        continuous_mass
    ) = continuous_hpd_region(
        tessellation=tessellation,
        p0=p0,
        represented_mass=represented_mass,
        gamma=gamma,
        n_grid=n_grid
    )

    if len(continuous_points) == 0:
        return {
            "gamma": gamma,
            "hpd_iou": np.nan,
            "continuous_hpd_mass": 0.0,
            "discrete_hpd_mass": 0.0
        }

    # --------------------------------------------------------
    # Discrete HPD
    # --------------------------------------------------------

    (
        selected_cells,
        discrete_mass
    ) = discrete_hpd_cells(
        tessellation=tessellation,
        belief=belief,
        gamma=gamma
    )

    if len(selected_cells) == 0:
        return {
            "gamma": gamma,
            "hpd_iou": 0.0,
            "continuous_hpd_mass":
                continuous_mass,
            "discrete_hpd_mass":
                discrete_mass
        }

    discrete_region = unary_union(
        [
            cell.valid_support
            for cell in selected_cells
        ]
    )

    # --------------------------------------------------------
    # Numerical IoU
    # --------------------------------------------------------

    continuous_set = {
        (
            round(
                point.x,
                12
            ),
            round(
                point.y,
                12
            )
        )
        for point in continuous_points
    }

    # Need all integration points again so we can determine
    # which ones belong to the discrete region.
    valid_supports = [
        cell.valid_support
        for cell in tessellation.cells.values()
        if not cell.valid_support.is_empty
    ]

    support_union = unary_union(
        valid_supports
    )

    minx, miny, maxx, maxy = (
        support_union.bounds
    )

    dx = (
        maxx - minx
    ) / n_grid

    dy = (
        maxy - miny
    ) / n_grid

    xs = (
        minx
        +
        (
            np.arange(n_grid)
            +
            0.5
        )
        *
        dx
    )

    ys = (
        miny
        +
        (
            np.arange(n_grid)
            +
            0.5
        )
        *
        dy
    )

    intersection_count = 0
    union_count = 0

    for x in xs:

        for y in ys:

            point = Point(
                float(x),
                float(y)
            )

            if not support_union.covers(
                point
            ):
                continue

            key = (
                round(
                    float(x),
                    12
                ),
                round(
                    float(y),
                    12
                )
            )

            in_continuous = (
                key
                in
                continuous_set
            )

            in_discrete = (
                discrete_region.covers(
                    point
                )
            )

            if (
                in_continuous
                or
                in_discrete
            ):
                union_count += 1

            if (
                in_continuous
                and
                in_discrete
            ):
                intersection_count += 1

    if union_count == 0:

        iou = np.nan

    else:

        iou = (
            intersection_count
            /
            union_count
        )

    return {
        "gamma":
            gamma,

        "hpd_iou":
            iou,

        "continuous_hpd_mass":
            continuous_mass,

        "discrete_hpd_mass":
            discrete_mass,

        "n_selected_cells":
            len(
                selected_cells
            )
    }
        
def represented_valid_support(tessellation):

    supports = [cell.valid_support for cell in tessellation.cells.values() if not cell.valid_support.is_empty]

    if len(supports) == 0:
        return Polygon()

    return union_all(supports)

def represented_probability_mass(tessellation, p0, n_grid=400):

    C_tau = (represented_valid_support(tessellation))

    return(integrate_density_over_geometry(C_tau, p0, n_grid=n_grid))

def support_loss(
    tessellation,
    p0,
    n_grid=120
):
    """
    Computes represented probability mass from the sum
    of the probability masses of the valid cell supports:

        M_tau = sum_i ∫_{C_i} p0(x) dx

    and

        epsilon_mass = 1 - M_tau
    """

    masses, represented_mass, _ = (
        cell_probability_masses(
            tessellation=tessellation,
            p0=p0,
            n_grid=n_grid
        )
    )

    represented_mass = float(
        np.clip(
            represented_mass,
            0.0,
            1.0
        )
    )

    return {
        "represented_mass":
            represented_mass,

        "support_loss":
            1.0 - represented_mass
    }


def cell_probability_masses(tessellation, p0, n_grid=150):

    masses = {}

    for cid, cell in tessellation.cells.items():

        if cell.valid_support.is_empty:
            masses[cid] = 0.0
            continue

        mass_i = integrate_density_over_geometry(geometry=cell.valid_support, density_fn=p0, n_grid=n_grid)

        masses[cid] = mass_i

    represented_mass = sum(masses.values())

    if represented_mass <= 0.0:
        raise ValueError("Represented probability mass is zero.")

    belief = {cid: mass_i/represented_mass for cid, mass_i in masses.items()}

    return (masses, represented_mass, belief)

def reconstructed_cell_densities(tessellation, belief):

    densities = {}

    for cid, cell in tessellation.cells.items():

        if cell.valid_area <= 0.0:
            densities[cid] = 0.0

        else:
            densities[cid] = (belief[cid]/cell.valid_area)
    return densities 


def total_variation_distance(
    tessellation,
    p0,
    n_grid=120
):
    """
    Computes the total variation distance between the
    continuous prior conditioned on the represented support
    and its piecewise-constant discrete reconstruction.

    D_TV = 1/2 * integral_{C_tau}
           | p0^C_tau(x) - p_hat_tau(x) | dx
    """

    # --------------------------------------------------------
    # 1. Cell masses and normalized discrete belief
    # --------------------------------------------------------

    masses, represented_mass, belief = (
        cell_probability_masses(
            tessellation=tessellation,
            p0=p0,
            n_grid=n_grid
        )
    )

    # --------------------------------------------------------
    # 2. Piecewise-constant reconstructed density
    # --------------------------------------------------------

    p_hat = reconstructed_cell_densities(
        tessellation=tessellation,
        belief=belief
    )

    # --------------------------------------------------------
    # 3. Integrate absolute difference cell by cell
    # --------------------------------------------------------

    total_absolute_error = 0.0

    for cid, cell in tessellation.cells.items():

        if cell.valid_support.is_empty:
            continue

        p_hat_i = p_hat[cid]

        def absolute_difference(
            x,
            p_hat_value=p_hat_i
        ):
            p_conditional = (
                p0(x)
                /
                represented_mass
            )

            return abs(
                p_conditional
                -
                p_hat_value
            )

        cell_error = (
            integrate_density_over_geometry(
                geometry=cell.valid_support,
                density_fn=absolute_difference,
                n_grid=n_grid
            )
        )

        total_absolute_error += (
            cell_error
        )

    # --------------------------------------------------------
    # 4. Total variation
    # --------------------------------------------------------

    tv_distance = (
        0.5
        *
        total_absolute_error
    )

    return {
        "tv_distance":
            float(tv_distance),

        "represented_mass":
            float(represented_mass),

        "belief":
            belief
    }