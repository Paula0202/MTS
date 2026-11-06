import numpy as np
import networkx as nx

from shapely.geometry import Polygon
from shapely.ops import unary_union


def represented_region(tessellation):
    """
    Omega_hat_tau = union_i h_{tau,i}

    Complete retained cells are used, not C_{tau,i}.
    """

    polygons = [
        cell.full_polygon
        for cell in tessellation.cells.values()
    ]

    if len(polygons) == 0:
        return Polygon()

    return unary_union(polygons)


def geometric_metrics(
    omega,
    tessellation
):
    """
    Main geometric metrics used in the paper.
    """

    represented = represented_region(
        tessellation
    )

    intersection_area = (
        omega
        .intersection(represented)
        .area
    )

    external_area = (
        represented
        .difference(omega)
        .area
    )

    union_area = (
        omega
        .union(represented)
        .area
    )

    coverage = (
        intersection_area
        / omega.area
    )

    external_excess = (
        external_area
        / omega.area
    )

    iou = (
        intersection_area
        / union_area
        if union_area > 0
        else np.nan
    )

    cells = list(
        tessellation.cells.values()
    )

    n_cells = len(cells)

    n_boundary = sum(
        cell.is_boundary
        for cell in cells
    )

    boundary_proportion = (
        n_boundary / n_cells
        if n_cells > 0
        else np.nan
    )

    valid_areas = np.array([
        cell.valid_area
        for cell in cells
    ])

    G = tessellation.graph

    degrees = np.array([
        degree
        for _, degree in G.degree()
    ])

    n_components = (
        nx.number_connected_components(G)
        if G.number_of_nodes() > 0
        else 0
    )

    if (
        G.number_of_nodes() > 0
        and n_components == 1
    ):
        graph_diameter = nx.diameter(G)
    else:
        graph_diameter = np.nan

    return {
        "tau":
            tessellation.tau,

        "N_tau":
            n_cells,

        "coverage":
            coverage,

        "external_excess":
            external_excess,

        "iou":
            iou,

        "n_boundary":
            n_boundary,

        "boundary_proportion":
            boundary_proportion,

        "valid_area_mean":
            (
                valid_areas.mean()
                if n_cells > 0
                else np.nan
            ),

        "valid_area_std":
            (
                valid_areas.std()
                if n_cells > 0
                else np.nan
            ),

        "valid_area_min":
            (
                valid_areas.min()
                if n_cells > 0
                else np.nan
            ),

        "valid_area_max":
            (
                valid_areas.max()
                if n_cells > 0
                else np.nan
            ),

        "n_nodes":
            G.number_of_nodes(),

        "n_edges":
            G.number_of_edges(),

        "mean_degree":
            (
                degrees.mean()
                if len(degrees) > 0
                else np.nan
            ),

        "degree_std":
            (
                degrees.std()
                if len(degrees) > 0
                else np.nan
            ),

        "n_components":
            n_components,

        "graph_diameter_hops":
            graph_diameter,
    }


def sample_boundary_points(
    geom,
    n_samples=1000
):
    boundary = geom.boundary
    total_length = boundary.length

    distances = np.linspace(
        0.0,
        total_length,
        n_samples,
        endpoint=False
    )

    return [
        boundary.interpolate(d)
        for d in distances
    ]


def average_symmetric_boundary_distance(
    omega,
    omega_hat,
    n_samples=1000
):
    points_omega = sample_boundary_points(
        omega,
        n_samples
    )

    points_hat = sample_boundary_points(
        omega_hat,
        n_samples
    )

    d_omega_hat = np.mean([
        omega_hat.boundary.distance(p)
        for p in points_omega
    ])

    d_hat_omega = np.mean([
        omega.boundary.distance(p)
        for p in points_hat
    ])

    return 0.5 * (
        d_omega_hat
        + d_hat_omega
    )


def physical_path_stretch(
    tessellation,
    s0,
    n_pairs=1000,
    seed=12345
):
    """
    Physical shortest-path stretch over sampled node pairs
    within the connected component containing s0.
    """

    G = tessellation.graph

    reachable = list(
        nx.node_connected_component(
            G,
            s0
        )
    )

    if len(reachable) < 2:
        return {
            "stretch_mean": np.nan,
            "stretch_p90": np.nan,
            "stretch_std": np.nan,
            "n_stretch_pairs": 0
        }

    rng = np.random.default_rng(seed)

    stretches = []

    for _ in range(n_pairs):

        i, j = rng.choice(
            reachable,
            size=2,
            replace=False
        )

        qi = tessellation.cells[i].centroid
        qj = tessellation.cells[j].centroid

        euclidean_distance = np.linalg.norm(
            qi - qj
        )

        if euclidean_distance <= 1e-12:
            continue

        graph_distance = nx.shortest_path_length(
            G,
            source=i,
            target=j,
            weight="weight"
        )

        stretches.append(
            graph_distance
            / euclidean_distance
        )

    stretches = np.asarray(stretches)

    return {
        "stretch_mean":
            float(np.mean(stretches)),

        "stretch_p90":
            float(np.percentile(
                stretches,
                90
            )),

        "stretch_std":
            float(np.std(
                stretches,
                ddof=1
            )),

        "n_stretch_pairs":
            len(stretches)
    }