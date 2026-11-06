from pathlib import Path
import math

import numpy as np
import pandas as pd
import networkx as nx

from domains import make_normalized_domains

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
    / "graph_robustness.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "graph_robustness_summary.csv"
)


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

A0 = 100.0

DOMAIN_ORDER = [
    "Square",
    "Circle",
    "Triangle",
    "Convex",
    "Concave",
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

# Normalized lattice phases (u, v)
PHASE_SET = [
    (0.0, 0.0),
    (0.5, 0.0),
    (0.0, 0.5),
    (0.5, 0.5),
]

# Symmetry-unique orientations
ORIENTATIONS = {

    "square": [
        0.0,
        22.5,
        45.0,
        67.5,
    ],

    "triangle": [
        0.0,
        15.0,
        30.0,
        45.0,
    ],

    "hex": [
        0.0,
        15.0,
        30.0,
        45.0,
    ],
}

N_STRETCH_PAIRS = 5000

STRETCH_SEED = 12345


# ============================================================
# PHASE CONVERSION
# ============================================================

def normalized_phase_to_physical(
    tau,
    nominal_cell_area,
    phase_u,
    phase_v,
):
    """
    Convert normalized phase coordinates (u, v) into the
    physical translation vector expected by build_tessellation():

        phi_tau(u,v) = u * a1_tau + v * a2_tau

    The basis vectors correspond to the unrotated lattice.
    Rotation is subsequently applied by build_tessellation().
    """

    A_c = nominal_cell_area

    # --------------------------------------------------------
    # Square
    # --------------------------------------------------------

    if tau == "square":

        side = math.sqrt(
            A_c
        )

        a1 = np.array([
            side,
            0.0,
        ])

        a2 = np.array([
            0.0,
            side,
        ])

    # --------------------------------------------------------
    # Triangular
    # --------------------------------------------------------

    elif tau == "triangle":

        side = math.sqrt(
            4.0
            * A_c
            / math.sqrt(3.0)
        )

        h_tri = (
            math.sqrt(3.0)
            * side
            / 2.0
        )

        a1 = np.array([
            side,
            0.0,
        ])

        a2 = np.array([
            side / 2.0,
            h_tri,
        ])

    # --------------------------------------------------------
    # Hexagonal
    # --------------------------------------------------------

    elif tau == "hex":

        side = math.sqrt(
            2.0
            * A_c
            / (
                3.0
                * math.sqrt(3.0)
            )
        )

        h_hex = (
            math.sqrt(3.0)
            * side
        )

        a1 = np.array([
            1.5 * side,
            h_hex / 2.0,
        ])

        a2 = np.array([
            0.0,
            h_hex,
        ])

    else:

        raise ValueError(
            f"Unknown tessellation: {tau}"
        )

    physical_phase = (
        phase_u * a1
        +
        phase_v * a2
    )

    return physical_phase


# ============================================================
# PHYSICAL PATH STRETCH
# ============================================================

def physical_path_stretch(
    tessellation,
    s0,
    n_pairs=1000,
    seed=12345,
):
    """
    Physical shortest-path stretch over sampled node pairs
    within the connected component containing s0.
    """

    G = tessellation.graph

    reachable = list(
        nx.node_connected_component(
            G,
            s0,
        )
    )

    if len(reachable) < 2:

        return {
            "stretch_mean": np.nan,
            "stretch_p90": np.nan,
            "stretch_std": np.nan,
            "n_stretch_pairs": 0,
        }

    rng = np.random.default_rng(
        seed
    )

    stretches = []

    for _ in range(
        n_pairs
    ):

        i, j = rng.choice(
            reachable,
            size=2,
            replace=False,
        )

        qi = (
            tessellation
            .cells[i]
            .centroid
        )

        qj = (
            tessellation
            .cells[j]
            .centroid
        )

        euclidean_distance = (
            np.linalg.norm(
                qi - qj
            )
        )

        if euclidean_distance <= 1e-12:
            continue

        graph_distance = (
            nx.shortest_path_length(
                G,
                source=i,
                target=j,
                weight="weight",
            )
        )

        stretches.append(
            graph_distance
            / euclidean_distance
        )

    stretches = np.asarray(
        stretches
    )

    return {

        "stretch_mean":
            float(
                np.mean(
                    stretches
                )
            ),

        "stretch_p90":
            float(
                np.percentile(
                    stretches,
                    90,
                )
            ),

        "stretch_std":
            float(
                np.std(
                    stretches,
                    ddof=1,
                )
            ),

        "n_stretch_pairs":
            len(
                stretches
            ),
    }


# ============================================================
# GRAPH METRICS
# ============================================================

def compute_graph_metrics(
    tessellation,
    x0,
):
    """
    Compute navigation-graph metrics for one lattice
    phase-orientation configuration.
    """

    G = tessellation.graph

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    s0 = initial_state_from_position(
        tessellation,
        x0,
    )

    # --------------------------------------------------------
    # Basic graph size
    # --------------------------------------------------------

    n_nodes = (
        G.number_of_nodes()
    )

    n_edges = (
        G.number_of_edges()
    )

    # --------------------------------------------------------
    # Degree statistics
    # --------------------------------------------------------

    degrees = np.array(
        [
            degree
            for _, degree
            in G.degree()
        ],
        dtype=float,
    )

    if len(degrees) > 0:

        mean_degree = float(
            np.mean(
                degrees
            )
        )

        std_degree = float(
            np.std(
                degrees,
                ddof=1,
            )
        )

        min_degree = int(
            np.min(
                degrees
            )
        )

        max_degree = int(
            np.max(
                degrees
            )
        )

    else:

        mean_degree = np.nan
        std_degree = np.nan
        min_degree = np.nan
        max_degree = np.nan

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    n_components = (
        nx.number_connected_components(
            G
        )
    )

    reachable = (
        nx.node_connected_component(
            G,
            s0,
        )
    )

    n_reachable = len(
        reachable
    )

    reachable_fraction = (
        n_reachable
        / n_nodes
        if n_nodes > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Physical shortest-path stretch
    # --------------------------------------------------------

    stretch = physical_path_stretch(
        tessellation=tessellation,
        s0=s0,
        n_pairs=N_STRETCH_PAIRS,
        seed=STRETCH_SEED,
    )

    return {

        "s0":
            s0,

        "n_nodes":
            n_nodes,

        "n_edges":
            n_edges,

        "mean_degree":
            mean_degree,

        "std_degree":
            std_degree,

        "min_degree":
            min_degree,

        "max_degree":
            max_degree,

        "n_components":
            n_components,

        "n_reachable":
            n_reachable,

        "reachable_fraction":
            reachable_fraction,

        "stretch_mean":
            stretch[
                "stretch_mean"
            ],

        "stretch_p90":
            stretch[
                "stretch_p90"
            ],

        "stretch_std":
            stretch[
                "stretch_std"
            ],

        "n_stretch_pairs":
            stretch[
                "n_stretch_pairs"
            ],
    }


# ============================================================
# RUN FULL ROBUSTNESS STUDY
# ============================================================

def run_graph_robustness():

    domains = make_normalized_domains(
        target_area=A0
    )

    rows = []

    expected_total = (
        len(DOMAIN_ORDER)
        * len(NREF_ORDER)
        * len(TAU_ORDER)
        * len(PHASE_SET)
        * 4
    )

    counter = 0

    # ========================================================
    # DOMAINS
    # ========================================================

    for domain_name in DOMAIN_ORDER:

        omega = domains[
            domain_name
        ]

        x0 = np.array([
            omega.centroid.x,
            omega.centroid.y,
        ])

        # ====================================================
        # RESOLUTIONS
        # ====================================================

        for N_ref in NREF_ORDER:

            nominal_cell_area = (
                omega.area
                / N_ref
            )

            # ================================================
            # TESSELLATIONS
            # ================================================

            for tau in TAU_ORDER:

                # ============================================
                # NORMALIZED PHASE
                # ============================================

                for (
                    phase_u,
                    phase_v,
                ) in PHASE_SET:

                    physical_phase = (
                        normalized_phase_to_physical(
                            tau=tau,
                            nominal_cell_area=(
                                nominal_cell_area
                            ),
                            phase_u=phase_u,
                            phase_v=phase_v,
                        )
                    )

                    # ========================================
                    # ORIENTATION
                    # ========================================

                    for theta_deg in (
                        ORIENTATIONS[tau]
                    ):

                        counter += 1

                        print(
                            f"[{counter:03d}/"
                            f"{expected_total}] "
                            f"{domain_name:8s} | "
                            f"N_ref={N_ref:3d} | "
                            f"{tau:8s} | "
                            f"phase="
                            f"({phase_u:.1f}, "
                            f"{phase_v:.1f}) | "
                            f"theta="
                            f"{theta_deg:5.1f}"
                        )

                        # ------------------------------------
                        # Build tessellation
                        # ------------------------------------

                        tess = (
                            build_tessellation(
                                omega=omega,
                                tau=tau,
                                nominal_cell_area=(
                                    nominal_cell_area
                                ),
                                anchor=x0,
                                phase=(
                                    physical_phase
                                ),
                                orientation_deg=(
                                    theta_deg
                                ),
                            )
                        )

                        # ------------------------------------
                        # Compute graph metrics
                        # ------------------------------------

                        metrics = (
                            compute_graph_metrics(
                                tessellation=tess,
                                x0=x0,
                            )
                        )

                        # ------------------------------------
                        # Store configuration
                        # ------------------------------------

                        rows.append({

                            "domain":
                                domain_name,

                            "N_ref":
                                N_ref,

                            "tau":
                                tau,

                            "phase_u":
                                phase_u,

                            "phase_v":
                                phase_v,

                            "phase_x":
                                physical_phase[0],

                            "phase_y":
                                physical_phase[1],

                            "theta_deg":
                                theta_deg,

                            **metrics,
                        })

    df = pd.DataFrame(
        rows
    )

    return df


# ============================================================
# SANITY CHECKS
# ============================================================

def sanity_check(
    df,
):

    expected_rows = (
        5
        * 4
        * 3
        * 16
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "SANITY CHECK"
    )

    print(
        "========================================"
    )

    print(
        f"Rows obtained: "
        f"{len(df)}"
    )

    print(
        f"Rows expected: "
        f"{expected_rows}"
    )

    if len(df) != expected_rows:

        raise AssertionError(
            "Unexpected number of "
            "robustness configurations."
        )

    # --------------------------------------------------------
    # 16 configurations for each condition
    # --------------------------------------------------------

    counts = (
        df.groupby(
            [
                "domain",
                "N_ref",
                "tau",
            ]
        )
        .size()
    )

    incorrect = counts[
        counts != 16
    ]

    if len(incorrect) > 0:

        print(
            "\nConditions without "
            "16 configurations:"
        )

        print(
            incorrect
        )

        raise AssertionError(
            "Some conditions do not contain "
            "exactly 16 configurations."
        )

    print(
        "OK: every domain-resolution-"
        "tessellation condition contains "
        "16 phase-orientation configurations."
    )

    # --------------------------------------------------------
    # Stretch pairs
    # --------------------------------------------------------

    if (
        df["n_stretch_pairs"]
        != N_STRETCH_PAIRS
    ).any():

        print(
            "\nWARNING: some configurations "
            "contain fewer stretch pairs."
        )

        print(
            df.loc[
                df["n_stretch_pairs"]
                != N_STRETCH_PAIRS,
                [
                    "domain",
                    "N_ref",
                    "tau",
                    "phase_u",
                    "phase_v",
                    "theta_deg",
                    "n_stretch_pairs",
                ],
            ]
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    check_columns = [
        "mean_degree",
        "reachable_fraction",
        "stretch_mean",
        "stretch_p90",
    ]

    if (
        df[check_columns]
        .isna()
        .any()
        .any()
    ):

        print(
            "\nWARNING: NaN values found:"
        )

        print(
            df.loc[
                df[
                    check_columns
                ]
                .isna()
                .any(
                    axis=1
                )
            ]
        )

    print(
        "\nSanity checks completed."
    )


# ============================================================
# SUMMARY
# ============================================================

def make_summary(
    df,
):
    """
    Aggregate results following the same logic used in the
    geometric robustness analysis.

    Step 1:
        For each domain / resolution / tessellation,
        average over the 16 phase-orientation configurations.

    Step 2:
        For each resolution / tessellation,
        average the five domain-specific means.

    The phase-orientation standard deviation reported for
    stretch is first computed within each domain and then
    averaged over the five domains.

    Connectivity quantities correspond to the worst observed
    configuration over domains, phases and orientations.
    """

    # ========================================================
    # PER-DOMAIN ROBUSTNESS STATISTICS
    # ========================================================

    domain_summary = (
        df.groupby(
            [
                "domain",
                "N_ref",
                "tau",
            ]
        )
        .agg(

            mean_degree=(
                "mean_degree",
                "mean",
            ),

            mean_degree_phase_std=(
                "mean_degree",
                "std",
            ),

            stretch_mean=(
                "stretch_mean",
                "mean",
            ),

            stretch_mean_phase_std=(
                "stretch_mean",
                "std",
            ),

            stretch_p90=(
                "stretch_p90",
                "mean",
            ),

            stretch_p90_phase_std=(
                "stretch_p90",
                "std",
            ),
        )
        .reset_index()
    )

    # ========================================================
    # AVERAGE OVER FIVE DOMAINS
    # ========================================================

    overall = (
        domain_summary
        .groupby(
            [
                "N_ref",
                "tau",
            ]
        )
        .agg(

            mean_degree=(
                "mean_degree",
                "mean",
            ),

            mean_degree_phase_std=(
                "mean_degree_phase_std",
                "mean",
            ),

            stretch_mean=(
                "stretch_mean",
                "mean",
            ),

            stretch_mean_phase_std=(
                "stretch_mean_phase_std",
                "mean",
            ),

            stretch_p90=(
                "stretch_p90",
                "mean",
            ),

            stretch_p90_phase_std=(
                "stretch_p90_phase_std",
                "mean",
            ),
        )
        .reset_index()
    )

    # ========================================================
    # WORST-CASE CONNECTIVITY
    # ========================================================

    connectivity = (
        df.groupby(
            [
                "N_ref",
                "tau",
            ]
        )
        .agg(

            n_components_max=(
                "n_components",
                "max",
            ),

            reachable_fraction_min=(
                "reachable_fraction",
                "min",
            ),
        )
        .reset_index()
    )

    # ========================================================
    # MERGE
    # ========================================================

    summary = (
        overall.merge(
            connectivity,
            on=[
                "N_ref",
                "tau",
            ],
            how="left",
        )
    )

    summary = summary[
        [
            "N_ref",
            "tau",

            "mean_degree",
            "mean_degree_phase_std",

            "n_components_max",
            "reachable_fraction_min",

            "stretch_mean",
            "stretch_mean_phase_std",

            "stretch_p90",
            "stretch_p90_phase_std",
        ]
    ]

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # RUN EXPERIMENT
    # ========================================================

    results = (
        run_graph_robustness()
    )

    # ========================================================
    # CHECK RESULTS
    # ========================================================

    sanity_check(
        results
    )

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    results.to_csv(
        RAW_OUTPUT,
        index=False,
    )

    print(
        "\nSaved raw results to:"
    )

    print(
        RAW_OUTPUT
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        make_summary(
            results
        )
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print(
        "\nSaved summary to:"
    )

    print(
        SUMMARY_OUTPUT
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "GRAPH ROBUSTNESS SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        summary.to_string(
            index=False
        )
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()