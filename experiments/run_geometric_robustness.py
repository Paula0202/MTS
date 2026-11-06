# ============================================================
# run_geometric_robustness.py
# ============================================================

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx


# ============================================================
# IMPORTS FROM PROJECT
# ============================================================
#
# CAMBIA ÚNICAMENTE ESTOS IMPORTS si tus módulos tienen
# nombres distintos.
#

from domains import make_normalized_domains

from tessellations import (
    build_tessellation,
    represented_region,
    initial_state_from_position,
)

from metrics_geometry import (
    geometric_metrics,
    average_symmetric_boundary_distance,
    physical_path_stretch,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PARTIAL_CSV = (
    RESULTS_DIR
    / "geometric_robustness_partial.csv"
)

PARTIAL_TMP_CSV = (
    RESULTS_DIR
    / "geometric_robustness_partial_tmp.csv"
)

FINAL_CSV = (
    RESULTS_DIR
    / "geometric_robustness.csv"
)

SUMMARY_CSV = (
    RESULTS_DIR
    / "geometric_robustness_summary.csv"
)

CANONICAL_CSV = (
    RESULTS_DIR
    / "geometric_robustness_canonical_subset.csv"
)


# ============================================================
# EXPERIMENTAL SETTINGS
# ============================================================

TARGET_AREA = 100.0

DOMAINS_TO_RUN = [
    "Square",
    "Circle",
    "Triangle",
    "Convex",
    "Concave",
]

N_REF_VALUES = [
    100,
    300,
    500,
    800,
]

TESSELLATIONS = [
    "square",
    "triangle",
    "hex",
]


# ------------------------------------------------------------
# Normalized lattice phases
# ------------------------------------------------------------
#
# phi_tau(u,v) = u a1_tau + v a2_tau
#

PHASES = [
    (0.0, 0.0),
    (0.5, 0.0),
    (0.0, 0.5),
    (0.5, 0.5),
]


# ------------------------------------------------------------
# Orientations
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# ASBD
# ------------------------------------------------------------

N_BOUNDARY_SAMPLES = 1000


# ------------------------------------------------------------
# Physical path stretch
# ------------------------------------------------------------

N_STRETCH_PAIRS = 1000
STRETCH_SEED = 12345


# ============================================================
# CELL GEOMETRY
# ============================================================


def side_length_from_area(
    tau,
    nominal_cell_area
):
    """
    Returns the cell side length corresponding to the nominal
    cell area for each regular tessellation.
    """

    if tau == "square":

        return math.sqrt(
            nominal_cell_area
        )

    if tau == "triangle":

        return math.sqrt(
            4.0
            * nominal_cell_area
            /
            math.sqrt(3.0)
        )

    if tau == "hex":

        return math.sqrt(
            2.0
            * nominal_cell_area
            /
            (
                3.0
                * math.sqrt(3.0)
            )
        )

    raise ValueError(
        f"Unknown tessellation: {tau}"
    )


# ============================================================
# PERIODIC BASIS
# ============================================================

def lattice_basis_vectors(
    tau,
    nominal_cell_area
):
    """
    Returns the canonical periodic basis vectors
    (a1_tau, a2_tau).

    The normalized phase coordinates (u,v) are converted as

        phi_tau(u,v) = u a1_tau + v a2_tau.

    The tessellation implementation applies translation first
    and rotation second, so this phase vector is subsequently
    rotated together with the lattice.
    """

    side = side_length_from_area(
        tau=tau,
        nominal_cell_area=nominal_cell_area
    )


    # --------------------------------------------------------
    # Square
    # --------------------------------------------------------

    if tau == "square":

        a1 = np.array([
            side,
            0.0
        ])

        a2 = np.array([
            0.0,
            side
        ])


    # --------------------------------------------------------
    # Triangular
    # --------------------------------------------------------

    elif tau == "triangle":

        h = (
            math.sqrt(3.0)
            * side
            / 2.0
        )

        a1 = np.array([
            side,
            0.0
        ])

        a2 = np.array([
            side / 2.0,
            h
        ])


    # --------------------------------------------------------
    # Hexagonal
    # --------------------------------------------------------

    elif tau == "hex":

        h = (
            math.sqrt(3.0)
            * side
        )

        a1 = np.array([
            1.5 * side,
            h / 2.0
        ])

        a2 = np.array([
            0.0,
            h
        ])


    else:

        raise ValueError(
            f"Unknown tessellation: {tau}"
        )


    return (
        a1.astype(float),
        a2.astype(float)
    )


# ============================================================
# PHASE
# ============================================================

def physical_phase_vector(
    tau,
    nominal_cell_area,
    u,
    v
):
    """
    Converts normalized lattice-phase coordinates (u,v) to
    Cartesian displacement.

        phi_tau(u,v)
        =
        u a1_tau + v a2_tau
    """

    a1, a2 = lattice_basis_vectors(
        tau=tau,
        nominal_cell_area=nominal_cell_area
    )

    return (
        float(u) * a1
        +
        float(v) * a2
    )


# ============================================================
# CONFIGURATION KEY
# ============================================================

def configuration_key(
    domain,
    n_ref,
    tau,
    phase_u,
    phase_v,
    theta_deg
):
    """
    Unique identifier of one experimental configuration.
    """

    return (
        str(domain),
        int(n_ref),
        str(tau),
        float(phase_u),
        float(phase_v),
        float(theta_deg),
    )


# ============================================================
# EXPERIMENT LIST
# ============================================================

def build_experiment_list():
    """
    Generates all combinations:

        domain
        x N_ref
        x tessellation
        x phase
        x orientation
    """

    experiments = []

    for domain_name in DOMAINS_TO_RUN:

        for n_ref in N_REF_VALUES:

            for tau in TESSELLATIONS:

                for phase_u, phase_v in PHASES:

                    for theta_deg in ORIENTATIONS[
                        tau
                    ]:

                        experiments.append({

                            "domain":
                                domain_name,

                            "N_ref":
                                n_ref,

                            "tau":
                                tau,

                            "phase_u":
                                phase_u,

                            "phase_v":
                                phase_v,

                            "theta_deg":
                                theta_deg,
                        })

    return experiments


# ============================================================
# PARTIAL RESULTS
# ============================================================

def load_partial_results():
    """
    Loads the partial CSV, if it exists, and returns both the
    existing dataframe and the set of completed configurations.
    """

    if not PARTIAL_CSV.exists():

        return (
            pd.DataFrame(),
            set()
        )


    print(
        f"Loading partial results from:\n"
        f"{PARTIAL_CSV}"
    )

    df = pd.read_csv(
        PARTIAL_CSV
    )


    completed = set()

    for _, row in df.iterrows():

        key = configuration_key(
            domain=row["domain"],
            n_ref=row["N_ref"],
            tau=row["tau"],
            phase_u=row["phase_u"],
            phase_v=row["phase_v"],
            theta_deg=row["theta_deg"],
        )

        completed.add(
            key
        )


    return (
        df,
        completed
    )


# ============================================================
# SAFE CHECKPOINT
# ============================================================

def save_partial_results(
    rows
):
    """
    Saves the partial CSV atomically.

    A temporary CSV is written first and then replaces the
    previous checkpoint. This reduces the risk of corrupting
    the partial results if execution stops during writing.
    """

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        PARTIAL_TMP_CSV,
        index=False
    )

    os.replace(
        PARTIAL_TMP_CSV,
        PARTIAL_CSV
    )


# ============================================================
# EVALUATE ONE CONFIGURATION
# ============================================================

def evaluate_configuration(
    domains,
    config
):
    """
    Builds one tessellation and evaluates geometric fidelity,
    boundary properties, graph structure, reachability and
    physical shortest-path stretch.
    """

    # --------------------------------------------------------
    # Read configuration
    # --------------------------------------------------------

    domain_name = config[
        "domain"
    ]

    n_ref = int(
        config["N_ref"]
    )

    tau = config[
        "tau"
    ]

    phase_u = float(
        config["phase_u"]
    )

    phase_v = float(
        config["phase_v"]
    )

    theta_deg = float(
        config["theta_deg"]
    )


    # --------------------------------------------------------
    # Continuous domain
    # --------------------------------------------------------

    omega = domains[
        domain_name
    ]


    # --------------------------------------------------------
    # Equivalent nominal cell area
    # --------------------------------------------------------

    nominal_cell_area = (
        omega.area
        /
        float(n_ref)
    )


    # --------------------------------------------------------
    # Physical phase vector
    # --------------------------------------------------------

    phase = physical_phase_vector(
        tau=tau,
        nominal_cell_area=nominal_cell_area,
        u=phase_u,
        v=phase_v
    )


    # --------------------------------------------------------
    # Build tessellation
    # --------------------------------------------------------

    tessellation = build_tessellation(
        omega=omega,
        tau=tau,
        nominal_cell_area=nominal_cell_area,

        # Common anchor:
        # domain centroid
        anchor=None,

        phase=phase,
        orientation_deg=theta_deg,
    )


    # --------------------------------------------------------
    # Existing geometric + graph metrics
    # --------------------------------------------------------

    metrics = geometric_metrics(
        omega=omega,
        tessellation=tessellation
    )


    # --------------------------------------------------------
    # Represented region
    # --------------------------------------------------------

    omega_hat = represented_region(
        tessellation
    )


    # --------------------------------------------------------
    # ASBD
    # --------------------------------------------------------

    asbd = (
        average_symmetric_boundary_distance(
            omega=omega,
            omega_hat=omega_hat,
            n_samples=N_BOUNDARY_SAMPLES,
        )
    )


    # --------------------------------------------------------
    # Common physical initial position x0
    # --------------------------------------------------------

    x0 = np.array([
        omega.centroid.x,
        omega.centroid.y
    ])


    # --------------------------------------------------------
    # Map x0 to discrete initial state s0
    # --------------------------------------------------------

    s0 = initial_state_from_position(
        tessellation=tessellation,
        x0=x0
    )


    # --------------------------------------------------------
    # Reachability
    # --------------------------------------------------------

    G = tessellation.graph

    if (
        G.number_of_nodes()
        > 0
    ):

        reachable = (
            nx.node_connected_component(
                G,
                s0
            )
        )

        n_reachable = len(
            reachable
        )

        reachable_fraction = (
            n_reachable
            /
            G.number_of_nodes()
        )

    else:

        n_reachable = 0

        reachable_fraction = (
            np.nan
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


    # --------------------------------------------------------
    # Collect results
    # --------------------------------------------------------

    result = {

        # Experimental condition
        "domain":
            domain_name,

        "N_ref":
            n_ref,

        "tau":
            tau,

        "phase_u":
            phase_u,

        "phase_v":
            phase_v,

        "theta_deg":
            theta_deg,

        "phase_x":
            float(
                phase[0]
            ),

        "phase_y":
            float(
                phase[1]
            ),

        "nominal_cell_area":
            nominal_cell_area,


        # ----------------------------------------------------
        # Metrics already returned by geometric_metrics()
        # ----------------------------------------------------

        **metrics,


        # ----------------------------------------------------
        # Additional robustness metrics
        # ----------------------------------------------------

        "asbd":
            float(asbd),

        "s0":
            int(s0),

        "n_reachable":
            int(n_reachable),

        "reachable_fraction":
            float(
                reachable_fraction
            ),

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


    return result


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    df
):
    """
    Aggregates the 16 phase-orientation configurations for each

        domain x N_ref x tessellation

    combination.

    Mean, standard deviation, minimum, maximum and range are
    reported.
    """

    metrics = [

        # Main geometric fidelity
        "coverage",
        "external_excess",
        "iou",
        "asbd",

        # Boundary representation
        "N_tau",
        "n_boundary",
        "boundary_proportion",

        "valid_area_mean",
        "valid_area_std",
        "valid_area_min",
        "valid_area_max",

        # Graph
        "n_nodes",
        "n_edges",
        "mean_degree",
        "degree_std",
        "n_components",
        "graph_diameter_hops",
        "reachable_fraction",

        # Physical path stretch
        "stretch_mean",
        "stretch_p90",
    ]


    # Keep only columns that actually exist.
    metrics = [
        metric
        for metric in metrics
        if metric in df.columns
    ]


    grouped = (
        df
        .groupby(
            [
                "domain",
                "N_ref",
                "tau",
            ]
        )[metrics]
        .agg(
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        )
        .reset_index()
    )


    # --------------------------------------------------------
    # Flatten MultiIndex column names
    # --------------------------------------------------------

    new_columns = []

    for col in grouped.columns:

        if isinstance(
            col,
            tuple
        ):

            parts = [
                str(part)
                for part in col
                if str(part) != ""
            ]

            new_columns.append(
                "_".join(parts)
            )

        else:

            new_columns.append(
                str(col)
            )


    grouped.columns = (
        new_columns
    )


    # --------------------------------------------------------
    # Add range
    # --------------------------------------------------------

    for metric in metrics:

        min_col = (
            f"{metric}_min"
        )

        max_col = (
            f"{metric}_max"
        )

        if (
            min_col
            in grouped.columns
            and
            max_col
            in grouped.columns
        ):

            grouped[
                f"{metric}_range"
            ] = (
                grouped[max_col]
                -
                grouped[min_col]
            )


    return grouped


# ============================================================
# SAVE CANONICAL SUBSET
# ============================================================

def save_canonical_subset(
    df
):
    """
    Extracts the phase=(0,0), theta=0 configurations.

    This is useful for checking that the robustness pipeline
    reproduces the existing canonical experiment.
    """

    canonical = df[
        (
            np.isclose(
                df["phase_u"],
                0.0
            )
        )
        &
        (
            np.isclose(
                df["phase_v"],
                0.0
            )
        )
        &
        (
            np.isclose(
                df["theta_deg"],
                0.0
            )
        )
    ].copy()


    canonical.to_csv(
        CANONICAL_CSV,
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Domains
    # --------------------------------------------------------

    domains = (
        make_normalized_domains(
            target_area=TARGET_AREA
        )
    )


    # --------------------------------------------------------
    # Configurations
    # --------------------------------------------------------

    experiments = (
        build_experiment_list()
    )

    total = len(
        experiments
    )


    print(
        "\n"
        "==============================================\n"
        " GEOMETRIC ROBUSTNESS EXPERIMENT\n"
        "=============================================="
    )

    print(
        f"Domains:        "
        f"{len(DOMAINS_TO_RUN)}"
    )

    print(
        f"Resolutions:    "
        f"{len(N_REF_VALUES)}"
    )

    print(
        f"Tessellations:  "
        f"{len(TESSELLATIONS)}"
    )

    print(
        f"Phases:         "
        f"{len(PHASES)}"
    )

    print(
        f"Orientations:   "
        f"4 per tessellation"
    )

    print(
        f"Total configs:  "
        f"{total}"
    )

    print(
        "==============================================\n"
    )


    # --------------------------------------------------------
    # Resume previous execution
    # --------------------------------------------------------

    partial_df, completed = (
        load_partial_results()
    )


    if len(partial_df) > 0:

        rows = (
            partial_df
            .to_dict(
                orient="records"
            )
        )

    else:

        rows = []


    # --------------------------------------------------------
    # Determine remaining configurations
    # --------------------------------------------------------

    remaining = []

    for config in experiments:

        key = configuration_key(
            domain=config[
                "domain"
            ],

            n_ref=config[
                "N_ref"
            ],

            tau=config[
                "tau"
            ],

            phase_u=config[
                "phase_u"
            ],

            phase_v=config[
                "phase_v"
            ],

            theta_deg=config[
                "theta_deg"
            ],
        )

        if key not in completed:

            remaining.append(
                config
            )


    print(
        f"Already completed: "
        f"{len(completed)}"
    )

    print(
        f"Remaining:         "
        f"{len(remaining)}\n"
    )


    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    for local_idx, config in enumerate(
        remaining,
        start=1
    ):

        global_idx = (
            len(completed)
            + local_idx
        )


        print(
            f"[{global_idx}/{total}] "
            f"{config['domain']} | "
            f"N_ref={config['N_ref']} | "
            f"{config['tau']} | "
            f"phase=("
            f"{config['phase_u']},"
            f"{config['phase_v']}"
            f") | "
            f"theta="
            f"{config['theta_deg']}"
        )


        try:

            result = (
                evaluate_configuration(
                    domains=domains,
                    config=config
                )
            )


        except Exception as exc:

            print(
                "\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                "ERROR while evaluating configuration:\n"
                f"{config}\n"
                f"{type(exc).__name__}: {exc}\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            )

            # Save everything obtained up to this point.
            save_partial_results(
                rows
            )

            raise


        rows.append(
            result
        )


        # ----------------------------------------------------
        # Save checkpoint after EVERY completed configuration
        # ----------------------------------------------------

        save_partial_results(
            rows
        )


        print(
            "    "
            f"IoU={result['iou']:.6f} | "
            f"ASBD={result['asbd']:.6f} | "
            f"Cov={result['coverage']:.6f} | "
            f"Eext={result['external_excess']:.6f} | "
            f"S={result['stretch_mean']:.6f} | "
            f"reach={result['reachable_fraction']:.6f}"
        )


    # ========================================================
    # FINAL DATAFRAME
    # ========================================================

    final_df = pd.DataFrame(
        rows
    )


    # Sort for readability
    final_df = (
        final_df
        .sort_values(
            by=[
                "domain",
                "N_ref",
                "tau",
                "phase_u",
                "phase_v",
                "theta_deg",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # SAVE FULL RESULTS
    # ========================================================

    final_df.to_csv(
        FINAL_CSV,
        index=False
    )


    # ========================================================
    # SAVE CANONICAL SUBSET
    # ========================================================

    save_canonical_subset(
        final_df
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = build_summary(
        final_df
    )


    summary.to_csv(
        SUMMARY_CSV,
        index=False
    )


    # ========================================================
    # FINAL INFORMATION
    # ========================================================

    print(
        "\n"
        "==============================================\n"
        " EXPERIMENT FINISHED\n"
        "=============================================="
    )

    print(
        f"Configurations completed: "
        f"{len(final_df)}"
    )

    print(
        f"\nFull results:\n"
        f"{FINAL_CSV}"
    )

    print(
        f"\nSummary:\n"
        f"{SUMMARY_CSV}"
    )

    print(
        f"\nCanonical subset:\n"
        f"{CANONICAL_CSV}"
    )

    print(
        f"\nPartial checkpoint:\n"
        f"{PARTIAL_CSV}"
    )

    print(
        "==============================================\n"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()