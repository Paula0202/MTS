import os

import numpy as np
import pandas as pd

from domains import make_normalized_domains

from priors import (
    make_single_hotspot_prior,
    make_multimodal_prior,
)

from tessellations import build_tessellation

from metrics_probability import (
    cell_probability_masses,
    mode_localization_error,
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_CONFIGS = None

TEST_CASES = None

A0 = 100.0
L0 = np.sqrt(A0)

N_REFS = [
    100,
    300,
    500,
    800,
]

TAUS = [
    "square",
    "triangle",
    "hex",
]

PRIORS = [
    "single_hotspot",
    "multimodal",
]

# Cell probability integration:
# already validated independently.
CELL_MASS_GRID = 120

# Auxiliary grid used only to detect candidate
# continuous probability modes.
MODE_GRID = 300


# ============================================================
# OUTPUT FILES
# ============================================================

RESULTS_DIR = "results"

CHECKPOINT_FILE = os.path.join(
    RESULTS_DIR,
    "mode_localization_partial.csv",
)

FINAL_FILE = os.path.join(
    RESULTS_DIR,
    "mode_localization.csv",
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)


# ============================================================
# PRIOR CONSTRUCTION
# ============================================================

def build_prior(
    prior_name,
    omega,
):
    """
    Builds the same continuous priors used in the
    probabilistic-fidelity experiments.
    """

    center = np.array(
        [
            omega.centroid.x,
            omega.centroid.y,
        ]
    )

    # --------------------------------------------------------
    # Single hotspot
    # --------------------------------------------------------

    if prior_name == "single_hotspot":

        mean = (
            center
            +
            np.array(
                [
                    0.15 * L0,
                    -0.10 * L0,
                ]
            )
        )

        sigma = (
            0.15 * L0
        )

        return make_single_hotspot_prior(
            omega=omega,
            mean=mean,
            sigma=sigma,
            n_grid=400,
        )

    # --------------------------------------------------------
    # Multimodal
    # --------------------------------------------------------

    if prior_name == "multimodal":

        means = [
            center
            +
            np.array(
                [
                    -0.20 * L0,
                    -0.12 * L0,
                ]
            ),

            center
            +
            np.array(
                [
                    +0.18 * L0,
                    +0.16 * L0,
                ]
            ),
        ]

        sigmas = [
            0.12 * L0,
            0.12 * L0,
        ]

        weights = [
            0.6,
            0.4,
        ]

        return make_multimodal_prior(
            omega=omega,
            means=means,
            sigmas=sigmas,
            weights=weights,
            n_grid=400,
        )

    raise ValueError(
        f"Unknown prior: {prior_name}"
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

if os.path.exists(
    CHECKPOINT_FILE
):

    df_existing = pd.read_csv(
        CHECKPOINT_FILE
    )

    print(
        f"Checkpoint found: "
        f"{len(df_existing)} rows"
    )

else:

    df_existing = pd.DataFrame()

    print(
        "No checkpoint found. "
        "Starting from scratch."
    )


# ============================================================
# COMPLETED CONFIGURATIONS
# ============================================================

completed = set()

if not df_existing.empty:

    for _, row in (
        df_existing.iterrows()
    ):

        key = (
            row["domain"],
            row["prior"],
            row["tau"],
            int(row["N_ref"]),
        )

        completed.add(
            key
        )


# ============================================================
# DOMAINS
# ============================================================

domains = make_normalized_domains(
    target_area=A0
)


# ============================================================
# TOTAL NUMBER OF RESULTS
# ============================================================

total_results = (
    len(domains)
    *
    len(PRIORS)
    *
    len(TAUS)
    *
    len(N_REFS)
)

print(
    "\n========================================"
)

print(
    f"Total expected mode-localization results: "
    f"{total_results}"
)

print(
    f"Already completed: "
    f"{len(completed)}"
)

print(
    "========================================\n"
)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

rows = []

if not df_existing.empty:

    rows = (
        df_existing
        .to_dict(
            orient="records"
        )
    )


configuration_counter = 0


for domain_name, omega in (
    domains.items()
):

    print(
        "\n########################################"
    )

    print(
        f"DOMAIN: {domain_name}"
    )

    print(
        "########################################"
    )

    anchor = np.array(
        [
            omega.centroid.x,
            omega.centroid.y,
        ]
    )

    # --------------------------------------------------------
    # Priors
    # --------------------------------------------------------

    for prior_name in PRIORS:

        print(
            f"\nPrior: {prior_name}"
        )

        p0 = build_prior(
            prior_name=prior_name,
            omega=omega,
        )

        expected_modes = (
            1
            if prior_name == "single_hotspot"
            else 2
        )

        # ----------------------------------------------------
        # Reference resolution
        # ----------------------------------------------------

        for N_ref in N_REFS:

            nominal_cell_area = (
                A0
                /
                N_ref
            )

            # ------------------------------------------------
            # Tessellation
            # ------------------------------------------------

            for tau in TAUS:

                configuration_counter += 1

                if (
                    MAX_CONFIGS is not None
                    and
                    configuration_counter > MAX_CONFIGS
                ):

                    print(
                        "\nTest limit reached."
                    )

                    raise SystemExit

                key = (
                    domain_name,
                    prior_name,
                    tau,
                    N_ref,
                )

                if TEST_CASES is not None and key not in TEST_CASES:
                    continue

                if key in completed:

                    print(
                        f"[SKIP] "
                        f"{domain_name} | "
                        f"{prior_name} | "
                        f"{tau} | "
                        f"N={N_ref}"
                    )

                    continue

                print(
                    f"\n[{configuration_counter}/"
                    f"{total_results}] "
                    f"{domain_name} | "
                    f"{prior_name} | "
                    f"{tau} | "
                    f"N={N_ref}"
                )

                # ============================================
                # Build tessellation
                # ============================================

                tess = build_tessellation(
                    omega=omega,
                    tau=tau,
                    nominal_cell_area=nominal_cell_area,
                    anchor=anchor,
                    phase=(0.0, 0.0),
                    orientation_deg=0.0,
                )

                # ============================================
                # Cell probabilities
                # ============================================

                (
                    masses,
                    represented_mass,
                    belief,
                ) = cell_probability_masses(
                    tessellation=tess,
                    p0=p0,
                    n_grid=CELL_MASS_GRID,
                )

                # ============================================
                # Mode localization
                # ============================================

                result = mode_localization_error(
                    tessellation=tess,
                    p0=p0,
                    belief=belief,
                    expected_modes=expected_modes,
                    n_grid=MODE_GRID,
                )

                row = {
                    "domain":
                        domain_name,

                    "prior":
                        prior_name,

                    "tau":
                        tau,

                    "N_ref":
                        N_ref,

                    "mode_error_mean":
                        result[
                            "mode_error_mean"
                        ],

                    "mode_error_max":
                        result[
                            "mode_error_max"
                        ],

                    "mode_error_mean_norm":
                        result[
                            "mode_error_mean_norm"
                        ],

                    "mode_error_max_norm":
                        result[
                            "mode_error_max_norm"
                        ],

                    "n_continuous_modes":
                        result[
                            "n_continuous_modes"
                        ],

                    "n_discrete_modes":
                        result[
                            "n_discrete_modes"
                        ],

                    "n_matched_modes":
                        result[
                            "n_matched_modes"
                        ],

                    "n_missing_modes":
                        result[
                            "n_missing_modes"
                        ],

                    "n_extra_modes":
                        result[
                            "n_extra_modes"
                        ],

                    "represented_mass":
                        represented_mass,

                    "cell_mass_grid":
                        CELL_MASS_GRID,

                    "mode_grid":
                        MODE_GRID,
                }

                rows.append(
                    row
                )

                completed.add(
                    key
                )

                print(
                    f"    mean error="
                    f"{result['mode_error_mean']:.6f}"
                    f" | max error="
                    f"{result['mode_error_max']:.6f}"
                    f" | continuous modes="
                    f"{result['n_continuous_modes']}"
                    f" | discrete modes="
                    f"{result['n_discrete_modes']}"
                    f" | matched="
                    f"{result['n_matched_modes']}"
                )

                # ============================================
                # Detailed diagnostics
                # ============================================

                print(
                    "    Continuous modes:"
                )

                for i, mode in enumerate(
                    result["continuous_modes"]
                ):

                    print(
                        f"      {i}: "
                        f"{mode['position']} "
                        f"| density="
                        f"{mode['density']:.6e}"
                    )

                print(
                    "    Discrete modes:"
                )

                for i, mode in enumerate(
                    result["discrete_modes"]
                ):

                    print(
                        f"      {i}: "
                        f"{mode['position']} "
                        f"| density="
                        f"{mode['density']:.6e}"
                        f" | cell="
                        f"{mode['cell_id']}"
                    )

                print(
                    "    Matches:"
                )

                for match in (
                    result["matches"]
                ):

                    print(
                        f"      continuous "
                        f"{match['continuous_index']}"
                        f" -> discrete "
                        f"{match['discrete_index']}"
                        f" | error="
                        f"{match['error']:.6f}"
                    )

                # ============================================
                # CHECKPOINT
                # ============================================

                df_checkpoint = pd.DataFrame(
                    rows
                )

                df_checkpoint.to_csv(
                    CHECKPOINT_FILE,
                    index=False,
                )

                print(
                    f"    Checkpoint: "
                    f"{len(df_checkpoint)}/"
                    f"{total_results} results"
                )


# ============================================================
# FINAL SAVE
# ============================================================

df_final = pd.DataFrame(
    rows
)

df_final = (
    df_final
    .sort_values(
        [
            "domain",
            "prior",
            "N_ref",
            "tau",
        ]
    )
    .reset_index(
        drop=True
    )
)

df_final.to_csv(
    FINAL_FILE,
    index=False,
)


print(
    "\n========================================"
)

print(
    "MODE-LOCALIZATION EXPERIMENT COMPLETE"
)

print(
    "========================================"
)

print(
    f"Rows: {len(df_final)}"
)

print(
    f"Final file: {FINAL_FILE}"
)

print(
    f"Checkpoint: {CHECKPOINT_FILE}"
)

print(
    "========================================"
)