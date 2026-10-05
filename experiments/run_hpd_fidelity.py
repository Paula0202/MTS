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
    hpd_iou,
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_CONFIGS = None

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

GAMMAS = [
    0.5,
    0.8,
]

# Cell probability integration:
# already validated independently.
CELL_MASS_GRID = 120

# Continuous HPD-region approximation:
# convergence validated from n_grid = 320 onward.
HPD_GRID = 320


# ============================================================
# OUTPUT FILES
# ============================================================

RESULTS_DIR = "results"

CHECKPOINT_FILE = os.path.join(
    RESULTS_DIR,
    "hpd_fidelity_partial.csv",
)

FINAL_FILE = os.path.join(
    RESULTS_DIR,
    "hpd_fidelity.csv",
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
            float(row["gamma"]),
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
    *
    len(GAMMAS)
)

print(
    "\n========================================"
)

print(
    f"Total expected HPD results: "
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

                if MAX_CONFIGS is not None and configuration_counter > MAX_CONFIGS:
                    print("\nTest limit reached.")
                    raise SystemExit

                gamma_keys = [
                    (
                        domain_name,
                        prior_name,
                        tau,
                        N_ref,
                        gamma,
                    )
                    for gamma in GAMMAS
                ]

                # If both gamma values already exist,
                # skip the whole configuration.
                if all(
                    key in completed
                    for key in gamma_keys
                ):

                    print(
                        f"[SKIP] "
                        f"{domain_name} | "
                        f"{prior_name} | "
                        f"{tau} | "
                        f"N={N_ref}"
                    )

                    continue

                print(
                    f"\n[{configuration_counter}/120] "
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
                #
                # Computed once and reused for both gamma
                # values.
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
                # HPD-IoU
                # ============================================

                for gamma in GAMMAS:

                    key = (
                        domain_name,
                        prior_name,
                        tau,
                        N_ref,
                        gamma,
                    )

                    # Allows recovery if one gamma was
                    # completed but the other was not.
                    if key in completed:

                        print(
                            f"    gamma={gamma}: "
                            f"already completed"
                        )

                        continue

                    result = hpd_iou(
                        tessellation=tess,
                        p0=p0,
                        belief=belief,
                        represented_mass=represented_mass,
                        gamma=gamma,
                        n_grid=HPD_GRID,
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

                        "gamma":
                            gamma,

                        "hpd_iou":
                            result[
                                "hpd_iou"
                            ],

                        "continuous_hpd_mass":
                            result[
                                "continuous_hpd_mass"
                            ],

                        "discrete_hpd_mass":
                            result[
                                "discrete_hpd_mass"
                            ],

                        "n_selected_cells":
                            result[
                                "n_selected_cells"
                            ],

                        "represented_mass":
                            represented_mass,

                        "cell_mass_grid":
                            CELL_MASS_GRID,

                        "hpd_grid":
                            HPD_GRID,
                    }

                    rows.append(
                        row
                    )

                    completed.add(
                        key
                    )

                    print(
                        f"    gamma={gamma}"
                        f" | IoU="
                        f"{result['hpd_iou']:.6f}"
                        f" | continuous mass="
                        f"{result['continuous_hpd_mass']:.6f}"
                        f" | discrete mass="
                        f"{result['discrete_hpd_mass']:.6f}"
                        f" | cells="
                        f"{result['n_selected_cells']}"
                    )

                # ============================================
                # CHECKPOINT
                #
                # Save after every domain-prior-tau-N
                # configuration.
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
            "gamma",
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
    "HPD EXPERIMENT COMPLETE"
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