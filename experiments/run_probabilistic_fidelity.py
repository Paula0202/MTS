import os
import numpy as np
import pandas as pd

from domains import make_normalized_domains
from priors import (
    make_uniform_prior,
    make_single_hotspot_prior,
    make_multimodal_prior
)
from metrics_probability import (
    support_loss,
    total_variation_distance
)
from tessellations import build_tessellation


# ============================================================
# CONFIGURATION
# ============================================================

A0 = 100.0

N_REF_VALUES = [
    100,
    300,
    500,
    800
]

TESSELLATIONS = [
    "square",
    "triangle",
    "hex"
]

N_GRID = 400
TV_GRID = 120

L0 = np.sqrt(A0)

SINGLE_SIGMA = 0.15 * L0
MULTI_SIGMA = 0.12 * L0

PARTIAL_FILE = (
    "results/probabilistic_fidelity_partial.csv"
)

FINAL_FILE = (
    "results/probabilistic_fidelity.csv"
)

SUMMARY_FILE = (
    "results/probabilistic_summary.csv"
)


# ============================================================
# PRIORS
# ============================================================

def make_prior_cases(omega):

    center = np.array([
        omega.centroid.x,
        omega.centroid.y
    ])

    prior_cases = {}

    # --------------------------------------------------------
    # Uniform prior
    # --------------------------------------------------------

    prior_cases["uniform"] = (
        make_uniform_prior(
            omega
        )
    )

    # --------------------------------------------------------
    # Single hotspot
    # --------------------------------------------------------

    single_mean = (
        center
        +
        np.array([
            0.15 * L0,
            -0.10 * L0
        ])
    )

    prior_cases["single_hotspot"] = (
        make_single_hotspot_prior(
            omega=omega,
            mean=single_mean,
            sigma=SINGLE_SIGMA,
            n_grid=N_GRID
        )
    )

    # --------------------------------------------------------
    # Multimodal prior
    # --------------------------------------------------------

    means = [
        center
        +
        np.array([
            -0.20 * L0,
            -0.12 * L0
        ]),

        center
        +
        np.array([
            +0.18 * L0,
            +0.16 * L0
        ])
    ]

    sigmas = [
        MULTI_SIGMA,
        MULTI_SIGMA
    ]

    weights = [
        0.6,
        0.4
    ]

    prior_cases["multimodal"] = (
        make_multimodal_prior(
            omega=omega,
            means=means,
            sigmas=sigmas,
            weights=weights,
            n_grid=N_GRID
        )
    )

    return prior_cases


# ============================================================
# CHECKPOINT UTILITIES
# ============================================================

def load_previous_results():

    if not os.path.exists(
        PARTIAL_FILE
    ):
        return pd.DataFrame()

    df = pd.read_csv(
        PARTIAL_FILE
    )

    print(
        f"\nLoaded checkpoint with "
        f"{len(df)} completed configurations."
    )

    return df


def configuration_key(
    domain_name,
    prior_name,
    tau,
    N_ref
):
    return (
        domain_name,
        prior_name,
        tau,
        int(N_ref)
    )


def completed_configurations(
    df
):

    if df.empty:
        return set()

    return set(
        configuration_key(
            row["domain"],
            row["prior"],
            row["tau"],
            row["N_ref"]
        )
        for _, row
        in df.iterrows()
    )


# ============================================================
# EXPERIMENT
# ============================================================

def run_probabilistic_fidelity():

    os.makedirs(
        "results",
        exist_ok=True
    )

    domains = (
        make_normalized_domains(
            target_area=A0
        )
    )

    previous_results = (
        load_previous_results()
    )

    completed = (
        completed_configurations(
            previous_results
        )
    )

    if previous_results.empty:
        rows = []
    else:
        rows = (
            previous_results
            .to_dict(
                orient="records"
            )
        )

    total_configurations = (
        len(domains)
        *
        3
        *
        len(N_REF_VALUES)
        *
        len(TESSELLATIONS)
    )

    completed_count = (
        len(completed)
    )

    print(
        f"\nTotal configurations: "
        f"{total_configurations}"
    )

    print(
        f"Already completed: "
        f"{completed_count}"
    )

    for (
        domain_name,
        omega
    ) in domains.items():

        print(
            f"\nDomain: "
            f"{domain_name}"
        )

        x0 = np.array([
            omega.centroid.x,
            omega.centroid.y
        ])

        prior_cases = (
            make_prior_cases(
                omega
            )
        )

        for (
            prior_name,
            p0
        ) in prior_cases.items():

            print(
                f"  Prior: "
                f"{prior_name}"
            )

            for N_ref in (
                N_REF_VALUES
            ):

                A_c = (
                    omega.area
                    /
                    N_ref
                )

                for tau in (
                    TESSELLATIONS
                ):

                    key = (
                        configuration_key(
                            domain_name,
                            prior_name,
                            tau,
                            N_ref
                        )
                    )

                    # ----------------------------------------
                    # Skip already-computed configurations
                    # ----------------------------------------

                    if key in completed:

                        print(
                            f"    SKIP "
                            f"{tau}, "
                            f"N_ref={N_ref}"
                        )

                        continue

                    print(
                        f"    RUN  "
                        f"{tau}, "
                        f"N_ref={N_ref}"
                    )

                    # ----------------------------------------
                    # Build tessellation
                    # ----------------------------------------

                    tess = (
                        build_tessellation(
                            omega=omega,
                            tau=tau,
                            nominal_cell_area=A_c,
                            anchor=x0,
                            phase=(0.0, 0.0),
                            orientation_deg=0.0
                        )
                    )

                    # ----------------------------------------
                    # Support metrics
                    # ----------------------------------------

                    metrics = (
                        support_loss(
                            tessellation=tess,
                            p0=p0,
                            n_grid=N_GRID
                        )
                    )

                    # ----------------------------------------
                    # Total variation
                    # ----------------------------------------

                    tv_metrics = (
                        total_variation_distance(
                            tessellation=tess,
                            p0=p0,
                            n_grid=TV_GRID
                        )
                    )

                    # ----------------------------------------
                    # Store row
                    # ----------------------------------------

                    row = {
                        "domain":
                            domain_name,

                        "prior":
                            prior_name,

                        "tau":
                            tau,

                        "N_ref":
                            N_ref,

                        "N_tau":
                            len(
                                tess.cells
                            ),

                        "represented_mass":
                            metrics[
                                "represented_mass"
                            ],

                        "support_loss":
                            metrics[
                                "support_loss"
                            ],

                        "tv_distance":
                            tv_metrics[
                                "tv_distance"
                            ]
                    }

                    rows.append(
                        row
                    )

                    completed.add(
                        key
                    )

                    completed_count += 1

                    # ----------------------------------------
                    # SAVE CHECKPOINT AFTER EVERY CONFIGURATION
                    # ----------------------------------------

                    current_df = (
                        pd.DataFrame(
                            rows
                        )
                    )

                    current_df.to_csv(
                        PARTIAL_FILE,
                        index=False
                    )

                    print(
                        f"      Saved checkpoint "
                        f"({completed_count}/"
                        f"{total_configurations})"
                    )

    # ========================================================
    # FINAL DATAFRAME
    # ========================================================

    results = (
        pd.DataFrame(
            rows
        )
    )

    results = (
        results
        .drop_duplicates(
            subset=[
                "domain",
                "prior",
                "tau",
                "N_ref"
            ],
            keep="last"
        )
        .sort_values(
            [
                "domain",
                "prior",
                "N_ref",
                "tau"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    results = (
        run_probabilistic_fidelity()
    )

    # --------------------------------------------------------
    # Save complete detailed results
    # --------------------------------------------------------

    results.to_csv(
        FINAL_FILE,
        index=False
    )

    print(
        "\nSaved final results to:"
    )

    print(
        FINAL_FILE
    )

    print(
        f"\nTotal rows saved: "
        f"{len(results)}"
    )

    # --------------------------------------------------------
    # Summary across domains
    # --------------------------------------------------------

    summary = (
        results
        .groupby(
            [
                "prior",
                "tau",
                "N_ref"
            ]
        )
        .agg(
            support_loss_mean=(
                "support_loss",
                "mean"
            ),

            support_loss_std=(
                "support_loss",
                "std"
            ),

            tv_mean=(
                "tv_distance",
                "mean"
            ),

            tv_std=(
                "tv_distance",
                "std"
            )
        )
        .reset_index()
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    print(
        "\nSaved summary to:"
    )

    print(
        SUMMARY_FILE
    )

    print(
        "\nSummary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )