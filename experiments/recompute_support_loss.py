import os
import numpy as np
import pandas as pd

from domains import make_normalized_domains
from priors import (
    make_uniform_prior,
    make_single_hotspot_prior,
    make_multimodal_prior
)
from metrics_probability import support_loss
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

INTEGRATION_GRID = 120

L0 = np.sqrt(A0)

SINGLE_SIGMA = 0.15 * L0
MULTI_SIGMA = 0.12 * L0

PARTIAL_FILE = (
    "results/support_loss_recomputed_partial.csv"
)

FINAL_SUPPORT_FILE = (
    "results/support_loss_recomputed.csv"
)

OLD_RESULTS_FILE = (
    "results/probabilistic_fidelity.csv"
)

MERGED_FILE = (
    "results/probabilistic_fidelity_final.csv"
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

    prior_cases["uniform"] = (
        make_uniform_prior(
            omega
        )
    )

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
            n_grid=400
        )
    )

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
            n_grid=400
        )
    )

    return prior_cases


# ============================================================
# CHECKPOINT
# ============================================================

def configuration_key(
    domain,
    prior,
    tau,
    N_ref
):
    return (
        domain,
        prior,
        tau,
        int(N_ref)
    )


def load_partial_results():

    if not os.path.exists(
        PARTIAL_FILE
    ):
        return pd.DataFrame()

    df = pd.read_csv(
        PARTIAL_FILE
    )

    print(
        f"\nLoaded checkpoint with "
        f"{len(df)} rows."
    )

    return df


# ============================================================
# RECOMPUTE SUPPORT LOSS
# ============================================================

def recompute_support_loss():

    os.makedirs(
        "results",
        exist_ok=True
    )

    domains = (
        make_normalized_domains(
            target_area=A0
        )
    )

    previous = (
        load_partial_results()
    )

    if previous.empty:
        rows = []
        completed = set()
    else:
        rows = previous.to_dict(
            orient="records"
        )

        completed = {
            configuration_key(
                row["domain"],
                row["prior"],
                row["tau"],
                row["N_ref"]
            )
            for row in rows
        }

    total = (
        len(domains)
        *
        3
        *
        len(N_REF_VALUES)
        *
        len(TESSELLATIONS)
    )

    print(
        f"\nTotal configurations: {total}"
    )

    print(
        f"Already completed: "
        f"{len(completed)}"
    )

    for (
        domain_name,
        omega
    ) in domains.items():

        print(
            f"\nDomain: {domain_name}"
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
                f"  Prior: {prior_name}"
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

                    metrics = (
                        support_loss(
                            tessellation=tess,
                            p0=p0,
                            n_grid=INTEGRATION_GRID
                        )
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

                        "N_tau":
                            len(
                                tess.cells
                            ),

                        "represented_mass_new":
                            metrics[
                                "represented_mass"
                            ],

                        "support_loss_new":
                            metrics[
                                "support_loss"
                            ]
                    }

                    rows.append(
                        row
                    )

                    completed.add(
                        key
                    )

                    # Save checkpoint
                    pd.DataFrame(
                        rows
                    ).to_csv(
                        PARTIAL_FILE,
                        index=False
                    )

                    print(
                        f"      Saved "
                        f"({len(completed)}/{total})"
                    )

    df = (
        pd.DataFrame(
            rows
        )
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

    df.to_csv(
        FINAL_SUPPORT_FILE,
        index=False
    )

    return df


# ============================================================
# MERGE WITH EXISTING TV RESULTS
# ============================================================

def merge_with_existing_tv(
    support_df
):

    old_df = pd.read_csv(
        OLD_RESULTS_FILE
    )

    keys = [
        "domain",
        "prior",
        "tau",
        "N_ref"
    ]

    merged = old_df.merge(
        support_df[
            keys
            +
            [
                "represented_mass_new",
                "support_loss_new"
            ]
        ],
        on=keys,
        how="inner"
    )

    # Replace old support metrics
    merged["represented_mass"] = (
        merged[
            "represented_mass_new"
        ]
    )

    merged["support_loss"] = (
        merged[
            "support_loss_new"
        ]
    )

    merged = merged.drop(
        columns=[
            "represented_mass_new",
            "support_loss_new"
        ]
    )

    merged.to_csv(
        MERGED_FILE,
        index=False
    )

    return merged


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    support_df = (
        recompute_support_loss()
    )

    print(
        "\nSupport-loss recomputation "
        "finished."
    )

    print(
        f"Saved to: "
        f"{FINAL_SUPPORT_FILE}"
    )

    merged = (
        merge_with_existing_tv(
            support_df
        )
    )

    print(
        "\nMerged final probabilistic "
        "results saved to:"
    )

    print(
        MERGED_FILE
    )

    print(
        f"\nRows in final file: "
        f"{len(merged)}"
    )

    print(
        "\nFirst rows:"
    )

    print(
        merged.head()
    )