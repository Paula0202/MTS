import numpy as np
import pandas as pd

from domains import make_normalized_domains
from priors import (
    make_single_hotspot_prior,
    make_multimodal_prior
)
from metrics_probability import (
    support_loss,
    total_variation_distance
)
from tessellations import build_tessellation


A0 = 100.0
L0 = np.sqrt(A0)

GRIDS = [60, 80, 120, 160, 240]

TEST_CASES = [
    ("Square", "single_hotspot", "square", 100),
    ("Square", "multimodal", "hex", 800),
    ("Triangle", "single_hotspot", "triangle", 300),
    ("Concave", "multimodal", "square", 100),
    ("Concave", "multimodal", "hex", 800),
]


def make_prior(
    omega,
    prior_name,
    normalization_grid=400
):
    center = np.array([
        omega.centroid.x,
        omega.centroid.y
    ])

    if prior_name == "single_hotspot":

        mean = (
            center
            +
            np.array([
                0.15 * L0,
                -0.10 * L0
            ])
        )

        return make_single_hotspot_prior(
            omega=omega,
            mean=mean,
            sigma=0.15 * L0,
            n_grid=normalization_grid
        )

    if prior_name == "multimodal":

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
            0.12 * L0,
            0.12 * L0
        ]

        weights = [
            0.6,
            0.4
        ]

        return make_multimodal_prior(
            omega=omega,
            means=means,
            sigmas=sigmas,
            weights=weights,
            n_grid=normalization_grid
        )

    raise ValueError(
        f"Unknown prior: {prior_name}"
    )


def run_convergence_study():

    domains = make_normalized_domains(
        target_area=A0
    )

    rows = []

    for (
        domain_name,
        prior_name,
        tau,
        N_ref
    ) in TEST_CASES:

        print(
            f"\nCase: {domain_name} / "
            f"{prior_name} / {tau} / "
            f"N_ref={N_ref}"
        )

        omega = domains[
            domain_name
        ]

        x0 = np.array([
            omega.centroid.x,
            omega.centroid.y
        ])

        p0 = make_prior(
            omega,
            prior_name
        )

        A_c = (
            omega.area
            /
            N_ref
        )

        tess = build_tessellation(
            omega=omega,
            tau=tau,
            nominal_cell_area=A_c,
            anchor=x0,
            phase=(0.0, 0.0),
            orientation_deg=0.0
        )




        for grid in GRIDS:

            print(
                f"  grid={grid}"
            )

            support = support_loss(
                tessellation=tess,
                p0=p0,
                n_grid=grid
            )

            tv = total_variation_distance(
                tessellation=tess,
                p0=p0,
                n_grid=grid
            )

            rows.append({
                "domain":
                    domain_name,

                "prior":
                    prior_name,

                "tau":
                    tau,

                "N_ref":
                    N_ref,

                "grid":
                    grid,

                "represented_mass":
                    support[
                        "represented_mass"
                    ],

                "support_loss":
                    support[
                        "support_loss"
                    ],

                "tv_distance":
                    tv[
                        "tv_distance"
                    ]
            })

    df = pd.DataFrame(
        rows
    )

    return df


if __name__ == "__main__":

    results = (
        run_convergence_study()
    )

    results.to_csv(
        "results/tv_convergence.csv",
        index=False
    )

    print(
        "\nSaved to "
        "results/tv_convergence.csv"
    )

    print(
        results.to_string(
            index=False
        )
    )
    results["tv_rel_change"] = (
    results
    .groupby(
        [
            "domain",
            "prior",
            "tau",
            "N_ref"
        ]
    )["tv_distance"]
    .pct_change()
    .abs()
)

results["support_abs_change"] = (
    results
    .groupby(
        [
            "domain",
            "prior",
            "tau",
            "N_ref"
        ]
    )["support_loss"]
    .diff()
    .abs()
)
results.to_csv(
    "results/tv_convergence.csv",
    index=False
)