import numpy as np

from domains import make_normalized_domains

from priors import (
    make_multimodal_prior
)

from tessellations import (
    build_tessellation
)

from metrics_probability import (
    cell_probability_masses,
    hpd_iou
)


A0 = 100.0

L0 = np.sqrt(
    A0
)

N_REF = 100

TAUS = [
    "square",
    "triangle",
    "hex"
]

GAMMAS = [
    0.5,
    0.8
]


domains = (
    make_normalized_domains(
        target_area=A0
    )
)

omega = (
    domains[
        "Square"
    ]
)

center = np.array([
    omega.centroid.x,
    omega.centroid.y
])


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


p0 = (
    make_multimodal_prior(
        omega=omega,
        means=means,
        sigmas=sigmas,
        weights=weights,
        n_grid=400
    )
)


x0 = np.array([
    omega.centroid.x,
    omega.centroid.y
])


A_c = (
    omega.area
    /
    N_REF
)


for tau in TAUS:

    print(
        "\n========================"
    )

    print(
        "Tessellation:",
        tau
    )

    print(
        "========================"
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

    (
        masses,
        represented_mass,
        belief
    ) = (
        cell_probability_masses(
            tessellation=tess,
            p0=p0,
            n_grid=120
        )
    )

    for gamma in GAMMAS:

        result = (
            hpd_iou(
                tessellation=tess,
                p0=p0,
                belief=belief,
                represented_mass=represented_mass,
                gamma=gamma,
                n_grid=240
            )
        )

        print(
            f"\ngamma = {gamma}"
        )

        print(
            "HPD IoU:",
            result[
                "hpd_iou"
            ]
        )

        print(
            "Continuous HPD mass:",
            result[
                "continuous_hpd_mass"
            ]
        )

        print(
            "Discrete HPD mass:",
            result[
                "discrete_hpd_mass"
            ]
        )

        print(
            "Selected cells:",
            result[
                "n_selected_cells"
            ]
        )