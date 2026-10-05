import numpy as np

from domains import make_normalized_domains

from priors import make_multimodal_prior

from tessellations import build_tessellation

from metrics_probability import (
    cell_probability_masses,
    hpd_iou,
)


# ============================================================
# CONFIGURATION
# ============================================================

A0 = 100.0

L0 = np.sqrt(A0)

DOMAIN_NAME = "Square"

TAU = "triangle"

N_REF = 100

GAMMAS = [
    0.5,
    0.8,
]

HPD_GRIDS = [
    120,
    160,
    240,
    320,
    400,
    480,
    640
]

# We keep the cell-mass integration fixed because this
# component was already validated independently.
CELL_MASS_GRID = 120


# ============================================================
# DOMAIN
# ============================================================

domains = make_normalized_domains(
    target_area=A0
)

omega = domains[
    DOMAIN_NAME
]


# ============================================================
# MULTIMODAL PRIOR
# ============================================================

center = np.array(
    [
        omega.centroid.x,
        omega.centroid.y,
    ]
)

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

p0 = make_multimodal_prior(
    omega=omega,
    means=means,
    sigmas=sigmas,
    weights=weights,
    n_grid=400,
)


# ============================================================
# TESSELLATION
# ============================================================

anchor = np.array(
    [
        omega.centroid.x,
        omega.centroid.y,
    ]
)

A_c = omega.area / N_REF

tess = build_tessellation(
    omega=omega,
    tau=TAU,
    nominal_cell_area=A_c,
    anchor=anchor,
    phase=(0.0, 0.0),
    orientation_deg=0.0,
)


# ============================================================
# DISCRETE BELIEF
# ============================================================

(
    masses,
    represented_mass,
    belief,
) = cell_probability_masses(
    tessellation=tess,
    p0=p0,
    n_grid=CELL_MASS_GRID,
)


# ============================================================
# HPD CONVERGENCE
# ============================================================

print(
    "\n========================================"
)

print(
    "HPD-IoU convergence test"
)

print(
    "========================================"
)

print(
    f"Domain: {DOMAIN_NAME}"
)

print(
    f"Tessellation: {TAU}"
)

print(
    f"N_ref: {N_REF}"
)

print(
    f"Represented mass: {represented_mass:.8f}"
)

print()


results = {}


for gamma in GAMMAS:

    print(
        f"\n--- gamma = {gamma} ---"
    )

    results[gamma] = []

    for n_grid in HPD_GRIDS:

        result = hpd_iou(
            tessellation=tess,
            p0=p0,
            belief=belief,
            represented_mass=represented_mass,
            gamma=gamma,
            n_grid=n_grid,
        )

        results[gamma].append(
            (
                n_grid,
                result["hpd_iou"],
                result["continuous_hpd_mass"],
                result["discrete_hpd_mass"],
            )
        )

        print(
            f"n_grid = {n_grid:3d}"
            f" | IoU = {result['hpd_iou']:.8f}"
            f" | continuous mass = "
            f"{result['continuous_hpd_mass']:.8f}"
            f" | discrete mass = "
            f"{result['discrete_hpd_mass']:.8f}"
        )


# ============================================================
# DIFFERENCES BETWEEN SUCCESSIVE GRIDS
# ============================================================

print(
    "\n========================================"
)

print(
    "Successive IoU differences"
)

print(
    "========================================"
)


for gamma in GAMMAS:

    print(
        f"\ngamma = {gamma}"
    )

    values = results[
        gamma
    ]

    for idx in range(
        1,
        len(values)
    ):

        previous_grid = (
            values[idx - 1][0]
        )

        current_grid = (
            values[idx][0]
        )

        previous_iou = (
            values[idx - 1][1]
        )

        current_iou = (
            values[idx][1]
        )

        difference = abs(
            current_iou
            -
            previous_iou
        )

        print(
            f"{previous_grid:3d}"
            f" -> "
            f"{current_grid:3d}"
            f" : "
            f"|ΔIoU| = "
            f"{difference:.8f}"
        )