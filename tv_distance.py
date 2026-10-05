import numpy as np

from domains import make_normalized_domains
from tessellations import build_tessellation
from priors import make_single_hotspot_prior

from priors import integrate_density_over_geometry

from metrics_probability import (support_loss, cell_probability_masses, reconstructed_cell_densities, total_variation_distance)

A0 = 100.0
N_REF = 100
DEBUG_GRID = 100

domains = make_normalized_domains(target_area=A0)

omega = domains["Square"]

x0 = np.array([omega.centroid.x, omega.centroid.y])
L0 = np.sqrt(A0)

single_mean = (
    x0
    +
    np.array([
        0.15 * L0,
        -0.10 * L0
    ])
)

sigma = 0.15 * L0


# ============================================================
# PRIOR
# ============================================================

p0 = make_single_hotspot_prior(
    omega=omega,
    mean=single_mean,
    sigma=sigma,
    n_grid=400
)


# ============================================================
# TESSELLATION
# ============================================================

A_c = (
    omega.area
    /
    N_REF
)

tess = build_tessellation(
    omega=omega,
    tau="square",
    nominal_cell_area=A_c,
    anchor=x0,
    phase=(0.0, 0.0),
    orientation_deg=0.0
)


# ============================================================
# SUPPORT MASS
# ============================================================

support = support_loss(
    tessellation=tess,
    p0=p0,
    n_grid=400
)

print("\n========================")
print("SUPPORT")
print("========================")

print(
    "Represented mass:",
    support["represented_mass"]
)

print(
    "Support loss:",
    support["support_loss"]
)


# ============================================================
# CELL MASSES
# ============================================================

masses, M_cells, belief = (
    cell_probability_masses(
        tessellation=tess,
        p0=p0,
        n_grid=120
    )
)

print("\n========================")
print("CELL MASSES")
print("========================")

print(
    "Mass from cells:",
    M_cells
)

print(
    "Mass from global support:",
    support["represented_mass"]
)

print(
    "Difference:",
    abs(
        M_cells
        -
        support["represented_mass"]
    )
)

print(
    "Sum belief:",
    sum(
        belief.values()
    )
)


# ============================================================
# RECONSTRUCTED DENSITY
# ============================================================

p_hat = reconstructed_cell_densities(
    tessellation=tess,
    belief=belief
)

cid_max = max(
    p_hat,
    key=p_hat.get
)

cell = tess.cells[
    cid_max
]

q = cell.centroid

continuous_at_centroid = (
    p0(q)
    /
    M_cells
)

reconstructed_at_cell = (
    p_hat[cid_max]
)

print("\n========================")
print("HIGHEST-DENSITY CELL")
print("========================")

print(
    "Cell ID:",
    cid_max
)

print(
    "Centroid:",
    q
)

print(
    "Valid area:",
    cell.valid_area
)

print(
    "Cell probability:",
    belief[cid_max]
)

print(
    "Continuous density at centroid:",
    continuous_at_centroid
)

print(
    "Reconstructed density:",
    reconstructed_at_cell
)

print(
    "Absolute difference:",
    abs(
        continuous_at_centroid
        -
        reconstructed_at_cell
    )
)

# ============================================================
# SINGLE-CELL TV DEBUG
# ============================================================

p_hat_i = reconstructed_at_cell


def absolute_difference(x):
    p_conditional = (
        p0(x)
        /
        M_cells
    )

    return abs(
        p_conditional
        -
        p_hat_i
    )


single_cell_error = (
    integrate_density_over_geometry(
        geometry=cell.valid_support,
        density_fn=absolute_difference,
        n_grid=DEBUG_GRID
    )
)

print("\n========================")
print("SINGLE-CELL TV DEBUG")
print("========================")

print(
    "Integral absolute difference:",
    single_cell_error
)

print(
    "Contribution to TV:",
    0.5 * single_cell_error
)

# ============================================================
# MANUAL TV OVER ALL CELLS
# ============================================================

manual_total_error = 0.0

for cid, current_cell in tess.cells.items():

    if current_cell.valid_support.is_empty:
        continue

    current_p_hat = p_hat[cid]

    def cell_absolute_difference(
        x,
        p_hat_value=current_p_hat
    ):
        return abs(
            p0(x) / M_cells
            -
            p_hat_value
        )

    error_i = (
        integrate_density_over_geometry(
            geometry=current_cell.valid_support,
            density_fn=cell_absolute_difference,
            n_grid=DEBUG_GRID
        )
    )

    manual_total_error += error_i


manual_tv = (
    0.5
    *
    manual_total_error
)

print("\n========================")
print("MANUAL TV")
print("========================")

print(
    "Manual total absolute error:",
    manual_total_error
)

print(
    "Manual TV distance:",
    manual_tv
)


# ============================================================
# CURRENT TV IMPLEMENTATION
# ============================================================

tv = total_variation_distance(
    tessellation=tess,
    p0=p0,
    n_grid=120
)

print("\n========================")
print("CURRENT TV")
print("========================")

print(
    "TV distance:",
    tv["tv_distance"]
)



DEBUG_GRID = 120


test_cell = next(
    cell
    for cell in tess.cells.values()
    if not cell.is_boundary
)

test_area = integrate_density_over_geometry(
    geometry=test_cell.valid_support,
    density_fn=lambda x: 1.0,
    n_grid=DEBUG_GRID
)

print("\n========================")
print("INTEGRATION SANITY CHECK")
print("========================")

print(
    "True area:",
    test_cell.valid_area
)

print(
    "Numerically integrated area:",
    test_area
)

print(
    "Difference:",
    abs(
        test_cell.valid_area
        -
        test_area
    )
)