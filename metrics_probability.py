from shapely.geometry import Polygon
from shapely import union_all 
import numpy as np

from priors import (integrate_density_over_geometry)

def represented_valid_support(tessellation):

    supports = [cell.valid_support for cell in tessellation.cells.values() if not cell.valid_support.is_empty]

    if len(supports) == 0:
        return Polygon()

    return union_all(supports)

def represented_probability_mass(tessellation, p0, n_grid=400):

    C_tau = (represented_valid_support(tessellation))

    return(integrate_density_over_geometry(C_tau, p0, n_grid=n_grid))

def support_loss(tessellation, p0, n_grid=400):

    M_tau = (represented_probability_mass(tessellation, p0, n_grid=n_grid))
    M_tau = float(np.clip(M_tau, 0.0, 1.0))

    epsilon_mass = (1.0 - M_tau)

    return {"represented_mass": M_tau, "support_loss": epsilon_mass}


def cell_probability_masses(tessellation, p0, n_grid=150):

    masses = {}

    for cid, cell in tessellation.cells.items():

        if cell.valid_support.is_empty:
            masses[cid] = 0.0
            continue

        mass_i = integrate_density_over_geometry(geometry=cell.valid_support, density_fn=p0, n_grid=n_grid)

        masses[cid] = mass_i

    represented_mass = sum(masses.values())

    if represented_mass <= 0.0:
        raise ValueError("Represented probability mass is zero.")

    belief = {cid: mass_i/represented_mass for cid, mass_i in masses.items()}

    return (masses, represented_mass, belief)

def reconstructed_cell_densities(tessellation, belief):

    densities = {}

    for cid, cell in tessellation.cells.items():

        if cell.valid_area <= 0.0:
            densities[cid] = 0.0

        else:
            densities[cid] = (belief[cid]/cell.valid_area)
    return densities 


def total_variation_distance(tessellation, p0, n_grid=120):
    (masses, represented_mass, belief) = cell_probability_masses(tessellation=tessellation, p0=p0, n_grid=n_grid)

    reconstructed_density = (reconstructed_cell_densities(tessellation=tessellation, belief=belief))

    total_error = 0.0

    for cid, cell in tessellation.cells.items():

        if cell.valid_support.is_empty:
            continue

        p_hat_i = (reconstructed_density[cid])

        def absolute_difference(x):

            p_conditional = (p0(x)/represented_mass)

            return abs(p_conditional - p_hat_i)

        cell_error = (integrate_density_over_geometry(geometry=cell.valid_support, density_fn=absolute_difference, n_grid=n_grid))
        total_error += (cell_error)

        return{"tv_distance": 0.5*total_error, "represented_mass": represented_mass, "belief": belief}