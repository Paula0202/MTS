import math
import numpy as np
from shapely.geometry import Point

def gaussian_2d(x, mean, sigma):
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    diff = (x - mean)
    coeff = (1.0/(2.0*math.pi*sigma**2))
    exponent = (-0.5*np.dot(diff, diff)/sigma**2)

    return(coeff*math.exp(exponent))

def gaussian_mixture_2d(x, means, sigmas, weights):
    weights = np.asarray(weights, dtype=float)
    weights = (weights/weights.sum())
    value = 0.0

    for(weight, mean, sigma) in zip(weights, means, sigmas):
        value += (weight*gaussian_2d(x=x, mean=mean, sigma=sigma))

    return value

def integrate_density_over_geometry(
    geometry,
    density_fn,
    n_grid=100
):
    """
    Numerical midpoint integration of a continuous function
    over a Shapely geometry.

    The bounding box is divided into n_grid x n_grid
    rectangular integration elements. The density is evaluated
    at the midpoint of each element.

    Midpoint sampling avoids systematic boundary loss caused by
    evaluating points exactly on polygon boundaries.
    """

    if geometry.is_empty:
        return 0.0

    minx, miny, maxx, maxy = geometry.bounds

    width = maxx - minx
    height = maxy - miny

    if width <= 0.0 or height <= 0.0:
        return 0.0

    dx = width / n_grid
    dy = height / n_grid

    xs = (
        minx
        +
        (np.arange(n_grid) + 0.5) * dx
    )

    ys = (
        miny
        +
        (np.arange(n_grid) + 0.5) * dy
    )

    total = 0.0

    for x in xs:
        for y in ys:

            point = Point(
                float(x),
                float(y)
            )

            if geometry.covers(point):

                total += (
                    density_fn(
                        np.array([
                            x,
                            y
                        ])
                    )
                    *
                    dx
                    *
                    dy
                )

    return float(total)

def normalize_prior_over_domain(omega, raw_density_fn, n_grid=400):

    Z = (integrate_density_over_geometry(omega, raw_density_fn, n_grid=n_grid))
    if Z <= 0:
        raise ValueError("Prior normalization constant must be positive")
    
    def p0(x):

        point = Point(x[0], x[1])

        if not omega.contains(point):
            return 0.0

        return (raw_density_fn(x)/Z)

    return p0, Z

def make_uniform_prior(omega):
    density = (1.0/omega.area)

    def p0(x):

        point = Point(x[0], x[1])

        if omega.contains(point):
            return density

        return 0.0

    return p0

def make_single_hotspot_prior(omega, mean, sigma, n_grid=400):
    raw = lambda x: gaussian_2d(x=x, mean=mean, sigma=sigma)

    return normalize_prior_over_domain(omega, raw, n_grid=n_grid)[0]

def make_multimodal_prior(omega, means, sigmas, weights, n_grid=400):
    raw = (lambda x: gaussian_mixture_2d(x=x, means=means, sigmas=sigmas, weights=weights))

    return normalize_prior_over_domain(omega, raw, n_grid=n_grid)[0]

