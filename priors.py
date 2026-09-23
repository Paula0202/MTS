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

def integrate_density_over_geometry(geometry, density_fn, n_grid=400):

    if geometry.is_empty:
        return 0.0
    
    minx, miny, maxx, maxy = (geometry.bounds)
    xs = np.linspace(minx, maxx, n_grid)
    ys = np.linspace(miny, maxy, n_grid)
    dx = (maxx - minx)/(n_grid - 1)
    dy = (maxy - miny)/(n_grid -1)

    total = 0.0

    for x in xs:
        for y in ys:

            point = Point(x, y)
            if geometry.contains(point):
                total += (density_fn(np.array([x, y]))*dx*dy)

    return total

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

