import math
from shapely.geometry import Polygon
from shapely.affinity import scale, translate


def make_circle_polygon(cx, cy, r, n_points=200):
    pts = [(cx + r*math.cos(2*math.pi* k /n_points ), 
           cy + r*math.sin(2*math.pi* k /n_points)) for k in range(n_points)]
    return Polygon(pts)

def make_equilateral_triangle(x0=0, y0=0, L=10):
    h = (math.sqrt(3)*L/2.0)
    return Polygon([(x0, y0), (x0 + L, y0), (x0 + L/2.0, y0 + h)])

def make_square(x0=0, y0=0, L=10):
    return Polygon([(x0, y0), (x0 + L, y0), (x0 + L, y0 + L), (x0, y0 + L)])

def make_convex_polygon():
    return Polygon([(0,0), (12,1), (11,8), (3,10), (-1,6)])

def make_concave_polygon():
    return Polygon([(0,0), (10,0), (10,5), (6,5), (6,9), (4,9), (4,5), (0,5)])

def make_domains():
    return {"Square": make_square(0,0,10), "Circle": make_circle_polygon(5,5,5,200), "Triangle":make_equilateral_triangle(0,0,10), "Convex":make_convex_polygon(), "Concave": make_concave_polygon()}

def normalize_domain(poly, target_area, target_centroid=(0.0,0.0)):
    factor = math.sqrt(target_area/poly.area)
    poly_scaled = scale(poly, xfact=factor, yfact=factor, origin=poly.centroid)

    cx = poly_scaled.centroid.x
    cy = poly_scaled.centroid.y

    return translate(poly_scaled, xoff=target_centroid[0] - cx, yoff=target_centroid[1] - cy)

def make_normalized_domains(target_area=100.0):
    domains=make_domains()
    return {name:normalize_domain(poly,target_area=target_area, target_centroid=(0.0,0.0)) for name, poly in domains.items()}
            

