import os
import numpy as np
import pandas as pd

from domains import(make_normalized_domains)
from priors import(make_uniform_prior, make_single_hotspot_prior, make_multimodal_prior)
from metrics_probability import(support_loss, total_variation_distance)
from tessellations import (build_tessellation)

## Configuración

A0 = 100.0

N_REF_VALUES = [100, 300, 500, 800]

TESSELLATIONS = ["square", "triangle", "hex"]

N_GRID = 400

L0 = np.sqrt(A0)

SINGLE_SIGMA = (0.15*L0)

MULTI_SIGMA = (0.12*L0)

def make_prior_cases(omega):
    center = np.array([omega.centroid.x, omega.centroid.y])
    prior_cases = {}

    prior_cases["uniform"] = make_uniform_prior(omega)

    single_mean = (center + np.array([0.15*L0, -0.10*L0]))

    prior_cases["single_hotspot"] = make_single_hotspot_prior(omega=omega, mean=single_mean, sigma=SINGLE_SIGMA, n_grid=N_GRID)

    means =  [center + np.array([-0.20*L0, -0.12*L0]), center + np.array([+0.18*L0, +0.16*L0])]

    sigmas = [MULTI_SIGMA, MULTI_SIGMA]

    weigths = [0.6, 0.4]

    prior_cases["multimodal"] = make_multimodal_prior(omega=omega, means=means, sigmas=sigmas, weights=weigths, n_grid=N_GRID)

    return prior_cases

def run_probabilistic_fidelity():

    domains = (make_normalized_domains(target_area=A0))

    rows = []

    for(domain_name, omega) in domains.items():

        print(f"\nDomain: {domain_name}")

        x0 = np.array([omega.centroid.x, omega.centroid.y])

        prior_cases = (make_prior_cases(omega))

        for (prior_name, p0) in prior_cases.items():
            print(f"  Prior: {prior_name}")

            for N_ref in (N_REF_VALUES):
                A_c = (omega.area/N_ref)

                for tau in (TESSELLATIONS):
                    tess = (build_tessellation(omega=omega, tau=tau, nominal_cell_area= A_c, anchor=x0, phase=(0.0,0.0), orientation_deg=0.0))
                    metrics = (support_loss(tessellation=tess, p0=p0, n_grid=N_GRID))
                    tv_metrics = (total_variation_distance(tessellation=tess, p0=p0, n_grid=120))

                    rows.append({"domain": domain_name, "prior": prior_name, "tau":tau, "N_ref": N_ref, "N_tau": len(tess.cells), "represented_mass": metrics["represented_mass"], "support_loss": metrics["support_loss"], "tv_distance": tv_metrics["tv_distance"]})


    return pd.DataFrame(rows)


if __name__ == "__main__":

    results = (run_probabilistic_fidelity())
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/probabilistic_fidelity.csv", index=False)
    print(results.head())
    print("\nSaved to: results/probabilistic_fidelity.csv")
    summary = (results.groupby(["prior", "tau", "N_ref"])["support_loss"].agg(["mean", "std"]).reset_index())
    print("\nSupport-loss summary:", summary.to_string(index=False))
    summary.to_csv("results/support_loss_summary.csv", index=False)
