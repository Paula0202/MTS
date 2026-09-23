import os
import numpy as np
import matplotlib.pyplot as plt

from domains import make_normalized_domains
from experiments.run_probabilistic_fidelity import make_prior_cases

A0 = 100.0
N_GRID_PLOT = 250

def evaluate_prior_grid(omega, p0, n_grid=N_GRID_PLOT):

    minx, miny, maxx, maxy = omega.bounds

    xs = np.linspace(minx, maxx, n_grid)
    xy = np.linspace(miny, maxy, n_grid)

    X, Y = np.meshgrid(xs, xy)

    Z = np.full_like(X, np.nan, dtype=float)

    for i in range(n_grid):
        for j in range(n_grid):

            point = np.array([X[i, j], Y[i, j]])

            value = p0(point)

            if value > 0:
                Z[i, j] = value 

    return X, Y, Z


def plot_priors():

    domains = make_normalized_domains(target_area=A0)

    for domain_name, omega in domains.items():

        prior_cases = make_prior_cases(omega)

        fig, axes = plt.subplots(1, len(prior_cases), figsize=(12, 4))

        for ax, (prior_name, p0) in zip(axes, prior_cases.items()):

            X, Y, Z = evaluate_prior_grid(omega, p0)

            contour = ax.contourf(X, Y, Z, levels=30)

            x, y = omega.exterior.xy

            ax.plot(x, y, linewidth=2)

            ax.set_title(prior_name.replace("_", " ").title())

            ax.set_aspect("equal")

            fig.colorbar(contour, ax=ax)

        fig.suptitle(f"Continues priors -- {domain_name}")
        fig.tight_layout()
        os.makedirs("figures/output", exist_ok=True)
        filename = (domain_name.lower().replace(" ", "_"))
        fig.savefig(f"figures/output/priors_{filename}.pdf", bbox_inches="tight")
        fig.savefig(f"figures/output/priors_{filename}.png", dpi=300, bbox_inches="tight")

        plt.show()

if __name__ == "__main__":
    plot_priors()

