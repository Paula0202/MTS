import os
import math
import numpy as np
import pandas as pd

from domains import make_normalized_domains
from priors import (
    make_single_hotspot_prior,
    make_multimodal_prior,
)
from tessellations import build_tessellation
from metrics_probability import (
    support_loss,
    total_variation_distance,
    cell_probability_masses,
    hpd_iou,
)


# ============================================================
# CONFIGURATION
# ============================================================

A0 = 100.0
L0 = np.sqrt(A0)

# Robustness subset: contrasting boundary geometries
DOMAIN_NAMES = [
    "Square",
    "Concave",
]

# Robustness subset: non-uniform priors only
PRIORS = [
    "single_hotspot",
    "multimodal",
]

# Robustness subset: coarse and fine extremes
N_REFS = [
    100,
    800,
]

TAUS = [
    "square",
    "triangle",
    "hex",
]

# Normalized phase coordinates in the tessellation-specific
# periodic basis. 0.5 is the maximally separated non-redundant
# phase relative to 0 along each periodic coordinate.
PHASES = [
    (0.0, 0.0),
    (0.5, 0.0),
    (0.0, 0.5),
    (0.5, 0.5),
]

ORIENTATIONS = {
    "square": [
        0.0,
        22.5,
        45.0,
        67.5,
    ],
    "triangle": [
        0.0,
        15.0,
        30.0,
        45.0,
    ],
    "hex": [
        0.0,
        15.0,
        30.0,
        45.0,
    ],
}

GAMMAS = [
    0.5,
    0.8,
]

# Same numerical settings as the validated canonical experiments.
PRIOR_GRID = 400
SUPPORT_GRID = 400
TV_GRID = 120
CELL_MASS_GRID = 120
HPD_GRID = 320

SINGLE_SIGMA = 0.15 * L0
MULTI_SIGMA = 0.12 * L0

# Optional debugging limit. Keep None for the full experiment.
MAX_CONFIGS = None


# ============================================================
# OUTPUT FILES
# ============================================================

RESULTS_DIR = "results"

CHECKPOINT_FILE = os.path.join(
    RESULTS_DIR,
    "probabilistic_robustness_partial.csv",
)

FINAL_FILE = os.path.join(
    RESULTS_DIR,
    "probabilistic_robustness.csv",
)

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "probabilistic_robustness_summary.csv",
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)


# ============================================================
# PRIOR CONSTRUCTION
# ============================================================

def build_prior(prior_name, omega):
    """
    Build exactly the same non-uniform priors used in the
    canonical probabilistic-fidelity experiments.
    """

    center = np.array([
        omega.centroid.x,
        omega.centroid.y,
    ])

    if prior_name == "single_hotspot":

        mean = (
            center
            + np.array([
                0.15 * L0,
                -0.10 * L0,
            ])
        )

        return make_single_hotspot_prior(
            omega=omega,
            mean=mean,
            sigma=SINGLE_SIGMA,
            n_grid=PRIOR_GRID,
        )

    if prior_name == "multimodal":

        means = [
            center
            + np.array([
                -0.20 * L0,
                -0.12 * L0,
            ]),
            center
            + np.array([
                +0.18 * L0,
                +0.16 * L0,
            ]),
        ]

        sigmas = [
            MULTI_SIGMA,
            MULTI_SIGMA,
        ]

        weights = [
            0.6,
            0.4,
        ]

        return make_multimodal_prior(
            omega=omega,
            means=means,
            sigmas=sigmas,
            weights=weights,
            n_grid=PRIOR_GRID,
        )

    raise ValueError(
        f"Unknown prior: {prior_name}"
    )


# ============================================================
# LATTICE PHASE
# ============================================================

def side_length_from_area(tau, nominal_cell_area):
    """
    Same equal-area side-length formulas used by tessellations.py.
    """

    if tau == "square":
        return math.sqrt(
            nominal_cell_area
        )

    if tau == "triangle":
        return math.sqrt(
            4.0
            * nominal_cell_area
            / math.sqrt(3.0)
        )

    if tau == "hex":
        return math.sqrt(
            2.0
            * nominal_cell_area
            / (
                3.0
                * math.sqrt(3.0)
            )
        )

    raise ValueError(
        f"Unknown tessellation: {tau}"
    )


def lattice_basis(tau, side):
    """
    Canonical periodicity vectors a1_tau^(0), a2_tau^(0).

    Important implementation detail:
    tessellations.transform_cell() applies phase translation first
    and rotation second. Therefore, passing the canonical phase

        phi^(0) = u a1^(0) + v a2^(0)

    and then orientation_deg=theta gives the effective world-frame
    translation

        R(theta) phi^(0),

    which is equivalent to using the rotated periodic basis defined
    in the methodology.
    """

    s = float(side)

    if tau == "square":

        a1 = np.array([
            s,
            0.0,
        ])

        a2 = np.array([
            0.0,
            s,
        ])

    elif tau == "triangle":

        h = (
            math.sqrt(3.0)
            * s
            / 2.0
        )

        a1 = np.array([
            s,
            0.0,
        ])

        a2 = np.array([
            s / 2.0,
            h,
        ])

    elif tau == "hex":

        h = (
            math.sqrt(3.0)
            * s
        )

        a1 = np.array([
            1.5 * s,
            h / 2.0,
        ])

        a2 = np.array([
            0.0,
            h,
        ])

    else:
        raise ValueError(
            f"Unknown tessellation: {tau}"
        )

    return a1, a2


def phase_from_uv(
    tau,
    nominal_cell_area,
    u,
    v,
):
    """
    Convert normalized phase coordinates (u,v) into the canonical
    physical translation passed to build_tessellation().
    """

    side = side_length_from_area(
        tau,
        nominal_cell_area,
    )

    a1, a2 = lattice_basis(
        tau,
        side,
    )

    return (
        u * a1
        + v * a2
    )


# ============================================================
# CHECKPOINT UTILITIES
# ============================================================

def configuration_key(
    domain_name,
    prior_name,
    tau,
    N_ref,
    u,
    v,
    theta_deg,
):
    """
    Stable key for one geometric robustness configuration.
    """

    return (
        str(domain_name),
        str(prior_name),
        str(tau),
        int(N_ref),
        round(float(u), 8),
        round(float(v), 8),
        round(float(theta_deg), 8),
    )


def load_previous_results():

    if not os.path.exists(
        CHECKPOINT_FILE
    ):
        return pd.DataFrame()

    df = pd.read_csv(
        CHECKPOINT_FILE
    )

    print(
        f"\nLoaded checkpoint with "
        f"{len(df)} completed configurations."
    )

    return df


def completed_configurations(df):

    if df.empty:
        return set()

    required = {
        "domain",
        "prior",
        "tau",
        "N_ref",
        "phase_u",
        "phase_v",
        "orientation_deg",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Existing robustness checkpoint has an incompatible schema. "
            f"Missing columns: {sorted(missing)}"
        )

    return {
        configuration_key(
            row["domain"],
            row["prior"],
            row["tau"],
            row["N_ref"],
            row["phase_u"],
            row["phase_v"],
            row["orientation_deg"],
        )
        for _, row in df.iterrows()
    }


# ============================================================
# EXPERIMENT
# ============================================================

def run_probabilistic_robustness():

    all_domains = make_normalized_domains(
        target_area=A0
    )

    missing_domains = [
        name
        for name in DOMAIN_NAMES
        if name not in all_domains
    ]

    if missing_domains:
        raise KeyError(
            f"Requested domains not found: {missing_domains}. "
            f"Available domains: {list(all_domains.keys())}"
        )

    domains = {
        name: all_domains[name]
        for name in DOMAIN_NAMES
    }

    previous_results = (
        load_previous_results()
    )

    completed = (
        completed_configurations(
            previous_results
        )
    )

    rows = (
        []
        if previous_results.empty
        else previous_results.to_dict(
            orient="records"
        )
    )

    total_configurations = (
        len(domains)
        * len(PRIORS)
        * len(N_REFS)
        * len(TAUS)
        * len(PHASES)
        * 4
    )

    print(
        "\n========================================"
    )
    print(
        "PROBABILISTIC ROBUSTNESS EXPERIMENT"
    )
    print(
        "========================================"
    )
    print(
        f"Total geometric configurations: "
        f"{total_configurations}"
    )
    print(
        f"Already completed: "
        f"{len(completed)}"
    )
    print(
        "========================================\n"
    )

    configuration_counter = 0

    for domain_name, omega in domains.items():

        print(
            "\n########################################"
        )
        print(
            f"DOMAIN: {domain_name}"
        )
        print(
            "########################################"
        )

        anchor = np.array([
            omega.centroid.x,
            omega.centroid.y,
        ])

        for prior_name in PRIORS:

            print(
                f"\nPrior: {prior_name}"
            )

            p0 = build_prior(
                prior_name,
                omega,
            )

            for N_ref in N_REFS:

                nominal_cell_area = (
                    omega.area
                    / N_ref
                )

                for tau in TAUS:

                    for u, v in PHASES:

                        # Canonical-basis phase. The tessellation
                        # constructor rotates this vector together
                        # with the lattice because translation is
                        # applied before rotation.
                        phase = phase_from_uv(
                            tau=tau,
                            nominal_cell_area=nominal_cell_area,
                            u=u,
                            v=v,
                        )

                        for theta_deg in ORIENTATIONS[tau]:

                            configuration_counter += 1

                            if (
                                MAX_CONFIGS is not None
                                and configuration_counter > MAX_CONFIGS
                            ):
                                print(
                                    "\nMAX_CONFIGS reached. "
                                    "Stopping test run."
                                )
                                return pd.DataFrame(rows)

                            key = configuration_key(
                                domain_name,
                                prior_name,
                                tau,
                                N_ref,
                                u,
                                v,
                                theta_deg,
                            )

                            if key in completed:

                                print(
                                    f"[SKIP] {domain_name} | "
                                    f"{prior_name} | {tau} | "
                                    f"N={N_ref} | "
                                    f"phase=({u:.1f},{v:.1f}) | "
                                    f"theta={theta_deg:g}"
                                )

                                continue

                            print(
                                f"[{configuration_counter}/"
                                f"{total_configurations}] "
                                f"{domain_name} | "
                                f"{prior_name} | {tau} | "
                                f"N={N_ref} | "
                                f"phase=({u:.1f},{v:.1f}) | "
                                f"theta={theta_deg:g}"
                            )

                            # =================================
                            # TESSELLATION
                            # =================================

                            tess = build_tessellation(
                                omega=omega,
                                tau=tau,
                                nominal_cell_area=nominal_cell_area,
                                anchor=anchor,
                                phase=phase,
                                orientation_deg=theta_deg,
                            )

                            # =================================
                            # SUPPORT LOSS
                            # =================================

                            support_metrics = support_loss(
                                tessellation=tess,
                                p0=p0,
                                n_grid=SUPPORT_GRID,
                            )

                            # =================================
                            # TOTAL VARIATION
                            # =================================

                            tv_metrics = total_variation_distance(
                                tessellation=tess,
                                p0=p0,
                                n_grid=TV_GRID,
                            )

                            # =================================
                            # CELL MASSES FOR HPD
                            # =================================

                            (
                                masses,
                                represented_mass_hpd,
                                belief,
                            ) = cell_probability_masses(
                                tessellation=tess,
                                p0=p0,
                                n_grid=CELL_MASS_GRID,
                            )

                            # =================================
                            # BASE ROW
                            # =================================

                            row = {
                                "domain": domain_name,
                                "prior": prior_name,
                                "tau": tau,
                                "N_ref": int(N_ref),
                                "phase_u": float(u),
                                "phase_v": float(v),
                                "orientation_deg": float(theta_deg),
                                "phase_x_canonical": float(phase[0]),
                                "phase_y_canonical": float(phase[1]),
                                "N_tau": int(len(tess.cells)),
                                "represented_mass": float(
                                    support_metrics[
                                        "represented_mass"
                                    ]
                                ),
                                "support_loss": float(
                                    support_metrics[
                                        "support_loss"
                                    ]
                                ),
                                "tv_distance": float(
                                    tv_metrics[
                                        "tv_distance"
                                    ]
                                ),
                            }

                            # =================================
                            # HPD-IoU
                            # =================================

                            for gamma in GAMMAS:

                                result = hpd_iou(
                                    tessellation=tess,
                                    p0=p0,
                                    belief=belief,
                                    represented_mass=represented_mass_hpd,
                                    gamma=gamma,
                                    n_grid=HPD_GRID,
                                )

                                suffix = str(gamma).replace(
                                    ".",
                                    "_",
                                )

                                row[
                                    f"hpd_iou_{suffix}"
                                ] = float(
                                    result[
                                        "hpd_iou"
                                    ]
                                )

                                row[
                                    f"hpd_continuous_mass_{suffix}"
                                ] = float(
                                    result[
                                        "continuous_hpd_mass"
                                    ]
                                )

                                row[
                                    f"hpd_discrete_mass_{suffix}"
                                ] = float(
                                    result[
                                        "discrete_hpd_mass"
                                    ]
                                )

                                row[
                                    f"hpd_n_cells_{suffix}"
                                ] = int(
                                    result[
                                        "n_selected_cells"
                                    ]
                                )

                            # =================================
                            # CHECK CONSISTENCY
                            # =================================

                            row[
                                "represented_mass_hpd_grid"
                            ] = float(
                                represented_mass_hpd
                            )

                            rows.append(row)
                            completed.add(key)

                            # Save after every configuration.
                            pd.DataFrame(rows).to_csv(
                                CHECKPOINT_FILE,
                                index=False,
                            )

                            print(
                                f"    support_loss="
                                f"{row['support_loss']:.6f} | "
                                f"DTV={row['tv_distance']:.6f} | "
                                f"HPD0.5={row['hpd_iou_0_5']:.6f} | "
                                f"HPD0.8={row['hpd_iou_0_8']:.6f}"
                            )

    results = pd.DataFrame(rows)

    results = (
        results
        .drop_duplicates(
            subset=[
                "domain",
                "prior",
                "tau",
                "N_ref",
                "phase_u",
                "phase_v",
                "orientation_deg",
            ],
            keep="last",
        )
        .sort_values(
            [
                "domain",
                "prior",
                "N_ref",
                "tau",
                "phase_u",
                "phase_v",
                "orientation_deg",
            ]
        )
        .reset_index(drop=True)
    )

    return results


# ============================================================
# SUMMARY
# ============================================================

def build_summary(results):
    """
    Summary over the 16 phase x orientation configurations for each
    domain-prior-resolution-tessellation condition.
    """

    metrics = [
        "support_loss",
        "tv_distance",
        "hpd_iou_0_5",
        "hpd_iou_0_8",
    ]

    aggregations = {}

    for metric in metrics:
        aggregations[
            f"{metric}_mean"
        ] = (
            metric,
            "mean",
        )
        aggregations[
            f"{metric}_std"
        ] = (
            metric,
            "std",
        )
        aggregations[
            f"{metric}_min"
        ] = (
            metric,
            "min",
        )
        aggregations[
            f"{metric}_max"
        ] = (
            metric,
            "max",
        )

    summary = (
        results
        .groupby(
            [
                "domain",
                "prior",
                "N_ref",
                "tau",
            ]
        )
        .agg(**aggregations)
        .reset_index()
    )

    for metric in metrics:
        summary[
            f"{metric}_range"
        ] = (
            summary[f"{metric}_max"]
            - summary[f"{metric}_min"]
        )

    return summary


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    results = run_probabilistic_robustness()

    if results.empty:
        raise RuntimeError(
            "No robustness results were generated."
        )

    results.to_csv(
        FINAL_FILE,
        index=False,
    )

    summary = build_summary(
        results
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print(
        "\n========================================"
    )
    print(
        "ROBUSTNESS EXPERIMENT COMPLETE"
    )
    print(
        "========================================"
    )
    print(
        f"Detailed rows: {len(results)}"
    )
    print(
        f"Expected rows: 384"
    )
    print(
        f"Detailed file: {FINAL_FILE}"
    )
    print(
        f"Summary file: {SUMMARY_FILE}"
    )
    print(
        "========================================"
    )
