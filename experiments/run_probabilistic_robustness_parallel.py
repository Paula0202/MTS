import os

# Prevent each worker process from spawning additional BLAS/OpenMP threads.
# These environment variables must be set before importing NumPy.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

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

# Number of independent configurations evaluated concurrently.
# Start with 8 on the i9-14900KF workstation and increase only after
# checking CPU utilisation, memory use, and temperatures.
MAX_WORKERS = 8





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
# PARALLEL WORKER STATE
# ============================================================

# These caches live independently inside each worker process.
# They avoid rebuilding normalized domains and priors for every
# single configuration handled by the same worker.
_WORKER_DOMAINS = None
_WORKER_PRIORS = {}


def _get_worker_domain(domain_name):
    """
    Return one normalized domain from a per-worker cache.
    """

    global _WORKER_DOMAINS

    if _WORKER_DOMAINS is None:
        _WORKER_DOMAINS = make_normalized_domains(
            target_area=A0
        )

    if domain_name not in _WORKER_DOMAINS:
        raise KeyError(
            f"Requested domain '{domain_name}' not found. "
            f"Available domains: {list(_WORKER_DOMAINS.keys())}"
        )

    return _WORKER_DOMAINS[domain_name]


def _get_worker_prior(domain_name, prior_name, omega):
    """
    Build each domain-prior pair at most once per worker process.
    """

    key = (
        str(domain_name),
        str(prior_name),
    )

    if key not in _WORKER_PRIORS:
        _WORKER_PRIORS[key] = build_prior(
            prior_name,
            omega,
        )

    return _WORKER_PRIORS[key]


# ============================================================
# ONE ROBUSTNESS CONFIGURATION
# ============================================================

def evaluate_configuration(config):
    """
    Evaluate one independent
    domain-prior-resolution-tessellation-phase-orientation
    configuration.

    This function is intentionally defined at module level so that it can
    be pickled by ProcessPoolExecutor on Windows.
    """

    domain_name = config["domain_name"]
    prior_name = config["prior_name"]
    N_ref = int(config["N_ref"])
    tau = config["tau"]
    u = float(config["u"])
    v = float(config["v"])
    theta_deg = float(config["theta_deg"])

    omega = _get_worker_domain(
        domain_name
    )

    p0 = _get_worker_prior(
        domain_name,
        prior_name,
        omega,
    )

    anchor = np.array([
        omega.centroid.x,
        omega.centroid.y,
    ])

    nominal_cell_area = (
        omega.area
        / N_ref
    )

    # Canonical-basis phase. build_tessellation() applies the
    # translation before the lattice rotation, so the phase is
    # rotated together with the lattice and therefore matches the
    # rotated-basis formulation in the methodology.
    phase = phase_from_uv(
        tau=tau,
        nominal_cell_area=nominal_cell_area,
        u=u,
        v=v,
    )

    # ========================================================
    # TESSELLATION
    # ========================================================

    tess = build_tessellation(
        omega=omega,
        tau=tau,
        nominal_cell_area=nominal_cell_area,
        anchor=anchor,
        phase=phase,
        orientation_deg=theta_deg,
    )

    # ========================================================
    # SUPPORT LOSS
    # ========================================================

    support_metrics = support_loss(
        tessellation=tess,
        p0=p0,
        n_grid=SUPPORT_GRID,
    )

    # ========================================================
    # TOTAL VARIATION
    # ========================================================

    tv_metrics = total_variation_distance(
        tessellation=tess,
        p0=p0,
        n_grid=TV_GRID,
    )

    # ========================================================
    # CELL MASSES FOR HPD
    # ========================================================

    (
        _masses,
        represented_mass_hpd,
        belief,
    ) = cell_probability_masses(
        tessellation=tess,
        p0=p0,
        n_grid=CELL_MASS_GRID,
    )

    # ========================================================
    # BASE ROW
    # ========================================================

    row = {
        "domain": domain_name,
        "prior": prior_name,
        "tau": tau,
        "N_ref": N_ref,
        "phase_u": u,
        "phase_v": v,
        "orientation_deg": theta_deg,
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

    # ========================================================
    # HPD-IoU
    # ========================================================

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

    row[
        "represented_mass_hpd_grid"
    ] = float(
        represented_mass_hpd
    )

    return row


# ============================================================
# CONFIGURATION GENERATION
# ============================================================

def build_pending_configurations(completed):
    """
    Build the deterministic list of configurations that are not already
    present in the checkpoint.
    """

    configs = []

    for domain_name in DOMAIN_NAMES:

        for prior_name in PRIORS:

            for N_ref in N_REFS:

                for tau in TAUS:

                    for u, v in PHASES:

                        for theta_deg in ORIENTATIONS[tau]:

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
                                continue

                            configs.append({
                                "domain_name": domain_name,
                                "prior_name": prior_name,
                                "N_ref": int(N_ref),
                                "tau": tau,
                                "u": float(u),
                                "v": float(v),
                                "theta_deg": float(theta_deg),
                            })

    if MAX_CONFIGS is not None:
        configs = configs[:MAX_CONFIGS]

    return configs


def _sorted_results_dataframe(rows):
    """
    De-duplicate and sort detailed rows in a deterministic order.
    """

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
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


def save_checkpoint(rows):
    """
    Save all completed rows. Only the parent process calls this function,
    so multiple workers never write to the same CSV concurrently.
    """

    results = _sorted_results_dataframe(
        rows
    )

    results.to_csv(
        CHECKPOINT_FILE,
        index=False,
    )

    return results


# ============================================================
# EXPERIMENT
# ============================================================

def run_probabilistic_robustness():

    # Validate requested domains in the parent process before
    # launching worker processes.
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

    total_configurations = sum(
        len(PHASES)
        * len(ORIENTATIONS[tau])
        for _domain_name in DOMAIN_NAMES
        for _prior_name in PRIORS
        for _N_ref in N_REFS
        for tau in TAUS
    )

    pending_configs = (
        build_pending_configurations(
            completed
        )
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
        f"Pending in this run: "
        f"{len(pending_configs)}"
    )
    print(
        f"Parallel workers: "
        f"{MAX_WORKERS}"
    )
    print(
        "========================================\n"
    )

    if not pending_configs:
        print(
            "No pending configurations. "
            "Checkpoint already contains the requested experiment."
        )

        return (
            _sorted_results_dataframe(rows),
            len(completed) >= total_configurations,
        )

    executor = ProcessPoolExecutor(
        max_workers=MAX_WORKERS
    )

    future_to_config = {}

    try:

        for config in pending_configs:

            future = executor.submit(
                evaluate_configuration,
                config,
            )

            future_to_config[
                future
            ] = config

        completed_this_run = 0

        for future in as_completed(
            future_to_config
        ):

            config = future_to_config[
                future
            ]

            try:

                row = future.result()

            except Exception as exc:

                print(
                    "\n[ERROR] "
                    f"{config['domain_name']} | "
                    f"{config['prior_name']} | "
                    f"{config['tau']} | "
                    f"N={config['N_ref']} | "
                    f"phase=({config['u']:.1f},"
                    f"{config['v']:.1f}) | "
                    f"theta={config['theta_deg']:g}"
                )

                print(
                    f"        {type(exc).__name__}: {exc}"
                )

                continue

            rows.append(row)

            key = configuration_key(
                row["domain"],
                row["prior"],
                row["tau"],
                row["N_ref"],
                row["phase_u"],
                row["phase_v"],
                row["orientation_deg"],
            )

            completed.add(key)

            completed_this_run += 1

            # IMPORTANT:
            # Only this parent process writes the checkpoint.
            # Worker processes never write to disk.
            save_checkpoint(
                rows
            )

            print(
                f"[DONE {len(completed)}/"
                f"{total_configurations}] "
                f"{row['domain']} | "
                f"{row['prior']} | "
                f"{row['tau']} | "
                f"N={row['N_ref']} | "
                f"phase=({row['phase_u']:.1f},"
                f"{row['phase_v']:.1f}) | "
                f"theta={row['orientation_deg']:g}"
            )

            print(
                f"    support_loss="
                f"{row['support_loss']:.6f} | "
                f"DTV={row['tv_distance']:.6f} | "
                f"HPD0.5={row['hpd_iou_0_5']:.6f} | "
                f"HPD0.8={row['hpd_iou_0_8']:.6f}"
            )

    except KeyboardInterrupt:

        print(
            "\n\nKeyboard interrupt received."
        )
        print(
            "Cancelling configurations that have not started..."
        )

        # Everything returned before the interrupt has already
        # been written to the checkpoint.
        for future in future_to_config:
            future.cancel()

        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )

        results = save_checkpoint(
            rows
        )

        print(
            f"Checkpoint safely contains "
            f"{len(results)} completed configurations:"
        )
        print(
            f"    {CHECKPOINT_FILE}"
        )

        return (
            results,
            False,
        )

    else:

        executor.shutdown(
            wait=True,
            cancel_futures=False,
        )

    results = save_checkpoint(
        rows
    )

    experiment_complete = (
        len(
            completed_configurations(
                results
            )
        )
        >= total_configurations
    )

    return (
        results,
        experiment_complete,
    )


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

    # Required for safe multiprocessing on Windows.
    mp.freeze_support()

    results, experiment_complete = (
        run_probabilistic_robustness()
    )

    if results.empty:
        raise RuntimeError(
            "No robustness results were generated."
        )

    # The checkpoint is always current. Only write the definitive
    # final and summary files once the full 384-configuration
    # experiment has actually completed.
    if experiment_complete:

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

    else:

        print(
            "\n========================================"
        )
        print(
            "ROBUSTNESS EXPERIMENT PAUSED / PARTIAL"
        )
        print(
            "========================================"
        )
        print(
            f"Completed rows currently stored: "
            f"{len(results)}"
        )
        print(
            f"Checkpoint file: {CHECKPOINT_FILE}"
        )
        print(
            "Run the same script again to resume."
        )
        print(
            "========================================"
        )
