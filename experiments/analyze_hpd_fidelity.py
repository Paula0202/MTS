import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = os.path.join(
    "results",
    "hpd_fidelity.csv",
)

OUTPUT_DIR = os.path.join(
    "results",
    "hpd_analysis",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# DISPLAY ORDER
# ============================================================

DOMAIN_ORDER = [
    "Square",
    "Circle",
    "Triangle",
    "Convex",
    "Concave",
]

PRIOR_ORDER = [
    "single_hotspot",
    "multimodal",
]

TAU_ORDER = [
    "square",
    "triangle",
    "hex",
]

TAU_LABELS = {
    "square": "Square",
    "triangle": "Triangular",
    "hex": "Hexagonal",
}

PRIOR_LABELS = {
    "single_hotspot": "Single hotspot",
    "multimodal": "Multimodal",
}

N_REF_ORDER = [
    100,
    300,
    500,
    800,
]

GAMMAS = [
    0.5,
    0.8,
]


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)

print(
    "\n========================================"
)
print(
    "HPD-IoU ANALYSIS"
)
print(
    "========================================"
)

print(
    f"Rows loaded: {len(df)}"
)


# ============================================================
# BASIC VALIDATION
# ============================================================

expected_rows = (
    len(DOMAIN_ORDER)
    * len(PRIOR_ORDER)
    * len(TAU_ORDER)
    * len(N_REF_ORDER)
    * len(GAMMAS)
)

print(
    f"Expected rows: {expected_rows}"
)

if len(df) != expected_rows:
    print(
        "WARNING: unexpected number of rows."
    )


key_columns = [
    "domain",
    "prior",
    "tau",
    "N_ref",
    "gamma",
]

duplicates = (
    df
    .duplicated(
        subset=key_columns
    )
    .sum()
)

print(
    f"Duplicate configurations: {duplicates}"
)


# ============================================================
# NUMERICAL SANITY CHECKS
# ============================================================

df["continuous_mass_error"] = (
    df["continuous_hpd_mass"]
    -
    df["gamma"]
).abs()

df["discrete_mass_overshoot"] = (
    df["discrete_hpd_mass"]
    -
    df["gamma"]
)

print(
    "\nNumerical checks"
)
print(
    "----------------"
)

print(
    "Maximum continuous HPD mass error:",
    f"{df['continuous_mass_error'].max():.8f}",
)

print(
    "Mean continuous HPD mass error:",
    f"{df['continuous_mass_error'].mean():.8f}",
)

print(
    "Maximum discrete HPD overshoot:",
    f"{df['discrete_mass_overshoot'].max():.8f}",
)

print(
    "Mean discrete HPD overshoot:",
    f"{df['discrete_mass_overshoot'].mean():.8f}",
)


# ============================================================
# GLOBAL SUMMARY:
# prior × gamma × resolution × tessellation
#
# Average over the five domains.
# ============================================================

summary = (
    df
    .groupby(
        [
            "prior",
            "gamma",
            "N_ref",
            "tau",
        ]
    )
    .agg(
        mean_hpd_iou=(
            "hpd_iou",
            "mean",
        ),
        std_hpd_iou=(
            "hpd_iou",
            "std",
        ),
        min_hpd_iou=(
            "hpd_iou",
            "min",
        ),
        max_hpd_iou=(
            "hpd_iou",
            "max",
        ),
        mean_discrete_mass=(
            "discrete_hpd_mass",
            "mean",
        ),
        mean_selected_cells=(
            "n_selected_cells",
            "mean",
        ),
    )
    .reset_index()
)

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "hpd_summary_by_resolution.csv",
    ),
    index=False,
)


# ============================================================
# PRINT TABLES FOR EACH PRIOR AND GAMMA
# ============================================================

for prior in PRIOR_ORDER:

    for gamma in GAMMAS:

        subset = summary[
            (summary["prior"] == prior)
            &
            (summary["gamma"] == gamma)
        ].copy()

        print(
            "\n========================================"
        )
        print(
            f"{PRIOR_LABELS[prior]} | gamma={gamma}"
        )
        print(
            "========================================"
        )

        pivot_mean = (
            subset
            .pivot(
                index="N_ref",
                columns="tau",
                values="mean_hpd_iou",
            )
            .reindex(
                index=N_REF_ORDER,
                columns=TAU_ORDER,
            )
        )

        pivot_std = (
            subset
            .pivot(
                index="N_ref",
                columns="tau",
                values="std_hpd_iou",
            )
            .reindex(
                index=N_REF_ORDER,
                columns=TAU_ORDER,
            )
        )

        formatted = pd.DataFrame(
            index=N_REF_ORDER
        )

        for tau in TAU_ORDER:

            formatted[
                TAU_LABELS[tau]
            ] = [
                (
                    f"{pivot_mean.loc[N, tau]:.5f}"
                    f" ± "
                    f"{pivot_std.loc[N, tau]:.5f}"
                )
                for N in N_REF_ORDER
            ]

        print(
            formatted.to_string()
        )

        filename = (
            f"hpd_table_"
            f"{prior}_"
            f"gamma_{gamma:.1f}.csv"
        )

        formatted.to_csv(
            os.path.join(
                OUTPUT_DIR,
                filename,
            )
        )


# ============================================================
# DOMAIN × TESSELLATION AVERAGES
#
# Average over resolution, keeping:
# prior, gamma, domain, tau
# ============================================================

domain_summary = (
    df
    .groupby(
        [
            "prior",
            "gamma",
            "domain",
            "tau",
        ]
    )
    .agg(
        mean_hpd_iou=(
            "hpd_iou",
            "mean",
        ),
        std_hpd_iou=(
            "hpd_iou",
            "std",
        ),
    )
    .reset_index()
)

domain_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "hpd_summary_by_domain.csv",
    ),
    index=False,
)


# ============================================================
# GLOBAL TESSELLATION AVERAGES
#
# Average over:
# domain, prior, resolution
# separately for gamma.
# ============================================================

global_tau_summary = (
    df
    .groupby(
        [
            "gamma",
            "tau",
        ]
    )
    .agg(
        mean_hpd_iou=(
            "hpd_iou",
            "mean",
        ),
        std_hpd_iou=(
            "hpd_iou",
            "std",
        ),
    )
    .reset_index()
)

global_tau_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "hpd_global_tessellation_summary.csv",
    ),
    index=False,
)

print(
    "\n========================================"
)
print(
    "GLOBAL TESSELLATION AVERAGES"
)
print(
    "========================================"
)

print(
    global_tau_summary.to_string(
        index=False
    )
)


# ============================================================
# IMPROVEMENT WITH RESOLUTION
#
# Compare N=100 vs N=800.
# ============================================================

resolution_change = []

for prior in PRIOR_ORDER:

    for gamma in GAMMAS:

        for tau in TAU_ORDER:

            subset = summary[
                (summary["prior"] == prior)
                &
                (summary["gamma"] == gamma)
                &
                (summary["tau"] == tau)
            ]

            value_100 = float(
                subset.loc[
                    subset["N_ref"] == 100,
                    "mean_hpd_iou",
                ].iloc[0]
            )

            value_800 = float(
                subset.loc[
                    subset["N_ref"] == 800,
                    "mean_hpd_iou",
                ].iloc[0]
            )

            resolution_change.append(
                {
                    "prior": prior,
                    "gamma": gamma,
                    "tau": tau,
                    "iou_N100": value_100,
                    "iou_N800": value_800,
                    "absolute_improvement":
                        value_800
                        -
                        value_100,
                    "relative_improvement_percent":
                        (
                            value_800
                            -
                            value_100
                        )
                        /
                        value_100
                        *
                        100.0,
                }
            )

resolution_change = pd.DataFrame(
    resolution_change
)

resolution_change.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "hpd_resolution_improvement.csv",
    ),
    index=False,
)

print(
    "\n========================================"
)
print(
    "N=100 -> N=800 IMPROVEMENT"
)
print(
    "========================================"
)

print(
    resolution_change.to_string(
        index=False
    )
)


# ============================================================
# WIN COUNTS
#
# For every:
# domain × prior × N_ref × gamma
#
# determine which tessellation achieves
# the largest HPD-IoU.
# ============================================================

winner_rows = []

group_columns = [
    "domain",
    "prior",
    "N_ref",
    "gamma",
]

for key, group in df.groupby(
    group_columns
):

    max_iou = group[
        "hpd_iou"
    ].max()

    winners = group[
        np.isclose(
            group["hpd_iou"],
            max_iou,
            atol=1e-12,
        )
    ]

    for _, row in winners.iterrows():

        winner_rows.append(
            {
                "domain": key[0],
                "prior": key[1],
                "N_ref": key[2],
                "gamma": key[3],
                "winner": row["tau"],
                "hpd_iou": row["hpd_iou"],
            }
        )

winner_df = pd.DataFrame(
    winner_rows
)

winner_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "hpd_winners.csv",
    ),
    index=False,
)

winner_counts = (
    winner_df
    .groupby(
        [
            "gamma",
            "winner",
        ]
    )
    .size()
    .reset_index(
        name="wins"
    )
)

winner_counts.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "hpd_winner_counts.csv",
    ),
    index=False,
)

print(
    "\n========================================"
)
print(
    "WIN COUNTS"
)
print(
    "========================================"
)

print(
    winner_counts.to_string(
        index=False
    )
)


# ============================================================
# FIGURES
#
# One figure per gamma:
#
# row 1 -> single hotspot
# row 2 -> multimodal
#
# columns -> domains
# ============================================================

for gamma in GAMMAS:

    fig, axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(15, 6),
        sharex=True,
        sharey=True,
    )

    for row_idx, prior in enumerate(
        PRIOR_ORDER
    ):

        for col_idx, domain in enumerate(
            DOMAIN_ORDER
        ):

            ax = axes[
                row_idx,
                col_idx,
            ]

            subset = df[
                (df["gamma"] == gamma)
                &
                (df["prior"] == prior)
                &
                (df["domain"] == domain)
            ]

            for tau in TAU_ORDER:

                tau_data = (
                    subset[
                        subset["tau"] == tau
                    ]
                    .sort_values(
                        "N_ref"
                    )
                )

                ax.plot(
                    tau_data["N_ref"],
                    tau_data["hpd_iou"],
                    marker="o",
                    label=TAU_LABELS[tau],
                )

            if row_idx == 0:

                ax.set_title(
                    domain
                )

            if col_idx == 0:

                ax.set_ylabel(
                    (
                        f"{PRIOR_LABELS[prior]}\n"
                        "HPD-IoU"
                    )
                )

            if row_idx == 1:

                ax.set_xlabel(
                    r"$N_{\mathrm{ref}}$"
                )

            ax.set_xticks(
                N_REF_ORDER
            )

            ax.grid(
                alpha=0.25
            )

    handles, labels = (
        axes[0, 0]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(
            0.5,
            1.01,
        ),
    )

    fig.suptitle(
        (
            "HPD-region preservation "
            f"($\\gamma={gamma}$)"
        ),
        y=1.06,
    )

    fig.tight_layout()

    output_figure = os.path.join(
        OUTPUT_DIR,
        (
            f"hpd_iou_gamma_"
            f"{gamma:.1f}.png"
        ),
    )

    fig.savefig(
        output_figure,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved figure: {output_figure}"
    )


# ============================================================
# DONE
# ============================================================

print(
    "\n========================================"
)
print(
    "ANALYSIS COMPLETE"
)
print(
    "========================================"
)

print(
    f"Output directory: {OUTPUT_DIR}"
)