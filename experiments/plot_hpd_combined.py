import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = os.path.join(
    "results",
    "hpd_fidelity.csv",
)

OUTPUT_FILE = os.path.join(
    "results",
    "hpd_analysis",
    "hpd_iou_combined.png",
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

N_REF_ORDER = [
    100,
    300,
    500,
    800,
]

ROW_CONFIGS = [
    ("single_hotspot", 0.5, r"Single hotspot, $\gamma=0.5$"),
    ("multimodal",     0.5, r"Multimodal, $\gamma=0.5$"),
    ("single_hotspot", 0.8, r"Single hotspot, $\gamma=0.8$"),
    ("multimodal",     0.8, r"Multimodal, $\gamma=0.8$"),
]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    nrows=4,
    ncols=5,
    figsize=(14.5, 9.0),
    sharex=True,
    sharey=True,
)


for row_idx, (
    prior,
    gamma,
    row_label,
) in enumerate(
    ROW_CONFIGS
):

    for col_idx, domain in enumerate(
        DOMAIN_ORDER
    ):

        ax = axes[
            row_idx,
            col_idx,
        ]

        subset = df[
            (df["prior"] == prior)
            &
            (df["gamma"] == gamma)
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
                linewidth=1.6,
                markersize=4.5,
                label=TAU_LABELS[tau],
            )

        # ----------------------------------------------------
        # Column titles
        # ----------------------------------------------------

        if row_idx == 0:

            ax.set_title(
                domain,
                fontsize=10,
            )

        # ----------------------------------------------------
        # Row labels
        # ----------------------------------------------------

        if col_idx == 0:

            ax.set_ylabel(
                (
                    row_label
                    +
                    "\nHPD-IoU"
                ),
                fontsize=9,
            )

        # ----------------------------------------------------
        # X labels
        # ----------------------------------------------------

        if row_idx == 3:

            ax.set_xlabel(
                r"$N_{\mathrm{ref}}$",
                fontsize=9,
            )

        ax.set_xticks(
            N_REF_ORDER
        )

        ax.tick_params(
            axis="both",
            labelsize=8,
        )

        ax.grid(
            alpha=0.25
        )


# ============================================================
# SHARED LEGEND
# ============================================================

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
        0.995,
    ),
    fontsize=9,
)


# ============================================================
# LAYOUT
# ============================================================

fig.suptitle(
    "Preservation of highest-probability-density regions",
    fontsize=12,
    y=1.025,
)

fig.tight_layout(
    rect=[
        0.02,
        0.02,
        0.99,
        0.965,
    ]
)


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig
)

print(
    f"Saved figure: {OUTPUT_FILE}"
)