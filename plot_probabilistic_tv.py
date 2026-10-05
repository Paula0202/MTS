import os 
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = ("results/probabilistic_fidelity_final.csv")
OUTPUT_FILE = ("results/dtv_by_domain_2x5.pdf")
PRIORS = ["single_hotspot", "multimodal"]
DOMAINS = ["Square", "Circle", "Triangle", "Convex", "Concave"]
TAUS = ["square", "triangle", "hex"]
TAU_LABELS = {"square": "Square", "triangle": "Triangular", "hex": "Hexagonal"}
PRIOR_LABELS = {
    "single_hotspot": "Single hotspot",
    "multimodal": "Multimodal"
}
MARKERS = {"square": "o", "triangle": "s", "hex": "^"}
LINESTYLES = {
    "square": "-",
    "triangle": "--",
    "hex": "-."
}

df = pd.read_csv(INPUT_FILE)

df = df[df["prior"].isin(PRIORS)].copy()
fig, axes = plt.subplots(
    2,
    5,
    figsize=(15.5, 6.8),
    sharex=True,
    sharey=True
)


for row_idx, prior in enumerate(
    PRIORS
):

    for col_idx, domain in enumerate(
        DOMAINS
    ):

        ax = axes[
            row_idx,
            col_idx
        ]

        subset = df[
            (df["prior"] == prior)
            &
            (df["domain"] == domain)
        ]

        for tau in TAUS:

            data = (
                subset[
                    subset["tau"] == tau
                ]
                .sort_values(
                    "N_ref"
                )
            )

            ax.plot(
                data["N_ref"],
                data["tv_distance"],
                marker=MARKERS[tau],
                linestyle=LINESTYLES[tau],
                linewidth=1.7,
                markersize=5,
                label=TAU_LABELS[tau]
            )

        # Column titles
        if row_idx == 0:
            ax.set_title(
                f"{domain} domain",
                fontsize=10
            )

        # Row labels
        if col_idx == 0:
            ax.set_ylabel(
                (
                    r"$D_{\mathrm{TV}}$"
                    "\n"
                    +
                    PRIOR_LABELS[prior]
                ),
                fontsize=10
            )

        # X labels only on bottom row
        if row_idx == 1:
            ax.set_xlabel(
                r"$N_{\mathrm{ref}}$",
                fontsize=10
            )

        ax.set_xticks(
            [100, 300, 500, 800]
        )

        ax.grid(
            alpha=0.25
        )

        ax.tick_params(
            labelsize=8
        )


# ============================================================
# COMMON LEGEND
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
    frameon=False,
    bbox_to_anchor=(0.5, 0.975)
)


# ============================================================
# LAYOUT
# ============================================================

fig.tight_layout(
    rect=[
        0.0,
        0.0,
        1.0,
        0.93
    ]
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

fig.savefig(
    OUTPUT_FILE,
    bbox_inches="tight"
)

print(
    f"Saved to: {OUTPUT_FILE}"
)

plt.show()  