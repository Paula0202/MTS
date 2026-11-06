from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = (
    BASE_DIR
    / "results"
    / "graph_robustness.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PDF_PATH = (
    OUTPUT_DIR
    / "mean_stretch_vs_resolution.pdf"
)

PNG_PATH = (
    OUTPUT_DIR
    / "mean_stretch_vs_resolution.png"
)


# ============================================================
# CONFIGURATION
# ============================================================

NREF_ORDER = [
    100,
    300,
    500,
    800,
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

MARKERS = {
    "square": "o",
    "triangle": "s",
    "hex": "D",
}


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    CSV_PATH
)


# ============================================================
# SANITY CHECKS
# ============================================================

required_columns = {
    "domain",
    "N_ref",
    "tau",
    "stretch_mean",
    "n_stretch_pairs",
}

missing = (
    required_columns
    - set(df.columns)
)

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


counts = (
    df.groupby(
        [
            "domain",
            "N_ref",
            "tau",
        ]
    )
    .size()
)

if not (counts == 16).all():
    raise ValueError(
        "Some domain-resolution-tessellation "
        "conditions do not contain 16 "
        "phase-orientation configurations."
    )


print(
    "OK: all conditions contain "
    "16 phase-orientation configurations."
)

print(
    "Stretch pairs:",
    sorted(
        df["n_stretch_pairs"]
        .unique()
    )
)


# ============================================================
# AGGREGATION
# ============================================================

# Step 1:
# For each domain-resolution-tessellation condition,
# compute:
#   - mean stretch over the 16 phase-orientation configurations
#   - standard deviation over those configurations

domain_summary = (
    df.groupby(
        [
            "domain",
            "N_ref",
            "tau",
        ]
    )
    .agg(
        stretch_mean=(
            "stretch_mean",
            "mean",
        ),

        stretch_phase_std=(
            "stretch_mean",
            "std",
        ),
    )
    .reset_index()
)


# Step 2:
# Average the five domain-specific means.
#
# Error bars correspond to the mean within-domain
# standard deviation across phase-orientation configurations.

summary = (
    domain_summary
    .groupby(
        [
            "N_ref",
            "tau",
        ]
    )
    .agg(
        stretch_mean=(
            "stretch_mean",
            "mean",
        ),

        stretch_phase_std=(
            "stretch_phase_std",
            "mean",
        ),
    )
    .reset_index()
)


print(
    "\nAggregated stretch results:"
)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        6.2,
        4.2,
    )
)


for tau in TAU_ORDER:

    subset = (
        summary[
            summary["tau"]
            == tau
        ]
        .set_index(
            "N_ref"
        )
        .loc[
            NREF_ORDER
        ]
        .reset_index()
    )

    ax.errorbar(
        subset["N_ref"],
        subset["stretch_mean"],
        yerr=subset["stretch_phase_std"],

        marker=MARKERS[tau],
        markersize=5,

        linewidth=1.4,
        capsize=3,
        elinewidth=1.0,

        label=TAU_LABELS[tau],
    )


# ============================================================
# DIRECT-EUCLIDEAN BASELINE
# ============================================================

ax.axhline(
    y=1.0,
    linestyle="--",
    linewidth=1.0,
    alpha=0.65,
)


# ============================================================
# AXES
# ============================================================

ax.set_xlabel(
    r"$N_{\mathrm{ref}}$"
)

ax.set_ylabel(
    r"Mean physical path stretch $\overline{S}$"
)

ax.set_xticks(
    NREF_ORDER
)

ax.set_ylim(
    0.98,
    1.31,
)

ax.grid(
    linewidth=0.5,
    alpha=0.25,
)

ax.legend(
    frameon=False,
    loc="center right",
)


# ============================================================
# LAYOUT
# ============================================================

fig.tight_layout()


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    PDF_PATH,
    bbox_inches="tight",
)

fig.savefig(
    PNG_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(
    f"\nSaved: {PDF_PATH}"
)

print(
    f"Saved: {PNG_PATH}"
)