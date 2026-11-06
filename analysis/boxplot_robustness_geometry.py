from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = (
    BASE_DIR
    / "experiments"
    / "results"
    / "geometric_robustness.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

DOMAIN_ORDER = [
    "Square",
    "Circle",
    "Triangle",
    "Convex",
    "Concave",
]

# Only coarse and fine resolutions are shown in the boxplots.
# Intermediate resolutions remain reported in the main table.
NREF_ORDER = [
    100,
    800,
]

TAU_ORDER = [
    "square",
    "triangle",
    "hex",
]

TAU_LABELS = {
    "square": "Sq.",
    "triangle": "Tri.",
    "hex": "Hex.",
}

# Winner in each subplot can be determined
# from either the median or the mean.
WINNER_CRITERION = "median"


# ============================================================
# HELPERS
# ============================================================

def get_winner_index(
    data,
    higher_is_better=True,
    criterion="median",
):
    """
    Return the index of the best tessellation.
    """

    if criterion == "median":

        values = np.array([
            np.median(d)
            if len(d) > 0
            else np.nan
            for d in data
        ])

    elif criterion == "mean":

        values = np.array([
            np.mean(d)
            if len(d) > 0
            else np.nan
            for d in data
        ])

    else:

        raise ValueError(
            "criterion must be 'median' or 'mean'"
        )

    if higher_is_better:

        return int(
            np.nanargmax(values)
        )

    return int(
        np.nanargmin(values)
    )


def emphasize_winner_median(
    bp,
    winner_idx,
):
    """
    Highlight only the median line of the winning tessellation.
    """

    bp["medians"][
        winner_idx
    ].set_linewidth(
        1.35
    )


# ============================================================
# FIGURE GENERATION
# ============================================================

def make_geometric_robustness_figure(
    df,
    metric,
    ylabel,
    output_basename,
    higher_is_better,
):
    """
    Create a 2 x 5 geometric-robustness boxplot figure.

    Rows:
        N_ref = 100
        N_ref = 800

    Columns:
        Square
        Circle
        Triangle
        Convex
        Concave

    Each subplot compares the square, triangular, and
    hexagonal tessellations.

    Each box contains the 16 phase-orientation
    configurations associated with the corresponding
    domain-resolution-tessellation condition.

    Each subplot uses its own y-axis scale for readability.
    """

    fig, axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(
            11.8,
            5.0,
        ),
    )

    # ========================================================
    # SUBPLOTS
    # ========================================================

    for row, n_ref in enumerate(
        NREF_ORDER
    ):

        for col, domain in enumerate(
            DOMAIN_ORDER
        ):

            ax = axes[
                row,
                col,
            ]

            # -----------------------------------------------
            # Select experimental condition
            # -----------------------------------------------

            subset = df[
                (
                    df["domain"]
                    == domain
                )
                & (
                    df["N_ref"]
                    == n_ref
                )
            ].copy()

            # -----------------------------------------------
            # Extract tessellation distributions
            # -----------------------------------------------

            data = []

            for tau in TAU_ORDER:

                values = (
                    subset.loc[
                        subset["tau"]
                        == tau,
                        metric,
                    ]
                    .dropna()
                    .to_numpy()
                )

                data.append(
                    values
                )

            # -----------------------------------------------
            # Sanity check
            # -----------------------------------------------

            for tau, values in zip(
                TAU_ORDER,
                data,
            ):

                if len(values) != 16:

                    print(
                        f"WARNING: "
                        f"{domain}, "
                        f"N_ref={n_ref}, "
                        f"{tau}: "
                        f"{len(values)} values "
                        f"instead of 16."
                    )

            # -----------------------------------------------
            # Determine winner
            # -----------------------------------------------

            winner_idx = (
                get_winner_index(
                    data=data,
                    higher_is_better=(
                        higher_is_better
                    ),
                    criterion=(
                        WINNER_CRITERION
                    ),
                )
            )

            # -----------------------------------------------
            # Boxplots
            # -----------------------------------------------

            bp = ax.boxplot(
                data,
                positions=[
                    1,
                    2,
                    3,
                ],
                widths=0.42,
                patch_artist=False,
                showfliers=False,

                boxprops=dict(
                    linewidth=0.55,
                ),

                medianprops=dict(
                    linewidth=0.80,
                ),

                whiskerprops=dict(
                    linewidth=0.50,
                ),

                capprops=dict(
                    linewidth=0.50,
                ),
            )

            emphasize_winner_median(
                bp,
                winner_idx,
            )

            # -----------------------------------------------
            # X axis
            # -----------------------------------------------

            ax.set_xticks([
                1,
                2,
                3,
            ])

            ax.set_xticklabels(
                [
                    TAU_LABELS[tau]
                    for tau
                    in TAU_ORDER
                ],
                fontsize=8,
            )

            # -----------------------------------------------
            # Automatic Y scale
            # -----------------------------------------------

            ax.autoscale(
                axis="y",
                tight=False,
            )

            ax.margins(
                y=0.08
            )

            # -----------------------------------------------
            # Grid
            # -----------------------------------------------

            ax.grid(
                axis="y",
                linewidth=0.40,
                alpha=0.20,
            )

            # -----------------------------------------------
            # Tick formatting
            # -----------------------------------------------

            ax.tick_params(
                axis="y",
                labelsize=7.5,
            )

            ax.tick_params(
                axis="x",
                labelsize=8,
            )

            # -----------------------------------------------
            # Column titles = domains
            # -----------------------------------------------

            if row == 0:

                ax.set_title(
                    domain,
                    fontsize=8.8,
                    pad=5,
                )

            # -----------------------------------------------
            # Row labels = resolution
            # -----------------------------------------------

            if col == 0:

                ax.set_ylabel(
                    (
                        rf"$N_{{\mathrm{{ref}}}}={n_ref}$"
                        "\n"
                        f"{ylabel}"
                    ),
                    fontsize=8.8,
                )

            else:

                ax.set_ylabel(
                    ""
                )

    # ========================================================
    # LAYOUT
    # ========================================================

    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.10,
        top=0.92,
        wspace=0.22,
        hspace=0.28,
    )

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    png_path = (
        OUTPUT_DIR
        / f"{output_basename}.png"
    )

    pdf_path = (
        OUTPUT_DIR
        / f"{output_basename}.pdf"
    )

    # ========================================================
    # SAVE
    # ========================================================

    fig.savefig(
        str(
            png_path
        ),
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        str(
            pdf_path
        ),
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {png_path}"
    )

    print(
        f"Saved: {pdf_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = pd.read_csv(
        CSV_PATH
    )

    # ========================================================
    # SANITY CHECK
    # ========================================================

    expected_columns = {
        "domain",
        "N_ref",
        "tau",
        "phase_u",
        "phase_v",
        "theta_deg",
        "iou",
        "asbd",
    }

    missing = (
        expected_columns
        - set(
            df.columns
        )
    )

    if missing:

        raise ValueError(
            f"Missing expected columns: "
            f"{missing}"
        )

    # ========================================================
    # IoU
    # ========================================================

    make_geometric_robustness_figure(
        df=df,
        metric="iou",
        ylabel="Geometric IoU",
        output_basename=(
            "robustness_boxplot_"
            "geometric_iou"
        ),
        higher_is_better=True,
    )

    # ========================================================
    # ASBD
    # ========================================================

    make_geometric_robustness_figure(
        df=df,
        metric="asbd",
        ylabel="ASBD",
        output_basename=(
            "robustness_boxplot_"
            "asbd"
        ),
        higher_is_better=False,
    )

    print(
        "\nDone."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()