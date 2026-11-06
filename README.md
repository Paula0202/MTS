# Spatial Discretization for Probabilistic UAV Search

Code, experimental data, and figures associated with the manuscript:

**A Systematic Experimental Study of the Impact of Spatial Discretization on UAV-Based Probabilistic Target Search**

This repository contains the experimental framework used to study how spatial tessellation affects the representation and performance of probabilistic UAV target-search problems.

The study compares **square, equilateral-triangular, and regular-hexagonal tessellations** under equivalent nominal cell-area constraints, considering their effects on geometric fidelity, probabilistic representation, navigation-graph structure, and search performance.

## Overview

The experimental evaluation is organized at two complementary levels.

### Representation-level analysis

The influence of spatial discretization is evaluated in terms of:

- geometric fidelity;
- probabilistic fidelity;
- navigation-graph structure;
- physical shortest-path stretch;
- sensitivity to lattice placement and orientation.

The robustness analysis considers four normalized lattice phases and four symmetry-unique lattice orientations for each tessellation.

### Search-level analysis

Two representative search strategies are evaluated:

- **Systematic lawnmower search**, defined in a common physical coordinate system;
- **Probability-guided greedy search**, driven by the evolving discrete belief map.

Search experiments use paired physical target realizations so that all tessellations are evaluated under equivalent continuous conditions.

## Experimental configuration

The main experimental settings are:

- **Tessellations:** square, equilateral triangular, regular hexagonal;
- **Reference resolutions:** `N_ref = {100, 300, 500, 800}`;
- **Domains:** square, circle, triangle, convex irregular, concave irregular;
- **Priors:** uniform, single hotspot, and multimodal;
- **Search strategies:** lawnmower and probability-guided greedy;
- **Monte Carlo targets:** 1000 paired physical target realizations per domain-prior-resolution condition.

For the search experiments, all tessellations use the canonical lattice configuration:

- zero lattice phase;
- zero lattice orientation;
- common physical UAV initialization.

The systematic lawnmower strategy is evaluated over the physical sweep orientations:


0°, 15°, 30°, ..., 165°


so that conclusions are not tied to a single alignment between the sweep direction and the navigation graph.

## Repository structure


MTS/
├── analysis/
│   └── scripts for result analysis and figure generation
│
├── experiments/
│   └── representation-level and robustness experiments
│
├── figures/
│   └── final figures used in the manuscript
│
├── results/
│   └── processed experimental results and summary CSV files
│
├── domains.py
├── tessellations.py
├── priors.py
├── metrics_geometry.py
├── metrics_probability.py
├── search_experiments.py
├── requirements.txt
└── README.md


The modules in the repository root define the main search domains, tessellations, prior distributions, metrics, and search experiment.

The `experiments/` directory contains the scripts used to generate the representation-level results, while `analysis/` contains scripts used for post-processing and visualization.

## Requirements

The experiments were developed and executed using:


Python 3.13


with the following package versions:


numpy==2.5.3
pandas==3.0.6
matplotlib==3.11.2
networkx==3.7
shapely==2.1.2
scipy==1.18.1


Install the required packages with:


pip install -r requirements.txt


## Running the experiments

Commands should be executed from the repository root.

### Geometric and graph robustness


python -m experiments.run_geometric_robustness


This experiment evaluates the complete combination of:

- five search domains;
- four spatial resolutions;
- three tessellations;
- four normalized lattice phases;
- four symmetry-unique lattice orientations.

The resulting metrics include geometric overlap, boundary fidelity, graph connectivity, and physical shortest-path stretch.

The main outputs are stored in `results/`, including:


geometric_robustness.csv
geometric_robustness_summary.csv


### Probabilistic robustness


python -m experiments.run_probabilistic_robustness_parallel


The probabilistic robustness analysis considers:

- square and concave domains;
- single-hotspot and multimodal priors;
- `N_ref = {100, 800}`;
- square, triangular, and hexagonal tessellations;
- 16 phase-orientation configurations per tessellation.

The evaluated metrics include:

- unrepresented probability mass;
- total variation distance;
- HPD-region IoU at probability levels 0.5 and 0.8.

This experiment comprises **384 configurations**.

The main outputs are:


probabilistic_robustness.csv
probabilistic_robustness_summary.csv


### Search experiments


python search_experiments.py


The search experiment compares the three tessellations using both systematic and probability-guided search.

For each domain-prior-resolution condition:

- 1000 physical target locations are sampled;
- the same targets are reused across tessellations;
- targets are sampled from the support jointly represented by all three tessellations;
- the UAV starts from the same physical reference position;
- the same continuous prior information is used across representations.

For systematic search, the same physical sweep orientations are also reused across tessellations.

The processed outputs include:


search_experiments_summary.csv
search_experiments_orientation_summary.csv


## Results

Final processed data used to produce the manuscript results are provided in the `results/` directory.

Representative files include:


geometric_robustness.csv
geometric_robustness_summary.csv

graph_robustness.csv
graph_robustness_summary.csv

probabilistic_robustness.csv
probabilistic_robustness_summary.csv

search_experiments_summary.csv
search_experiments_orientation_summary.csv


Intermediate checkpoint files generated during long-running experiments are excluded from version control.

The complete raw Monte Carlo output from the search experiments, `search_experiments.csv`, is approximately 550 MB and therefore exceeds the standard GitHub file-size limit. It can be regenerated using `search_experiments.py`.

A persistent archived version of the research artifact will also be made available through Zenodo.

## Figures

The `figures/` directory contains the final figures included in the manuscript.

Scripts used to generate quantitative figures and perform additional analysis are available in `analysis/`.

## Reproducibility

Several measures are used to support reproducible comparisons:

- equivalent nominal cell-area constraints across tessellations;
- common continuous search domains and priors;
- fixed random seeds where applicable;
- paired physical target realizations;
- common UAV initialization;
- canonical lattice configuration for search experiments;
- matched physical lawnmower sweep orientations;
- deterministic tie-breaking rules.

The robustness experiments also use checkpoint files so that long executions can be safely resumed. These intermediate files are intentionally excluded from the public repository.

## Data availability

The source code and processed data supporting the study are publicly available at:

**https://github.com/Paula0202/MTS**

A persistent Zenodo archive and DOI will be added once the repository version associated with the manuscript is finalized.

## Authors

- Paula López Álvarez
- Jesús García
- José M. Molina
- Juan P. Llerena

## Citation

Citation information for the associated article and archived repository version will be added once available.

## License

License information will be added before the archived release of the repository.
