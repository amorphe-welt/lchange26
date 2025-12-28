# Semantic Drift Analysis Repository (Anonymous)

This repository contains the code, data organization, and experimental outputs used for analyzing **sense-level semantic drift in contextualized word embeddings** across multiple neural language models and historical time spans.

The repository is organized to support **fully reproducible drift analyses**, including trajectory-based tests, distributional divergence metrics, and cross-model comparisons.

---

## Repository Structure

```
.
├── pipeline_<modelname>.sh   # Model-specific end-to-end pipelines
├── scripts/                  # Entry-point analysis scripts (CLI)
├── src/                      # Core libraries
├── data/                     # Dataset metadata (JSON synset lists)
├── embeddings/               # Precomputed embeddings (HDF5)
├── results/                  # Drift results (model / lexeme / synset)
└── README.md
```

---

## End-to-End Pipelines

The repository provides **one bash pipeline per model** at the repository root:

```
pipeline_<modelname>.sh
```

Each pipeline script:

* Runs the **full drift analysis** for a single embedding model
* Calls the appropriate Python scripts in `scripts/`
* Writes all outputs to `results/<modelname>/`

These pipelines serve as the **primary entry point** for reproducing experiments.

### Example

```bash
bash pipeline_qwen3-4b.sh
```

This executes the complete analysis pipeline for the specified model using precomputed embeddings.

---

## Directory Details

### `scripts/`

Command-line Python scripts that perform individual analysis stages, including:

* Sense-level drift computation
* Trajectory-based statistical tests
* Distributional divergence (Wasserstein, KL)
* KDE and CDF estimation
* Cross-model aggregation
* Plot and table generation

Scripts are modular and are orchestrated by the pipeline `.sh` files.

---

### `src/`

Core Python libraries implementing:

* Embedding alignment and normalization
* Drift and divergence metrics
* Statistical testing procedures
* KDE-based density estimation
* Visualization utilities

All reusable logic is contained here.

---

### `data/`

Dataset metadata and configuration files:

* JSON lists of evaluated synsets
* Lexeme–synset mappings
* Time span definitions

No raw corpora are included.

---

### `embeddings/`

Precomputed contextual embeddings for all evaluated models, stored as HDF5 (`.h5`) files.

* One file per model
* Indexed by lexeme, synset, and time span
* Model-specific dimensionality

This enables full reproducibility without recomputing embeddings.

---

### `results/`

All experimental outputs are written here.

Structure:

```
results/
└── <model_name>/
    └── <lexeme>/
        └── <synset>/
            ├── synset_drift.csv
            ├── divergence.csv
            ├── statistics.json
            └── plots/
```

Each synset directory contains:

* Per-dimension drift magnitudes
* Wasserstein and KL divergence across time spans
* Statistical test outputs
* Generated figures

---

## Reproducibility

* Pipelines are deterministic given fixed embeddings
* Random seeds are fixed where applicable
* No training or fine-tuning is performed
* All results in `results/` can be regenerated via the pipeline scripts

---

## Anonymization

This repository is anonymized for peer review:

* No author names or affiliations
* No institution-specific references
* Model names are used only as neutral identifiers
