#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# --------------------------------------------------
# Load all per-dimension drift magnitudes for a model
# --------------------------------------------------

def load_model_dimension_drifts(results_root, model):
    rows = []
    base = Path(results_root) / model / "lexemes"

    for drift_csv in base.rglob("*_drift.csv"):
        df = pd.read_csv(drift_csv)

        if not {"embedding_dim", "drift_magnitude"}.issubset(df.columns):
            continue

        df = df.copy()
        df["model"] = model
        df["synset"] = drift_csv.parent.name
        rows.append(df)

    if not rows:
        raise RuntimeError(f"No drift files found for {model}")

    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------
# Summaries used in the paper / appendix
# --------------------------------------------------

def summarize_model(df):
    x = df["drift_magnitude"].values

    # log-scale is essential for heavy-tailed drift
    x_log = np.log1p(x)
    kde = gaussian_kde(x_log)

    return {
        "synset_count": df["synset"].nunique(),
        "num_dimensions": len(x),

        "mean_drift": x.mean(),
        "median_drift": np.median(x),
        "iqr_drift": np.percentile(x, 75) - np.percentile(x, 25),
        "p95_drift": np.percentile(x, 95),
        "p99_drift": np.percentile(x, 99),

        "kde_peak": kde(0.0)[0],  # density at zero drift
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--outdir", default="results/global_drift")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_models = {}
    summaries = []

    # ----------------------
    # Load + summarize
    # ----------------------
    for model in args.models:
        df = load_model_dimension_drifts(args.results_root, model)
        all_models[model] = df

        summary = summarize_model(df)
        summary["model"] = model
        summaries.append(summary)

        df.to_csv(outdir / f"{model}_all_dimension_drifts.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(outdir / "global_dimension_drift_summary.csv", index=False)

    # ----------------------
    # KDE plot
    # ----------------------
    plt.figure(figsize=(6.5, 4.5))

    for model, df in all_models.items():
        x_log = np.log1p(df["drift_magnitude"].values)
        kde = gaussian_kde(x_log)

        xs = np.linspace(x_log.min(), x_log.max(), 600)
        plt.plot(xs, kde(xs), label=model, linewidth=2)

    plt.xlabel("log(1 + per-dimension drift magnitude)")
    plt.ylabel("Density")
    plt.title("Global Per-Dimension Drift Distributions")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outdir / "global_dimension_drift_kde.pdf")
    plt.close()

    # ----------------------
    # CDF plot
    # ----------------------
    plt.figure(figsize=(6.5, 4.5))

    for model, df in all_models.items():
        x_log = np.sort(np.log1p(df["drift_magnitude"].values))
        y = np.linspace(0, 1, len(x_log))
        plt.plot(x_log, y, label=model, linewidth=2)

    plt.xlabel("log(1 + per-dimension drift magnitude)")
    plt.ylabel("CDF")
    plt.title("Cumulative Per-Dimension Drift")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outdir / "global_dimension_drift_cdf.pdf")
    plt.close()

    print("✔ Global per-dimension drift comparison complete")
    print(f"→ {outdir / 'global_dimension_drift_summary.csv'}")


if __name__ == "__main__":
    main()
