#!/usr/bin/env python
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
from scipy.stats import spearmanr
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_synset_mapping(json_path):
    """Parses the synsets.json into a flat dictionary for lookup."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        mapping = {}
        for entry in data:
            for synset in entry['synsets']:
                # Store the short definition/label
                mapping[synset['id']] = synset['definition']
        return mapping
    except Exception as e:
        logger.error(f"Could not load synset mapping: {e}")
        return {}

def run_comparison(file_a, file_b, name_a, name_b, output_dir, mapping_path):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)
    
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = {s['id']: s['definition'] for entry in json.load(f) for s in entry['synsets']}

    # 2. Merge
    merged = pd.merge(df_a, df_b, on=['lexeme', 'synset'], suffixes=(f'_{name_a}', f'_{name_b}'))
    merged['definition'] = merged['synset'].map(mapping).fillna("N/A")

    # 3. NORMALIZATION (Z-Score)
    # Formula: (x - mean) / std
    for name in [name_a, name_b]:
        col = f'max_drift_magnitude_{name}'
        z_col = f'z_score_{name}'
        merged[z_col] = (merged[col] - merged[col].mean()) / merged[col].std()

    # 4. Visualization (Plotting Z-Scores instead of raw values)
    plt.figure(figsize=(10, 8))
    sns.regplot(data=merged, x=f'z_score_{name_a}', y=f'z_score_{name_b}', 
                scatter_kws={'alpha':0.5}, line_kws={'color':'red', 'ls':'--'})
    
    plt.axhline(0, color='grey', lw=1, ls='-') # Mean lines
    plt.axvline(0, color='grey', lw=1, ls='-')
    
    plt.title(f'Normalized Semantic Drift: {name_a} vs {name_b}')
    plt.xlabel(f'{name_a} (Standard Deviations from Mean)')
    plt.ylabel(f'{name_b} (Standard Deviations from Mean)')
    plt.tight_layout()
    plt.savefig(out_path / "normalized_comparison.png")

    # 5. Report with Normalized Values
    report_file = out_path / "normalized_alignment_report.txt"
    with open(report_file, 'w') as f:
        f.write(f"NORMALIZED REPORT: {name_a} vs {name_b}\n")
        f.write("Units are Standard Deviations (Z-Scores). 0.0 = Average Drift.\n")
        f.write("="*90 + "\n")
        
        # Sort by those where models DISAGREE most on Z-scale
        merged['z_diff'] = (merged[f'z_score_{name_a}'] - merged[f'z_score_{name_b}']).abs()
        top_diff = merged.sort_values('z_diff', ascending=False).head(10)
        
        f.write(f"{'LEXEME':<12} {'DEF':<30} {name_a+' (Z)':<12} {name_b+' (Z)':<12} {'DIVERGENCE':<10}\n")
        for _, r in top_diff.iterrows():
            f.write(f"{r['lexeme']:<12} {r['definition'][:28]:<30} "
                    f"{r[f'z_score_{name_a}']:<12.2f} {r[f'z_score_{name_b}']:<12.2f} "
                    f"{r['z_diff']:<10.2f}\n")

def run_comparison_old(file_a, file_b, name_a, name_b, output_dir, mapping_path):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)
    synset_defs = load_synset_mapping(mapping_path)

    # 2. Merge on unique identifiers
    merged = pd.merge(
        df_a, df_b, 
        on=['lexeme', 'synset'], 
        suffixes=(f'_{name_a}', f'_{name_b}')
    )
    
    # Add definitions to the merged dataframe
    merged['definition'] = merged['synset'].map(synset_defs).fillna("No definition found")

    if merged.empty:
        logger.error("No overlapping lexemes/synsets found.")
        return

    # 3. Statistical Analysis
    col_a = f'max_drift_magnitude_{name_a}'
    col_b = f'max_drift_magnitude_{name_b}'
    corr, p_value = spearmanr(merged[col_a], merged[col_b])
    
    # 4. Visualization
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    sns.regplot(
        data=merged, x=col_a, y=col_b,
        scatter_kws={'alpha':0.4, 's':60, 'edgecolor':'w'},
        line_kws={'color':'#e74c3c', 'ls':'--'}
    )
    
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    plt.title(f'Semantic Drift Alignment: {name_a} vs {name_b}', fontsize=14, weight='bold')
    plt.xlabel(f'{name_a} Magnitude (Log Scale)')
    plt.ylabel(f'{name_b} Magnitude (Log Scale)')
    
    # Text box for correlation info
    plt.text(0.05, 0.95, f'Spearman ρ: {corr:.3f}\np-value: {p_value:.2e}', 
             transform=ax.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig(out_path / f'model_comparison_{name_a}_{name_b}.png', dpi=300)
    plt.close()

    # 5. Generate Enriched Text Report
    report_file = out_path / f'alignment_report_{name_a}_{name_b}.txt'
    with open(report_file, 'w') as f:
        f.write(f"CROSS-MODEL ALIGNMENT REPORT: {name_a} vs {name_b}\n")
        f.write("="*80 + "\n")
        f.write(f"Total overlapping senses: {len(merged)}\n")
        f.write(f"Spearman Correlation: {corr:.4f}\n\n")
        
        f.write(f"{'LEXEME':<15} {'ID':<8} {'DEF (GERMANET)':<40} {name_a:<10} {name_b:<10}\n")
        f.write("-" * 85 + "\n")
        
        merged['diff'] = (merged[col_a] - merged[col_b]).abs()
        top_diff = merged.sort_values('diff', ascending=False).head(15)
        
        for _, row in top_diff.iterrows():
            f.write(f"{row['lexeme']:<15} {row['synset']:<8} {row['definition'][:38]:<40} "
                    f"{row[col_a]:<10.1f} {row[col_b]:<10.1f}\n")

    logger.info(f"Comparison complete. Results in {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare semantic drift results with definitions.")
    parser.add_argument('--path_a', required=True)
    parser.add_argument('--path_b', required=True)
    parser.add_argument('--name_a', default="ModelA")
    parser.add_argument('--name_b', default="ModelB")
    parser.add_argument('--out_dir', default="model_comparison_results")
    parser.add_argument('--synsets', default="data/synsets.json")

    args = parser.parse_args()
    run_comparison(args.path_a, args.path_b, args.name_a, args.name_b, args.out_dir, args.synsets)

if __name__ == "__main__":
    main()
