#!/usr/bin/env python
"""
Visualize aggregated drift analysis results with specialized LSCD metrics.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging
import matplotlib.ticker as ticker
from collections import Counter
from adjustText import adjust_text # You may need: pip install adjust_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set global plotting style for academic papers
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 200})

def plot_drift_overview(df, output_dir):
    """Create comprehensive overview visualizations."""
    
    # Filter to significant results
    sig_df = df[df['any_significant']].copy()
    
    if sig_df.empty:
        logger.warning("No significant results to visualize")
        return
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Top lexemes by number of significant synsets
    ax1 = fig.add_subplot(gs[0, 0])
    lexeme_counts = sig_df['lexeme'].value_counts().head(15)
    lexeme_counts.plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_xlabel('Number of Synsets with Significant Drift')
    ax1.set_ylabel('Lexeme')
    ax1.set_title('Top 15 Lexemes by Significant Synsets', weight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Distribution of drift magnitudes
    ax2 = fig.add_subplot(gs[0, 1])
    if 'max_drift_magnitude' in sig_df.columns:
        sig_df['max_drift_magnitude'].dropna().hist(bins=50, ax=ax2, color='coral', edgecolor='black')
        ax2.set_xlabel('Maximum Drift Magnitude')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Drift Magnitudes', weight='bold')
        ax2.axvline(sig_df['max_drift_magnitude'].median(), color='red', 
                   linestyle='--', linewidth=2, label=f"Median: {sig_df['max_drift_magnitude'].median():.3f}")
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    
    # 3. P-value distribution
    ax3 = fig.add_subplot(gs[0, 2])
    if 'min_pvalue' in sig_df.columns:
        pvals = sig_df['min_pvalue'].dropna()
        if len(pvals) > 0:
            # ax3.hist(pvals, bins=50, ax=ax3, color='lightgreen', edgecolor='black')
            ax3.set_xlabel('Minimum P-value')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of P-values', weight='bold')
            ax3.axvline(0.05, color='red', linestyle='--', linewidth=2, label='α=0.05')
            ax3.axvline(0.01, color='orange', linestyle='--', linewidth=2, label='α=0.01')
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)
    
    # 4. Number of significant pairs per synset
    ax4 = fig.add_subplot(gs[1, 0])
    pairs_data = sig_df['num_significant_pairs']
    pairs_data[pairs_data > 0].hist(bins=30, ax=ax4, color='purple', edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Number of Significant Timespan Pairs')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Significant Pairs Distribution', weight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Impact duration distribution
    ax5 = fig.add_subplot(gs[1, 1])
    if 'max_impact_duration' in sig_df.columns:
        impact_data = sig_df[sig_df['max_impact_duration'] > 0]['max_impact_duration']
        if len(impact_data) > 0:
            impact_data.hist(bins=20, ax=ax5, color='orange', edgecolor='black', alpha=0.7)
            ax5.set_xlabel('Maximum Impact Duration (timespans)')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Drift Event Impact Duration', weight='bold')
            ax5.grid(axis='y', alpha=0.3)
    
    # 6. Event types breakdown
    ax6 = fig.add_subplot(gs[1, 2])
    event_types = []
    for types_str in sig_df['drift_event_types']:
        if isinstance(types_str, str) and types_str:
            event_types.extend(types_str.split(','))
    if event_types:
        event_counts = pd.Series(event_types).value_counts()
        event_counts.plot(kind='bar', ax=ax6, color='teal', edgecolor='black')
        ax6.set_xlabel('Event Type')
        ax6.set_ylabel('Count')
        ax6.set_title('Drift Event Types', weight='bold')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(axis='y', alpha=0.3)
    
    # 7. Scatter: drift magnitude vs p-value
    ax7 = fig.add_subplot(gs[2, 0])
    if 'max_drift_magnitude' in sig_df.columns and 'min_pvalue' in sig_df.columns:
        valid_data = sig_df[sig_df['min_pvalue'].notna() & sig_df['max_drift_magnitude'].notna()]
        if not valid_data.empty:
            scatter = ax7.scatter(valid_data['max_drift_magnitude'], 
                                 valid_data['min_pvalue'],
                                 c=valid_data['num_significant_pairs'], 
                                 cmap='viridis', 
                                 s=100, 
                                 alpha=0.6,
                                 edgecolors='black')
            ax7.set_xlabel('Maximum Drift Magnitude')
            ax7.set_ylabel('Minimum P-value')
            ax7.set_title('Drift Magnitude vs Significance', weight='bold')
            ax7.set_yscale('log')
            ax7.axhline(0.05, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax7.grid(alpha=0.3)
            cbar = plt.colorbar(scatter, ax=ax7)
            cbar.set_label('# Significant Pairs')
    
    # 8. Lexeme-Synset heatmap (top lexemes only)
    ax8 = fig.add_subplot(gs[2, 1:])
    top_lexemes = sig_df['lexeme'].value_counts().head(10).index
    heatmap_data = sig_df[sig_df['lexeme'].isin(top_lexemes)].pivot_table(
        index='lexeme',
        columns='synset',
        values='num_significant_pairs',
        fill_value=0
    )
    if not heatmap_data.empty:
        sns.heatmap(heatmap_data, ax=ax8, cmap='YlOrRd', annot=True, fmt='.0f', 
                   cbar_kws={'label': '# Significant Pairs'})
        ax8.set_title('Significant Pairs: Top 10 Lexemes', weight='bold')
        ax8.set_xlabel('Synset')
        ax8.set_ylabel('Lexeme')
    
    plt.savefig(Path(output_dir) / 'drift_overview_dashboard.png', dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved overview dashboard to {output_dir}/drift_overview_dashboard.png")

def plot_timeline_overview(df, output_dir):
    """Create timeline visualization showing when drift occurred."""
    sig_df = df[df['any_significant']].copy()
    
    if sig_df.empty or 'affected_timespans' not in sig_df.columns:
        logger.warning("No timeline data available")
        return
    
    # Extract all affected timespans
    timespan_counts = {}
    for idx, row in sig_df.iterrows():
        if isinstance(row['affected_timespans'], str) and row['affected_timespans']:
            timespans = row['affected_timespans'].split(',')
            for ts in timespans:
                ts = ts.strip()
                if ts:
                    timespan_counts[ts] = timespan_counts.get(ts, 0) + 1
    
    if not timespan_counts:
        logger.warning("No timespan data to visualize")
        return
    
    # Sort timespans
    sorted_timespans = sorted(timespan_counts.items(), key=lambda x: x[0])
    timespans, counts = zip(*sorted_timespans)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(timespans)), counts, color='steelblue', edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(timespans)))
    ax.set_xticklabels(timespans, rotation=45, ha='right')
    ax.set_xlabel('Timespan', fontsize=12, weight='bold')
    ax.set_ylabel('Number of Synsets Affected', fontsize=12, weight='bold')
    ax.set_title('Semantic Drift Activity Over Time', fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count}', ha='center', va='bottom', fontsize=9, weight='bold')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'drift_timeline_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved timeline overview to {output_dir}/drift_timeline_overview.png")


def plot_top_intensity(df, output_dir):
    """FIGURE 1: Top 15 Senses by Magnitude (Intensity)."""
    sig_df = df[df['any_significant']].sort_values('max_drift_magnitude', ascending=False).head(15)
    
    plt.figure(figsize=(10, 8))
    # Create a label combining lexeme and synset for clarity
    sig_df['label'] = sig_df['lexeme'] + " (" + sig_df['synset'] + ")"
    
    sns.barplot(data=sig_df, x='max_drift_magnitude', y='label', palette='rocket', hue='label')
    plt.title('Highest Magnitude Semantic Drift Events (Intensity)', weight='bold')
    plt.xlabel('Max Euclidean Drift Magnitude (Latent Space)')
    plt.ylabel('Lexical Unit (Synset)')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'scientific_top_intensity.png')
    plt.close()
    
def plot_intra_lexeme_comparison(df, output_dir):
    """FIGURE 2: Comparing different senses of the same lexeme with Log Scale and Labels."""
    # Filter to significant results
    sig_df = df[df['any_significant']].copy()
    
    # We'll focus on lexemes with multiple senses to show divergence, 
    # but the logic works for single-sense too if you prefer.
    counts = sig_df['lexeme'].value_counts()
    multi_sense = counts[counts > 1].index
    
    if multi_sense.empty:
        logger.warning("No multi-sense lexemes found for comparison.")
        return


    target_lexemes = ['bank', 'atmosphäre', 'klappe', 'blatt', 'decke', 'zeitung'] 
    comp_df = df[df['lexeme'].isin(target_lexemes) & df['any_significant']].copy()
    
    # comp_df = sig_df[sig_df['lexeme'].isin(multi_sense)]
    
    plt.figure(figsize=(6, 4))
    
    # 1. Apply Log Scale to handle the >3000 outlier
    plt.yscale('log')
    
    # 2. Main Scatter Plot
    ax =sns.scatterplot(
        data=comp_df, 
        x='num_significant_pairs', 
        y='max_drift_magnitude', 
        hue='lexeme', 
        size='max_impact_duration',
        sizes=(100, 500),
        alpha=0.6,
        palette='tab10' # High contrast for different lexemes
    )
    
    # Collect all text objects in a list
    texts = []
    for i in range(comp_df.shape[0]):
        t = plt.text(
            x=comp_df.num_significant_pairs.iloc[i], 
            y=comp_df.max_drift_magnitude.iloc[i], 
            s=f"{comp_df.lexeme.iloc[i]}\n({comp_df.synset.iloc[i]})", 
            fontsize=8
        )
        texts.append(t)
    adjust_text(texts)
    
    plt.title('Intra-Lexeme Sense Divergence (Log Scale)', weight='bold', fontsize=12)
    plt.xlabel('Frequency of Change (# Significant Pairs)', fontsize=11)
    plt.ylabel('Intensity of Change (Max Drift Magnitude)', fontsize=11)
    plt.tick_params(labelsize=9)

    ax = plt.gca() # Get current axis

    # 1. Set major ticks to show 100, 1000, etc.
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False) # Disable 1e3 notation

    # 2. Force specific major ticks if the auto-generator is too sparse
    # Adjust these numbers based on your actual data range
    ax.set_yticks([1, 5, 10, 15, 20, 25, 30, 40, 80, 100, 200])
    ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

    # 3. Add minor ticks (the small lines between major numbers)
    # In log scale, these show 200, 300, 400, etc.
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=12))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter()) # Keep minor labels hidden for cleanliness

    # 4. Turn on the grid for both major and minor ticks
    ax.grid(True, which="major", ls="-", alpha=0.3)
    ax.grid(True, which="minor", ls=":", alpha=0.1)

    # 5. Set the limits slightly wider so labels at the edges don't get cut off
    ax.set_ylim(bottom=df['max_drift_magnitude'].min() * 0.8, 
                top=df['max_drift_magnitude'].max() * 1.2)

    # 3. Create the small, clean legend
    plt.legend(
        loc='upper right', 
        bbox_to_anchor=(0.99, 0.99),
        #title="Lexeme",
        #title_fontsize='10',
        fontsize='8',           # Makes the font smaller
        markerscale=0.7,        # Makes the colored dots in the legend smaller
        frameon=True,
        framealpha=0.9,
        edgecolor='gray'
    )
    plt.legend().set_visible(False)
    
    # Move legend outside to keep the plot area clean
    #plt.legend(bbox_to_anchor=(0.98, 0.98), loc='upper right')
    
    # Add minor gridlines for the log axis to help the reader
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'scientific_sense_divergence.pdf', dpi=600)
    plt.close()

def plot_decadal_velocity(df, output_dir):
    """FIGURE 3: Continuous Decadal Velocity across the 20th Century."""
    sig_df = df[df['any_significant']].copy()
    
    all_ts = []
    for entry in sig_df['affected_timespans'].dropna():
        all_ts.extend([t.strip() for t in entry.split(',')])
    
    if not all_ts: return
    
    counts = Counter(all_ts)
    timeline = pd.DataFrame.from_dict(counts, orient='index', columns=['freq']).sort_index()
    
    plt.figure(figsize=(12, 5))
    plt.plot(timeline.index, timeline['freq'], marker='s', color='#2c3e50', linewidth=2, markersize=8)
    plt.fill_between(timeline.index, timeline['freq'], color='#34495e', alpha=0.1)
    
    plt.title('Latent Decadal Velocity: Distribution of Usage Shifts', weight='bold')
    plt.ylabel('Total Significant Drift Events')
    plt.xlabel('Decade')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'scientific_decadal_velocity.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize aggregated drift analysis results.")
    parser.add_argument('--summary_dir', default='drift_summaries')
    parser.add_argument('--output_dir', default='drift_summaries')
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_file = Path(args.summary_dir) / 'all_drift_results.csv'
    if not csv_file.exists():
        logger.error(f"Summary CSV not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded {len(df)} results. Significant: {df['any_significant'].sum()}")

    # 1. Your original Dashboard & Timeline
    logger.info("Generating drift overview...")
    plot_drift_overview(df, args.output_dir)
    
    logger.info("Generating timeline overview...")
    plot_timeline_overview(df, args.output_dir)
    
    # 2. New Scientific Paper Visualizations
    logger.info("Generating scientific visualizations...")
    plot_top_intensity(df, args.output_dir)
    plot_intra_lexeme_comparison(df, args.output_dir)
    plot_decadal_velocity(df, args.output_dir)
    
    logger.info(f"All visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
