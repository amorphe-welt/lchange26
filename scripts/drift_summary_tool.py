#!/usr/bin/env python
"""
Aggregate and summarize drift analysis results across all lexemes and synsets.
Creates a consolidated view of significant semantic changes.
"""
import os
import argparse
import logging
import pandas as pd
import json
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde, iqr

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
    
class DriftSummaryAggregator:
    """Aggregates drift analysis results across multiple experiments."""
    
    def __init__(self, base_output_dir):
        """
        Initialize aggregator.
        
        Args:
            base_output_dir: Root directory containing all drift analysis results
        """
        self.base_dir = Path(base_output_dir)
        self.summary_data = []

    def summarize_kde_by_significance(self, kde_df, output_file=None):
        """
        Summarize KDE characteristics at the synset level,
        contrasting significant vs non-significant drift.
        
        Args:
            kde_df: DataFrame returned by compute_kde_metrics()
            output_file: Optional path to save summary CSV
        """
        if kde_df.empty:
            logger.warning("KDE dataframe is empty; cannot summarize.")
            return pd.DataFrame()

        summary = (
            kde_df
            .groupby("any_significant")
            .agg(
                mean_std_drift=("std_drift", "mean"),
                median_std_drift=("std_drift", "median"),
                mean_iqr_drift=("iqr_drift", "mean"),
                median_iqr_drift=("iqr_drift", "median"),
                mean_kde_peak=("kde_peak", "mean"),
                median_kde_peak=("kde_peak", "median"),
                synset_count=("synset", "count")
            )
            .reset_index()
            .rename(columns={"any_significant": "significant_drift"})
        )

        logger.info("KDE summary by significance:")
        logger.info("\n" + summary.to_string(index=False))

        if output_file:
            summary.to_csv(output_file, index=False)
            logger.info(f"KDE significance summary saved to {output_file}")

        return summary
    
    def summarize_kde_by_timespan(self, kde_df):
        """
        Aggregate KDE metrics across timespans to detect trends.
        Args:
            kde_df: DataFrame with columns ['lexeme','synset','timespan','std_drift','iqr_drift','kde_peak']
        Returns:
            summary_df: DataFrame with mean/std of metrics per timespan
        """
        if kde_df.empty:
            logger.warning("No KDE metrics available for timespan summary")
            return pd.DataFrame()

        # Split comma-separated timespans into separate rows
        expanded_rows = []
        for _, row in kde_df.iterrows():
            ts_list = str(row['timespan']).split(',')
            for ts in ts_list:
                if ts:  # skip empty strings
                    new_row = row.copy()
                    new_row['timespan'] = ts
                    expanded_rows.append(new_row)

        expanded_df = pd.DataFrame(expanded_rows)
        expanded_df['timespan'] = expanded_df['timespan'].astype(str)

        # Aggregate metrics per timespan
        summary_df = expanded_df.groupby('timespan').agg(
            mean_std=('std_drift','mean'),
            median_std=('std_drift','median'),
            mean_iqr=('iqr_drift','mean'),
            median_iqr=('iqr_drift','median'),
            mean_kde_peak=('kde_peak','mean'),
            median_kde_peak=('kde_peak','median'),
            count=('lexeme','count')
        ).sort_index()

        return summary_df

    
    def collect_all_results(self, alpha=0.05):
        """
        Recursively collect all drift analysis results.
        
        Args:
            alpha: Significance threshold for filtering
            
        Returns:
            DataFrame with all results
        """
        logger.info(f"Scanning {self.base_dir} for drift analysis results...")
        
        # Find all lexeme directories
        for lexeme_dir in self.base_dir.iterdir():
            if not lexeme_dir.is_dir():
                continue
            
            lexeme = lexeme_dir.name
            logger.info(f"Processing lexeme: {lexeme}")
            
            # Find all synset directories within this lexeme
            for synset_dir in lexeme_dir.iterdir():
                if not synset_dir.is_dir():
                    continue
                
                synset = synset_dir.name
                self._process_synset(lexeme, synset, synset_dir, alpha)
        
        if not self.summary_data:
            logger.warning("No drift analysis results found!")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.summary_data)
        logger.info(f"Collected results for {len(df)} synsets across {df['lexeme'].nunique()} lexemes")
        return df
    
    def _process_synset(self, lexeme, synset, synset_dir, alpha):
        """Process a single synset's results."""
        result = {
            'lexeme': lexeme,
            'synset': synset,
            'has_consecutive_test': False,
            'consecutive_pvalue': None,
            'consecutive_significant': False,
            'has_pairwise_test': False,
            'num_significant_pairs': 0,
            'max_drift_magnitude': None,
            'min_pvalue': None,
            'num_drift_events': 0,
            'max_impact_duration': 0,
            'total_timespans': 0,
            'drift_event_types': [],
            'affected_timespans': [],
            'result_path': str(synset_dir)
        }
        
        # Check for pairwise test results
        sig_pairs_file = synset_dir / "significant_drift_pairs.csv"
        if sig_pairs_file.exists():
            result['has_pairwise_test'] = True
            try:
                sig_pairs = pd.read_csv(sig_pairs_file)
                if not sig_pairs.empty:
                    result['num_significant_pairs'] = len(sig_pairs)
                    result['max_drift_magnitude'] = sig_pairs['drift_magnitude'].max()
                    result['min_pvalue'] = sig_pairs['p_value'].min()
                    result['total_timespans'] = sig_pairs[['timespan_1', 'timespan_2']].nunique().sum()
            except Exception as e:
                logger.warning(f"Error reading {sig_pairs_file}: {e}")
        
        # Check for drift events
        events_file = synset_dir / "drift_events.csv"
        if events_file.exists():
            try:
                events = pd.read_csv(events_file)
                if not events.empty:
                    result['num_drift_events'] = len(events)
                    result['max_impact_duration'] = events['impact_duration'].max()
                    result['drift_event_types'] = events['event_type'].unique().tolist() if 'event_type' in events.columns else []
                    
                    # Collect all affected timespans
                    if 'affected_timespans' in events.columns:
                        all_affected = []
                        for ts_str in events['affected_timespans'].dropna():
                            all_affected.extend(str(ts_str).split(','))
                        result['affected_timespans'] = sorted(set(all_affected))
            except Exception as e:
                logger.warning(f"Error reading {events_file}: {e}")
        
        # Check for null distribution (consecutive test)
        null_dist_file = synset_dir / "null_distribution.csv"
        if null_dist_file.exists():
            result['has_consecutive_test'] = True
            # Try to extract p-value from metadata file if it exists
            metadata_file = synset_dir / "test_metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        result['consecutive_pvalue'] = metadata.get('consecutive_pvalue')
                        result['consecutive_significant'] = result['consecutive_pvalue'] < alpha if result['consecutive_pvalue'] else False
                except Exception as e:
                    logger.warning(f"Error reading {metadata_file}: {e}")
        
        # Check for sample counts
        samples_file = synset_dir / f"{synset}_samples_per_timespan.png"
        if samples_file.exists():
            # Try to get actual counts from CSV if available
            drift_csv = synset_dir / f"{synset}_drift.csv"
            if drift_csv.exists():
                try:
                    drift_data = pd.read_csv(drift_csv)
                    result['num_dimensions'] = len(drift_data)
                except:
                    pass
        
        # Determine overall significance
        result['any_significant'] = (
            result['consecutive_significant'] or 
            result['num_significant_pairs'] > 0 or
            result['num_drift_events'] > 0
        )
        
        self.summary_data.append(result)

    def compute_kde_metrics(self):
        """
        Compute KDE-based spread metrics for each synset using per-dimension drift.
        """
        records = []

        for result in self.summary_data:
            synset_dir = Path(result['result_path'])
            drift_csv = synset_dir / f"{result['synset']}_drift.csv"

            if not drift_csv.exists():
                continue

            try:
                #drift = pd.read_csv(drift_csv, header=None).values.flatten()
                #drift = drift[np.isfinite(drift)]

                df = pd.read_csv(drift_csv)
                # Flatten all numeric columns
                drift = pd.to_numeric(df.values.flatten(), errors="coerce")
                # Drop NaNs and infinities
                drift = drift[np.isfinite(drift)]


                if len(drift) < 10:
                    continue  # too small for KDE

                kde = gaussian_kde(drift)
                x = np.linspace(drift.min(), drift.max(), 1000)

                records.append({
                    "lexeme": result["lexeme"],
                    "synset": result["synset"],
                    "std_drift": np.std(drift),
                    "iqr_drift": iqr(drift),
                    "kde_peak": kde(x).max(),
                    "num_dimensions": len(drift),
                    "any_significant": result["any_significant"]
                })

            except Exception as e:
                logger.warning(f"KDE failed for {result['synset']}: {e}")

        return pd.DataFrame(records)

    
    def create_summary_report(self, df, output_file=None):
        """
        Create a comprehensive summary report.
        
        Args:
            df: DataFrame with collected results
            output_file: Optional path to save report
        """
        if df.empty:
            logger.warning("No data to summarize")
            return
        
        report = []
        report.append("=" * 80)
        report.append("DRIFT ANALYSIS SUMMARY REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall statistics
        report.append("OVERALL STATISTICS")
        report.append("-" * 80)
        report.append(f"Total lexemes analyzed: {df['lexeme'].nunique()}")
        report.append(f"Total synsets analyzed: {len(df)}")
        report.append(f"Synsets with any significant drift: {df['any_significant'].sum()}")
        report.append(f"Synsets with pairwise tests: {df['has_pairwise_test'].sum()}")
        report.append(f"Synsets with consecutive tests: {df['has_consecutive_test'].sum()}")
        report.append("")
        
        # Most significant changes
        report.append("TOP 20 SYNSETS BY SIGNIFICANCE")
        report.append("-" * 80)
        
        # Filter to significant results
        sig_df = df[df['any_significant']].copy()
        
        if not sig_df.empty:
            # Sort by multiple criteria
            sig_df['significance_score'] = (
                sig_df['num_significant_pairs'] * 10 +
                sig_df['num_drift_events'] * 5 +
                sig_df['max_impact_duration']
            )
            sig_df = sig_df.sort_values('significance_score', ascending=False)
            
            for idx, row in sig_df.head(20).iterrows():
                report.append(f"\n{row['lexeme']} / {row['synset']}")
                report.append(f"  Significant pairs: {row['num_significant_pairs']}")
                report.append(f"  Drift events: {row['num_drift_events']}")
                report.append(f"  Max impact duration: {row['max_impact_duration']}")
                if row['min_pvalue'] is not None:
                    report.append(f"  Min p-value: {row['min_pvalue']:.6f}")
                if row['max_drift_magnitude'] is not None:
                    report.append(f"  Max drift magnitude: {row['max_drift_magnitude']:.4f}")
                report.append(f"  Path: {row['result_path']}")
        else:
            report.append("No significant changes detected.")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        print(report_text)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_file}")
    
    def export_filtered_results(self, df, output_file, significance_only=True, alpha=0.05):
        """
        Export filtered results to CSV for further analysis.
        
        Args:
            df: DataFrame with results
            output_file: Path to save CSV
            significance_only: Only include significant results
            alpha: Significance threshold
        """
        if df.empty:
            logger.warning("No data to export")
            return
        
        export_df = df.copy()
        
        if significance_only:
            export_df = export_df[export_df['any_significant']]
        
        # Convert lists to strings for CSV
        if 'drift_event_types' in export_df.columns:
            export_df['drift_event_types'] = export_df['drift_event_types'].apply(
                lambda x: ','.join(x) if isinstance(x, list) else ''
            )
        if 'affected_timespans' in export_df.columns:
            export_df['affected_timespans'] = export_df['affected_timespans'].apply(
                lambda x: ','.join(map(str, x)) if isinstance(x, list) else ''
            )
        
        export_df.to_csv(output_file, index=False)
        logger.info(f"Exported {len(export_df)} results to {output_file}")
    
    def create_comparison_matrix(self, df, output_file=None):
        """
        Create a matrix comparing drift across lexemes and synsets.
        
        Args:
            df: DataFrame with results
            output_file: Optional path to save matrix
        """
        if df.empty:
            logger.warning("No data for comparison matrix")
            return None
        
        # Create pivot table
        matrix = df.pivot_table(
            index='lexeme',
            columns='synset',
            values='num_significant_pairs',
            fill_value=0
        )
        
        if output_file:
            matrix.to_csv(output_file)
            logger.info(f"Comparison matrix saved to {output_file}")
        
        return matrix


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate drift analysis results across all lexemes and synsets."
    )
    parser.add_argument(
        '--input_dir',
        default='figures',
        help='Base directory containing drift analysis results (default: figures)'
    )
    parser.add_argument(
        '--output_dir',
        default='drift_summaries',
        help='Directory to save summary reports (default: drift_summaries)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance threshold (default: 0.05)'
    )
    parser.add_argument(
        '--all_results',
        action='store_true',
        help='Include non-significant results in exports'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize aggregator
    aggregator = DriftSummaryAggregator(args.input_dir)
    
    # Collect all results
    logger.info("Collecting drift analysis results...")
    df = aggregator.collect_all_results(alpha=args.alpha)
    
    if df.empty:
        logger.error("No results found. Have you run drift analysis yet?")
        return
    
    # Create summary report
    report_file = os.path.join(args.output_dir, "drift_summary_report.txt")
    aggregator.create_summary_report(df, output_file=report_file)
    
    # Export detailed results
    csv_file = os.path.join(args.output_dir, "all_drift_results.csv")
    aggregator.export_filtered_results(
        df, 
        csv_file, 
        significance_only=not args.all_results,
        alpha=args.alpha
    )
    
    # Create comparison matrix
    matrix_file = os.path.join(args.output_dir, "lexeme_synset_matrix.csv")
    aggregator.create_comparison_matrix(df, output_file=matrix_file)
    
    # Additional statistics
    sig_df = df[df['any_significant']]
    
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Lexemes with significant drift: {sig_df['lexeme'].nunique()}/{df['lexeme'].nunique()}")
    logger.info(f"Synsets with significant drift: {len(sig_df)}/{len(df)}")
    
    if not sig_df.empty:
        logger.info(f"\nTop 5 lexemes by number of significant synsets:")
        lexeme_counts = sig_df['lexeme'].value_counts().head(5)
        for lexeme, count in lexeme_counts.items():
            logger.info(f"  {lexeme}: {count} synsets")
        
        if 'max_drift_magnitude' in sig_df.columns:
            logger.info(f"\nDrift magnitude statistics (significant changes):")
            logger.info(f"  Mean: {sig_df['max_drift_magnitude'].mean():.4f}")
            logger.info(f"  Median: {sig_df['max_drift_magnitude'].median():.4f}")
            logger.info(f"  Max: {sig_df['max_drift_magnitude'].max():.4f}")
    
    kde_df = aggregator.compute_kde_metrics()

    if not kde_df.empty:
        kde_csv = os.path.join(args.output_dir, "kde_summary_metrics.csv")
        kde_df.to_csv(kde_csv, index=False)
        logger.info(f"KDE metrics saved to {kde_csv}")

        aggregator.summarize_kde_by_significance(
            kde_df,
            output_file=os.path.join(args.output_dir, "kde_significance_summary.csv")
        )


    logger.info("\n" + "=" * 80)
    logger.info(f"All results saved to: {args.output_dir}")
    logger.info("=" * 80)
    
if __name__ == "__main__":
    main()
