#!/usr/bin/env python
"""
Embedding drift analysis per synset over timespans.
Analyzes semantic drift with pairwise testing and event detection.
"""
import os
import argparse
import logging
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, entropy

from src.dataset import load_jsonl
from src.embedding_store import EmbeddingStore
from src.drift_visualizer import DriftVisualizer

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ensure_dir_exists(path: str):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def permutation_test_drift(embeddings, timespans, n_iter=1000, random_state=None):
    """
    Test if drift between consecutive timespans is significantly larger than expected by chance.
    
    Returns:
        observed_magnitude: Mean drift magnitude observed
        null_drifts: Array of null distribution values
        p_value: Statistical significance
    """
    if not embeddings:
        logger.warning("No embeddings provided for permutation test.")
        return None, None, None
    
    rng = np.random.default_rng(random_state)
    
    ids = list(embeddings.keys())
    # X = np.stack([embeddings[i] for i in ids])
    X = np.stack([embeddings[i] for i in ids]).astype(np.float64)
    ts = np.array([timespans[i] for i in ids])
    
    unique_ts = sorted(set(ts))
    mean_per_ts = np.array([X[ts == t].mean(axis=0) for t in unique_ts])
    
    # Observed drift between consecutive timespans
    #observed_drift = mean_per_ts[1:] - mean_per_ts[:-1]
    #observed_magnitude = np.linalg.norm(observed_drift, axis=1).mean()
    
    drift = mean_per_ts[1:] - mean_per_ts[:-1]
    observed_magnitude = np.mean(np.sqrt(np.sum(drift * drift, axis=1, dtype=np.float64)))
    
    logger.info(f"Observed mean drift magnitude: {observed_magnitude:.4f}")
    
    # Null distribution via permutation
    null_drifts = []
    for _ in range(n_iter):
        shuffled_ts = rng.permutation(ts)
        mean_per_ts_shuff = np.array([X[shuffled_ts == t].mean(axis=0) for t in unique_ts])
        
        #drift_shuff = mean_per_ts_shuff[1:] - mean_per_ts_shuff[:-1]
        #drift_mag_shuff = np.linalg.norm(drift_shuff, axis=1).mean()
        
        drift = mean_per_ts_shuff[1:] - mean_per_ts_shuff[:-1]
        drift_mag_shuff = np.mean(
            np.sqrt(np.sum(drift * drift, axis=1, dtype=np.float64))
        )
        
        null_drifts.append(drift_mag_shuff)
    
    null_drifts = np.array(null_drifts)
    p_value = np.mean(null_drifts >= observed_magnitude)
    logger.info(f"Permutation test p-value: {p_value:.4f}")
    
    return observed_magnitude, null_drifts, p_value


def pairwise_drift_test(embeddings, timespans, n_iter=1000, random_state=None):
    """
    Test drift significance between ALL pairs of timespans.
    
    Returns:
        drift_matrix: Matrix of drift magnitudes
        p_value_matrix: Matrix of p-values
        unique_ts: List of timespan labels
    """
    if not embeddings:
        logger.warning("No embeddings provided for pairwise drift test.")
        return None, None, None
    
    rng = np.random.default_rng(random_state)
    
    ids = list(embeddings.keys())
    X = np.stack([embeddings[i] for i in ids])
    ts = np.array([timespans[i] for i in ids])
    
    unique_ts = sorted(set(ts))
    n_ts = len(unique_ts)
    
    # Precompute mean embeddings
    #mean_per_ts = {t: X[ts == t].mean(axis=0) for t in unique_ts}
    
    mean_per_ts = {
        t: X[ts == t].mean(axis=0).astype(np.float64)
        for t in unique_ts
    }
    
    p_value_matrix = np.zeros((n_ts, n_ts))
    drift_matrix = np.zeros((n_ts, n_ts))
    
    # Test all pairs
    for i, ts1 in enumerate(unique_ts):
        for j, ts2 in enumerate(unique_ts):
            if i >= j:
                continue
            
            #observed_drift = np.linalg.norm(mean_per_ts[ts2] - mean_per_ts[ts1])
            diff = mean_per_ts[ts2] - mean_per_ts[ts1]
            observed_drift = np.sqrt(np.sum(diff * diff, dtype=np.float64))
            
            
            drift_matrix[i, j] = observed_drift
            drift_matrix[j, i] = observed_drift
            
            # Null distribution for this pair
            mask = (ts == ts1) | (ts == ts2)
            X_pair = X[mask]
            ts_pair = ts[mask]
            
            null_drifts = []
            for _ in range(n_iter):
                shuffled_ts = rng.permutation(ts_pair)
                
                #mean1 = X_pair[shuffled_ts == ts1].mean(axis=0)
                #mean2 = X_pair[shuffled_ts == ts2].mean(axis=0)
                #null_drift = np.linalg.norm(mean2 - mean1)
                mean1 = X_pair[shuffled_ts == ts1].mean(axis=0).astype(np.float64)
                mean2 = X_pair[shuffled_ts == ts2].mean(axis=0).astype(np.float64)
                diff = mean2 - mean1
                null_drift = np.sqrt(np.sum(diff * diff, dtype=np.float64))

                null_drifts.append(null_drift)
            
            null_drifts = np.array(null_drifts)
            p_value = np.mean(null_drifts >= observed_drift)
            p_value_matrix[i, j] = p_value
            p_value_matrix[j, i] = p_value
            
            logger.info(f"  {ts1} vs {ts2}: drift={observed_drift:.4f}, p={p_value:.4f}")
    
    return drift_matrix, p_value_matrix, unique_ts


def identify_drift_events(drift_matrix, p_value_matrix, timespans, alpha=0.05):
    """
    Identify when drift events occurred and their temporal impact.
    
    Returns:
        drift_events: List of detected drift events with timing and impact
        event_matrix: Matrix marking transitions with lasting impact
    """
    n = len(timespans)
    event_matrix = np.zeros((n-1, n))
    drift_events = []
    
    # Check consecutive transitions
    for transition_idx in range(n - 1):
        before_idx = transition_idx
        after_idx = transition_idx + 1
        
        if p_value_matrix[before_idx, after_idx] < alpha:
            event = {
                'event_id': len(drift_events),
                'transition_from': timespans[before_idx],
                'transition_to': timespans[after_idx],
                'transition_idx': transition_idx,
                'immediate_drift': drift_matrix[before_idx, after_idx],
                'immediate_pvalue': p_value_matrix[before_idx, after_idx],
                'impact_duration': 0,
                'affected_timespans': []
            }
            
            # Measure impact duration
            for future_idx in range(after_idx, n):
                if p_value_matrix[before_idx, future_idx] < alpha:
                    duration = future_idx - after_idx + 1
                    event['impact_duration'] = duration
                    event['affected_timespans'].append(timespans[future_idx])
                    event_matrix[transition_idx, future_idx] = 1
                else:
                    break
            
            drift_events.append(event)
    
    # Detect sudden shifts (non-gradual)
    for i in range(n):
        for j in range(i + 2, n):
            if p_value_matrix[i, j] < alpha:
                intermediate_significant = any(
                    p_value_matrix[k, k+1] < alpha for k in range(i, j)
                )
                
                if not intermediate_significant:
                    event = {
                        'event_id': len(drift_events),
                        'transition_from': timespans[i],
                        'transition_to': timespans[j],
                        'transition_idx': i,
                        'immediate_drift': drift_matrix[i, j],
                        'immediate_pvalue': p_value_matrix[i, j],
                        'impact_duration': j - i,
                        'affected_timespans': [timespans[k] for k in range(i+1, j+1)],
                        'event_type': 'sudden_shift'
                    }
                    drift_events.append(event)
    
    return drift_events, event_matrix


def analyze_synset_drift(synset, embeddings, timespans, lexeme, output_root="figures",
                         exclude_timespans=None, window=1):
    """
    Perform comprehensive drift analysis for a single synset.
    
    Args:
        synset: Synset identifier
        embeddings: Dict of embeddings by ID
        timespans: Dict of timespans by ID
        lexeme: Lexeme being analyzed
        output_root: Base output directory
        exclude_timespans: List of timespans to exclude
        window: Sliding window size
    """
    if synset == "None":
        logger.info(f"Skipping synset 'None'")
        return
    
    if exclude_timespans is None:
        exclude_timespans = []

    synset_safe = str(synset)
    synset_dir = os.path.join(output_root, lexeme, synset_safe)
    ensure_dir_exists(synset_dir)
    
    # Initialize visualizer
    viz = DriftVisualizer(synset_dir, synset_safe, lexeme)

    ids = list(embeddings.keys())
    #X = np.stack([embeddings[i] for i in ids])
    X = np.stack([embeddings[i] for i in ids]).astype(np.float64)
    ts = np.array([timespans[i] for i in ids])

    # Filter excluded timespans
    mask = np.array([t not in exclude_timespans for t in ts])
    X = X[mask]
    ts = ts[mask]

    timespans_unique = sorted(set(ts))
    if len(timespans_unique) < 2:
        logger.warning(f"Not enough timespans for synset {synset_safe}, skipping.")
        return

    dim = X.shape[1]

    # Compute mean embeddings per timespan
    mean_per_timespan = np.stack([X[ts == t].mean(axis=0) for t in timespans_unique])

    # Sliding window drift
    drift_window = np.stack([
        mean_per_timespan[i + window] - mean_per_timespan[i]
        for i in range(len(timespans_unique) - window)
    ])

    # Visualizations
    viz.plot_drift_heatmap(drift_window, window)
    
    #drift_magnitude = np.linalg.norm(drift_window, axis=0)
    drift_magnitude = np.sqrt(
        np.sum(drift_window * drift_window, axis=0, dtype=np.float64)
    )
    
    viz.plot_drift_distribution(drift_magnitude)
    
    # Save drift magnitude CSV
    drift_df = pd.DataFrame({"embedding_dim": np.arange(dim), "drift_magnitude": drift_magnitude})
    drift_df.to_csv(os.path.join(synset_dir, f"{synset_safe}_drift.csv"), index=False)

    # Drift distributions per timespan
    drift_dists = []
    drift_labels = []
    for i in range(1, len(timespans_unique)):
        prev = X[ts == timespans_unique[i - 1]]
        curr = X[ts == timespans_unique[i]]
        diff = curr.mean(axis=0) - prev.mean(axis=0)
        drift_dists.append(diff)
        drift_labels.append(f"{timespans_unique[i]}")

    viz.plot_drift_overlay(drift_dists, drift_labels)
    
    # Save distributions
    dist_df = pd.DataFrame({label: drift_dists[i] for i, label in enumerate(drift_labels)})
    dist_df.to_csv(os.path.join(synset_dir, f"{synset_safe}_drift_distributions.csv"), index=False)

    # Wasserstein + KL divergence
    div_rows = []
    for i in range(1, len(drift_dists)):
        p = drift_dists[i - 1]
        q = drift_dists[i]
        hist_p, bins = np.histogram(p, bins=100, density=True)
        hist_q, _ = np.histogram(q, bins=bins, density=True)
        hist_p += 1e-12
        hist_q += 1e-12

        w = wasserstein_distance(p, q)
        kl = entropy(hist_q, hist_p)
        div_rows.append({
            "synset": synset,
            "lexeme": lexeme,
            "from_timespan": timespans_unique[i - 1],
            "to_timespan": timespans_unique[i],
            "wasserstein": w,
            "kl_divergence": kl
        })

    div_df = pd.DataFrame(div_rows)
    div_df.to_csv(os.path.join(synset_dir, f"{synset_safe}_distribution_divergence.csv"), index=False)
    viz.plot_divergence(div_df)

    # Sample counts
    viz.plot_samples_per_timespan(ts)

    # Enhanced distribution visualizations
    viz.plot_kde_cdf_combined(drift_dists, drift_labels)
    viz.plot_extreme_dimensions_analysis(drift_dists, drift_labels)
    viz.plot_ridgeline(drift_dists, drift_labels)
    viz.plot_box_violin(drift_dists, drift_labels)
    viz.plot_quantiles(drift_dists, drift_labels)

    logger.info(f"Completed drift analysis for synset {synset_safe}")


def main():
    parser = argparse.ArgumentParser(description="Embedding drift analysis per synset over timespans.")
    parser.add_argument('-d', '--dataset', required=True, help='Path to dataset (JSONL).')
    parser.add_argument('-l', '--lexeme', required=True, help='Lexeme to filter.')
    parser.add_argument('-s', '--synsets', required=False, help='Comma-separated synsets to analyze.')
    parser.add_argument('--emb_h5', required=True, help='HDF5 file containing embeddings.')
    parser.add_argument('--window', type=int, default=1, help='Sliding window size (default 1).')
    parser.add_argument('--output_dir', default='figures', help='Root output directory.')
    parser.add_argument('--exclude_timespans', nargs='*', default=[], help='Timespans to skip.')
    parser.add_argument('--pairwise_test', action='store_true', help='Perform pairwise drift tests.')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level (default 0.05).')
    args = parser.parse_args()

    logger.info(f"Loading dataset from {args.dataset}")
    samples = list(load_jsonl(args.dataset, lexem_filter=args.lexeme))
    if not samples:
        logger.error("No samples found.")
        return

    ids = [s['id'] for s in samples]
    synset_by_id = {s['id']: s['synset'] for s in samples}
    timespan_by_id = {s['id']: s['timespan'] for s in samples}

    logger.info("Loading embeddings")
    with EmbeddingStore(args.emb_h5) as store:
        all_embeddings = store.load(ids)

    # Determine synsets to analyze
    synsets = set(synset_by_id.values()) - {"None"}
    if args.synsets:
        synsets &= set(args.synsets.split(','))

    for synset in sorted(synsets):
        emb_subset = {sid: all_embeddings[sid] for sid in ids 
                     if synset_by_id[sid] == synset and sid in all_embeddings}
        ts_subset = {sid: timespan_by_id[sid] for sid in emb_subset}
        
        # Filter excluded timespans
        if args.exclude_timespans:
            emb_subset = {sid: emb for sid, emb in emb_subset.items() 
                         if ts_subset[sid] not in args.exclude_timespans}
            ts_subset = {sid: ts for sid, ts in ts_subset.items() if sid in emb_subset}
        
        if not emb_subset:
            logger.warning(f"No embeddings for synset {synset} after filtering, skipping.")
            continue
        
        synset_dir = os.path.join(args.output_dir, args.lexeme, synset)
        ensure_dir_exists(synset_dir)
        viz = DriftVisualizer(synset_dir, synset, args.lexeme)
        
        # Pairwise drift test
        if args.pairwise_test:
            logger.info(f"Performing pairwise drift tests for synset {synset}...")
            drift_matrix, p_value_matrix, unique_timespans = pairwise_drift_test(
                emb_subset, ts_subset, n_iter=1000, random_state=42
            )
            
            if drift_matrix is not None:
                # Save matrices
                pd.DataFrame(drift_matrix, index=unique_timespans, columns=unique_timespans).to_csv(
                    os.path.join(synset_dir, "pairwise_drift_matrix.csv"))
                pd.DataFrame(p_value_matrix, index=unique_timespans, columns=unique_timespans).to_csv(
                    os.path.join(synset_dir, "pairwise_pvalue_matrix.csv"))
                
                # Visualizations
                viz.plot_pairwise_heatmap(drift_matrix, p_value_matrix, unique_timespans, args.alpha)
                viz.plot_drift_connections(drift_matrix, p_value_matrix, unique_timespans, args.alpha)
                
                # Significant pairs
                n = len(unique_timespans)
                sig_pairs = [
                    {
                        'timespan_1': unique_timespans[i],
                        'timespan_2': unique_timespans[j],
                        'distance': j - i,
                        'drift_magnitude': drift_matrix[i, j],
                        'p_value': p_value_matrix[i, j]
                    }
                    for i in range(n) for j in range(i + 1, n)
                    if p_value_matrix[i, j] < args.alpha
                ]
                
                if sig_pairs:
                    sig_df = pd.DataFrame(sig_pairs).sort_values('p_value')
                    sig_df.to_csv(os.path.join(synset_dir, "significant_drift_pairs.csv"), index=False)
                    logger.info(f"Found {len(sig_pairs)} significant drift pairs for synset {synset}")
                    
                    # Identify drift events
                    logger.info(f"Identifying drift events for synset {synset}...")
                    drift_events, event_matrix = identify_drift_events(
                        drift_matrix, p_value_matrix, unique_timespans, args.alpha
                    )
                    
                    if drift_events:
                        events_df = pd.DataFrame([
                            {
                                'event_id': e['event_id'],
                                'transition_from': e['transition_from'],
                                'transition_to': e['transition_to'],
                                'drift_magnitude': e['immediate_drift'],
                                'p_value': e['immediate_pvalue'],
                                'impact_duration': e['impact_duration'],
                                'affected_timespans': ','.join(map(str, e['affected_timespans'])),
                                'event_type': e.get('event_type', 'consecutive')
                            }
                            for e in drift_events
                        ])
                        events_df.to_csv(os.path.join(synset_dir, "drift_events.csv"), index=False)
                        logger.info(f"Identified {len(drift_events)} drift events")
                        
                        for event in drift_events:
                            logger.info(f"  Event {event['event_id']}: "
                                      f"{event['transition_from']}→{event['transition_to']}, "
                                      f"drift={event['immediate_drift']:.4f}, "
                                      f"impact_duration={event['impact_duration']}")
                        
                        viz.plot_event_timeline(drift_events, drift_matrix, p_value_matrix,
                                              unique_timespans, args.alpha)
                    else:
                        logger.info(f"No drift events detected for synset {synset}")
                else:
                    logger.info(f"No significant drift pairs found for synset {synset}")
        
        # Consecutive permutation test
        obs_mag, null_dist, p_val = permutation_test_drift(emb_subset, ts_subset, n_iter=1000, random_state=42)
        logger.info(f"Consecutive permutation test for synset {synset}: p-value={p_val:.4f}")
        
        null_df = pd.DataFrame({"null_drift_magnitude": null_dist})
        null_df.to_csv(os.path.join(synset_dir, "null_distribution.csv"), index=False)
        viz.plot_null_distribution(null_dist, obs_mag, p_val)
        
        # Save test metadata
        metadata = {
            'lexeme': args.lexeme,
            'synset': synset,
            'consecutive_pvalue': float(p_val) if p_val is not None else None,
            'observed_magnitude': float(obs_mag) if obs_mag is not None else None,
            'n_permutations': 1000,
            'alpha': args.alpha,
            'window_size': args.window,
            'excluded_timespans': args.exclude_timespans,
            'n_samples': len(emb_subset),
            'n_timespans': len(set(ts_subset.values()))
        }
        
        import json
        with open(os.path.join(synset_dir, "test_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved test metadata to {synset_dir}/test_metadata.json")
        
        # Standard drift analysis
        analyze_synset_drift(synset, emb_subset, ts_subset, args.lexeme, args.output_dir,
                            args.exclude_timespans, args.window)

    logger.info("Embedding drift analysis completed.")


if __name__ == "__main__":
    main()
