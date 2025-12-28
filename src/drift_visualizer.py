"""
Visualization utilities for embedding drift analysis.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import logging

logger = logging.getLogger(__name__)


class DriftVisualizer:
    """Handles all visualization tasks for drift analysis."""
    
    def __init__(self, output_dir, synset, lexeme):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Base output directory for plots
            synset: Synset identifier
            lexeme: Lexeme being analyzed
        """
        self.output_dir = output_dir
        self.synset = str(synset)
        self.lexeme = lexeme
        self._ensure_dir_exists(output_dir)
    
    @staticmethod
    def _ensure_dir_exists(path):
        """Create directory if it doesn't exist."""
        if not os.path.exists(path):
            os.makedirs(path)
    
    def _save_plot(self, filename, dpi=150):
        """Save plot and close figure."""
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=dpi)
        plt.close()
        logger.info(f"Saved plot to {filepath}")
    
    def plot_drift_heatmap(self, drift_window, window=1):
        """Plot heatmap of drift per dimension over time."""
        plt.figure(figsize=(12, 6))
        plt.imshow(
            drift_window.T,
            aspect="auto",
            cmap="bwr",
            vmin=-np.max(np.abs(drift_window)),
            vmax=np.max(np.abs(drift_window)),
        )
        plt.colorbar(label=f"Change vs previous {window} timespan(s)")
        plt.xlabel("Sliding window step")
        plt.ylabel("Embedding dimension")
        plt.title(f"Sliding Window Drift per Dimension – {self.lexeme} / {self.synset}")
        self._save_plot(f"{self.synset}_drift_heatmap.png")
    
    def plot_drift_distribution(self, drift_magnitude):
        """Plot histogram of drift magnitude distribution."""
        plt.figure(figsize=(10, 4))
        plt.hist(drift_magnitude, bins=50, color="skyblue", edgecolor="k")
        plt.xlabel("Drift magnitude")
        plt.ylabel("Number of dimensions")
        plt.title(f"Distribution of embedding drift magnitude – {self.lexeme} / {self.synset}")
        self._save_plot(f"{self.synset}_drift_distribution.png")
    
    def plot_drift_overlay(self, drift_dists, drift_labels):
        """Plot overlay of drift distributions across timespans."""
        plt.figure(figsize=(10, 5))
        for i, label in enumerate(drift_labels):
            plt.plot(drift_dists[i], label=label, alpha=0.7)
        plt.xlabel("Embedding dimension")
        plt.ylabel("Drift per dimension")
        plt.title(f"Drift distributions – {self.lexeme} / {self.synset}")
        plt.legend()
        self._save_plot(f"{self.synset}_drift_distributions.png")
    
    def plot_divergence(self, div_df):
        """Plot Wasserstein and KL divergence over time."""
        plt.figure(figsize=(8, 4))
        x = range(len(div_df))
        plt.plot(x, div_df["wasserstein"], marker="o", label="Wasserstein")
        plt.plot(x, div_df["kl_divergence"], marker="s", label="KL divergence")
        plt.xticks(x, div_df["to_timespan"], rotation=45)
        plt.xlabel("Timespan")
        plt.ylabel("Divergence")
        plt.title(f"Distribution shift – {self.lexeme} / {self.synset}")
        plt.legend()
        self._save_plot(f"{self.synset}_distribution_divergence.png")
    
    def plot_samples_per_timespan(self, timespans):
        """Plot bar chart of sample counts per timespan."""
        timespan_counts = pd.Series(timespans).value_counts().sort_index()
        plt.figure(figsize=(10, 6))
        timespan_counts.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.xlabel("Timespan")
        plt.ylabel("Number of samples")
        plt.title(f"Samples per Timespan for {self.lexeme} / {self.synset}")
        self._save_plot(f"{self.synset}_samples_per_timespan.png")
    
    def plot_null_distribution(self, null_dist, observed_mag, p_value):
        """Plot null distribution with observed drift."""
        plt.figure(figsize=(10, 6))
        plt.hist(null_dist, bins=50, alpha=0.7, color='gray', label='Null distribution')
        plt.axvline(observed_mag, color='red', linestyle='dashed', linewidth=2, 
                   label=f'Observed drift (p-value={p_value:.4f})')
        plt.xlabel("Drift Magnitude")
        plt.ylabel("Frequency")
        plt.title(f"Null distribution vs. Observed Drift for Synset {self.synset}")
        plt.legend()
        self._save_plot("null_distribution_plot.png")
    
    def plot_kde_cdf_combined(self, drift_dists, drift_labels):
        """Plot combined KDE and CDF with improved scale for small drifts."""
        colors = plt.cm.viridis(np.linspace(0, 1, len(drift_dists)))
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Standard KDE (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        for i, (drift_dist, label) in enumerate(zip(drift_dists, drift_labels)):
            sns.kdeplot(drift_dist, ax=ax1, label=label, color=colors[i], 
                       linewidth=2.5, fill=True, alpha=0.3)
        ax1.set_xlabel("Drift per Dimension", fontsize=11, weight='bold')
        ax1.set_ylabel("Density", fontsize=11, weight='bold')
        ax1.set_title("KDE: Full Distribution", fontsize=12, weight='bold')
        ax1.legend(title="To Timespan", fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # 2. Zoomed KDE around 0 (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        for i, (drift_dist, label) in enumerate(zip(drift_dists, drift_labels)):
            sns.kdeplot(drift_dist, ax=ax2, label=label, color=colors[i], 
                       linewidth=2.5, fill=True, alpha=0.3)
        # Zoom to central region
        all_data = np.concatenate(drift_dists)
        p25, p75 = np.percentile(all_data, [25, 75])
        iqr = p75 - p25
        zoom_range = max(iqr * 2, 0.1)  # At least 0.1 range
        ax2.set_xlim(-zoom_range, zoom_range)
        ax2.set_xlabel("Drift per Dimension (zoomed)", fontsize=11, weight='bold')
        ax2.set_ylabel("Density", fontsize=11, weight='bold')
        ax2.set_title("KDE: Zoomed to Central Region", fontsize=12, weight='bold')
        ax2.legend(title="To Timespan", fontsize=9, loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # 3. Log-scale KDE for tails (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        for i, (drift_dist, label) in enumerate(zip(drift_dists, drift_labels)):
            # Separate positive and negative
            pos_drift = drift_dist[drift_dist > 1e-6]
            neg_drift = -drift_dist[drift_dist < -1e-6]
            
            if len(pos_drift) > 0:
                sns.kdeplot(pos_drift, ax=ax3, label=f"{label} (pos)", color=colors[i], 
                           linewidth=2, alpha=0.7)
            if len(neg_drift) > 0:
                sns.kdeplot(neg_drift, ax=ax3, color=colors[i], 
                           linewidth=2, alpha=0.7, linestyle='--')
        
        ax3.set_xscale('log')
        ax3.set_xlabel("Absolute Drift (log scale)", fontsize=11, weight='bold')
        ax3.set_ylabel("Density", fontsize=11, weight='bold')
        ax3.set_title("KDE: Tail Behavior (Log Scale)", fontsize=12, weight='bold')
        ax3.legend(fontsize=8, loc='best')
        ax3.grid(True, alpha=0.3, linestyle='--', which='both')
        
        # 4. Standard CDF (middle right)
        ax4 = fig.add_subplot(gs[1, 1])
        for i, (drift_dist, label) in enumerate(zip(drift_dists, drift_labels)):
            sorted_data = np.sort(drift_dist)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax4.plot(sorted_data, cdf, label=label, color=colors[i], linewidth=2.5, 
                    marker='o', markevery=len(sorted_data)//20, markersize=4)
        ax4.set_xlabel("Drift per Dimension", fontsize=11, weight='bold')
        ax4.set_ylabel("Cumulative Probability", fontsize=11, weight='bold')
        ax4.set_title("CDF: Full Distribution", fontsize=12, weight='bold')
        ax4.legend(title="To Timespan", fontsize=9, loc='best')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax4.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        # 5. Complementary CDF (focusing on tails) (bottom left)
        ax5 = fig.add_subplot(gs[2, 0])
        for i, (drift_dist, label) in enumerate(zip(drift_dists, drift_labels)):
            sorted_data = np.sort(np.abs(drift_dist))
            ccdf = 1 - (np.arange(1, len(sorted_data) + 1) / len(sorted_data))
            # Only plot where ccdf > 0.01 to focus on tails
            mask = ccdf > 0.01
            ax5.plot(sorted_data[mask], ccdf[mask], label=label, color=colors[i], 
                    linewidth=2.5, marker='s', markevery=max(1, len(sorted_data[mask])//15), markersize=4)
        ax5.set_xlabel("Absolute Drift", fontsize=11, weight='bold')
        ax5.set_ylabel("P(|Drift| > x)", fontsize=11, weight='bold')
        ax5.set_title("Complementary CDF: Tail Probabilities", fontsize=12, weight='bold')
        ax5.set_yscale('log')
        ax5.legend(title="To Timespan", fontsize=9, loc='best')
        ax5.grid(True, alpha=0.3, linestyle='--', which='both')
        
        # 6. Quantile-Quantile comparison (bottom right)
        ax6 = fig.add_subplot(gs[2, 1])
        if len(drift_dists) > 1:
            # Compare each timespan to the first one
            base_dist = np.sort(drift_dists[0])
            base_quantiles = np.linspace(0, 1, len(base_dist))
            
            for i in range(1, len(drift_dists)):
                comp_dist = np.sort(drift_dists[i])
                comp_quantiles = np.linspace(0, 1, len(comp_dist))
                
                # Interpolate to match lengths
                comp_interp = np.interp(base_quantiles, comp_quantiles, comp_dist)
                
                ax6.scatter(base_dist, comp_interp, alpha=0.5, s=20, 
                           color=colors[i], label=f"{drift_labels[0]} vs {drift_labels[i]}")
            
            # Add diagonal reference line
            lims = [
                np.min([ax6.get_xlim(), ax6.get_ylim()]),
                np.max([ax6.get_xlim(), ax6.get_ylim()]),
            ]
            ax6.plot(lims, lims, 'r--', alpha=0.75, zorder=0, linewidth=2)
            ax6.set_xlabel(f"Drift Quantiles: {drift_labels[0]}", fontsize=11, weight='bold')
            ax6.set_ylabel("Drift Quantiles: Other Timespans", fontsize=11, weight='bold')
            ax6.set_title("Q-Q Plot: Distribution Comparison", fontsize=12, weight='bold')
            ax6.legend(fontsize=9, loc='best')
            ax6.grid(True, alpha=0.3, linestyle='--')
            ax6.set_aspect('equal')
        else:
            ax6.text(0.5, 0.5, 'Need 2+ timespans for Q-Q plot', 
                    ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        
        self._save_plot(f"{self.synset}_{self.lexeme}_kde_cdf_combined.png", dpi=200)
    
    def plot_extreme_dimensions_analysis(self, drift_dists, drift_labels):
        """Analyze and visualize the most extreme drifting dimensions."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        colors = plt.cm.viridis(np.linspace(0, 1, len(drift_dists)))
        
        # 1. Top drifting dimensions per timespan
        ax1 = axes[0, 0]
        top_k = 20
        for i, (drift_dist, label) in enumerate(zip(drift_dists, drift_labels)):
            top_indices = np.argsort(np.abs(drift_dist))[-top_k:][::-1]
            top_values = drift_dist[top_indices]
            
            x_offset = i * (top_k + 2)
            ax1.bar(x_offset + np.arange(top_k), top_values, color=colors[i], 
                   alpha=0.7, label=label, width=0.8)
        
        ax1.set_xlabel("Dimension Rank (within timespan)", fontsize=11, weight='bold')
        ax1.set_ylabel("Drift Magnitude", fontsize=11, weight='bold')
        ax1.set_title(f"Top {top_k} Drifting Dimensions per Timespan", fontsize=12, weight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(0, color='red', linestyle='--', linewidth=1)
        
        # 2. Distribution of extreme values
        ax2 = axes[0, 1]
        for i, (drift_dist, label) in enumerate(zip(drift_dists, drift_labels)):
            abs_drift = np.abs(drift_dist)
            p90, p95, p99 = np.percentile(abs_drift, [90, 95, 99])
            
            ax2.bar(i, p90, color=colors[i], alpha=0.3, label='90th %ile' if i == 0 else '')
            ax2.bar(i, p95, color=colors[i], alpha=0.6, label='95th %ile' if i == 0 else '')
            ax2.bar(i, p99, color=colors[i], alpha=0.9, label='99th %ile' if i == 0 else '')
            
            # Add text labels
            ax2.text(i, p99, f'{p99:.3f}', ha='center', va='bottom', fontsize=8, weight='bold')
        
        ax2.set_xticks(range(len(drift_labels)))
        ax2.set_xticklabels(drift_labels, rotation=45, ha='right')
        ax2.set_ylabel("Absolute Drift", fontsize=11, weight='bold')
        ax2.set_title("Extreme Drift Percentiles by Timespan", fontsize=12, weight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Proportion of dimensions with "significant" drift
        ax3 = axes[1, 0]
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
        width = 0.15
        
        for t_idx, thresh in enumerate(thresholds):
            proportions = []
            for drift_dist in drift_dists:
                prop = np.mean(np.abs(drift_dist) > thresh)
                proportions.append(prop * 100)  # Convert to percentage
            
            x_pos = np.arange(len(drift_labels))
            offset = (t_idx - len(thresholds)/2) * width
            ax3.bar(x_pos + offset, proportions, width, alpha=0.8, 
                   label=f'|drift| > {thresh}')
        
        ax3.set_xticks(range(len(drift_labels)))
        ax3.set_xticklabels(drift_labels, rotation=45, ha='right')
        ax3.set_ylabel("% of Dimensions", fontsize=11, weight='bold')
        ax3.set_title("Proportion of Dimensions Exceeding Thresholds", fontsize=12, weight='bold')
        ax3.legend(fontsize=8, ncol=2)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Spread metrics over time
        ax4 = axes[1, 1]
        x_pos = np.arange(len(drift_labels))
        
        metrics = {
            'Std Dev': [np.std(d) for d in drift_dists],
            'IQR': [np.percentile(np.abs(d), 75) - np.percentile(np.abs(d), 25) for d in drift_dists],
            'MAD': [np.median(np.abs(d - np.median(d))) for d in drift_dists],
            'Max': [np.max(np.abs(d)) for d in drift_dists]
        }
        
        width = 0.2
        for m_idx, (metric_name, values) in enumerate(metrics.items()):
            offset = (m_idx - len(metrics)/2) * width
            ax4.bar(x_pos + offset, values, width, alpha=0.8, label=metric_name)
        
        ax4.set_xticks(range(len(drift_labels)))
        ax4.set_xticklabels(drift_labels, rotation=45, ha='right')
        ax4.set_ylabel("Metric Value", fontsize=11, weight='bold')
        ax4.set_title("Drift Spread Metrics by Timespan", fontsize=12, weight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self._save_plot(f"{self.synset}_{self.lexeme}_extreme_dimensions.png", dpi=200)
    
    def plot_ridgeline(self, drift_dists, drift_labels):
        """Plot ridgeline (stacked KDE) for temporal progression."""
        colors = plt.cm.viridis(np.linspace(0, 1, len(drift_dists)))
        fig, axes = plt.subplots(len(drift_dists), 1, figsize=(12, 2 * len(drift_dists)), sharex=True)
        if len(drift_dists) == 1:
            axes = [axes]
        
        for i, (drift_dist, label) in enumerate(zip(drift_dists, drift_labels)):
            ax = axes[i]
            sns.kdeplot(drift_dist, ax=ax, color=colors[i], fill=True, alpha=0.6, linewidth=2)
            ax.set_ylabel(label, fontsize=11, weight='bold', rotation=0, ha='right', va='center')
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2, axis='x')
            ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            
            # Add statistics
            mean_val = np.mean(drift_dist)
            std_val = np.std(drift_dist)
            ax.text(0.98, 0.85, f'μ={mean_val:.3f}\nσ={std_val:.3f}', 
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[-1].set_xlabel("Drift per Dimension", fontsize=12, weight='bold')
        axes[0].set_title(f"Ridgeline Plot: Drift Distributions Over Time\n{self.lexeme} / {self.synset}", 
                         fontsize=14, weight='bold', pad=20)
        
        self._save_plot(f"{self.synset}_{self.lexeme}_ridgeline.png")
    
    def plot_box_violin(self, drift_dists, drift_labels):
        """Plot box and violin plots side-by-side."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Prepare data
        plot_data = []
        for dist, label in zip(drift_dists, drift_labels):
            for val in dist:
                plot_data.append({'Timespan': label, 'Drift': val})
        plot_df = pd.DataFrame(plot_data)
        
        # Box plot
        sns.boxplot(data=plot_df, x='Timespan', y='Drift', ax=ax1, palette='viridis')
        ax1.set_xlabel("To Timespan", fontsize=12, weight='bold')
        ax1.set_ylabel("Drift per Dimension", fontsize=12, weight='bold')
        ax1.set_title(f"Box Plot: Drift Distribution Statistics\n{self.lexeme} / {self.synset}", 
                     fontsize=14, weight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.tick_params(axis='x', rotation=45)
        
        # Violin plot
        sns.violinplot(data=plot_df, x='Timespan', y='Drift', ax=ax2, palette='viridis', inner='quartile')
        ax2.set_xlabel("To Timespan", fontsize=12, weight='bold')
        ax2.set_ylabel("Drift per Dimension", fontsize=12, weight='bold')
        ax2.set_title(f"Violin Plot: Drift Distribution Shape\n{self.lexeme} / {self.synset}", 
                     fontsize=14, weight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.tick_params(axis='x', rotation=45)
        
        self._save_plot(f"{self.synset}_{self.lexeme}_box_violin.png")
    
    def plot_quantiles(self, drift_dists, drift_labels):
        """Plot quantile comparison across timespans."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        x_pos = np.arange(len(drift_labels))
        width = 0.15
        
        for q_idx, q in enumerate(quantiles):
            q_values = [np.quantile(dist, q) for dist in drift_dists]
            offset = (q_idx - len(quantiles)/2) * width
            ax.bar(x_pos + offset, q_values, width, label=f'{int(q*100)}th percentile', alpha=0.8)
        
        ax.set_xlabel("To Timespan", fontsize=12, weight='bold')
        ax.set_ylabel("Drift Magnitude", fontsize=12, weight='bold')
        ax.set_title(f"Quantile Comparison Across Timespans\n{self.lexeme} / {self.synset}", 
                    fontsize=14, weight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(drift_labels, rotation=45, ha='right')
        ax.legend(title="Quantiles", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        self._save_plot(f"{self.synset}_{self.lexeme}_quantiles.png")
    
    def plot_pairwise_heatmap(self, drift_matrix, p_value_matrix, timespans, alpha=0.05):
        """Plot heatmap of pairwise drift magnitudes and p-values."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Drift magnitude heatmap
        im1 = axes[0].imshow(drift_matrix, cmap='YlOrRd', aspect='auto')
        axes[0].set_xticks(range(len(timespans)))
        axes[0].set_yticks(range(len(timespans)))
        axes[0].set_xticklabels(timespans, rotation=45, ha='right')
        axes[0].set_yticklabels(timespans)
        axes[0].set_title(f'Drift Magnitude Between Timespans\n{self.lexeme} / {self.synset}')
        plt.colorbar(im1, ax=axes[0], label='Drift Magnitude')
        
        # Add drift values
        for i in range(len(timespans)):
            for j in range(len(timespans)):
                if i < j:
                    axes[0].text(j, i, f'{drift_matrix[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        # P-value heatmap
        p_matrix_display = p_value_matrix.copy()
        p_matrix_display[p_matrix_display == 0] = np.nan
        
        im2 = axes[1].imshow(p_matrix_display, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.1)
        axes[1].set_xticks(range(len(timespans)))
        axes[1].set_yticks(range(len(timespans)))
        axes[1].set_xticklabels(timespans, rotation=45, ha='right')
        axes[1].set_yticklabels(timespans)
        axes[1].set_title(f'P-values for Drift Significance\n{self.lexeme} / {self.synset}')
        plt.colorbar(im2, ax=axes[1], label='P-value')
        
        # Add p-values and significance markers
        for i in range(len(timespans)):
            for j in range(len(timespans)):
                if i < j:
                    p_val = p_value_matrix[i, j]
                    marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < alpha else ''
                    axes[1].text(j, i, f'{p_val:.3f}\n{marker}',
                               ha="center", va="center", 
                               color="white" if p_val < 0.05 else "black", 
                               fontsize=8, weight='bold' if marker else 'normal')
        
        self._save_plot(f"{self.synset}_pairwise_drift_heatmap.png")
    
    def plot_drift_connections(self, drift_matrix, p_value_matrix, timespans, alpha=0.05):
        """Plot network-style visualization of significant drift connections."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        n = len(timespans)
        positions = np.array(range(n))
        
        # Plot timespan points
        ax.scatter(positions, [0] * n, s=200, c='skyblue', edgecolors='black', linewidths=2, zorder=3)
        
        # Add labels
        for i, ts in enumerate(timespans):
            ax.text(i, -0.15, str(ts), ha='center', va='top', fontsize=10, weight='bold')
        
        # Draw arcs for significant drifts
        for i in range(n):
            for j in range(i + 1, n):
                p_val = p_value_matrix[i, j]
                if p_val < alpha:
                    drift = drift_matrix[i, j]
                    linewidth = 1 + (drift / drift_matrix.max()) * 5
                    alpha_line = 1.0 - (p_val / alpha)
                    height = 0.3 + 0.1 * (j - i)
                    
                    x = np.linspace(i, j, 100)
                    y = height * np.sin(np.pi * (x - i) / (j - i))
                    
                    color = 'red' if p_val < 0.001 else 'orange' if p_val < 0.01 else 'yellow'
                    ax.plot(x, y, linewidth=linewidth, alpha=alpha_line, color=color, zorder=1)
                    
                    mid_x = (i + j) / 2
                    mid_y = height * 1.1
                    ax.text(mid_x, mid_y, f'd={drift:.3f}\np={p_val:.3f}', 
                           ha='center', va='bottom', fontsize=7, 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, 2)
        ax.set_xlabel('Timespan Index', fontsize=12)
        ax.set_title(f'Significant Drift Connections (p < {alpha})\n{self.lexeme} / {self.synset}', 
                    fontsize=14, weight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=3, label='p < 0.001'),
            Line2D([0], [0], color='orange', linewidth=3, label='p < 0.01'),
            Line2D([0], [0], color='yellow', linewidth=3, label='p < 0.05')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        self._save_plot(f"{self.synset}_significant_drift_timeline.png")
    
    def plot_event_timeline(self, drift_events, drift_matrix, p_value_matrix, timespans, alpha=0.05):
        """Plot timeline showing when drift events occurred and their impact duration."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        n = len(timespans)
        positions = np.array(range(n))
        
        # Top panel: Event timeline
        ax1.scatter(positions, [0] * n, s=200, c='skyblue', edgecolors='black', linewidths=2, zorder=3)
        
        for i, ts in enumerate(timespans):
            ax1.text(i, -0.15, str(ts), ha='center', va='top', fontsize=10, weight='bold')
        
        # Draw arcs for significant drifts
        for i in range(n):
            for j in range(i + 1, n):
                p_val = p_value_matrix[i, j]
                if p_val < alpha:
                    drift = drift_matrix[i, j]
                    linewidth = 1 + (drift / drift_matrix.max()) * 5
                    alpha_line = 1.0 - (p_val / alpha)
                    height = 0.3 + 0.1 * (j - i)
                    
                    x = np.linspace(i, j, 100)
                    y = height * np.sin(np.pi * (x - i) / (j - i))
                    
                    color = 'red' if p_val < 0.001 else 'orange' if p_val < 0.01 else 'yellow'
                    ax1.plot(x, y, linewidth=linewidth, alpha=alpha_line, color=color, zorder=1)
        
        # Highlight drift events
        consecutive_events = [e for e in drift_events if e.get('event_type') != 'sudden_shift']
        for event in consecutive_events:
            trans_idx = event['transition_idx']
            impact_end = trans_idx + 1 + event['impact_duration']
            
            ax1.axvspan(trans_idx + 0.5, min(impact_end, n-1) + 0.5, 
                       alpha=0.2, color='red', zorder=0)
            
            ax1.plot([trans_idx + 0.5, trans_idx + 0.5], [-0.3, 1.5], 
                    'r--', linewidth=3, alpha=0.8, zorder=2)
            
            ax1.text(trans_idx + 0.5, 1.6, f"Event {event['event_id']}\nd={event['immediate_drift']:.3f}", 
                    ha='center', va='bottom', fontsize=9, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
        
        ax1.set_xlim(-0.5, n - 0.5)
        ax1.set_ylim(-0.5, 2)
        ax1.set_xlabel('Timespan Index', fontsize=12, weight='bold')
        ax1.set_title(f'Drift Events and Impact Duration\n{self.lexeme} / {self.synset}', 
                     fontsize=14, weight='bold')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.set_yticks([])
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=3, label='p < 0.001'),
            Line2D([0], [0], color='orange', linewidth=3, label='p < 0.01'),
            Line2D([0], [0], color='yellow', linewidth=3, label='p < 0.05'),
            Line2D([0], [0], color='red', linewidth=3, linestyle='--', label='Drift event'),
            Patch(facecolor='red', alpha=0.2, label='Impact duration')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Bottom panel: Impact duration chart
        if consecutive_events:
            event_ids = [e['event_id'] for e in consecutive_events]
            durations = [e['impact_duration'] for e in consecutive_events]
            transitions = [f"{e['transition_from']}→{e['transition_to']}" for e in consecutive_events]
            colors_bar = ['red' if e['immediate_pvalue'] < 0.001 else 'orange' if e['immediate_pvalue'] < 0.01 else 'yellow' 
                         for e in consecutive_events]
            
            bars = ax2.bar(event_ids, durations, color=colors_bar, edgecolor='black', linewidth=1.5)
            ax2.set_xlabel('Event ID', fontsize=12, weight='bold')
            ax2.set_ylabel('Impact Duration\n(# timespans)', fontsize=12, weight='bold')
            ax2.set_title('Duration of Each Drift Event Impact', fontsize=12, weight='bold')
            ax2.set_xticks(event_ids)
            ax2.set_xticklabels([f"E{i}\n{t}" for i, t in zip(event_ids, transitions)], fontsize=9)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
            
            for bar, dur in zip(bars, durations):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{dur}', ha='center', va='bottom', fontsize=10, weight='bold')
        else:
            ax2.text(0.5, 0.5, 'No consecutive drift events detected', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        self._save_plot(f"{self.synset}_drift_events_timeline.png")
