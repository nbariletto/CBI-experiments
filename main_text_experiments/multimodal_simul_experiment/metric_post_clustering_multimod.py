# ==============================================
# 0. Imports
# ==============================================
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import multivariate_t
from scipy.special import multigammaln
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, mutual_info_score
import os
import math
from matplotlib.patches import Ellipse
from joblib import Parallel, delayed
import sys
import pickle
import pandas as pd

from pyrichlet import mixture_models

plt.rc('font', size=18) # controls default text sizes
plt.rc('axes', titlesize=18) # fontsize of the (SUBPLOT) axes title
plt.rc('axes', labelsize=18) # fontsize of the x and y labels
plt.rc('xtick', labelsize=16) # fontsize of the tick labels
plt.rc('ytick', labelsize=16) # fontsize of the tick labels
plt.rc('legend', fontsize=16) # legend fontsize
plt.rc('figure', titlesize=22) # fontsize of the figure title

# ==============================================
# 1. MCMC Sampling Module (UPDATED TO PYRICHLET)
# ==============================================

def simulate_gmm_data(n_nodes, p_dim, n_clusters, seed=42):
    """Simulates a single dataset from a Gaussian Mixture Model."""
    np.random.seed(seed)
    # Using more separated means for clearer clusters
    means = np.array([[-3, -3], [-3, 3], [3, 0]])
    cov = np.eye(p_dim) * 1.5
    true_labels = np.random.randint(0, n_clusters, n_nodes)
    X = np.zeros((n_nodes, p_dim))
    for i in range(n_nodes):
        X[i, :] = np.random.multivariate_normal(means[true_labels[i]], cov)
    return X, true_labels

def run_mcmc_and_save(config, filepath):
    """Runs MCMC using pyrichlet and saves the results."""
    X, true_labels = simulate_gmm_data(
        n_nodes=config['n_nodes'], p_dim=config['p_dim'],
        n_clusters=config['n_clusters_true'], seed=config['seed']
    )
    
    print("--- Using Pyrichlet for MCMC Sampling (Pitman-Yor Process) ---")
    
    # Map the simulation configuration to pyrichlet's parameters
    total_iter = config['burn_in'] + (config['n_final_samples'] * config['thinning'])
    burn_in = config['burn_in']
    subsample_steps = config['thinning']
    p_dim = config['p_dim']

    # Initialize the Pitman-Yor Mixture model from pyrichlet
    mm = mixture_models.PitmanYorMixture(
        alpha=config['alpha'],
        pyd=config.get('py_sigma', 0.0), # Map py_sigma to pyd
        mu_prior=X.mean(axis=0),
        lambda_prior=0.01, # Weaker prior to allow cluster means to diverge
        psi_prior=np.eye(p_dim) * 1.5,
        nu_prior=p_dim + 2,
        rng=config['seed'],
        total_iter=total_iter,
        burn_in=burn_in,
        subsample_steps=subsample_steps
    )

    # Fit the model using Gibbs sampling
    mm.fit_gibbs(y=X, show_progress=True)
    
    print("\nProcessing Pyrichlet posterior samples...")
    # Extract partitions directly from the saved simulation steps
    final_partitions = [samp['d'] for samp in tqdm(mm.sim_params, desc="Extracting samples")]
    
    print(f"Completed. Total final samples collected: {len(final_partitions)}")

    results = {
        'X': X, 'true_labels': true_labels, 'mcmc_partitions': final_partitions,
        'log_likelihood_trace': [], # Pyrichlet does not track this by default
        'config': config
    }
    save_full_path = os.path.join(filepath, 'mcmc_results_multimod.pkl')
    with open(save_full_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"MCMC results saved to {save_full_path}")


# ==============================================
# 2. Kernel Density Estimation Module
# ==============================================

def _entropy(labels):
    """Computes the entropy of a labeling."""
    if len(labels) == 0: return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return -np.sum(probabilities * np.log2(probabilities))

def _variation_of_information(p1, p2):
    """Computes the Variation of Information metric."""
    # Mutual information in scipy is in nats, convert to bits
    mi_nats = mutual_info_score(p1, p2)
    mi_bits = mi_nats / np.log(2)
    h1 = _entropy(p1)
    h2 = _entropy(p2)
    vi = h1 + h2 - 2 * mi_bits
    return vi

def _binder_loss(p1, p2):
    """Computes the Binder loss (fraction of disagreeing pairs)."""
    n = len(p1)
    if n != len(p2): raise ValueError("Partitions must have the same number of elements.")
    disagreements = 0
    for i in range(n):
        for j in range(i + 1, n):
            in_same_cluster1 = (p1[i] == p1[j])
            in_same_cluster2 = (p2[i] == p2[j])
            if in_same_cluster1 != in_same_cluster2:
                disagreements += 1
    total_pairs = n * (n - 1) / 2
    return disagreements / total_pairs if total_pairs > 0 else 0

class MetricKDE:
    def __init__(self, train_partitions, metric='ari', gamma=1.0, subsample_size=None):
        self.train_partitions_ = np.array(train_partitions, dtype=object)
        self.m_ = len(self.train_partitions_)
        self.gamma_ = gamma
        self.metric_ = metric
        self.subsample_size_ = subsample_size
        if self.metric_ not in ['ari', 'binder', 'vi']: raise ValueError("Metric must be 'ari', 'binder', or 'vi'.")
        init_msg = f"MetricKDE initialized with {self.m_} training partitions using '{self.metric_}' metric."
        if self.subsample_size_ is not None:
            init_msg += f" Will use subsamples of size {self.subsample_size_} for scoring."
        print(init_msg)

    def _get_distance(self, p1, p2):
        dist = 0.0
        if self.metric_ == 'ari':
            dist = 1.0 - adjusted_rand_score(p1, p2)
        elif self.metric_ == 'binder':
            dist = _binder_loss(p1, p2)
        elif self.metric_ == 'vi':
            dist = _variation_of_information(p1, p2)
        # Enforce non-negativity for all metrics to handle floating-point issues.
        return max(0.0, dist)

    def log_prob(self, test_partition):
        if self.m_ == 0: return -np.inf
        if self.subsample_size_ and self.subsample_size_ < self.m_:
            indices = np.random.choice(self.m_, self.subsample_size_, replace=False)
            partitions_to_use = self.train_partitions_[indices]
            denominator = self.subsample_size_
        else:
            partitions_to_use = self.train_partitions_
            denominator = self.m_

        distances = np.array([self._get_distance(test_partition, train_part) for train_part in partitions_to_use])
        total_kernel_similarity = np.sum(np.exp(-self.gamma_ * distances))
        density = total_kernel_similarity / denominator
        return np.log(density + 1e-9)

def score_partition(model, partition):
    return -model.log_prob(partition)

def _test_random_partition(k_values, k_probs, n_nodes, kde_model, tau):
    """Helper for parallel random partition testing. Returns 1 if in set, 0 otherwise."""
    # Each worker process gets its own independent random state.
    sampled_k = np.random.choice(k_values, p=k_probs)
    if sampled_k == 0:
        return 0 # 0 inclusions
    
    random_partition = np.random.randint(0, sampled_k, size=n_nodes)
    score_rand = -kde_model.log_prob(random_partition)
    
    return 1 if score_rand <= tau else 0

# ==============================================
# 3. Analysis & Plotting Module
# ==============================================

def plot_clusters_fixed_points(x_fixed, labels_list, titles, n_std=2.0, save_path=None):
    n = len(labels_list)
    if n == 0:
        print("Warning: No labels to plot.")
        return
    n_cols = 4
    n_rows = math.ceil(n / n_cols)
    plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    max_clusters = max((len(np.unique(l)) for l in labels_list if l is not None), default=1)
    colors = plt.cm.viridis(np.linspace(0, 1, max(max_clusters, 3)))
    x_min, x_max = x_fixed[:, 0].min() - 1.5, x_fixed[:, 0].max() + 1.5
    y_min, y_max = x_fixed[:, 1].min() - 1.5, x_fixed[:, 1].max() + 1.5

    for i in range(n):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        labels_i = labels_list[i]
        if labels_i is None:
            ax.set_title(titles[i] + "\n(No samples in set)", pad=20); continue

        unique_labels = np.unique(labels_i)
        # Sort labels by cluster centroid for consistent coloring
        centroids = [x_fixed[labels_i == c].mean(axis=0) for c in unique_labels]
        sorted_pairs = sorted(zip(centroids, unique_labels), key=lambda p: (p[0][0], p[0][1]))
        color_map = {label: color_idx for color_idx, (_, label) in enumerate(sorted_pairs)}

        for c in unique_labels:
            idx = labels_i == c
            cluster_points = x_fixed[idx]
            color_idx = color_map[c]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[color_idx], s=100, alpha = 0.7)

            if len(cluster_points) > 1:
                cov = np.cov(cluster_points, rowvar=False)
                mean = np.mean(cluster_points, axis=0)
                try:
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    order = eigenvalues.argsort()[::-1]
                    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
                    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
                    width, height = 2 * n_std * np.sqrt(np.maximum(0, eigenvalues)) # Ensure non-negative
                    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor=colors[color_idx], facecolor='none', linestyle='--', alpha=0.7, lw=2)
                    ax.add_patch(ellipse)
                except np.linalg.LinAlgError: pass

        ax.set_title(titles[i], pad=20); ax.set_xlabel("X1"); ax.set_ylabel("X2")
        ax.grid(True); ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

    plt.tight_layout(pad=3.0, w_pad=3.0) # Added w_pad for horizontal spacing
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved cluster plot to {save_path}")
    plt.show()

def _compute_distances_for_row(i, partitions, model):
    """Helper function for parallel computation of one row of the distance matrix."""
    n = len(partitions)
    # Only computes the upper triangle part of the row
    return [model._get_distance(partitions[i], partitions[j]) for j in range(i, n)]

def run_analysis(filepath, config):
    seed = 42
    np.random.seed(seed)

    full_filepath = os.path.join(filepath, 'mcmc_results_multimod.pkl')
    figure_dir = filepath
    with open(full_filepath, 'rb') as f:
        results = pickle.load(f)
        
    X, true_labels, mcmc_partitions, log_likelihood_trace, mcmc_config = \
        results['X'], results['true_labels'], results['mcmc_partitions'], results['log_likelihood_trace'], results['config']

    partitions = mcmc_partitions

    # --- Plot the true partition separately at the beginning ---
    print("\n--- Ground Truth Data Clustering ---")
    plot_clusters_fixed_points(X, [true_labels], [""],
                               save_path=os.path.join(figure_dir, 'true_partition_multimod.eps'))

    num_clusters_posterior = [len(np.unique(p)) for p in partitions]
    k_counts = Counter(num_clusters_posterior)
    k_values, k_probs = sorted(k_counts.keys()), [k_counts[k] / len(num_clusters_posterior) for k in sorted(k_counts.keys())]
    plt.figure(figsize=(8, 5)); plt.bar(k_values, k_probs, color='skyblue')
    plt.xlabel("Number of Clusters (K)"); plt.ylabel("Posterior Probability")
    plt.xticks(k_values); plt.grid(axis='y', linestyle='--'); plt.savefig(os.path.join(figure_dir, 'posterior_k_multimod.eps'), bbox_inches='tight'); plt.show()

    indices = list(range(len(partitions))); np.random.shuffle(indices)
    split_idx = int(len(partitions) * 5/6)
    train_indices, calib_indices = indices[:split_idx], indices[split_idx:]
    train_partitions = [partitions[i] for i in train_indices]
    calib_partitions = [partitions[i] for i in calib_indices]

    print("--- Initializing MetricKDE Model ---")
    kde_model = MetricKDE(
        train_partitions=train_partitions,
        metric=config['kde_metric'],
        gamma=config['kde_gamma'],
        subsample_size=config.get('kde_subsample_size')
    )

    print("\nScoring calibration set in parallel...")
    tasks = [delayed(score_partition)(kde_model, p) for p in calib_partitions]
    scores = np.array(Parallel(n_jobs=-1)(tqdm(tasks, desc="Scoring calibration set")))
    
    # --- FINITE SAMPLE CORRECTED CONFORMAL THRESHOLD ---
    n_calib = len(scores)
    alpha = config['alpha_conf']
    k_hat = int(np.ceil((n_calib + 1) * (1 - alpha)))
    if k_hat > n_calib: k_hat = n_calib # Sanity check
    
    if n_calib > 0:
        tau = np.sort(scores)[k_hat - 1]
    else:
        tau = np.inf # No calibration samples, set threshold to infinity
        
    print(f"\nConformal threshold (alpha={alpha}, n_calib={n_calib}, finite-sample corrected): {tau:.4f}")

    print("\n--- Finding and Plotting the Highest Ranked Partition ---")
    sorted_calib = sorted(zip(scores, calib_partitions), key=lambda item: item[0])
    if sorted_calib:
        # Plot only the highest ranked (first) partition with no title
        highest_ranked_partition = sorted_calib[0][1]
        plot_clusters_fixed_points(X, [highest_ranked_partition], [""], 
                                   save_path=os.path.join(figure_dir, f'top_partition_kde_{config["kde_metric"]}_multimod.eps'))
    else:
        print("No partitions in calibration set to plot.")

    # --- Running Density Peak Clustering on Calibration Partitions ---
    print("\n--- Running Density Peak Clustering on Calibration Partitions ---")
    
    print(f"Computing pairwise distance matrix for {n_calib} samples using '{config['kde_metric']}' metric (optimized parallel)...")
    tasks = [delayed(_compute_distances_for_row)(i, calib_partitions, kde_model) for i in range(n_calib)]
    results_dist = Parallel(n_jobs=-1)(tqdm(tasks, desc="Building distance matrix (parallel)"))
    
    calib_dist_matrix = np.zeros((n_calib, n_calib))
    for i, row_distances in enumerate(results_dist):
        for j_offset, dist in enumerate(row_distances):
            j = i + j_offset
            calib_dist_matrix[i, j] = dist
            calib_dist_matrix[j, i] = dist

    print("Calculating s and delta for Density Peak algorithm...")
    s = np.exp(-scores) 
    
    sorted_s_indices = np.argsort(s)[::-1]
    
    delta = np.zeros(n_calib)
    
    if n_calib > 0: # Avoid error on empty calib set
        delta[sorted_s_indices[0]] = np.max(calib_dist_matrix[sorted_s_indices[0], :]) if n_calib > 1 else 1.0

        for i in range(1, n_calib):
            current_idx = sorted_s_indices[i]
            higher_density_indices = sorted_s_indices[:i]
            
            dist_to_higher_density = calib_dist_matrix[current_idx, higher_density_indices]
            
            delta[current_idx] = np.min(dist_to_higher_density)
    
    gamma = s * delta
    print("Identifying cluster centers from decision graph...")

    s_threshold = 0.75 # Renamed from rho_threshold
    delta_threshold = 0.6
    candidate_indices = np.where((s > s_threshold) & (delta > delta_threshold))[0] # Using s

    if len(candidate_indices) > 0:
        candidate_gammas = gamma[candidate_indices]
        sorted_candidate_indices_of_indices = np.argsort(candidate_gammas)[::-1]
        center_indices = candidate_indices[sorted_candidate_indices_of_indices]
        print(f"Found {len(center_indices)} candidate(s) satisfying thresholds.")
    else:
        print("Quantile-based method found no centers. Falling back to the point with the highest gamma score.")
        if len(gamma) > 0:
            center_indices = [np.argmax(gamma)]
        else:
            center_indices = [] # No centers if no data
    
    plt.figure(figsize=(10, 7))
    if n_calib > 0:
        calib_k = np.array([len(np.unique(p)) for p in calib_partitions])
        unique_ks = sorted(np.unique(calib_k))
        colors = plt.cm.plasma(np.linspace(0, 1, len(unique_ks)))
        color_map = {k: color for k, color in zip(unique_ks, colors)}

        # Plot points for each K value separately to build the legend
        for k in unique_ks:
            indices_k = np.where(calib_k == k)[0]
            plt.scatter(s[indices_k], delta[indices_k], c=[color_map[k]], label=f'K={k}', s=30, alpha=0.7)

        plt.scatter(s[center_indices], delta[center_indices], s=150, alpha=0.9, 
                    edgecolor='k', facecolor='red', marker='o', label='Modes')
        plt.legend(title="Number of Clusters")
    
    plt.xlabel('Density (s)'); plt.ylabel('Distance to higher density (δ)') # Label changed to s
    plt.grid(True, linestyle='--', alpha=0.6)
    decision_graph_path = os.path.join(figure_dir, f'density_peak_decision_graph_{config["kde_metric"]}_multimod.eps')
    plt.savefig(decision_graph_path, bbox_inches='tight')
    plt.show()


    if len(center_indices) > 0:
        plot_indices = center_indices[:min(3, len(center_indices))]
        
        center_partitions_and_gammas = [(calib_partitions[i], gamma[i]) for i in plot_indices]
        # Sort the identified centers by their gamma score before plotting
        sorted_centers = sorted(center_partitions_and_gammas, key=lambda x: x[1], reverse=True)
        
        plot_labels = [p for p, g in sorted_centers]
        plot_titles = [f"Mode #{i+1}" for i in range(len(plot_labels))]

        plot_clusters_fixed_points(
            X, plot_labels, plot_titles,
            save_path=os.path.join(figure_dir, f'density_peak_centers_{config["kde_metric"]}_multimod.eps')
        )
    else:
        print("Density Peak algorithm did not identify any cluster centers.")
    
    # --- Test specific hypotheses ---
    print("\n--- Conformal Hypothesis Tests ---")
    # Test 1: Ground-truth partition
    score_true = -kde_model.log_prob(true_labels)
    print(f"Ground-Truth (K=3) Partition Score: {score_true:.4f} -> In Set: {score_true <= tau}")
    # Test 2: Collapsed partition
    collapsed_labels = true_labels.copy()
    collapsed_labels[collapsed_labels == 1] = 0 # Merge cluster 1 into 0
    _, collapsed_labels = np.unique(collapsed_labels, return_inverse=True) # Relabel to be dense (0, 1)
    score_collapsed = -kde_model.log_prob(collapsed_labels)
    print(f"Collapsed (K=2) Partition Score: {score_collapsed:.4f} -> In Set: {score_collapsed <= tau}")

    print(f"\n--- Testing {num_random_tests} Random Partitions Sampled from Posterior on K (Parallel) ---")
    num_random_tests = 1000
    
    # Create tasks for the parallel execution
    tasks = [
        delayed(_test_random_partition)(
            k_values, 
            k_probs, 
            mcmc_config['n_nodes'], 
            kde_model, 
            tau
        ) 
        for _ in range(num_random_tests)
    ]
    
    # Run tasks in parallel and get results (list of 0s and 1s)
    results_list = Parallel(n_jobs=-1)(tqdm(tasks, desc="Testing random partitions"))
    
    # Aggregate results
    random_partitions_in_set = np.sum(results_list)

    print("\n--- Random Partition Test Summary ---")
    print("=" * 35)
    print(f"| Total partitions tested: {num_random_tests:<9} |")
    print(f"| Partitions IN the set:   {random_partitions_in_set:<9} |")
    print(f"| Partitions OUT of set:   {num_random_tests - random_partitions_in_set:<9} |")
    print(f"| Inclusion Rate:          {random_partitions_in_set / num_random_tests:<9.2%} |")
    print("=" * 35)


# ==============================================
# 4. Main Execution
# ==============================================
if __name__ == "__main__":
    # --- Configuration ---
    CHOSEN_METRIC = 'vi'

    simulation_config = {
        'n_nodes': 100, 'p_dim': 2, 'n_clusters_true': 3,
        'seed': 12345,
        'n_final_samples': 6000,
        'burn_in': 1000,
        'thinning': 5,
        # PYMM-specific
        'alpha': 0.03,
        'py_sigma': 0.01,
    }

    analysis_config = {
        'alpha_conf': 0.1,
        'kde_metric': CHOSEN_METRIC,
        'kde_gamma': 0.5,
        'kde_subsample_size': None 
    }

    RERUN_MCMC = True
    output_filepath = './multimodal_simul_experiment/'
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)

    if RERUN_MCMC or not os.path.exists(os.path.join(output_filepath, 'mcmc_results_multimod.pkl')):
        print("Running MCMC sampler and saving results...")
        run_mcmc_and_save(simulation_config, output_filepath)
    else:
        print(f"MCMC results file found at '{output_filepath}'. Skipping sampling.")

    print("\nLoading data and running the KDE Analysis workflow...")
    run_analysis(
        output_filepath,
        analysis_config
    )