import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, mutual_info_score
import os
import math
from joblib import Parallel, delayed
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
# 1. Data Loading
# ==============================================

def load_galaxy_data(filepath, log_transform=False):
    """
    Loads the galaxy velocity dataset from a local CSV file.
    Optionally applies a natural log transformation.
    """
    print(f"Loading galaxy data from local file: {filepath}...")
    try:
        df = pd.read_csv(filepath, index_col=0)
        velocities = df['x'].values / 1000.0
        
        if log_transform:
            print("Applying natural log transformation to the data.")
            # Ensure no non-positive values before logging
            if np.any(velocities <= 0):
                print("Warning: Data contains non-positive values. Skipping log transform.")
                return velocities.reshape(-1, 1)
            return np.log(velocities).reshape(-1, 1)
        else:
            return velocities.reshape(-1, 1)

    except FileNotFoundError:
        print(f"Error: The file was not found at {filepath}")
        return None
    except Exception as e:
        print(f"Failed to load data. Error: {e}")
        return None

# ==============================================
# 2. MCMC Sampler (UPDATED TO PITMAN-YOR PROCESS)
# ==============================================

def run_mcmc_with_pyrichlet(X, config, filepath):
    """
    Runs MCMC using the pyrichlet library with a Pitman-Yor Process,
    pre-computes the posterior density, and saves all results to a single file.
    """
    print("--- Using Pyrichlet for MCMC Sampling (Pitman-Yor Process) ---")
    
    # Map the simulation configuration to pyrichlet's parameters
    total_iter = config['burn_in'] + (config['n_final_samples'] * config['thinning'])
    burn_in = config['burn_in']
    subsample_steps = config['thinning']

    # Set weaker priors and use the Pitman-Yor Process for more flexibility
    mm = mixture_models.PitmanYorMixture(
        alpha=config.get('alpha', 1.0),
        pyd=config.get('pyd', 0.5), # Correct discount parameter is 'pyd'
        mu_prior=X.mean(axis=0),
        lambda_prior=config.get('lambda_prior', 0.1),
        psi_prior=np.atleast_2d(config.get('psi_prior', 1.0)),
        nu_prior=config.get('nu_prior', X.shape[1]),
        rng=config['seed'],
        total_iter=total_iter,
        burn_in=burn_in,
        subsample_steps=subsample_steps
    )

    mm.fit_gibbs(y=X, show_progress=True)
    
    print("\nProcessing Pyrichlet posterior samples...")
    final_partitions = []
    final_params = []

    for samp in tqdm(mm.sim_params, desc="Extracting samples"):
        assignments = samp['d']
        final_partitions.append(assignments)

        weights = samp['w']
        theta = samp['theta']
        
        active_means = {k: v[0][0] for k, v in theta.items()}
        active_vars = {k: v[1][0, 0] for k, v in theta.items()}
        active_weights = {k: weights[k] for k in theta.keys() if k < len(weights)}
        
        final_params.append({
            'means': active_means, 'vars': active_vars, 'weights': active_weights
        })

    print(f"Completed. Total final samples collected: {len(final_partitions)}")
    
    # --- Pre-compute density estimate before saving ---
    print("Pre-computing average posterior predictive density...")
    # Expand plot range
    x_range_avg = np.linspace(X.min() - 5, X.max() + 5, 400)
    avg_density = mm.gibbs_eap_density(y=x_range_avg)
    
    # Save all results, including the pre-computed density.
    results = {
        'X': X,
        'mcmc_partitions': final_partitions,
        'mcmc_params': final_params,
        'config': config,
        'density_grid': (x_range_avg, avg_density)
    }
    
    save_full_path = os.path.join(filepath, 'mcmc_results_galaxy.pkl')
    with open(save_full_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"MCMC results and density curve saved to {save_full_path}")


# ==============================================
# 3. Analysis & Plotting Module (DENSITY PART UPDATED)
# ==============================================

def plot_clusters_1d(x_data, labels_list, titles, save_path=None, avg_density_line=None, on_log_scale=False):
    """
    Plots 1D cluster assignments. The density line shown is the PRE-CALCULATED
    average posterior predictive density.
    """
    n = len(labels_list)
    if n == 0: return
    n_cols = min(5, n); n_rows = math.ceil(n / n_cols)
    plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    
    x_flat = x_data.flatten()
    
    for i in range(n):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        labels_i = labels_list[i]
        if labels_i is None: continue
            
        ax.hist(x_flat, bins=60, density=True, color='lightgray', alpha=0.6)

        if avg_density_line is not None:
            x_range, density_estimate = avg_density_line
            ax.plot(x_range, density_estimate, color='black', lw=2, alpha=0.8)
            # Find max density for rug plot placement, handle empty density case
            max_density_val = np.max(density_estimate) if density_estimate.size > 0 else 1.0
        else:
            max_density_val = 1.0 # Default height if no density line

        unique_labels = np.unique(labels_i)
        colors = plt.cm.viridis(np.linspace(0, 1, max(1, len(unique_labels))))
        
        for c_idx, c_label in enumerate(unique_labels):
            cluster_points = x_flat[labels_i == c_label]
            ax.plot(cluster_points, np.zeros_like(cluster_points) - 0.05 * max_density_val, '|', color=colors[c_idx], markersize=20, markeredgewidth=2.0)
            
        ax.set_yticks([]); ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False)
        
        xlabel = "Log Velocity (1000 km/s)" if on_log_scale else "Velocity (1000 km/s)"
        ax.set_xlabel(xlabel)
        ax.set_title(titles[i], pad=20)

    plt.tight_layout(pad=2.0, w_pad=3.0) # Added w_pad for horizontal spacing
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='eps')
        print(f"Saved cluster plot to {save_path}")
    plt.show()

def score_partition(model, partition):
    """Helper function to score a single partition for parallel execution."""
    return -model.log_prob(partition)

def _compute_distances_for_row(i, partitions, model):
    """Helper function for parallel computation of one row of the distance matrix."""
    n = len(partitions)
    return [model._get_distance(partitions[i], partitions[j]) for j in range(i, n)]

def _test_random_partition(k_values, k_probs, n_samples, kde_model, tau):
    """Helper function to generate, score, and test a single random partition for parallel execution."""
    # 1. Sample K from the empirical posterior
    sampled_k = np.random.choice(k_values, p=k_probs)
    
    # 2. Generate a random partition with K clusters
    if sampled_k > n_samples: # Sanity check
        sampled_k = n_samples
    
    if sampled_k <= 0: # Sanity check for empty or invalid K
        random_partition = np.zeros(n_samples, dtype=int)
        sampled_k = 1
    
    if sampled_k == 1:
        random_partition = np.zeros(n_samples, dtype=int)
    else:
        # Assign one sample to each of the k clusters to guarantee occupation
        labels = np.arange(sampled_k) 
        # Assign the remaining n_samples - k samples randomly
        remaining_labels = np.random.randint(0, sampled_k, size=n_samples - sampled_k)
        # Combine and shuffle
        random_partition = np.concatenate((labels, remaining_labels))
        np.random.shuffle(random_partition)
    
    # 3. Score the partition
    score_random = -kde_model.log_prob(random_partition)
    
    # 4. Check if it's in the set and return 1 or 0
    return 1 if score_random <= tau else 0

def run_analysis(filepath, config, on_log_scale=False):
    """
    Analysis workflow that loads pre-computed MCMC samples and the density curve.
    """
    seed = 42
    np.random.seed(seed)

    full_filepath = os.path.join(filepath, 'mcmc_results_galaxy.pkl')
    with open(full_filepath, 'rb') as f: results = pickle.load(f)
        
    # Unpack results, including the pre-computed density_grid
    X, partitions, params, mcmc_config, avg_density_line = \
        results['X'], results['mcmc_partitions'], results['mcmc_params'], results['config'], results['density_grid']

    # Count clusters based on the unique labels in the partition, not the parameter dictionary.
    k_counts = Counter(len(np.unique(p)) for p in partitions)
    
    k_values, k_probs = zip(*sorted(k_counts.items()))
    k_probs = np.array(k_probs) / sum(k_probs)
    plt.figure(figsize=(8, 5)); plt.bar(k_values, k_probs, color='skyblue')
    plt.xlabel("Number of Clusters (K)"); plt.ylabel("Posterior Probability")
    plt.xticks(range(min(k_values), max(k_values) + 1)); plt.grid(axis='y', linestyle='--')
    plt.savefig(os.path.join(filepath, 'posterior_k_galaxy.eps'), bbox_inches='tight'); plt.show()
    
    # --- DENSITY PLOTTING (USING PRE-COMPUTED DATA) ---
    print("\nPlotting pre-computed average posterior predictive density...")
    x_range_avg, avg_density = avg_density_line
    
    xlabel = "Log Velocity (1000 km/s)" if on_log_scale else "Velocity (1000 km/s)"

    plt.figure(figsize=(10, 6))
    plt.hist(X.flatten(), bins=60, density=True, color='lightgray', alpha=0.7, label='Data Histogram')
    plt.plot(x_range_avg, avg_density, color='black', lw=2, label='Avg. Predictive Density')
    plt.xlabel(xlabel); plt.ylabel("Density")
    plt.grid(axis='y', linestyle='--');
    plt.savefig(os.path.join(output_filepath, 'average_density_galaxy.eps'), bbox_inches='tight')
    plt.show()
    
    # --- The rest of the post-processing remains unchanged ---
    indices = np.random.permutation(len(partitions))
    split_idx = int(len(partitions) * 5/6)
    train_indices, calib_indices = indices[:split_idx], indices[split_idx:]
    train_partitions = [partitions[i] for i in train_indices]
    calib_partitions = [partitions[i] for i in calib_indices]

    print("\n--- Finding Top Partitions using MetricKDE density estimation ---")
    kde_model = MetricKDE(
        train_partitions, metric=config['kde_metric'], gamma=config['kde_gamma'],
        subsample_size=config.get('kde_subsample_size')
    )
    
    print("Scoring calibration set with MetricKDE in parallel...")
    tasks = [delayed(score_partition)(kde_model, p) for p in calib_partitions]
    scores = np.array(Parallel(n_jobs=-1)(tqdm(tasks, desc="Scoring Partitions")))
    
    sorted_calib_indices = np.argsort(scores)
    
    labels_to_plot, titles_to_plot = [], []
    for rank, idx in enumerate(sorted_calib_indices[:1]):
        original_calib_index = indices[split_idx:][idx]
        labels_to_plot.append(partitions[original_calib_index])
        titles_to_plot.append("")

    plot_clusters_1d(X, labels_to_plot, titles_to_plot, 
                       save_path=os.path.join(filepath, 'top_partitions_kde_galaxy.eps'),
                       avg_density_line=avg_density_line,
                       on_log_scale=on_log_scale)
    
    # --- FINITE SAMPLE CORRECTED CONFORMAL THRESHOLD ---
    print("\n--- Calculating Conformal Threshold ---")
    n_calib = len(scores)
    alpha = config['alpha_conf']
    k_hat = int(np.ceil((n_calib + 1) * (1 - alpha)))
    if k_hat > n_calib: k_hat = n_calib # Sanity check
    
    if n_calib > 0:
        tau = np.sort(scores)[k_hat - 1]
    else:
        tau = np.inf # Handle case with no calibration samples

    print(f"Conformal threshold (alpha={alpha}, n_calib={n_calib}, finite-sample corrected) calculated: {tau:.4f}")

    # --- K-MEANS TESTING IS NOW MOVED TO THE END ---

    print("\n--- Running Density Peak Clustering on Calibration Partitions ---")
    n_calib = len(calib_partitions)
    print(f"Computing pairwise distance matrix for {n_calib} samples...")
    tasks_dist = [delayed(_compute_distances_for_row)(i, calib_partitions, kde_model) for i in range(n_calib)]
    results_dist = Parallel(n_jobs=-1)(tqdm(tasks_dist, desc="Building distance matrix"))
    
    calib_dist_matrix = np.zeros((n_calib, n_calib))
    for i, row_distances in enumerate(results_dist):
        for j_offset, dist in enumerate(row_distances):
            j = i + j_offset
            calib_dist_matrix[i, j] = calib_dist_matrix[j, i] = dist

    print("Calculating rho and delta for Density Peak algorithm...")
    rho = np.exp(-scores)
    sorted_rho_indices = np.argsort(rho)[::-1]
    
    delta = np.zeros(n_calib)
    if n_calib > 0: # Avoid error on empty calib set
        delta[sorted_rho_indices[0]] = np.max(calib_dist_matrix[sorted_rho_indices[0], :]) if n_calib > 1 else 1.0

        for i in range(1, n_calib):
            current_idx = sorted_rho_indices[i]
            higher_density_indices = sorted_rho_indices[:i]
            dist_to_higher = calib_dist_matrix[current_idx, higher_density_indices]
            delta[current_idx] = np.min(dist_to_higher)

    gamma = rho * delta
    rho_threshold = np.quantile(rho, 0.85); delta_threshold = np.quantile(delta, 0.85)
    center_indices = np.where((rho > rho_threshold) & (delta > delta_threshold))[0]

    if len(center_indices) == 0:
        print("Quantile method found no centers. Falling back to highest gamma score.")
        if n_calib > 0:
            center_indices = np.array([np.argmax(gamma)])
        else:
            center_indices = np.array([]) # No centers if no calib data
    else:
        print(f"Found {len(center_indices)} center(s) via quantile method.")

    # Calculate the number of clusters for each calibration partition to use for coloring
    if n_calib > 0:
        calib_k_values = np.array([len(np.unique(p)) for p in calib_partitions])
    else:
        calib_k_values = np.array([]) # Empty if no calib data
    
    plt.figure(figsize=(10, 7))
    if n_calib > 0: # Only plot if there is data
        # Color the scatter plot points by the number of clusters, using a non-viridis colormap
        scatter = plt.scatter(rho, delta, s=30, alpha=0.6, c=calib_k_values, cmap='plasma')
        plt.colorbar(scatter, label="Number of Clusters (K)")
        
        # Plot the identified modes in red on top
        plt.scatter(rho[center_indices], delta[center_indices], s=150, alpha=0.9,
                    edgecolor='k', facecolor='red', marker='o', label='Modes')
        plt.legend()
            
    plt.xlabel('Density (ρ)'); plt.ylabel('Distance to higher density (δ)'); plt.grid(True)
    plt.savefig(os.path.join(output_filepath, f'dpc_decision_graph_galaxy.eps'), bbox_inches='tight'); plt.show()
    
    if len(center_indices) > 0:
        # Sort the identified centers by their gamma score in descending order
        center_gammas = gamma[center_indices]
        sorted_order = np.argsort(center_gammas)[::-1]
        sorted_center_indices = center_indices[sorted_order]
        
        # Select the top 3 centers based on the new sorted order
        dpc_plot_indices = sorted_center_indices[:min(3, len(sorted_center_indices))]
        
        center_partitions = [calib_partitions[i] for i in dpc_plot_indices]
        plot_titles_dpc = [f"Mode #{i+1}" for i in range(len(dpc_plot_indices))]
        plot_clusters_1d(X, center_partitions, plot_titles_dpc,
                         save_path=os.path.join(filepath, f'dpc_centers_galaxy.eps'),
                         avg_density_line=avg_density_line,
                         on_log_scale=on_log_scale)

    # ==============================================
    # 5. Conformal Testing (Moved to the end)
    # ==============================================
    print("\n" + "="*30)
    print("--- Conformal Test Results ---")
    print(f"Alpha: {alpha}, N_calib: {n_calib}")
    print(f"Conformal Threshold (tau): {tau:.4f}")
    print("="*30)

    # --- Test 1: K-Means Partitions ---
    print("\nTesting K-Means Partitions:")
    kmeans_results = {}
    for k in [3, 4, 5, 6]:
        kmeans_partition = KMeans(n_clusters=k, random_state=42, n_init='auto').fit_predict(X)
        score_kmeans = -kde_model.log_prob(kmeans_partition)
        in_set = score_kmeans <= tau
        kmeans_results[k] = (score_kmeans, in_set)
        print(f"   K-Means (k={k}) Score: {score_kmeans:.4f} -> In Set: {in_set}")

    # --- Test 2: Random Partitions from Posterior ---
    print("\nTesting Random Partitions (from K posterior) in parallel:")
    n_random_tests = 1000
    n_samples = len(X)
    
    # k_values and k_probs were defined earlier when plotting the posterior
    tasks_random = [delayed(_test_random_partition)(k_values, k_probs, n_samples, kde_model, tau) for _ in range(n_random_tests)]
    results_random = Parallel(n_jobs=-1)(tqdm(tasks_random, desc="Testing random partitions"))
    n_in_set = np.sum(results_random)
    
    proportion_in_set = n_in_set / n_random_tests if n_random_tests > 0 else 0
    print(f"\nProportion of {n_random_tests} random partitions in the conformal set: {proportion_in_set:.4f} ({n_in_set} / {n_random_tests})")
    print("="*30)


class MetricKDE:
    def __init__(self, train_partitions, metric='vi', gamma=1.0, subsample_size=None):
        self.train_partitions_ = np.array(train_partitions, dtype=object)
        self.m_ = len(self.train_partitions_)
        self.gamma_ = gamma
        self.metric_ = metric
        self.subsample_size_ = subsample_size
        init_msg = f"MetricKDE initialized with {self.m_} partitions using '{metric}' metric."
        if subsample_size: init_msg += f" Will use subsamples of size {subsample_size}."
        print(init_msg)

    def _get_distance(self, p1, p2):
        if self.metric_ == 'vi': return _variation_of_information(p1, p2)
        return 1.0 - adjusted_rand_score(p1, p2)

    def log_prob(self, test_partition):
        if self.m_ == 0: return -np.inf
        partitions_to_use = self.train_partitions_
        if self.subsample_size_ and self.subsample_size_ < self.m_:
            indices = np.random.choice(self.m_, self.subsample_size_, replace=False)
            partitions_to_use = self.train_partitions_[indices]
        
        distances = np.array([self._get_distance(test_partition, p) for p in partitions_to_use])
        density = np.sum(np.exp(-self.gamma_ * distances)) / len(partitions_to_use)
        return np.log(density + 1e-9)

def _variation_of_information(p1, p2):
    n = len(p1)
    mi_nats = mutual_info_score(p1, p2)
    mi = mi_nats / np.log(2)
    h1_counts = Counter(p1); h1 = -sum( (c/n)*np.log2(c/n) for c in h1_counts.values()) if n > 0 else 0
    h2_counts = Counter(p2); h2 = -sum( (c/n)*np.log2(c/n) for c in h2_counts.values()) if n > 0 else 0
    return h1 + h2 - 2*mi

# ==============================================
# 4. Main Execution
# ==============================================
if __name__ == "__main__":
    CHOSEN_METRIC = 'vi'
    LOG_TRANSFORM_DATA = False # Set to True to use log-transformed data

    simulation_config = {
        'seed': 12345,
        'n_final_samples': 6000,
        'burn_in': 2000,
        'thinning': 5,
        'alpha': 2,
        'pyd': 0.05,          # Discount parameter for Pitman-Yor (pyd)
        'lambda_prior': 0.001,
        'psi_prior': 0.5,
        'nu_prior': 2          # Degrees of freedom for variance prior
    }
    analysis_config = {
        'alpha_conf': 0.1, 'kde_metric': CHOSEN_METRIC, 'kde_gamma': 0.5,
        'kde_subsample_size': None,
    }

    RERUN_MCMC = True # Set to false to re-run analysis on existing file
    output_filepath = './galaxies_experiment/'
    if not os.path.exists(output_filepath): os.makedirs(output_filepath)
    mcmc_results_path = os.path.join(output_filepath, 'mcmc_results_galaxy.pkl')
    galaxy_csv_path = os.path.join(output_filepath, 'galaxies.csv')

    if RERUN_MCMC or not os.path.exists(mcmc_results_path):
        X = load_galaxy_data(galaxy_csv_path, log_transform=LOG_TRANSFORM_DATA)
        if X is not None:
            plt.figure(figsize=(10, 6))
            xlabel = "Log Galaxy Velocity (1000 km/s)" if LOG_TRANSFORM_DATA else "Galaxy Velocity (1000 km/s)"
            plt.hist(X.flatten(), bins=60, color='skyblue', edgecolor='black', alpha=0.7)
            plt.xlabel(xlabel); plt.ylabel("Frequency"); plt.grid(axis='y', linestyle='--')
            plt.savefig(os.path.join(output_filepath, 'galaxy_histogram.eps'), bbox_inches='tight')
            plt.show()
            print("Running MCMC sampler and saving results...")
            run_mcmc_with_pyrichlet(X, simulation_config, output_filepath)
    else:
        print("MCMC results file found. Skipping sampling.")

    if os.path.exists(mcmc_results_path):
        print("\nLoading data and running the revised analysis workflow...")
        run_analysis(output_filepath, analysis_config, on_log_scale=LOG_TRANSFORM_DATA)
    else:
        print("\nCould not find MCMC results file. Aborting analysis.")