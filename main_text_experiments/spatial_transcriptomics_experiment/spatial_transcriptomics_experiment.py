# ==============================================
# 0. Imports
# ==============================================
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import adjusted_rand_score, mutual_info_score
from scipy.spatial.distance import cdist # For spatial test
import os
import math
from matplotlib.patches import Ellipse
from joblib import Parallel, delayed
import sys
import pandas as pd  # <-- We will use this to read CSVs
import warnings

plt.rc('font', size=18) # controls default text sizes
plt.rc('axes', titlesize=18) # fontsize of the (SUBPLOT) axes title
plt.rc('axes', labelsize=18) # fontsize of the x and y labels
plt.rc('xtick', labelsize=16) # fontsize of the tick labels
plt.rc('ytick', labelsize=16) # fontsize of the tick labels
plt.rc('legend', fontsize=16) # legend fontsize
plt.rc('figure', titlesize=22) # fontsize of the figure title

# ==============================================
# 1. Kernel Density Estimation Module
# ==============================================

def _entropy(labels):
    """Computes the entropy of a labeling."""
    if len(labels) == 0: return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return -np.sum(probabilities * np.log2(probabilities))

def _variation_of_information(p1, p2):
    """Computes the Variation of Information metric."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
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
        # print(f"MetricKDE initialized...") # Silenced

    def _get_distance(self, p1, p2):
        dist = 0.0
        if self.metric_ == 'ari':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                dist = 1.0 - adjusted_rand_score(p1, p2)
        elif self.metric_ == 'binder':
            dist = _binder_loss(p1, p2)
        elif self.metric_ == 'vi':
            dist = _variation_of_information(p1, p2)
        return max(0.0, dist) # Enforce non-negativity

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

# ==============================================
# 2. Analysis & Plotting Module
# ==============================================

def plot_clusters_fixed_points(x_fixed, labels_list, titles, xlabel="X", ylabel="Y", n_std=2.0, save_path=None):
    """
    Plots a list of clusterings on the same fixed 2D points (x_fixed).
    """
    n = len(labels_list)
    if n == 0:
        print("Warning: No labels to plot.")
        return
        
    x_fixed_normalized = x_fixed / 1000.0
    
    n_cols = min(n, 4) 
    n_rows = math.ceil(n / n_cols)
    plt.figure(figsize=(6 * n_cols, 6 * n_rows))
    
    all_labels = set()
    for l in labels_list:
        if l is not None:
            all_labels.update(np.unique(l))
    
    max_clusters = len(all_labels)
    if max_clusters == 0:
        print("Warning: No clusters found in any labels.")
        return

    colors = plt.cm.tab20(np.linspace(0, 1, max(20, max_clusters)))
    if max_clusters > 20: 
        colors = plt.cm.gist_ncar(np.linspace(0, 1, max_clusters))
        
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(sorted(list(all_labels)))}

    for i in range(n):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        labels_i = labels_list[i]
        if labels_i is None:
            ax.set_title(titles[i] + "\n(No samples in set)", pad=20); continue

        unique_labels = np.unique(labels_i)

        for c in unique_labels:
            idx = labels_i == c
            cluster_points = x_fixed_normalized[idx]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color_map[c], s=10, alpha = 0.7)

            if len(cluster_points) > 1:
                cov = np.cov(cluster_points, rowvar=False)
                mean = np.mean(cluster_points, axis=0)
                try:
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    order = eigenvalues.argsort()[::-1]
                    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
                    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
                    width, height = 2 * n_std * np.sqrt(np.maximum(0, eigenvalues)) 
                    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor=color_map[c], facecolor='none', linestyle='--', alpha=0.7, lw=2)
                    ax.add_patch(ellipse)
                except np.linalg.LinAlgError: pass
        
        ax.set_title(titles[i], pad=20); 
        ax.set_xlabel(f"{xlabel}"); 
        ax.set_ylabel(f"{ylabel}")
        
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)
            
        ax.grid(True, linestyle='--', alpha=0.5)
    
    # if max_clusters > 10 or n > 1:
    #   ... (legend code removed) ...

    plt.tight_layout(pad=3.0, w_pad=3.0)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved cluster plot to {save_path}")
    plt.show()

def plot_combined_dpc_graphs(results_A, results_B, save_path):
    """Plots the DPC decision graphs for two results side-by-side."""
    if not results_A or not results_B:
        print("Skipping combined DPC plot: Missing results.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7)) # 1 row, 2 cols.
    
    # --- Plot for A ---
    s_A = results_A['dpc_s']
    delta_A = results_A['dpc_delta']
    center_indices_A = results_A['dpc_center_indices']
    title_A = results_A['cell_type']
    
    ax1.scatter(s_A, delta_A, s=30, alpha=0.7)
    ax1.scatter(s_A[center_indices_A], delta_A[center_indices_A], s=150, alpha=0.9, 
                edgecolor='k', facecolor='red', marker='o', label='Modes')
    ax1.set_xlabel('Density (s)')
    ax1.set_ylabel('Distance to higher density (δ)')
    ax1.set_title(title_A)
    ax1.grid(True, linestyle='--', alpha=0.6)
    # ax1.legend() # Legend removed as in original script

    # --- Plot for B ---
    s_B = results_B['dpc_s']
    delta_B = results_B['dpc_delta']
    center_indices_B = results_B['dpc_center_indices']
    title_B = results_B['cell_type']
    
    ax2.scatter(s_B, delta_B, s=30, alpha=0.7)
    ax2.scatter(s_B[center_indices_B], delta_B[center_indices_B], s=150, alpha=0.9, 
                edgecolor='k', facecolor='red', marker='o', label='Modes')
    ax2.set_xlabel('Density (s)')
    ax2.set_ylabel('Distance to higher density (δ)')
    ax2.set_title(title_B)
    ax2.grid(True, linestyle='--', alpha=0.6)
    # ax2.legend() # Legend removed as in original script
    
    # --- Save and Show ---
    plt.tight_layout(pad=3.0, w_pad=3.0)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved combined DPC plot to {save_path}")
    plt.show()

def _compute_distances_for_row(i, partitions, model):
    """Helper function for parallel computation of one row of the distance matrix."""
    n = len(partitions)
    return [model._get_distance(partitions[i], partitions[j]) for j in range(i, n)]

# ==============================================
# 3. Data Loading Module
# ==============================================

def load_sarp_data_from_csv(cell_type_prefix, config):
    """
    Loads pre-processed data from .csv files for a given cell type.
    
    These .csv files should be generated by the R script.
    """
    base_path = config['output_filepath']
    print(f"\n--- Loading data for {cell_type_prefix} from {base_path} ---")
    
    # Define file paths
    file_paths = {
        'mcmc_partitions': os.path.join(base_path, f"{cell_type_prefix}_mcmc_partitions.csv"),
        'spatial_coords': os.path.join(base_path, f"{cell_type_prefix}_spatial_coords.csv"),
    }
    
    data_dict = {'cell_type': cell_type_prefix}
    
    try:
        # Load MCMC partitions (List of np.arrays)
        print(f"Loading 'mcmc_partitions'...")
        mcmc_data = pd.read_csv(file_paths['mcmc_partitions']).to_numpy()
        data_dict['mcmc_partitions'] = [row for row in mcmc_data]
        print(f"Loaded 'mcmc_partitions': {len(data_dict['mcmc_partitions'])} samples.")

        # Load Spatial coordinates (np.array, shape [N, 2])
        print(f"Loading 'spatial_coords'...")
        data_dict['spatial_coords'] = pd.read_csv(file_paths['spatial_coords']).to_numpy()
        print(f"Loaded 'spatial_coords': shape {data_dict['spatial_coords'].shape}.")
        
        # Check for consistency
        n_cells_spatial = data_dict['spatial_coords'].shape[0]
        n_cells_mcmc = len(data_dict['mcmc_partitions'][0])
        if n_cells_spatial != n_cells_mcmc:
            warnings.warn(f"Cell count mismatch! Spatial data has {n_cells_spatial}, MCMC has {n_cells_mcmc}.")
            
    except FileNotFoundError as e:
        print(f"ERROR: Could not find required data file: {e.filename}")
        print("Please ensure you have run the R conversion script first.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred loading data: {e}")
        sys.exit(1)

    print(f"Data loading for {cell_type_prefix} complete.\n")
    return data_dict


# ==============================================
# 4. Analysis Pipeline & Spatial Test
# ==============================================

def run_conformal_pipeline(data_dict, config, figure_prefix, train_indices, calib_indices):
    """
    Runs the full KDE-VI pipeline on a given dataset.
    Plots point estimate and DPC modes.
    """
    cell_type = data_dict['cell_type']
    X_spatial = data_dict['spatial_coords']
    partitions = data_dict['mcmc_partitions']
    
    figure_dir = config['output_filepath']
    os.makedirs(figure_dir, exist_ok=True)
    
    print(f"\n==============================================")
    print(f"Running Conformal Pipeline for: {cell_type}")
    print(f"==============================================")

    # --- Train/Calib Split (using pre-computed indices) ---
    train_partitions = [partitions[i] for i in train_indices]
    calib_partitions = [partitions[i] for i in calib_indices]
    print(f"Using {len(train_partitions)} pre-selected training samples.")
    print(f"Using {len(calib_partitions)} pre-selected calibration samples (thinned).")

    print("--- Initializing MetricKDE Model ---")
    kde_model = MetricKDE(
        train_partitions=train_partitions,
        metric=config['kde_metric'],
        gamma=config['kde_gamma'],
        subsample_size=config.get('kde_subsample_size')
    )

    print("\nScoring calibration set in parallel...")
    tasks = [delayed(score_partition)(kde_model, p) for p in calib_partitions]
    scores = np.array(Parallel(n_jobs=-1)(tqdm(tasks, desc=f"Scoring {cell_type} calib set")))
    
    # --- Conformal Threshold ---
    n_calib = len(scores)
    alpha = config['alpha_conf']
    k_hat = int(np.ceil((n_calib + 1) * (1 - alpha)))
    if k_hat > n_calib: k_hat = n_calib
    
    if n_calib == 0:
        print("ERROR: No calibration samples. Cannot proceed.")
        return None
        
    tau = np.sort(scores)[k_hat - 1] if n_calib > 0 else np.inf
    
    print(f"\nConformal threshold (alpha={alpha}, n_calib={n_calib}, finite-sample corrected): {tau:.4f}")

    # --- Find Highest Ranked Partition (Point Estimate) ---
    print("\nFinding Highest Ranked Partition (KDE-VI Point Estimate)...")
    sorted_calib = sorted(zip(scores, calib_partitions), key=lambda item: item[0])
    if sorted_calib:
        point_estimate_partition = sorted_calib[0][1]
        # Standalone point estimate plot REMOVED
    else:
        print("No partitions in calibration set to plot.")
        point_estimate_partition = None

    # --- Running Density Peak Clustering on Calibration Partitions ---
    print("\n--- Running Density Peak Clustering (DPC) on Calibration Partitions ---")
    
    print(f"Computing pairwise distance matrix for {n_calib} samples (optimized parallel)...")
    tasks = [delayed(_compute_distances_for_row)(i, calib_partitions, kde_model) for i in range(n_calib)]
    results_dist = Parallel(n_jobs=-1)(tqdm(tasks, desc=f"Building {cell_type} dist matrix"))
    
    calib_dist_matrix = np.zeros((n_calib, n_calib))
    for i, row_distances in enumerate(results_dist):
        for j_offset, dist in enumerate(row_distances):
            j = i + j_offset
            calib_dist_matrix[i, j] = dist
            calib_dist_matrix[j, i] = dist

    print("Calculating s (density) and delta (distance) for DPC...")
    s = np.exp(-scores) 
    sorted_s_indices = np.argsort(s)[::-1]
    delta = np.zeros(n_calib)
    
    if n_calib > 1:
        delta[sorted_s_indices[0]] = np.max(calib_dist_matrix[sorted_s_indices[0], :])
        for i in range(1, n_calib):
            current_idx = sorted_s_indices[i]
            higher_density_indices = sorted_s_indices[:i]
            dist_to_higher_density = calib_dist_matrix[current_idx, higher_density_indices]
            delta[current_idx] = np.min(dist_to_higher_density)
    elif n_calib == 1:
        delta[0] = 1.0 # Only one point
        
    gamma = s * delta
    
    # --- DPC Decision Graph ---
    
    # --- Identify and Plot Modes (DPC Centers) ---
    print("Identifying DPC modes from decision graph...")
    s_threshold = np.quantile(s, 0.9)
    delta_threshold = np.quantile(delta, 0.9)
    
    center_indices = np.where((s > s_threshold) & (delta > delta_threshold))[0]
    
    if len(center_indices) == 0:
        print("Quantile method found no modes. Falling back to highest gamma score.")
        center_indices = [np.argmax(gamma)]
    
    print(f"Found {len(center_indices)} candidate mode(s).")
    
    # plt.figure(figsize=(10, 7))
    # ... (plotting code) ...
    # plt.savefig(decision_graph_path, bbox_inches='tight')
    # plt.show()

    # --- Plot the DPC Modes ---
    if len(center_indices) > 1: # Only plot if MORE than 1 mode is found
        sorted_center_indices = center_indices[np.argsort(gamma[center_indices])][::-1]
        plot_indices = sorted_center_indices[:min(4, len(sorted_center_indices))]
        
        plot_labels = [calib_partitions[i] for i in plot_indices]
        plot_titles = [f"Mode #{i+1}" for i in range(len(plot_labels))] # New titles

        print(f"Plotting top {len(plot_labels)} DPC modes...")
        plot_clusters_fixed_points(
            X_spatial, plot_labels, plot_titles,
            xlabel="X", ylabel="Y", # These will be modified by the plot function
            save_path=os.path.join(figure_dir, f'{figure_prefix}_dpc_modes.eps')
        )
    else:
        print("DPC algorithm identified only one mode (the point estimate), so no mode plot will be generated.")

    print(f"\n--- Pipeline for {cell_type} complete. ---")
    
    return {
        'kde_model': kde_model,
        'tau': tau,
        'point_estimate': point_estimate_partition, # This is the KDE-VI point estimate
        'spatial_coords': X_spatial,
        'dpc_s': s,
        'dpc_delta': delta,
        'dpc_center_indices': center_indices,
        'cell_type': cell_type
    }


def _spatial_cluster_assignment(spatial_coords_A, labels_A, spatial_coords_B):
    """
    Helper function to assign labels to B based on the *closest cell* from A.
    """
    
    if spatial_coords_A.shape[0] == 0:
        print("Warning: No source coordinates (A) found.")
        return np.zeros(spatial_coords_B.shape[0], dtype=int)
        
    if labels_A.shape[0] == 0:
        print("Warning: No source labels (A) found.")
        return np.zeros(spatial_coords_B.shape[0], dtype=int)
        
    if spatial_coords_B.shape[0] == 0:
        print("Warning: No target coordinates (B) found.")
        return np.array([], dtype=int)

    # Compute the pairwise distances between each cell in B and each cell in A
    # The cdist(B, A) function will return a matrix of shape [n_cells_B, n_cells_A]
    distances = cdist(spatial_coords_B, spatial_coords_A)
    
    # For each cell in B (each row), find the index of the closest cell in A (minimum value in that row)
    # np.argmin(distances, axis=1) will return an array of shape [n_cells_B]
    # Each value is the index (in A) of the closest cell.
    closest_A_indices = np.argmin(distances, axis=1)
    
    # Use these indices to look up the corresponding cluster label in labels_A
    translated_partition_B = labels_A[closest_A_indices]
    
    return translated_partition_B
    

def perform_spatial_translation_test(results_A, results_B, name_A, name_B, config):
    """
    Performs the spatial translation and conformal test.
    Plots B's point estimate vs. the translated partition.
    """
    
    print(f"\n==============================================")
    print(f"Test: {name_A} centroids -> {name_B} cells")
    print(f"==============================================")
    
    point_estimate_A = results_A['point_estimate']
    spatial_coords_A = results_A['spatial_coords']
    
    point_estimate_B = results_B['point_estimate']
    spatial_coords_B = results_B['spatial_coords']
    kde_model_B = results_B['kde_model']
    tau_B = results_B['tau']
    
    if point_estimate_A is None or point_estimate_B is None:
        print(f"Cannot perform test: A or B has no KDE-VI point estimate.")
        return
        
    print(f"Source ({name_A}): {spatial_coords_A.shape[0]} cells, {len(np.unique(point_estimate_A))} clusters.")
    print(f"Target ({name_B}): {spatial_coords_B.shape[0]} cells, Conf. Threshold = {tau_B:.4f}")

    print(f"Finding closest cell in {name_A} for each cell in {name_B}...")
    translated_partition_B = _spatial_cluster_assignment(
        spatial_coords_A, 
        point_estimate_A, 
        spatial_coords_B
    )
    print(f"Assigning {name_B} cells to {name_A} closest cell's cluster...")

    print(f"Scoring translated partition using {name_B}'s KDE model...")
    score_translated = -kde_model_B.log_prob(translated_partition_B)
    
    is_in_set = score_translated <= tau_B
    
    print("\n--- TEST RESULT ---")
    print(f"Translated Partition Score: {score_translated:.4f}")
    print(f"Conformal Threshold (tau) for {name_B}: {tau_B:.4f}")
    if is_in_set:
        print(f"✅ RESULT: The {name_A}-derived clustering IS IN the {name_B} credible region.")
    else:
        print(f"❌ RESULT: The {name_A}-derived clustering IS NOT IN the {name_B} credible region.")

    # --- Plot B's point estimate side-by-side with the translated partition ---
    print(f"\nPlotting comparison for {name_B}...")
    
    plot_labels = [point_estimate_B, translated_partition_B]
    plot_titles = [
        f"{name_B} Cells Clustering Point Estimate", 
        f"{name_A}-Translated Clustering of {name_B} Cells"
    ]
    
    plot_clusters_fixed_points(
        spatial_coords_B,
        plot_labels,
        plot_titles,
        xlabel="X", ylabel="Y", # These will be modified by the plot function
        save_path=os.path.join(config['output_filepath'], f'comparison_{name_A}_to_{name_B}.eps')
    )
    
    print(f"==============================================\n")


# ==============================================
# 5. Main Execution
# ==============================================
if __name__ == "__main__":

    SEED = 12
    np.random.seed(SEED)
    
    # --- Configuration ---
    # This is the base folder where your data files are and where plots will be saved.
    OUTPUT_FILEPATH = './spatial_transcriptomics_experiment/' 
    
    # --- Expected File Names ---
    # The script will look for files generated by the R conversion script:
    # - "Tumor_mcmc_partitions.csv"
    # - "Tumor_spatial_coords.csv"
    # - "Immune_mcmc_partitions.csv"
    # - "Immune_spatial_coords.csv"

    # Analysis config from user
    analysis_config = {
        'alpha_conf': 0.1,
        'kde_metric': 'vi', # Use 'vi' as requested
        'kde_gamma': 0.5,
        'kde_subsample_size': None,
        'output_filepath': OUTPUT_FILEPATH
    }

    # --- 1. Load Data From .csv Files ---
    os.makedirs(analysis_config['output_filepath'], exist_ok=True)
    
    tumor_data = None
    immune_data = None

    print("--- 1. Loading and Preparing Data from .csv Files ---")
    print(f"Looking for .csv files in: {OUTPUT_FILEPATH}")
    
    try:
        tumor_data = load_sarp_data_from_csv(
            cell_type_prefix='Tumor', 
            config=analysis_config
        )
        
        immune_data = load_sarp_data_from_csv(
            cell_type_prefix='Immune', 
            config=analysis_config
        )
        print("Data loading successful.")
    except FileNotFoundError as e:
        print(f"\n\nERROR: Could not find required data file: {e.filename}")
        print("Please make sure you have run the R conversion script (Step 1) first.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        import traceback
        traceback.print_exc() # Print full error trace
        sys.exit(1)


    # --- 2. Prepare Shared Train/Calibration Indices ---
    try:
        if tumor_data and immune_data:
            if len(tumor_data['mcmc_partitions']) != len(immune_data['mcmc_partitions']):
                print("ERROR: MCMC partition lists have different lengths. Cannot proceed with shared indexing.")
                sys.exit(1)
                
            n_partitions_total = len(tumor_data['mcmc_partitions'])
            print(f"\n--- 2. Preparing Shared Train/Calibration Indices ---")
            print(f"Total partitions per cell type: {n_partitions_total}")
            
            all_indices = list(range(n_partitions_total))
            
            # (a) Calibration samples by thinning
            calib_indices = list(range(0, n_partitions_total, 5)) # 0, 10, 20...
            print(f"Selected {len(calib_indices)} calibration samples (every 5th, starting at 1).")

            # (b) Remainder are test samples
            calib_indices_set = set(calib_indices)
            train_indices = [i for i in all_indices if i not in calib_indices_set]
            
        else:
            print("Missing data for Tumor or Immune, cannot create shared indices.")
            sys.exit(1)
            
    except Exception as e:
        print(f"An error occurred during index preparation: {e}")
        import traceback
        traceback.print_exc() # Print full error trace
        sys.exit(1)


    # --- 3. Run Conformal Pipeline for each cell type ---
    tumor_results = None
    immune_results = None
    
    try:
        if tumor_data:
            tumor_results = run_conformal_pipeline(
                tumor_data, 
                analysis_config,
                figure_prefix="Tumor",
                train_indices=train_indices,   
                calib_indices=calib_indices  
            )
        
        if immune_data:
            immune_results = run_conformal_pipeline(
                immune_data, 
                analysis_config,
                figure_prefix="Immune",
                train_indices=train_indices,   
                calib_indices=calib_indices  
            )
    except Exception as e:
        print(f"\n\nAn error occurred during the main analysis pipeline: {e}")
        import traceback
        traceback.print_exc() # Print full error trace
        sys.exit(1)

    # --- 4. Plot Combined DPC Decision Graphs ---
    try:
        if tumor_results and immune_results:
            print("\n--- Plotting Combined DPC Decision Graphs ---")
            combined_dpc_path = os.path.join(
                analysis_config['output_filepath'], 
                'Tumor_Immune_dpc_decision_graph_combined.eps' # Save as .eps
            )
            plot_combined_dpc_graphs(
                tumor_results,
                immune_results,
                combined_dpc_path
            )
    except Exception as e:
        print(f"An error occurred during combined DPC plotting: {e}")
        import traceback
        traceback.print_exc() # Print full error trace

    # --- 5. Perform the requested Translation Tests ---
    
    try:
        if tumor_results and immune_results:
            # Test 1: (a=Tumor, b=Immune)
            perform_spatial_translation_test(
                results_A=tumor_results, 
                results_B=immune_results,
                name_A="Tumor",
                name_B="Immune",
                config=analysis_config
            )
            
            # Test 2: (a=Immune, b=Tumor)
            perform_spatial_translation_test(
                results_A=immune_results, 
                results_B=tumor_results,
                name_A="Immune",
                name_B="Tumor",
                config=analysis_config
            )
        else:
            print("\nSkipping translation tests due to error in pipeline.")
    except Exception as e:
        print(f"\n\nAn error occurred during the translation test: {e}")
        import traceback
        traceback.print_exc() # Print full error trace
        sys.exit(1)

    print("\n--- Full Workflow Complete ---")