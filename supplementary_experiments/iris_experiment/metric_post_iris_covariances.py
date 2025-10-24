# ==============================================
# 0. Imports
# ==============================================
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from tqdm import tqdm
from joblib import Parallel, delayed # Make sure delayed is imported
import math
import os

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

# Tell numpyro to use a single CPU core to avoid warnings
numpyro.set_host_device_count(1)

# --- MODIFICATION: Increased font sizes for paper readability ---
plt.rc('font', size=18) # controls default text sizes
plt.rc('axes', titlesize=18) # fontsize of the (SUBPLOT) axes title
plt.rc('axes', labelsize=18) # fontsize of the x and y labels
plt.rc('xtick', labelsize=16) # fontsize of the tick labels
plt.rc('ytick', labelsize=16) # fontsize of the tick labels
plt.rc('legend', fontsize=16) # legend fontsize
plt.rc('figure', titlesize=22) # fontsize of the figure title
# -----------------------------------------------------------------

# ==============================================
# 1. Load and Prepare the Data
# ==============================================

def load_and_prepare_iris_data():
    """
    Loads the Iris dataset, filters for a single species, and then
    centers and scales it for a mean=0 model.
    """
    print("Loading Iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names

    # Revert to using a single species (species 0: Iris setosa)
    species_name = "setosa"
    X_subset = X[y == 0]
    print(f"Using the Iris {species_name} subset.")

    # IMPORTANT: Center the data since the model assumes a mean of 0
    data_centered = X_subset - X_subset.mean(axis=0)

    # Scale the data by its standard deviation to make it a correlation matrix
    data_scaled = data_centered / data_centered.std(axis=0)

    n, d = data_scaled.shape
    print(f"Data prepared and scaled: {n} samples, {d} variables")

    return data_scaled, feature_names, species_name

# ==============================================
# 2. NumPyro Model for a Single Covariance
# ==============================================

def simple_covariance_model(data, d):
    """
    A simple Multivariate Normal model with a mean of 0 and a
    standard LKJ prior on the covariance matrix. This is suitable
    for unimodal data.
    """
    sigma = numpyro.sample("sigma", dist.HalfCauchy(1.0).expand([d]))
    L_corr = numpyro.sample("L_corr", dist.LKJCholesky(d, concentration=1.0))
    cov = jnp.diag(sigma) @ L_corr @ L_corr.T @ jnp.diag(sigma)

    # Only sample observations if data is provided (for prior sampling)
    if data is not None:
        numpyro.sample(
            "obs",
            dist.MultivariateNormal(loc=jnp.zeros(d), covariance_matrix=cov),
            obs=data
        )

# ==============================================
# 3. Run MCMC Sampler
# ==============================================

def run_mcmc_and_get_samples(data, d, num_samples=6000, num_warmup=2000, seed=0, thinning=5):
    """
    Initializes and runs NUTS, then reconstructs and returns the
    posterior samples for the full covariance matrix.
    """
    print("\n--- Running Numpyro MCMC Sampler ---")
    rng_key = jax.random.PRNGKey(seed)

    nuts_kernel = NUTS(simple_covariance_model)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, thinning=thinning, progress_bar=True)
    mcmc.run(rng_key, data=data, d=d)

    print("\nMCMC sampling complete. Reconstructing covariance matrices...")
    posterior_samples = mcmc.get_samples()

    sigma_samples = posterior_samples['sigma']
    L_corr_samples = posterior_samples['L_corr']

    cov_samples = []
    for i in tqdm(range(num_samples), desc="Reconstructing posterior covariances"):
        S = jnp.diag(sigma_samples[i])
        L = L_corr_samples[i]
        cov = S @ L @ L.T @ S
        cov_samples.append(np.array(cov))

    return np.array(cov_samples)


# ==============================================
# 4. Analysis Components
# ==============================================

def spectral_norm_distance(m1, m2):
    """Computes the operator norm (spectral norm) distance between two matrices."""
    return np.linalg.norm(m1 - m2, ord=2)

class MetricKDE:
    def __init__(self, train_samples, metric_fn, gamma=1.0):
        self.train_samples_ = train_samples
        self.m_ = len(self.train_samples_)
        self.gamma_ = gamma
        self.metric_fn_ = metric_fn
        print(f"MetricKDE initialized with {self.m_} training samples.")

    def log_prob(self, test_sample):
        if self.m_ == 0: return -np.inf
        distances = np.array([self.metric_fn_(test_sample, p) for p in self.train_samples_])
        density = np.sum(np.exp(-self.gamma_ * distances)) / self.m_
        return np.log(density + 1e-9)

# --- MODIFICATION: Added helper for parallel scoring ---
def score_sample(model, sample):
    """Helper function to score a single sample for parallel execution."""
    return -model.log_prob(sample)
# --- END MODIFICATION ---

def _compute_distances_for_row(i, samples, metric_fn):
    """Helper for parallel computation of one row of the distance matrix for DPC."""
    n = len(samples)
    return [metric_fn(samples[i], samples[j]) for j in range(i, n)]

def plot_matrix_heatmaps(matrices, titles, feature_names, filepath=None):
    """Utility to plot and save covariance matrix heatmaps using Matplotlib."""
    n = len(matrices)
    if n == 0: return
    n_cols = min(n, 4)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 7*n_rows), squeeze=False)

    axes_flat = axes.flatten()
    for i, (matrix, title) in enumerate(zip(matrices, titles)):
        ax = axes_flat[i]

        im = ax.imshow(matrix, cmap="viridis", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_yticks(np.arange(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha="right")
        ax.set_yticklabels(feature_names)

        # ax.set_title(title, pad=20) # Title removed

    for i in range(n, n_rows*n_cols):
        axes_flat[i].axis('off')
    plt.tight_layout(pad=3.0, w_pad=3.0)
    if filepath:
        plt.savefig(filepath, bbox_inches='tight', format='eps')
        print(f"Saved heatmap plot to {filepath}")
    plt.show()

# ==============================================
# 5. Full Analysis Pipeline
# ==============================================

def run_full_analysis(mcmc_samples, X_data, d_vars, feature_names, species_name, config, output_dir):
    """
    Performs the full analysis pipeline: KDE, Point Estimation, Conformal, DPC, Prior Check.
    """
    print("\n" + "="*50)
    print("### Starting Full Posterior Analysis ###")
    print("="*50)

    num_total_samples = len(mcmc_samples)

    # --- Train / Calibration Split ---
    np.random.seed(42)
    indices = np.random.permutation(num_total_samples)
    split_idx = int(num_total_samples * 5 / 6)
    train_indices, calib_indices = indices[:split_idx], indices[split_idx:]
    train_samples = [mcmc_samples[i] for i in train_indices]
    calib_samples = [mcmc_samples[i] for i in calib_indices]

    print(f"\nTotal samples: {num_total_samples} | Training: {len(train_samples)} | Calibration: {len(calib_samples)}")

    print("\n--- Building KDE and Scoring Calibration Samples (in parallel) ---")
    kde_model = MetricKDE(train_samples, metric_fn=spectral_norm_distance, gamma=config['kde_gamma'])
    # --- MODIFICATION: Use score_sample helper for calibration ---
    tasks_calib = [delayed(score_sample)(kde_model, s) for s in calib_samples]
    scores = np.array(Parallel(n_jobs=-1)(tqdm(tasks_calib, desc="Scoring Calibration Samples")))
    log_probs = -scores # Recalculate log_probs from scores

    # --- 1. Point Estimation ---
    print("\n--- Finding Posterior Mode (Point Estimate) ---")
    mode_idx_calib = np.argmax(log_probs) # Use log_probs to find max density
    posterior_mode = calib_samples[mode_idx_calib]
    print(f"Found posterior mode at calibration index {mode_idx_calib} (Max KDE score).")
    plot_matrix_heatmaps(
        [posterior_mode],
        [""], # No title
        feature_names,
        filepath=os.path.join(output_dir, f'posterior_mode_{species_name}.eps')
    )

    # --- 2. Credible Region Construction ---
    print("\n--- Building Credible Set using Conformal Inference ---")
    n_calib = len(scores)
    alpha = config['alpha_conf']
    k_hat = int(np.ceil((n_calib + 1) * (1 - alpha)))
    if k_hat > n_calib: k_hat = n_calib

    if n_calib > 0:
        tau = np.sort(scores)[k_hat - 1]
    else:
        tau = np.inf

    print(f"Conformal threshold (alpha={alpha}): {tau:.4f}")

    # --- Prior Predictive Check ---
    print("\n--- Generating and Scoring Prior Samples (in parallel) ---")
    num_prior_samples = 1000
    prior_seed = 43
    prior_predictive = Predictive(simple_covariance_model, num_samples=num_prior_samples)
    prior_samples_raw = prior_predictive(jax.random.PRNGKey(prior_seed), data=None, d=d_vars)

    prior_sigma = prior_samples_raw['sigma']
    prior_L_corr = prior_samples_raw['L_corr']

    prior_cov_samples = []
    for i in tqdm(range(num_prior_samples), desc="Reconstructing prior covariances"):
        S = jnp.diag(prior_sigma[i])
        L = prior_L_corr[i]
        cov = S @ L @ L.T @ S
        prior_cov_samples.append(np.array(cov))

    print(f"Scoring {num_prior_samples} prior samples (in parallel)...")
    # --- MODIFICATION: Parallelize prior scoring ---
    tasks_prior = [delayed(score_sample)(kde_model, s) for s in prior_cov_samples]
    prior_scores = np.array(Parallel(n_jobs=-1)(tqdm(tasks_prior, desc="Scoring Prior Samples")))
    # --- END MODIFICATION ---

    prior_in_set_count = np.sum(prior_scores <= tau)
    prior_in_set_fraction = prior_in_set_count / num_prior_samples if num_prior_samples > 0 else 0

    # --- Test Empirical Covariance ---
    empirical_cov = np.cov(X_data, rowvar=False)
    score_empirical = score_sample(kde_model, empirical_cov) # Use helper
    empirical_in_set = score_empirical <= tau

    # --- 3. Multimodality Analysis ---
    print("\n--- Multimodality Analysis using Density Peak Clustering ---")
    dist_matrix_calib = np.zeros((len(calib_samples), len(calib_samples)))
    tasks_dist = [delayed(_compute_distances_for_row)(i, calib_samples, kde_model.metric_fn_) for i in range(len(calib_samples))]
    results_dist = Parallel(n_jobs=-1)(tqdm(tasks_dist, desc="Building DPC matrix"))
    for i, row_distances in enumerate(results_dist):
        for j_offset, dist_val in enumerate(row_distances):
            j = i + j_offset
            dist_matrix_calib[i, j] = dist_matrix_calib[j, i] = dist_val

    rho = np.exp(log_probs)
    sorted_rho_indices = np.argsort(rho)[::-1]
    delta = np.zeros(n_calib)
    if n_calib > 0:
        delta[sorted_rho_indices[0]] = np.max(dist_matrix_calib[sorted_rho_indices[0], :]) if n_calib > 1 else 1.0
        for i in range(1, n_calib):
            current_idx = sorted_rho_indices[i]
            higher_density_indices = sorted_rho_indices[:i]
            delta[current_idx] = np.min(dist_matrix_calib[current_idx, higher_density_indices])

    rho_threshold = 0.5
    delta_threshold = 0.8
    center_indices = np.where((rho > rho_threshold) & (delta > delta_threshold))[0]
    print(f"Found {len(center_indices)} potential modes (Expecting 1 for this dataset).")

    plt.figure(figsize=(10, 7))
    plt.scatter(rho, delta, s=30, alpha=0.6, label='Calibration Samples')
    if len(center_indices) > 0:
        plt.scatter(rho[center_indices], delta[center_indices], s=150, alpha=0.9, edgecolor='k', facecolor='red', marker='o', label='Modes')
    plt.xlabel('Density (s)'); plt.ylabel('Distance (δ)'); plt.grid(True); plt.legend()
    plt.savefig(os.path.join(output_dir, f'dpc_decision_plot_{species_name}.eps'), bbox_inches='tight')
    plt.show()

    if len(center_indices) > 0:
        matrices_to_plot = [calib_samples[i] for i in center_indices]
        titles = ["" for i in range(len(center_indices))] # No titles
        plot_matrix_heatmaps(
            matrices_to_plot,
            titles,
            feature_names,
            filepath=os.path.join(output_dir, f'dpc_modes_{species_name}.eps')
        )

    # --- Print Summary Results ---
    print("\n" + "="*50)
    print("### Conformal Test Summary ###")
    print("="*50)
    print(f" - Empirical Covariance Score: {score_empirical:.4f}")
    print(f" - Empirical Covariance In Set (<= {tau:.4f}): {empirical_in_set}")
    print("-" * 50)
    print(f" - Prior Samples Tested: {num_prior_samples}")
    print(f" - Prior Samples In Set (<= {tau:.4f}): {prior_in_set_count}")
    print(f" - Prior Inclusion Fraction: {prior_in_set_fraction:.4f}")
    print("="*50)

# ==============================================
# 6. Main Execution
# ==============================================
if __name__ == "__main__":
    # Define configurations
    mcmc_config = {
        'num_samples': 6000,
        'num_warmup': 2000,
        'seed': 42,
        'thinning': 5
    }
    analysis_config = {
        'alpha_conf': 0.1,
        'kde_gamma': 0.5
    }

    # Define and create the output directory for plots
    output_dir = r'.\iris_experiment'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir}")

    # 1. Get the data
    data_centered, feature_names, species_name = load_and_prepare_iris_data()
    n_obs, d_vars = data_centered.shape

    # 2. Run the MCMC sampler
    mcmc_samples = run_mcmc_and_get_samples(
        data_centered,
        d_vars,
        num_samples=mcmc_config['num_samples'],
        num_warmup=mcmc_config['num_warmup'],
        seed=mcmc_config['seed'],
        thinning=mcmc_config['thinning']
    )

    # 3. Run the full analysis pipeline
    run_full_analysis(mcmc_samples, data_centered, d_vars, feature_names, species_name, analysis_config, output_dir)