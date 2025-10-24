# ==============================================
# 0. Imports
# ==============================================
import os
import pickle
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D

# --- JAX and Numpyro for MCMC ---
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as jax_norm, gamma
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive # Added Predictive

# --- SciPy, Scikit-learn, and OT for Analysis ---
from scipy.stats import norm as scipy_norm
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import ot # Python Optimal Transport for EMD
import warnings # Added

# Tell numpyro to use a single CPU core to avoid warnings
numpyro.set_host_device_count(1)

plt.rc('font', size=18) # controls default text sizes
plt.rc('axes', titlesize=18) # fontsize of the (SUBPLOT) axes title (will be ignored)
plt.rc('axes', labelsize=18) # fontsize of the x and y labels
plt.rc('xtick', labelsize=16) # fontsize of the tick labels
plt.rc('ytick', labelsize=16) # fontsize of the tick labels
plt.rc('legend', fontsize=16) # legend fontsize
plt.rc('figure', titlesize=22) # fontsize of the figure title (will be ignored)


# ==============================================
# 1. Data Loading & MCMC (Shared Components)
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

def pyp_gmm_stick_breaking(K, alpha, discount):
    """ Pitman-Yor Process stick-breaking construction """
    with numpyro.plate("beta_plate", K - 1):
        beta = numpyro.sample("beta", dist.Beta(1 - discount, alpha + jnp.arange(K - 1) * discount))

    beta_padded = jnp.concatenate([beta, jnp.array([1.0])])
    log_sticks = jnp.log(beta_padded)
    log_rest_sticks = jnp.log1p(-beta_padded)
    log_cum_rest_sticks = jnp.cumsum(log_rest_sticks)
    log_w = log_sticks + jnp.concatenate([jnp.array([0.0]), log_cum_rest_sticks[:-1]])
    # Store weights explicitly for prior sampling if needed
    numpyro.deterministic("weights", jnp.exp(log_w))
    return log_w


def pyp_gmm_model(y, K, alpha, discount, lambda_prior, psi_prior, nu_prior):
    """ Numpyro model for a Pitman-Yor Process Gaussian Mixture Model. """
    # Use a fixed reference point (e.g., 0) for mu0 if y is None during prior sampling
    mu0 = jnp.mean(y) if y is not None else 0.0

    with numpyro.plate("components", K):
        sigma2 = numpyro.sample("sigma2", dist.InverseGamma(nu_prior / 2, psi_prior / 2))
        mu = numpyro.sample("mu", dist.Normal(mu0, jnp.sqrt(sigma2 / lambda_prior)))

    log_w = pyp_gmm_stick_breaking(K, alpha, discount)

    # Only compute likelihood if data is provided
    if y is not None:
        log_likelihood = dist.Normal(mu, jnp.sqrt(sigma2)).log_prob(y[:, None]) + log_w
        log_mix_likelihood = jax.scipy.special.logsumexp(log_likelihood, axis=1)
        numpyro.factor("log_likelihood", jnp.sum(log_mix_likelihood))

def run_numpyro_mcmc(X, config, filepath):
    """ Runs MCMC using Numpyro and saves the posterior samples. """
    print("--- Running Numpyro MCMC Sampler for PYP-GMM ---")
    num_samples_to_generate = config['desired_thinned_samples'] * config['thinning']
    print(f"Desired thinned samples: {config['desired_thinned_samples']}, Thinning factor: {config['thinning']}")
    print(f"--> Numpyro will generate {num_samples_to_generate} raw samples.")

    kernel = NUTS(pyp_gmm_model)
    mcmc = MCMC(kernel, num_warmup=config['burn_in'], num_samples=num_samples_to_generate, num_chains=1, progress_bar=True)

    rng_key = jax.random.PRNGKey(config['seed'])
    # Pass flattened data as y
    mcmc.run(rng_key, y=jnp.array(X.flatten()), K=config['truncation_level'], alpha=config['alpha'],
             discount=config['pyd'], lambda_prior=config['lambda_prior'],
             psi_prior=config['psi_prior'], nu_prior=config['nu_prior'])

    print("\nExtracting and thinning samples...")
    posterior_samples = mcmc.get_samples()

    # Reconstruct weights from beta samples (as done in original script)
    betas = posterior_samples['beta']
    betas_padded = jnp.pad(betas, ((0, 0), (0, 1)), constant_values=1.0)
    remaining_stick_lengths = jnp.concatenate([jnp.ones((betas.shape[0], 1)), jnp.cumprod(1 - betas, axis=1)], axis=1)
    weights = betas_padded * remaining_stick_lengths

    thinned_indices = np.arange(0, num_samples_to_generate, config['thinning'])
    final_samples = {
        'mu': posterior_samples['mu'][thinned_indices],
        'sigma2': posterior_samples['sigma2'][thinned_indices],
        'weights': weights[thinned_indices], # Use reconstructed weights
        'config': config, 'X': X
    }

    print(f"Completed. Total final samples collected: {len(thinned_indices)}")

    save_full_path = os.path.join(filepath, 'mcmc_results_mixing_measure.pkl')
    with open(save_full_path, 'wb') as f:
        pickle.dump(final_samples, f)
    print(f"MCMC results saved to {save_full_path}")


# ==============================================
# 2. Analysis of Mixing Measures (Earth Mover's Distance)
# ==============================================

def _emd_distance_mixing_measure(sample1, sample2):
    """
    Computes the Earth Mover's Distance (Wasserstein-1) between two discrete
    mixing measures on the 2D parameter space (mean, variance).
    """
    w1, m1, s2_1 = sample1['weights'], sample1['mu'], sample1['sigma2']
    w2, m2, s2_2 = sample2['weights'], sample2['mu'], sample2['sigma2']

    # Filter negligible weights BEFORE normalization
    idx1 = w1 > 1e-6; w1, m1, s2_1 = w1[idx1], m1[idx1], s2_1[idx1]
    idx2 = w2 > 1e-6; w2, m2, s2_2 = w2[idx2], m2[idx2], s2_2[idx2]

    if len(w1) == 0 or len(w2) == 0: return np.inf # Return inf if one is empty

    # Normalize weights AFTER filtering
    w1_norm = np.array(w1 / w1.sum(), dtype=np.float64)
    w2_norm = np.array(w2 / w2.sum(), dtype=np.float64)

    locs1 = np.vstack((m1, s2_1)).T
    locs2 = np.vstack((m2, s2_2)).T
    cost_matrix = pairwise_distances(locs1, locs2)

    return ot.emd2(w1_norm, w2_norm, cost_matrix)


def plot_mixing_measure_3d(ax, sample, title, on_log_scale=False):
    """ Creates a 3D stem plot of a single mixing measure sample. """
    w, m, s2 = sample['weights'], sample['mu'], sample['sigma2']
    idx = w > 1e-3
    w, m, s2 = w[idx], m[idx], s2[idx]
    for i in range(len(w)):
        ax.plot([m[i], m[i]], [s2[i], s2[i]], [0, w[i]], marker='o', color='b', markersize=5, linestyle='-')
    # ax.set_title(title) # Title removed
    ax.set_xlabel("Log Mean" if on_log_scale else "Mean (μ)")
    ax.set_ylabel("Variance (σ²)"); ax.set_zlabel("Weight (w)")
    ax.view_init(elev=20., azim=-65)

class MetricKDE:
    def __init__(self, train_samples, metric_fn, gamma=1.0):
        self.train_samples_ = train_samples
        self.m_ = len(self.train_samples_)
        self.gamma_ = gamma
        self.metric_fn_ = metric_fn
        print(f"MetricKDE initialized with {self.m_} samples.")

    def log_prob(self, test_sample):
        if self.m_ == 0: return -np.inf
        distances = np.array([self.metric_fn_(test_sample, p) for p in self.train_samples_])
        # Handle potential inf distances
        distances = np.nan_to_num(distances, nan=np.inf)
        if np.any(np.isinf(distances)):
            print("Warning: Infinite distance encountered in KDE scoring.") # Debug print
        density = np.sum(np.exp(-self.gamma_ * distances[np.isfinite(distances)])) / self.m_ # Only sum finite
        return np.log(density + 1e-9)

def score_sample(model, sample):
    """ Helper function to score a single sample (negative log probability). """
    return -model.log_prob(sample)


def _compute_distances_for_row(i, samples, metric_fn):
    """ Helper for parallel computation of one row of the distance matrix for DPC. """
    n = len(samples)
    return [metric_fn(samples[i], samples[j]) for j in range(i, n)]

def run_analysis(filepath, config, X, all_samples, mcmc_run_config, on_log_scale=False): # Added mcmc_run_config
    """ Analysis workflow for MCMC samples of mixing measures. """
    print("\n" + "="*50)
    print("### Starting Analysis of Mixing Measures (EMD) ###")
    print("="*50)

    num_total_samples = len(all_samples['mu'])
    mcmc_samples = [{'weights': np.array(all_samples['weights'][i]), 'mu': np.array(all_samples['mu'][i]), 'sigma2': np.array(all_samples['sigma2'][i])} for i in range(num_total_samples)]

    np.random.seed(42)
    indices = np.random.permutation(num_total_samples)
    split_idx = int(num_total_samples * 5/6)
    train_indices, calib_indices = indices[:split_idx], indices[split_idx:]
    train_samples = [mcmc_samples[i] for i in train_indices]
    calib_samples = [mcmc_samples[i] for i in calib_indices]

    print(f"\nTotal samples: {num_total_samples} | Training: {len(train_samples)} | Calibration: {len(calib_samples)}")

    print("\n--- Building KDE and Scoring Samples (in parallel) ---")
    kde_model = MetricKDE(train_samples, metric_fn=_emd_distance_mixing_measure, gamma=config['kde_gamma_mixing'])
    tasks = [delayed(score_sample)(kde_model, s) for s in calib_samples] # Use score_sample
    scores = np.array(Parallel(n_jobs=-1)(tqdm(tasks, desc="Scoring Mixing Measures")))
    log_probs = -scores # Recalculate log_probs

    # --- 1. Point Estimation ---
    print("\n--- Finding Posterior Mode (Point Estimate) ---")
    mode_idx = np.argmax(log_probs)
    posterior_mode = calib_samples[mode_idx]
    print(f"Found posterior mode at calibration index {mode_idx} (Max KDE score).")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_mixing_measure_3d(ax, posterior_mode, "", on_log_scale) # No title
    plt.savefig(os.path.join(filepath, 'mode_mixing_measure_3d.eps'), bbox_inches='tight'); plt.show() # Save as eps

    # --- 2. Credible Region Construction ---
    print("\n--- Building Credible Set (EMD) using Conformal Inference ---")
    n_calib = len(scores)
    alpha = config['alpha_conf']
    k_hat = int(np.ceil((n_calib + 1) * (1 - alpha)))
    if k_hat > n_calib: k_hat = n_calib

    if n_calib > 0:
        tau = np.sort(scores)[k_hat - 1]
    else:
        tau = np.inf

    print(f"Conformal threshold (alpha={alpha}): {tau:.4f}")

    print("\n--- Generating and Scoring Prior Mixing Measures (in parallel) ---")
    num_prior_samples = 1000
    prior_seed = 43 # Different seed
    prior_predictive = Predictive(pyp_gmm_model, num_samples=num_prior_samples)
    # Sample from prior by setting y=None
    prior_samples_raw = prior_predictive(
        jax.random.PRNGKey(prior_seed),
        y=None,
        K=mcmc_run_config['truncation_level'], # Use K from MCMC config
        alpha=mcmc_run_config['alpha'],
        discount=mcmc_run_config['pyd'],
        lambda_prior=mcmc_run_config['lambda_prior'],
        psi_prior=mcmc_run_config['psi_prior'],
        nu_prior=mcmc_run_config['nu_prior']
    )

    # Construct prior mixing measure samples (weights are directly sampled now)
    prior_mixing_samples = [
        {'weights': np.array(prior_samples_raw['weights'][i]),
         'mu': np.array(prior_samples_raw['mu'][i]),
         'sigma2': np.array(prior_samples_raw['sigma2'][i])}
        for i in range(num_prior_samples)
    ]

    print(f"Scoring {num_prior_samples} prior mixing measures (in parallel)...")
    tasks_prior = [delayed(score_sample)(kde_model, s) for s in prior_mixing_samples]
    prior_scores = np.array(Parallel(n_jobs=-1)(tqdm(tasks_prior, desc="Scoring Prior Mixing Measures")))

    prior_in_set_count = np.sum(prior_scores <= tau)
    prior_in_set_fraction = prior_in_set_count / num_prior_samples if num_prior_samples > 0 else 0


    conformal_results = [{'type': 'threshold', 'tau': tau}]
    print("\nTesting if simple K-Means GMMs are in the credible set...") # Moved print statement
    for k in [3, 4, 5, 6]:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X)
            cluster_sizes = np.bincount(kmeans.labels_, minlength=k)
            if np.any(cluster_sizes == 0):
                print(f"   - K-Means GMM (k={k}) Skipped: Resulted in empty cluster(s).")
                conformal_results.append({'type': 'test', 'name': f'K-Means (k={k})', 'score': np.nan, 'in_set': False, 'skipped': True})
                continue

            kmeans_sample = {'mu': kmeans.cluster_centers_.flatten(),
                             'sigma2': np.array([np.var(X[kmeans.labels_ == i]) for i in range(k)]),
                             'weights': np.array([np.sum(kmeans.labels_ == i) / len(X) for i in range(k)])}
            score_kmeans = score_sample(kde_model, kmeans_sample) # Use helper
            in_set = score_kmeans <= tau
            print(f"   - K-Means GMM (k={k}) Score: {score_kmeans:.4f} -> In Set: {in_set}")
            conformal_results.append({'type': 'test', 'name': f'K-Means (k={k})', 'score': score_kmeans, 'in_set': in_set})
        except Exception as e:
            print(f"   - K-Means GMM (k={k}) Failed with error: {e}")
            conformal_results.append({'type': 'test', 'name': f'K-Means (k={k})', 'score': np.nan, 'in_set': False, 'error': str(e)})


    # --- 3. Multimodality Analysis ---
    print("\n--- Multimodality Analysis (EMD) using Density Peak Clustering ---")
    dist_matrix_calib = np.zeros((len(calib_samples), len(calib_samples)))
    tasks_dist = [delayed(_compute_distances_for_row)(i, calib_samples, kde_model.metric_fn_) for i in range(len(calib_samples))]
    results_dist = Parallel(n_jobs=-1)(tqdm(tasks_dist, desc="Building DPC matrix (EMD)"))
    for i, row_distances in enumerate(results_dist):
        for j_offset, dist in enumerate(row_distances):
            j = i + j_offset
            dist_matrix_calib[i, j] = dist_matrix_calib[j, i] = dist

    rho = np.exp(log_probs)
    sorted_rho_indices = np.argsort(rho)[::-1]

    delta = np.zeros(n_calib)
    if n_calib > 0:
        delta[sorted_rho_indices[0]] = np.max(dist_matrix_calib[sorted_rho_indices[0], :]) if n_calib > 1 else 1.0
        for i in range(1, n_calib):
            current_idx = sorted_rho_indices[i]
            higher_density_indices = sorted_rho_indices[:i]
            dist_to_higher = dist_matrix_calib[current_idx, higher_density_indices]
            delta[current_idx] = np.min(dist_to_higher) if len(dist_to_higher) > 0 else delta[sorted_rho_indices[0]]

    rho_threshold = np.quantile(rho, 0.90) if n_calib > 0 else 0
    delta_threshold = np.quantile(delta, 0.90) if n_calib > 0 else 0
    center_indices = np.where((rho > rho_threshold) & (delta > delta_threshold))[0]
    print(f"Found {len(center_indices)} potential modes in the posterior.")

    plt.figure(figsize=(10, 7))
    plt.scatter(rho, delta, s=30, alpha=0.6)
    if len(center_indices) > 0:
        plt.scatter(rho[center_indices], delta[center_indices], s=150, alpha=0.9, edgecolor='k', facecolor='red', marker='o', label='Modes')
    plt.xlabel('Density (s)'); plt.ylabel('Distance (δ)'); plt.grid(True)
    plt.legend();
    plt.savefig(os.path.join(filepath, 'dpc_decision_graph_mixing_measure.eps'), bbox_inches='tight'); plt.show() # Save as eps

    if len(center_indices) > 0:
        num_modes = len(center_indices)
        fig = plt.figure(figsize=(8 * num_modes, 8))
        for i, center_idx in enumerate(center_indices):
            ax = fig.add_subplot(1, num_modes, i + 1, projection='3d')
            plot_mixing_measure_3d(ax, calib_samples[center_idx], "", on_log_scale) # No title
        plt.savefig(os.path.join(filepath, 'dpc_modes_mixing_measure_3d.eps'), bbox_inches='tight'); plt.show() # Save as eps

    # --- Add prior results to the conformal_results list for printing ---
    conformal_results.append({
        'type': 'prior_check',
        'num_tested': num_prior_samples,
        'num_in_set': prior_in_set_count,
        'fraction_in_set': prior_in_set_fraction
    })

    return conformal_results

# ==============================================
# 3. Main Execution
# ==============================================
if __name__ == "__main__":
    LOG_TRANSFORM_DATA = False

    simulation_config = {
        'seed': 12345,
        'desired_thinned_samples': 6000,
        'burn_in': 1000,
        'thinning': 5,
        'truncation_level': 10,
        'alpha': 2,
        'pyd': 0.05,
        'lambda_prior': 0.001,
        'psi_prior': 0.5,
        'nu_prior': 2
    }
    analysis_config = {
        'alpha_conf': 0.1,
        'kde_gamma_mixing': 2.0
    }

    RERUN_MCMC = True # Set to True to re-run MCMC
    output_filepath = './galaxies_experiment_supplement/'
    if not os.path.exists(output_filepath): os.makedirs(output_filepath)
    mcmc_results_path = os.path.join(output_filepath, 'mcmc_results_mixing_measure.pkl')

    # Use a relative path assuming the script is run from the project root or the file is in the same dir
    galaxy_csv_path = 'galaxies.csv'

    if RERUN_MCMC or not os.path.exists(mcmc_results_path):
        X = load_galaxy_data(galaxy_csv_path, log_transform=LOG_TRANSFORM_DATA)
        if X is not None:
            # Optionally plot histogram (no title)
            plt.figure(figsize=(10, 6))
            xlabel_hist = "Log Galaxy Velocity (1000 km/s)" if LOG_TRANSFORM_DATA else "Galaxy Velocity (1000 km/s)"
            plt.hist(X.flatten(), bins=60, color='skyblue', edgecolor='black', alpha=0.7)
            plt.xlabel(xlabel_hist); plt.ylabel("Frequency"); plt.grid(axis='y', linestyle='--')
            plt.savefig(os.path.join(output_filepath, 'galaxy_histogram.eps'), bbox_inches='tight') # Save as eps
            plt.show()

            run_numpyro_mcmc(X, simulation_config, output_filepath)
    else:
        print("MCMC results file found. Skipping sampling.")

    if os.path.exists(mcmc_results_path):
        print("\nLoading MCMC samples for analysis...")
        with open(mcmc_results_path, 'rb') as f:
            results = pickle.load(f)
        X = results['X'] # Load X from the results file
        mcmc_run_config = results['config'] # Load MCMC config for prior sampling
        all_samples = {k: v for k, v in results.items() if k not in ['config', 'X']}

        mixing_results = run_analysis(output_filepath, analysis_config, X, all_samples, mcmc_run_config, on_log_scale=LOG_TRANSFORM_DATA)

        # --- FINAL CONFORMAL TESTING SUMMARY ---
        print("\n" + "="*60)
        print("### Final Conformal Testing Summary ###")
        print("="*60)

        print("\n--- Analysis on Mixing Measures (Earth Mover's Distance) ---")
        mixing_tau = next((item['tau'] for item in mixing_results if item['type'] == 'threshold'), np.inf)
        print(f"Conformal Threshold (tau): {mixing_tau:.4f}")

        # Print K-Means results
        print("\nK-Means Tests:")
        for result in mixing_results:
            if result['type'] == 'test':
                if result.get('skipped'):
                    print(f"   - {result['name']:<18}: SKIPPED (empty clusters)")
                elif result.get('error'):
                    print(f"   - {result['name']:<18}: FAILED ({result['error']})")
                else:
                    print(f"   - {result['name']:<18}: Score={result['score']:.4f} -> In Set: {result['in_set']}")

        # Print Prior Check results
        print("\nPrior Predictive Check:")
        prior_result = next((item for item in mixing_results if item['type'] == 'prior_check'), None)
        if prior_result:
            print(f"   - Prior Samples Tested: {prior_result['num_tested']}")
            print(f"   - Prior Samples In Set: {prior_result['num_in_set']}")
            print(f"   - Prior Inclusion Fraction: {prior_result['fraction_in_set']:.4f}")
        else:
            print("   - Prior check results not found.")

        print("\n" + "="*60)

    else:
        print("\nCould not find MCMC results file. Aborting analysis.")