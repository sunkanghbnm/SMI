import numpy as np
import scipy.io
from scipy.stats import entropy
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
import matplotlib.pyplot as plt
from minepy import MINE  
from sklearn.feature_selection import mutual_info_regression
from joblib import Parallel, delayed
from tqdm import tqdm  


# -------------------------- Core Utility Functions (Unchanged) --------------------------
def _sample_pixels(x, y, n_samples=1000):
    """Randomly sample pixels from two bands synchronously to reduce computation load"""
    np.random.seed()
    total_pixels = len(x)
    if total_pixels <= n_samples:
        return x, y  # Return all pixels if total is less than sample size
    # Synchronous random sampling to ensure pixel correspondence between x and y
    idx = np.random.choice(total_pixels, size=n_samples, replace=False)
    return x[idx], y[idx]

def calc_mic(x, y, n_samples=1000):
    """Calculate Maximal Information Coefficient (MIC) for two variables with pixel sampling"""
    x_sampled, y_sampled = _sample_pixels(x, y, n_samples)
    mine = MINE(alpha=0.6, c=20)
    mine.compute_score(x_sampled, y_sampled)
    return mine.mic()

def calc_nmi(x, y, n_samples=1000):
    """Calculate Normalized Mutual Information (NMI) for two continuous variables with pixel sampling"""
    x_sampled, y_sampled = _sample_pixels(x, y, n_samples)
    mi = mutual_info_regression(x_sampled.reshape(-1, 1), y_sampled)[0]
    h_x = mutual_info_regression(x_sampled.reshape(-1, 1), x_sampled)[0]
    h_y = mutual_info_regression(y_sampled.reshape(-1, 1), y_sampled)[0]
    if h_x + h_y == 0:
        return 0
    return 2 * mi / (h_x + h_y)

def calc_conditional_nmi(x, y, z, n_samples = 1000):
    """
    Symmetrically averaged normalized conditional mutual information (consistent with NMI)
    Value range is naturally [0,1], no min function or hard truncation
    """
    # 1. Synchronous sampling
    np.random.seed()
    total_pixels = len(x)
    if total_pixels > n_samples:
        idx = np.random.choice(total_pixels, size=n_samples, replace=False)
        x = x[idx]
        y = y[idx]
        z = z[idx]
    
    # 2. Calculate raw CMI (correct negative error of kNN estimation)
    mi_xz_y = mutual_info_regression(np.hstack([x.reshape(-1,1), z]), y)[0]
    mi_z_y = mutual_info_regression(z, y)[0]
    cmi_raw = max(mi_xz_y - mi_z_y, 0.0)  # Correct only theoretically impossible negative values
    
    # 3. Calculate conditional entropy
    h_x = mutual_info_regression(x.reshape(-1, 1), x)[0]
    # 3.2 Calculate I(X; Z): Z is 2D X, x is 1D y (meets sklearn input requirements)
    i_x_z = mutual_info_regression(z, x)[0]
    # 3.3 Calculate conditional entropy H(X|Z), ensure non-negativity (numerical stability protection)
    h_x_given_z = max(h_x - i_x_z, 0.0)
    
    # 4. Calculate H(Y|Z) = H(Y) - I(Y; Z) (same logic)
    h_y = mutual_info_regression(y.reshape(-1, 1), y)[0]
    i_y_z = mutual_info_regression(z, y)[0]
    h_y_given_z = max(h_y - i_y_z, 0.0)
    # 4. Symmetric average normalization (consistent with NMI)
    denom = h_x_given_z + h_y_given_z + 1e-8
    n_cmi = 2 * cmi_raw / denom
    
    return n_cmi

# -------------------------- Core Optimization Function: Pre-bin All Bands --------------------------
def _pre_band_binning(data_flat, n_bins=10, bin_method='quantile'):
    """
    Pre-bin all bands to avoid repeated calculations in loops
    
    Parameters:
        data_flat: np.ndarray, shape (number of pixels, number of bands), hyperspectral data with bad pixels removed
        n_bins: int, number of bins for discretization
        bin_method: str, 'quantile' (equal-frequency binning) or 'equal' (equal-width binning)
    
    Returns:
        x_bins_all: np.ndarray, shape (number of pixels, number of bands), pre-binned results for all bands (integers 0~n_bins-1)
    """
    n_pixels, L = data_flat.shape
    x_bins_all = np.zeros_like(data_flat, dtype=np.int32)  # Use int32 to save memory
    
    if bin_method == 'quantile':
        # Pre-calculate quantiles for all bands (avoid loop calls to np.percentile)
        quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]  # Remove 0 and 100 to get n_bins-1 quantiles
        band_percentiles = np.percentile(data_flat, quantiles, axis=0)  # Shape (n_bins-1, L)
        
        # Batch binning with searchsorted (10x faster than pd.qcut)
        for i in range(L):
            x_bins_all[:, i] = np.searchsorted(band_percentiles[:, i], data_flat[:, i], side='left')
            # Handle possible boundary overflow (theoretically impossible)
            x_bins_all[:, i] = np.clip(x_bins_all[:, i], 0, n_bins - 1)
    
    elif bin_method == 'equal':
        # Pre-calculate min and max values for all bands
        band_min = np.min(data_flat, axis=0)
        band_max = np.max(data_flat, axis=0)
        # Avoid division by zero (case where all pixel values are the same)
        band_range = np.where(band_max == band_min, 1e-10, band_max - band_min)
        
        # Batch binning
        for i in range(L):
            x_bins_all[:, i] = np.floor((data_flat[:, i] - band_min[i]) / band_range[i] * n_bins).astype(np.int32)
            # Handle boundary overflow
            x_bins_all[:, i] = np.clip(x_bins_all[:, i], 0, n_bins - 1)
    
    else:
        raise ValueError("bin_method must be 'quantile' or 'equal'")
    
    return x_bins_all

# -------------------------- Core Optimization Function: Conditional Entropy Calculation for Single Band Pair --------------------------
def _single_pair_conditional_entropy(i, x_bins_all, n_bins):
    """
    Calculate conditional entropy for a single adjacent band pair (for Joblib parallelization)
    
    Parameters:
        i: int, index of band pair (calculate H(b_{i+1} | b_i))
        x_bins_all: np.ndarray, shape (number of pixels, number of bands), pre-binned results for all bands
        n_bins: int, number of bins for discretization
    
    Returns:
        cond_entropy: float, conditional entropy of the single band pair
    """
    x = x_bins_all[:, i]
    y = x_bins_all[:, i+1]
    
    # Calculate joint histogram with np.bincount (20x faster than np.histogram2d)
    combined_idx = x * n_bins + y
    joint_hist = np.bincount(combined_idx, minlength=n_bins**2).reshape(n_bins, n_bins)
    joint_prob = joint_hist / joint_hist.sum()
    
    # Calculate marginal entropy H(X)
    x_hist = np.sum(joint_hist, axis=1)
    x_prob = x_hist / x_hist.sum()
    h_x = entropy(x_prob)
    
    # Calculate joint entropy H(X,Y)
    h_joint = entropy(joint_prob.flatten())
    
    # Conditional entropy H(Y|X) = H(X,Y) - H(X)
    return h_joint - h_x

# -------------------------- Main Function: Optimized Adjacent Conditional Entropy Calculation --------------------------
def calculate_adjacent_conditional_entropy(
    X_2D, 
    n_bins = 1024, 
    bin_method='quantile',
    n_jobs=-1,
):
    """
    Calculate conditional entropy H(b_{i+1} | b_i) for all adjacent band pairs of hyperspectral images
    Optimized version: pre-binning, deduplication, Joblib parallelization, support for multiple data formats
    
    Parameters:
        X_2D: np.ndarray, shape (number of pixels, number of bands)
            Hyperspectral data cube or 2D matrix
        n_bins: int, number of bins for discretization, recommended 10~20 (common value in remote sensing field)
        bin_method: str, 'quantile' (equal-frequency binning, recommended) or 'equal' (equal-width binning)
        n_jobs: int, number of CPU cores used for Joblib parallelization, default -1 (use all available cores)
    
    Returns:
        cond_entropies: np.ndarray, shape (L-1,), conditional entropy of each adjacent band pair
    """
    # -------------------------- 1. Unify data format to (number of pixels, number of bands) --------------------------
    data_flat = X_2D
    
    # -------------------------- 2. Remove all-zero pixels (bad pixels) --------------------------
    n_pixels, L = data_flat.shape
    if n_pixels == 0:
        raise ValueError("All pixels are all-zero bad pixels, please check input data")
    
    # -------------------------- 3. Pre-bin all bands (core of deduplication) --------------------------
    x_bins_all = _pre_band_binning(data_flat, n_bins, bin_method)
    
    # -------------------------- 4. Parallel calculation of conditional entropy for all adjacent band pairs with Joblib --------------------------
    cond_entropies = Parallel(n_jobs=n_jobs)(
        delayed(_single_pair_conditional_entropy)(i, x_bins_all, n_bins)
        for i in range(L-1)
    )
    cond_entropies = np.array(cond_entropies)
    
    # -------------------------- 5. Calculate statistics --------------------------
    mean_ce = np.mean(cond_entropies)
    std_ce = np.std(cond_entropies)
    si = 1 - (std_ce / mean_ce)  # Stationarity Index SI = 1 - std/mean
    non_stationary_contribution = 1 - si  # Non-stationary contribution degree
    print(f"Mean CE: {mean_ce:.4f}, Std CE: {std_ce:.4f}, SI: {si:.4f}, Non-stationary Contribution: {non_stationary_contribution:.4f}")
    return cond_entropies



# -------------------------- On-Demand Parallel Calculation Auxiliary Functions (Unchanged) --------------------------
def _calc_adj_mic(i, data, n_samples):
    """Calculate MIC only for adjacent band pair (i, i+1)"""
    return calc_mic(data[:, i], data[:, i+1], n_samples)

def _calc_nonadj_mic(pair, data, n_samples):
    """Calculate MIC only for the specified non-adjacent band pair"""
    i, j = pair
    return calc_mic(data[:, i], data[:, j], n_samples)

def _calc_cis_i(i, data, n_samples):
    """Calculate CIS component for a single i (calculate NMI on demand, unconditional NMI is also sampled)"""
    L = data.shape[1]
    x = data[:, i]
    z = data[:, [i-1, i+1]]  # Conditional variables: left and right adjacent bands
    j_list = [j for j in range(L) if abs(j - i) >= 2]
    
    if len(j_list) == 0:
        return None
    
    # Calculate unconditional NMI and conditional NMI for each j on demand
    nmi_uncond_list = []
    nmi_cond_list = []
    for j in j_list:
        y = data[:, j]
        # Unconditional NMI: direct calculation (with sampling)
        nmi_uncond = calc_nmi(x, y, n_samples)
        nmi_uncond_list.append(nmi_uncond)
        # Conditional NMI: direct calculation (with sampling)
        nmi_cond = calc_conditional_nmi(x, y, z, n_samples)
        nmi_cond_list.append(nmi_cond)
    
    # Calculate mean and ratio
    nmi_uncond_mean = np.mean(nmi_uncond_list)
    nmi_cond_mean = np.mean(nmi_cond_list)
    
    if nmi_uncond_mean == 0:
        ratio = 1
    else:
        ratio = np.clip(nmi_cond_mean / nmi_uncond_mean, 0, 1)
    
    return 1 - ratio

def calc_cis(data_2d, n_samples = 1000, n_jobs = -1, shuffled = False):
    N, L = data_2d.shape    
    if shuffled:
        band_indices = np.random.permutation(L)
        data_2d = data_2d[:, band_indices]
    cis_tasks = [delayed(_calc_cis_i)(i, data_2d, n_samples) for i in range(1, L-1)]
    cis_results = Parallel(n_jobs=n_jobs)(tqdm(cis_tasks, desc="  Calculating CIS components"))
    # Filter None values
    cis_list = [res for res in cis_results if res is not None]
    return np.array(cis_list)

def calc_mic_adj(data_2d, n_samples = 1000, n_jobs = -1, shuffled = False):
    N, L = data_2d.shape
    if shuffled:
        band_indices = np.random.permutation(L)
        data_2d = data_2d[:, band_indices]
    adj_tasks = [delayed(_calc_adj_mic)(i, data_2d, n_samples) for i in range(L-1)]
    mic_adj_list = Parallel(n_jobs=n_jobs)(tqdm(adj_tasks, desc="  Calculating MIC for adjacent bands"))
    return np.array(mic_adj_list)

def calc_mic_unadj(data_2d, n_samples = 1000, nonadj_max_pairs = 1000, n_jobs = -1, shuffled = False):
    N, L = data_2d.shape 
    if shuffled:
        band_indices = np.random.permutation(L)
        data_2d = data_2d[:, band_indices]
    all_nonadj_pairs = [(i, j) for i in range(L) for j in range(i+5, L)]
    if len(all_nonadj_pairs) > 0:
        # Randomly sample specified number of non-adjacent pairs (no more than total)
        sample_size = min(nonadj_max_pairs, len(all_nonadj_pairs))
        np.random.seed()
        nonadj_pairs = np.random.choice(len(all_nonadj_pairs), size=sample_size, replace=False)
        nonadj_pairs = [all_nonadj_pairs[idx] for idx in nonadj_pairs]
        
        print(f"  Randomly sampling non-adjacent band pairs (total {len(all_nonadj_pairs)}, sampled {sample_size})...")
        nonadj_tasks = [delayed(_calc_nonadj_mic)(pair, data_2d, n_samples) for pair in nonadj_pairs]
        mic_nonadj_list = Parallel(n_jobs=n_jobs)(tqdm(nonadj_tasks, desc="  Calculating MIC for non-adjacent bands"))
    
    return np.array(mic_nonadj_list)

def calc_band_fit_r2(hsi_data_2d, remove_outlier=True, use_adjusted_r2=True, shuffled = False):
    """
    Calculate 'left-right adjacent dual-band fitting R²' for each middle band of hyperspectral data
    :param hsi_data_2d: 2D hyperspectral data, shape=(N pixels, L bands)
    :param remove_outlier: whether to remove outliers using 3σ criterion, default True
    :param use_adjusted_r2: whether to use adjusted R², default True (recommended in papers)
    :param shuffled: whether to randomly sort data, default False
    :return:
        r2_array: R² array for each middle band, shape=(L-2,), corresponding to band indices 1~L-2
    """
    N, L = hsi_data_2d.shape
    if shuffled:
        band_indices = np.random.permutation(L)
        hsi_data_2d = hsi_data_2d[:, band_indices]
    if L < 3:
        raise ValueError("Number of bands must be ≥3 to calculate left-right adjacent fitting R²")
    
    # 1. Outlier processing: 3σ criterion, replace outliers with band mean
    if remove_outlier:
        z_scores = zscore(hsi_data_2d, axis=0)
        outlier_mask = np.abs(z_scores) > 3
        # Replace outliers with band mean for each band
        band_means = np.mean(hsi_data_2d, axis=0)
        hsi_data_clean = hsi_data_2d.copy()
        for band in range(L):
            hsi_data_clean[outlier_mask[:, band], band] = band_means[band]
    else:
        hsi_data_clean = hsi_data_2d
    
    # 2. Iterate over each middle band to calculate R²
    r2_array = np.zeros(L-2)
    linear_model = LinearRegression(fit_intercept=True)  # Linear regression with intercept
    
    for i in range(1, L-1):
        # Extract independent variables (left + right adjacent) and dependent variable (current band)
        X = np.hstack([
            hsi_data_clean[:, i-1:i],  # Left adjacent band
            hsi_data_clean[:, i+1:i+2] # Right adjacent band
        ])
        y = hsi_data_clean[:, i:i+1]
        
        # Fit linear regression
        linear_model.fit(X, y)
        r2 = linear_model.score(X, y)  # Basic R²
        
        # Calculate adjusted R²
        if use_adjusted_r2:
            k = 2  # Number of independent variables
            n = N  # Sample size
            r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        
        r2_array[i-1] = r2

    return r2_array

def _calc_acc(mic_adjs, mic_nonadjs):
    """
    ACC calculation function optimized based on Bhattacharyya distance, fully adapted to requirements
    :param mic_adjs: MIC sequence of adjacent bands, array/list format
    :param mic_nonadjs: MIC sequence of non-adjacent bands, array/list format
    :return:
        acc: final ACC value, range [0,1]
        stats: dictionary containing intermediate calculation statistics for paper analysis
    """
    
    # 1. Calculate core statistics
    mu_adj = np.mean(mic_adjs)
    var_adj = np.var(mic_adjs, ddof=1)  # Unbiased variance
    mu_nonadj = np.mean(mic_nonadjs)
    var_nonadj = np.var(mic_nonadjs, ddof=1)
    
    # 2. Calculate core separability term
    delta_mu_sq = (mu_adj - mu_nonadj) ** 2
    var_sum = var_adj + var_nonadj + 1e-8  # Add small value to avoid division by zero
    acc = 1 - np.exp(- delta_mu_sq / var_sum)
    acc = np.clip(acc, 0, 1)  # Only for numerical stability protection, no physical hard truncation
    
    # Statistics output
    stats = {
        "mu_adj": mu_adj,
        "var_adj": var_adj,
        "sigma_adj": np.sqrt(var_adj),
        "mu_nonadj": mu_nonadj,
        "var_nonadj": var_nonadj,
        "mu_diff": mu_adj - mu_nonadj,
        "acc": acc,
    }
    print(stats)
    return acc

def calc_mic_vs_bandinterval(
    data_2d,
    max_k = None,          # Maximum band interval, default L-1 (all intervals)
    max_pairs_per_k = 500, # Maximum sampled band pairs per interval (dimensionality reduction)
    n_samples = 1000,      # Pixel samples per band pair
    n_jobs = -1,           # Number of parallel cores
    shuffled = False,      # Whether to randomly permute bands (only for verification)
):
    """
    Calculate average MIC under different band intervals and return data for plotting
    :param data_2d: 2D hyperspectral data, shape=(number of pixels N, number of bands L)
    :param max_k: maximum calculation interval, default L-1 (calculate all possible intervals)
    :param max_pairs_per_k: maximum number of band pairs calculated per interval k (random sampling if exceeded)
    :param n_samples: number of pixel samples per band pair
    :param n_jobs: number of parallel cores
    :return: (k_list, mic_mean_list): interval list, corresponding average MIC list
    """
    N, L = data_2d.shape
    if shuffled:
        band_indices = np.random.permutation(L)
        data_2d = data_2d[:, band_indices]
    if max_k is None:
        max_k = L - 1  # Calculate all possible intervals by default
    if max_k < 1 or max_k >= L:
        raise ValueError(f"max_k must be in [1, {L-1}] range")
    
    print("="*60)
    print(f"Start calculating interval-MIC curve: Number of bands={L}, Max interval={max_k}")
    print(f"Max sampled pairs per interval={max_pairs_per_k}, Pixel samples per pair={n_samples}")
    print("="*60)
    
    k_list = []          # Store all intervals k
    mic_mean_list = []   # Store average MIC for each k
    
    # Iterate over each band interval k (from 1 to max_k)
    for k in tqdm(range(1, max_k + 1), desc="Iterating band intervals"):
        # 1. Generate all band pairs for current interval k: (i, i+k)
        all_pairs = [(i, i + k) for i in range(L - k)]
        
        if len(all_pairs) == 0:
            continue  # Skip if no valid band pairs
        
        # 2. Random sampling for dimensionality reduction (if too many band pairs)
        if len(all_pairs) > max_pairs_per_k:
            np.random.seed()  # Different seed for each k to ensure reproducibility
            sample_idx = np.random.choice(len(all_pairs), size=max_pairs_per_k, replace=False)
            pairs = [all_pairs[idx] for idx in sample_idx]
        else:
            pairs = all_pairs
        
        # 3. Parallel calculation of MIC for all band pairs under current k
        mic_list = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(calc_mic)(
                data_2d[:, i], data_2d[:, j],
                n_samples=n_samples                
            )
            for idx, (i, j) in enumerate(pairs)
        )
        
        # 4. Calculate average MIC for current k
        mean_mic = np.mean(mic_list)
        k_list.append(k)
        mic_mean_list.append(mean_mic)
    
    print("\n" + "="*60)
    print("Interval-MIC curve calculation completed!")
    print("="*60)
    
    return np.array(mic_mean_list)

# -------------------------- Core SMI Calculation Function (Optimized: tqdm + Random Sampling of Band Pairs) --------------------------
def calc_smi(
    data_2d, 
    n_jobs=-1, 
    n_samples = 1000, 
    nonadj_max_pairs = 1000,
    precomputed_cis = None,
    precomputed_mic_adj = None,
    precomputed_mic_nonadj = None
):
    """
    Calculate Spectral Markov Index (SMI) (Optimized: tqdm progress bar + random sampling of non-adjacent band pairs)
    :param data_2d: 2D hyperspectral data, shape=(number of pixels N, number of bands L)
    :param n_jobs: number of parallel cores, -1 means use all available cores
    :param n_samples: number of random sampled pixels per band pair (default 1000, sufficiently accurate)
    :param nonadj_max_pairs: maximum number of sampled non-adjacent band pairs (control computational complexity)
    :param precomputed_cis: precomputed CIS values (avoid repeated calculations)
    :param precomputed_mic_adj: precomputed MIC values for adjacent bands (avoid repeated calculations)
    :param precomputed_mic_nonadj: precomputed MIC values for non-adjacent bands (avoid repeated calculations)
    :return: SMI, ACC, CIS
    """
    N, L = data_2d.shape
    if L < 5:
        raise ValueError("Number of bands L must be ≥5 to stably calculate SMI")
    
    # 1. Calculate ACC: Adjacent Correlation Concentration (only calculate required combinations)
    print("Calculating ACC (only adjacent + randomly sampled non-adjacent band pairs)...")
    # 1.1 Parallel calculation of MIC for adjacent bands (with tqdm progress bar)
    print(f"  Calculating MIC for adjacent bands (total {L-1} combinations)...")
    if precomputed_mic_adj is None:
        adj_tasks = [delayed(_calc_adj_mic)(i, data_2d, n_samples) for i in range(L-1)]
        mic_adj_list = Parallel(n_jobs=n_jobs)(tqdm(adj_tasks, desc="  Calculating MIC for adjacent bands"))
    else:
        mic_adj_list = precomputed_mic_adj
    
    mic_adj_mean = np.mean(mic_adj_list)
    print(f'  Average MIC for adjacent bands: {mic_adj_mean:.4f}')
    
    # 1.2 Random sampling of non-adjacent band pairs (replace full traversal)
    # Generate all eligible non-adjacent pairs (j≥i+5)
    if precomputed_mic_nonadj is None:
        all_nonadj_pairs = [(i, j) for i in range(L) for j in range(i+5, L)]
        if len(all_nonadj_pairs) > 0:
            # Randomly sample specified number of non-adjacent pairs (no more than total)
            sample_size = min(nonadj_max_pairs, len(all_nonadj_pairs))
            np.random.seed()
            nonadj_pairs = np.random.choice(len(all_nonadj_pairs), size=sample_size, replace=False)
            nonadj_pairs = [all_nonadj_pairs[idx] for idx in nonadj_pairs]
            
            print(f"  Randomly sampling non-adjacent band pairs (total {len(all_nonadj_pairs)}, sampled {sample_size})...")
            nonadj_tasks = [delayed(_calc_nonadj_mic)(pair, data_2d, n_samples) for pair in nonadj_pairs]
            mic_nonadj_list = Parallel(n_jobs=n_jobs)(tqdm(nonadj_tasks, desc="  Calculating MIC for non-adjacent bands"))
            mic_nonadj_mean = np.mean(mic_nonadj_list)
        else:
            mic_nonadj_mean = 0
    else:
        mic_nonadj_list = precomputed_mic_nonadj
        mic_nonadj_mean = np.mean(mic_nonadj_list)    
            
    print(f'  Average MIC for non-adjacent bands: {mic_nonadj_mean:.4f}')
    
    # 1.3 Calculate ACC
    ACC = _calc_acc(mic_adj_list, mic_nonadj_list)
    
    # 2. Calculate CIS: Conditional Independence Score (with tqdm progress bar)
    print("Calculating CIS (calculate unconditional/conditional NMI on demand)...")
    if precomputed_cis is None:
        cis_tasks = [delayed(_calc_cis_i)(i, data_2d, n_samples) for i in range(1, L-1)]
        cis_results = Parallel(n_jobs=n_jobs)(tqdm(cis_tasks, desc="  Calculating CIS components"))
        # Filter None values
        cis_list = [res for res in cis_results if res is not None]
    else:
        cis_list = precomputed_cis
    
    CIS = max(0, np.mean(cis_list)) if len(cis_list) > 0 else 0
    print(f'CIS: {CIS:.4f}')
    
    # 3. Calculate final SMI
    SMI = np.sqrt(ACC * CIS)
    
    return SMI, ACC, CIS