import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


def self_representation_bs(X_2D, num_selected_bands, correlation_matrix = None):
    """
    Perform feature selection using self-representation.
    
    Parameters:
    X_2D : ndarray, shape (n_samples, n_features)
        2D data matrix where rows are samples, columns are features.
    num_selected_feature : int
        Number of features to select.
    correlation_matrix : ndarray, shape (n_features, n_features), optional
    
    Returns:
    selected_indices : ndarray, shape (num_selected_feature,)
        Indices of selected features.
    """
    _, n_bands = X_2D.shape
    if(correlation_matrix is None):
        correlation_matrix = X_2D.T @ X_2D
    
     #initialize
    selected_bands = []
    unselected_bands = list(range(n_bands))
    
    # step 1: choose the first band with maximum variance
    variances = np.var(X_2D, axis=0)
    first_band = np.argmax(variances)
    selected_bands.append(first_band)
    unselected_bands.remove(first_band)
    #err_new = np.linalg.norm(X_2D)**2
    for selected_id in tqdm(range(1, num_selected_bands),desc = 'Self representation band selection'):
        #best_band = unselected_bands[selected_id]
        def compute_error(candidate):
            tmp_selected_bands = selected_bands.copy()
            tmp_selected_bands.append(candidate)
            XTX = correlation_matrix
            XTy = correlation_matrix[:,tmp_selected_bands]
            yTy = correlation_matrix[np.ix_(tmp_selected_bands, tmp_selected_bands)]
            yTy_inv = np.linalg.inv(yTy)
            err_tmp = np.trace(XTX)- np.trace(XTy @ yTy_inv @ XTy.T)
            return candidate, err_tmp

        results = Parallel(n_jobs = -1)(
            delayed(compute_error)(candidate) for candidate in unselected_bands
        )
        # choose error minimization band
        best_band, min_err = min(results, key=lambda x: x[1])
        #err_new = min_err
        selected_bands.append(best_band)
        unselected_bands.remove(best_band)
    return np.array(selected_bands)

def self_representation_smi_bs(X_2D, num_selected_bands, band_subsets, correlation_matrix = None):

    _, n_bands = X_2D.shape
    if(correlation_matrix is None):
        correlation_matrix = X_2D.T @ X_2D
    
     
    selected_bands = []
    unselected_bands = list(range(n_bands))
    
    # step 1: choose the first band with maximum variance
    variances = np.var(X_2D, axis=0)
    first_band = np.argmax(variances)
    selected_bands.append(first_band)
    for subset in band_subsets:
        if (np.isin(first_band, subset)):
            for i in subset:
                unselected_bands.remove(i)

    #err_new = np.linalg.norm(X_2D)**2
    for selected_id in tqdm(range(1, num_selected_bands),desc = 'Self representation band selection'):
        #best_band = unselected_bands[selected_id]
        def compute_error(candidate):
            tmp_selected_bands = selected_bands.copy()
            tmp_selected_bands.append(candidate)
            XTX = correlation_matrix
            XTy = correlation_matrix[:,tmp_selected_bands]
            yTy = correlation_matrix[np.ix_(tmp_selected_bands, tmp_selected_bands)]
            yTy_inv = np.linalg.inv(yTy)
            err_tmp = np.trace(XTX)- np.trace(XTy @ yTy_inv @ XTy.T)
            return candidate, err_tmp

        results = Parallel(n_jobs = -1)(
            delayed(compute_error)(candidate) for candidate in unselected_bands
        )
        # choose error minimization band
        best_band, min_err = min(results, key=lambda x: x[1])
        #err_new = min_err
        selected_bands.append(best_band)
        for subset in band_subsets:
            if (np.isin(best_band, subset)):
                for i in subset:
                    unselected_bands.remove(i)
    return np.array(selected_bands)