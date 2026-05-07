import numpy as np

def mvpca_bs(X_2D, num_selected_feature, variances = None):

    _, n_bands = X_2D.shape
    if  (variances is None):
        variances = np.var(X_2D, axis = 0)
    selected_indices = np.argsort(variances)[-num_selected_feature:]# Select top features
    return selected_indices

def mvpca_smi_bs(X_2D, num_selected_feature, band_subsets,variances = None):
    if  (variances is None):
        variances = np.var(X_2D, axis = 0)
    selected_indices = []
    
    for subset in band_subsets:
       variance_tmp = variances[subset]
       tmp_max_ind = subset[np.argmax(variance_tmp)]
       selected_indices.append(tmp_max_ind)
    
    return np.array(selected_indices)