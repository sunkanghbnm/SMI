import numpy as np
from sklearn.metrics import pairwise_distances

def eca_bs(X_2D, distance_matrix = None, sigma = None):
    
    """
    实现ECA波段选择算法
    参数：
        X : 高光谱数据 (N, L)
        k : 选择波段数（None时自动确定）
    返回：
        selected_bands : 选择波段索引
        ES_values : 所有波段ES值
    """
    # 1. 计算距离矩阵
    if distance_matrix is None:
        distance_matrix = pairwise_distances(X_2D.T, metric = 'euclidean')
    D = distance_matrix
    # 2. 自适应参数设置
    if sigma is None:     
        sigma = np.sqrt(np.mean(D) / 30)
    # 3. 计算局部密度ρ
    rho = np.sum(np.exp(-D / (2 * sigma**2)), axis = 1)
    # 4. 计算最小距离δ
    delta = np.zeros_like(rho)
    max_density_idx = np.argmax(rho)
    
    for i in range(len(rho)):
        if i == max_density_idx:
            delta[i] = np.max(D[i])
        else:
            higher_density_idxs = np.where(rho > rho[i])[0]
            delta[i] = np.min(D[i, higher_density_idxs]) if len(higher_density_idxs) > 0 else 0
    
    # 5. 计算ES值
    ES = rho * delta
    
    # 6. 波段选择
    sorted_indices = np.argsort(ES)[::-1]
    
    return sorted_indices, ES

def eca_smi_bs(X_2D, band_subsets, distance_matrix = None, sigma = None):
    """
    实现ECA波段选择算法
    参数：
        X : 高光谱数据 (N, L)
        k : 选择波段数（None时自动确定）
    返回：
        selected_bands : 选择波段索引
        ES_values : 所有波段ES值
    """
    # 1. 计算距离矩阵
    if distance_matrix is None:
        distance_matrix = pairwise_distances(X_2D.T, metric = 'euclidean')
    D = distance_matrix
    # 2. 自适应参数设置
    if sigma is None:     
        sigma = np.sqrt(np.mean(D) / 30)
    # 3. 计算局部密度ρ
    rho = np.sum(np.exp(-D / (2 * sigma**2)), axis = 1)
    # 4. 计算最小距离δ
    delta = np.zeros_like(rho)
    max_density_idx = np.argmax(rho)
    
    for i in range(len(rho)):
        if i == max_density_idx:
            delta[i] = np.max(D[i])
        else:
            higher_density_idxs = np.where(rho > rho[i])[0]
            delta[i] = np.min(D[i, higher_density_idxs]) if len(higher_density_idxs) > 0 else 0
    
    # 5. 计算ES值
    ES = rho * delta
    
    selected_indices = []
    
    for subset in band_subsets:
       variance_tmp = ES[subset]
       tmp_max_ind = subset[np.argmax(variance_tmp)]
       selected_indices.append(tmp_max_ind)
    
    return np.array(selected_indices)
