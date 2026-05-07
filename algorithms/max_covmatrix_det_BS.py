import numpy as np
import math
import warnings
# 忽略运行时警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

def max_covmatrix_det_bs(X_2D, num_selected_bands, covariance_matrix = None):
    """
    Performs feature selection using the Maximum Determinant Covariance Matrix (MDCM) algorithm.
    This method iteratively selects a subset of features that maximize the product of singular values (akin to the determinant)
    of the covariance submatrix aligning to the chosen features. The approach prefer subsets whose corresponding covariance 
    submatrix is well-conditioned in terms of spread of data.
    Args:
        X_2D (numpy.ndarray): 2D data matrix of shape (n_samples, n_features).
        num_selected_feature (int): Number of features to select.
        covariance_matrix (numpy.ndarray, optional): Covariance matrix of shape (n_features, n_features). 
            If None, it will be computed from X.
    Returns:
        numpy.ndarray: Indices of selected features (length = num_selected_feature).
    """
    _, num_bands = X_2D.shape
    if(covariance_matrix is None):
        covariance_matrix = np.cov(X_2D, rowvar = False)
    np.random.seed(42)
    selected_indices = np.sort(np.random.choice(num_bands, size = num_selected_bands, replace = False))        
    iter_times = 10
    for iter in range(iter_times):
        sub_conv_matrix = covariance_matrix[np.ix_(selected_indices, selected_indices)]
        _, s, _ = np.linalg.svd(sub_conv_matrix)
        err_old = math.prod(s)
        err_new = err_old
        for i in range(num_selected_bands):
            for j in range(num_bands):
                if (np.isin(j, selected_indices)):
                    continue
                selected_indices_tmp = selected_indices.copy()
                selected_indices_tmp[i] = j
                sub_conv_matrix = covariance_matrix[np.ix_(selected_indices_tmp, selected_indices_tmp)]
                _, s, _ = np.linalg.svd(sub_conv_matrix)
                err_tmp = math.prod(s)
                if(err_tmp > err_new):
                    err_new = err_tmp
                    selected_indices[i] = j
        if(np.abs(err_new - err_old)/np.abs(err_old + 1e-10) < 1e-6):
            break
    return selected_indices

def max_covmatrix_det_smi_bs(X_2D, num_selected_bands, band_subsets, covariance_matrix = None):
    """
    从band_subsets的每个子集选择一个波段，使得选中波段组成的协方差矩阵行列式最大。
    约束：必须从band_subsets中的每个子集各选一个波段，最终选中num_selected_bands个波段（band_subsets长度需等于num_selected_bands）。

    Args:
        X_2D (numpy.ndarray): 2D数据矩阵，形状为(n_samples, n_features)
        num_selected_bands (int): 要选择的波段总数（需等于band_subsets的长度）
        band_subsets (list/numpy.ndarray): 波段子集列表，每个元素是一个包含若干波段索引的子集（数组/列表），
                                          长度需等于num_selected_bands，代表每个位置的可选波段范围
        covariance_matrix (numpy.ndarray, optional): 预计算的协方差矩阵，形状为(n_features, n_features)。
                                                    若为None，将从X_2D计算。

    Returns:
        numpy.ndarray: 选中的波段索引（长度=num_selected_bands），满足从每个子集选一个且行列式最大
    """
    # 输入合法性校验
    if len(band_subsets) != num_selected_bands:
        raise ValueError(f"band_subsets的长度({len(band_subsets)})必须等于要选择的波段数num_selected_bands({num_selected_bands})")
    for idx, subset in enumerate(band_subsets):
        if len(subset) == 0:
            raise ValueError(f"band_subsets的第{idx}个子集为空，无法选择波段")
    
    _, num_bands = X_2D.shape
    # 计算/验证协方差矩阵
    if covariance_matrix is None:
        covariance_matrix = np.cov(X_2D, rowvar = False)
    
    # 初始化：从每个子集随机选一个波段作为初始解
    np.random.seed(42)
    selected_indices = np.array([np.random.choice(subset, size=1)[0] for subset in band_subsets])
    iter_times = 10  # 迭代优化次数，与原函数保持一致
    
    for iter in range(iter_times):
        # 计算当前选中波段的协方差矩阵行列式（奇异值乘积）
        sub_cov_matrix = covariance_matrix[np.ix_(selected_indices, selected_indices)]
        _, s, _ = np.linalg.svd(sub_cov_matrix)
        err_old = math.prod(s)
        err_new = err_old  # 记录本轮最优行列式值
        
        # 遍历每个子集位置，尝试替换为该子集中的其他波段
        for i in range(num_selected_bands):
            current_subset = band_subsets[i]  # 当前位置对应的可选子集
            current_band = selected_indices[i]  # 当前位置选中的波段
            
            # 遍历该子集中的所有候选波段（排除当前已选的）
            for j in current_subset:
                if j == current_band:
                    continue  # 跳过当前已选波段
                
                # 临时替换当前位置的波段
                selected_indices_tmp = selected_indices.copy()
                selected_indices_tmp[i] = j
                
                # 计算替换后的行列式
                sub_cov_matrix_tmp = covariance_matrix[np.ix_(selected_indices_tmp, selected_indices_tmp)]
                _, s_tmp, _ = np.linalg.svd(sub_cov_matrix_tmp)
                err_tmp = math.prod(s_tmp)
                
                # 如果替换后行列式更大，更新最优解
                if err_tmp > err_new:
                    err_new = err_tmp
                    selected_indices[i] = j  # 永久替换
        
        # 收敛判断：相对变化小于阈值则提前终止迭代
        if np.abs(err_new - err_old) / np.abs(err_old + 1e-10) < 1e-6:
            break
    
    # 返回排序后的选中索引（与原函数输出格式保持一致）
    return np.sort(selected_indices)