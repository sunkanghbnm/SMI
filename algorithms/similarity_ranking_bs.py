import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.signal import convolve2d
from joblib import Parallel, delayed
import warnings
# 忽略运行时警告
warnings.filterwarnings("ignore", category = RuntimeWarning)

def _generate_gaussian_window(size = 11, sigma = 1.5):
   
    """生成高斯权重窗口"""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def _compute_single_image_stats(img, window, L = 255):
    """计算单个图像的统计量（并行化版本）"""
    # 转换为浮点型
    img_float = img.astype(np.float64)
    # 计算加权均值
    mu = convolve2d(img_float, window, mode='same', boundary='symm')
    
    # 计算加权平方均值
    img_sq = img_float ** 2
    img_sq_conv = convolve2d(img_sq, window, mode='same', boundary='symm')
    
    # 计算加权方差
    #sigma_sq = np.maximum(img_sq_conv - mu**2, 0)  # 防止负方差
    return {
        'img': img_float,
        'mu': mu,
        'mu_sq': mu ** 2,
        'img_sq_conv': img_sq_conv
    }


def _precompute_image_stats(X_3D, window_size = 11, sigma = 1.5, L = 255, n_jobs =None):
    """使用joblib并行预计算所有图像的统计量"""
    window = _generate_gaussian_window(window_size, sigma)
    num_bands = X_3D.shape[2]
    stats = Parallel(n_jobs = n_jobs, verbose = 10)(
        delayed(_compute_single_image_stats)(X_3D[:, :, i], window, L)
        for i in range(num_bands)
    )
    return stats

def _calculate_mssim_with_stats(stats_x, stats_y, window, L = 255):
    """
    使用预计算的统计量计算两幅图像的MSSIM
    避免重复计算每个图像对中相同图像的统计量
    
    参数:
    stats_x, stats_y: 预计算的图像统计量
    window: 高斯窗口
    L: 像素值动态范围
    
    返回:
    MSSIM值
    """
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    C3 = C2 / 2
    
    # 从预计算的统计量中获取数据
    mu_x = stats_x['mu']
    mu_y = stats_y['mu']
    mu_x_sq = stats_x['mu_sq']
    mu_y_sq = stats_y['mu_sq']
    
    # 计算方差 - 使用预计算结果
    sigma_x_sq = stats_x['img_sq_conv'] - mu_x_sq
    sigma_y_sq = stats_y['img_sq_conv'] - mu_y_sq
    
    # 计算协方差 - 只需要计算一次
    sigma_xy = convolve2d(
        stats_x['img'] * stats_y['img'], 
        window, 
        mode='same', 
        boundary='symm'
    ) - mu_x * mu_y
    
    # 计算SSIM分量
    luminance = (2 * mu_x * mu_y + C1) / (mu_x_sq + mu_y_sq + C1)
    
    # 避免负数方差导致NaN
    contrast_num = 2 * np.sqrt(np.maximum(sigma_x_sq, 0)) * np.sqrt(np.maximum(sigma_y_sq, 0)) + C2
    contrast_denom = np.maximum(sigma_x_sq, 0) + np.maximum(sigma_y_sq, 0) + C2
    contrast = contrast_num / contrast_denom
    
    # 结构分量
    structure_num = sigma_xy + C3
    structure_denom = np.sqrt(np.maximum(sigma_x_sq, 0)) * np.sqrt(np.maximum(sigma_y_sq, 0)) + C3
    structure = structure_num / structure_denom
    
    # 综合SSIM图
    ssim_map = luminance * contrast * structure
    
    # 计算平均值时忽略NaN值
    valid_pixels = ~np.isnan(ssim_map)
    if np.any(valid_pixels):
        return np.mean(ssim_map[valid_pixels])
    return 0.0

def compute_mssim_matrix(X_3D, window_size = 11, sigma = 1.5, L = 255, n_jobs = 20):
    """
    计算多图像间的MSSIM相似度矩阵 (joblib并行版)
    参数:
    X_3D: 高光谱数据立方体 (H, W, Bands)
    window_size: 滑动窗口大小
    sigma: 高斯核标准差
    L: 像素值动态范围
    n_jobs: 并行核数
    返回:
    N×N的相似度矩阵
    """
    num_bands = X_3D.shape[2]
    sim_matrix = np.zeros((num_bands, num_bands))
    np.fill_diagonal(sim_matrix, 1.0)
    window = _generate_gaussian_window(window_size, sigma)
    precomputed_stats = _precompute_image_stats(X_3D, window_size=window_size, sigma=sigma, L=L, n_jobs = n_jobs)

    # 任务列表
    tasks = [(i, j, precomputed_stats[i], precomputed_stats[j])
             for i in range(num_bands - 1)
             for j in range(i + 1, num_bands)]

    def mssim_job(args):
        i, j, stats_i, stats_j = args
        sim = _calculate_mssim_with_stats(stats_i, stats_j, window, L)
        return i, j, sim

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(mssim_job)(task) for task in tasks
    )

    for i, j, sim in results:
        sim_matrix[i, j] = sim
        sim_matrix[j, i] = sim
    return sim_matrix

def similarity_ranking_bs(similarity_matrix, similarity_threshold, num_selected_bands,sr_eta = None):
    """
    SR算法实现：基于相似度矩阵选择最具代表性的K个波段作为聚类中心
    参数:
    S: 相似度矩阵，形状为(L, L)，L为波段数量
    s_c: 相似度阈值，用于筛选显著相似的波段对
    K: 需要选择的波段数量
    
    返回:
    M: 选中的波段索引列表，长度为K
    B. Xu, X. Li, W. Hou, Y. Wang, and Y. Wei, 
    "A Similarity-Based Ranking Method for Hyperspectral Band Selection, " 
    IEEE Transactions on Geoscience and Remote Sensing, vol. 59, pp. 9585-9599, (2021).
    """
    if sr_eta is None:
        L = similarity_matrix.shape[0]  # 波段数量
        
        # Step 1: 计算每个波段的平均相似度α
        alpha = np.zeros(L)
        for i in range(L):
            # 计算超过阈值s_c的相似度值的总和和数量
            valid_similarities = similarity_matrix[i, similarity_matrix[i] > similarity_threshold]
            if len(valid_similarities) > 0:
                alpha[i] = np.mean(valid_similarities)
            else:
                alpha[i] = 0  # 如果没有超过阈值的相似度，设为0
        
        # 按α降序排序，获取排序后的索引
        sorted_indices = np.argsort(alpha)[::-1]
        
        # Step 2: 计算显著相异性φ
        phi = np.zeros(L)
        A = np.zeros(L, dtype=int)  # 存储与每个波段最相似的高α波段索引
        
        # 初始化φ: 第一个波段(最高α)的φ设为1，其余设为0
        phi[sorted_indices[0]] = 1.0
        
        # 对于排序后的每个波段(从第二个开始)
        for i in range(1, L):
            current_idx = sorted_indices[i]  # 当前波段索引
            
            # 在所有α更高的波段中寻找与当前波段最相似的
            max_similarity = -np.inf
            best_match_idx = -1
            
            for j in range(i):  # 只考虑排序在前的波段(α更高)
                candidate_idx = sorted_indices[j]
                similarity = similarity_matrix[current_idx, candidate_idx]
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_idx = candidate_idx
            
            # 记录找到的最大相似度和对应波段索引
            phi[current_idx] = max_similarity
            A[current_idx] = best_match_idx
        
        # 将最高α波段的φ值设为所有φ中的最小值
        min_phi = np.min(phi[phi > 0])  # 只考虑非零的φ值
        phi[sorted_indices[0]] = min_phi
        A[sorted_indices[0]] = sorted_indices[0]  # 最高α波段与自身最相似
        
        # 计算θ = √(1 - φ²)，表示相异性
        theta = np.sqrt(1 - np.square(phi))
        
        # Step 3: 计算每个波段的综合得分η
        # 归一化α和θ
        norm_alpha = (alpha - np.min(alpha)) / (np.max(alpha) - np.min(alpha) + 1e-10)
        norm_theta = (theta - np.min(theta)) / (np.max(theta) - np.min(theta) + 1e-10)
        eta = norm_alpha * norm_theta
    else:
        eata = sr_eta
    # 计算得分η = norm(α) × norm(θ)
    
    # Step 4: 选择得分最高的K个波段
    selected_indices = np.argsort(eta)[::-1][:num_selected_bands]
    
    return selected_indices, eta

def similarity_ranking_smi_bs(band_subsets, sr_eta = None):
    """
    """
    selected_indices = []
    for subset in band_subsets:
       sr_eta_tmp = sr_eta[subset]
       tmp_max_ind = subset[np.argmax(sr_eta_tmp)]
       selected_indices.append(tmp_max_ind)
    
    return np.array(selected_indices)