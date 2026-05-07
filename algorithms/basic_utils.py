import numpy as np
import pandas as pd
import copy
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import entropy,pearsonr, chi2_contingency
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed
import pywt
from scipy.cluster.hierarchy import linkage, fcluster
from skimage.metrics import structural_similarity
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

def compute_entropy(X_2D):
    """计算每个波段的信息熵"""
    _, n_bands = X_2D.shape
    entropy_vals = np.zeros(n_bands)
    for i in range(n_bands):
        band_data = X_2D[:, i]
        hist, _ = np.histogram(band_data, bins = 256)
        prob = hist / np.sum(hist)
        entropy_vals[i] = entropy(prob)
    return entropy_vals
def euclidean_distance_matrix(X_2D):
    """计算欧氏距离矩阵"""

    return squareform(pdist(X_2D.T, metric = 'euclidean'))

def compute_variances(X_2D):
    """计算方差"""
    variances = np.var(X_2D, axis = 0)
    return variances

def rbf_kernel_dismat(distance_matrix, sigma = 500):
    """
    计算RBF核矩阵
    :param X_3D: 高光谱数据 (height, width, n_bands)
    :return: 核矩阵 (n_bands, n_bands)
    """
    # 计算RBF核
    return np.exp(-distance_matrix / (2 * sigma ** 2))

def compute_correlation_matrix_unsupervised(X_2D):
    """计算相关系数矩阵"""
    corr_matrix = X_2D.T @ X_2D
    return corr_matrix


def compute_corrcoef_matrix(X_2D):
    """
    计算波段间相关系数矩阵（矩阵运算优化版）
    :param data: 高光谱数据 (h, w, d)
    :return: 相关系数矩阵 (d, d)
    """
    corrcoef_matrix = np.abs(np.corrcoef(X_2D, rowvar = False))
    return corrcoef_matrix

def compute_covariance_matrix(X_2D):
    """计算协方差矩阵"""

    cov_matrix = np.cov(X_2D, rowvar = False)
    return cov_matrix


def _compute_single_ssim(args):
    """
    辅助函数：计算单个波段对的SSIM值
    """
    hyperspectral_image, i, j, window_size = args
    band_img_i = hyperspectral_image[:, :, i].astype(np.int32)
    band_img_j = hyperspectral_image[:, :, j].astype(np.int32)
    ssim_value = structural_similarity(band_img_i, band_img_j, gaussian_weight=True, win_size=window_size)
    return i, j, ssim_value

def compute_ssim_matrix(X_3D, window_size=3, n_jobs=-1):
    """
    计算高光谱图像的SSIM（结构相似性）矩阵（对称矩阵）
    Input:
        X_3D (np.array): 高光谱图像矩阵 (行×列×波段数)
        window_size (int): SSIM计算的窗口大小，默认为3
        n_jobs (int): 并行任务数，-1表示使用所有CPU核心
    Output:
        ssim_matrix (np.array): 波段间的SSIM矩阵 (波段数×波段数)
    """
    total_band_count = X_3D.shape[-1]
    ssim_matrix = np.zeros((total_band_count, total_band_count))
    np.fill_diagonal(ssim_matrix, 1.0)
    
    # 生成所有需要计算的波段对 (i, j)，其中 i < j
    band_pairs = []
    for i in range(total_band_count):
        for j in range(i + 1, total_band_count):
            band_pairs.append((X_3D, i, j, window_size))
    
    # 使用joblib并行计算所有波段对的SSIM，并使用tqdm显示进度
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_single_ssim)(pair) for pair in tqdm(band_pairs, desc="Computing SSIM", unit="pair")
    )
    
    # 填充SSIM矩阵
    for i, j, ssim_value in results:
        ssim_matrix[i, j] = ssim_value
        ssim_matrix[j, i] = ssim_value
    
    return ssim_matrix


def _histogram_prob(data, bins=256, eps=1e-10):
    """计算一维数据的直方图概率分布（论文指定方案）"""
    counts, _ = np.histogram(data, bins=bins, density=False)
    p = counts / counts.sum()
    p = p + eps
    return p / p.sum()

def _joint_histogram_prob(data1, data2, bins=256, eps=1e-10):
    """计算两个一维数据的联合直方图概率分布"""
    counts, _, _ = np.histogram2d(data1, data2, bins=bins, density=False)
    p = counts / counts.sum()
    p = p + eps
    return p / p.sum()

def _shannon_entropy(p):
    """计算香农熵 H(X)"""
    return -np.sum(p * np.log2(p))

def mutual_info_distance(data1, data2, bins=256, eps=1e-10):
    """计算WaLuMI归一化互信息距离 D_NI（论文公式7）"""
    p1 = _histogram_prob(data1, bins, eps)
    p2 = _histogram_prob(data2, bins, eps)
    p_joint = _joint_histogram_prob(data1, data2, bins, eps)
    
    h1 = _shannon_entropy(p1)
    h2 = _shannon_entropy(p2)
    h_joint = _shannon_entropy(p_joint)
    
    mi = h1 + h2 - h_joint
    ni = 2 * mi / (h1 + h2 + eps)
    return (1 - np.sqrt(ni)) ** 2

def symmetric_kl_distance(data1, data2, bins=256, eps=1e-10):
    """计算WaLuDi对称KL散度距离 D_KL（论文公式8）"""
    data_min = min(data1.min(), data2.min())
    data_max = max(data1.max(), data2.max())
    bins_edges = np.linspace(data_min, data_max, bins + 1)
    
    counts1, _ = np.histogram(data1, bins=bins_edges, density=False)
    counts2, _ = np.histogram(data2, bins=bins_edges, density=False)
    
    p1 = counts1 / counts1.sum() + eps
    p2 = counts2 / counts2.sum() + eps
    p1 = p1 / p1.sum()
    p2 = p2 / p2.sum()
    
    kl1 = np.sum(p1 * np.log2(p1 / p2))
    kl2 = np.sum(p2 * np.log2(p2 / p1))
    return kl1 + kl2

# ==============================================
# 
# ==============================================
def _compute_single_distance(i, j, data_flat, dist_func, bins, eps):
    """计算单个波段对的距离（供并行调用）"""
    return i, j, dist_func(data_flat[:, i], data_flat[:, j], bins, eps)

def compute_distance_matrix_parallel(X_2D, dist_func, bins=256, eps=1e-10, n_jobs=-1):
    """
    并行构建L×L波段间不相似性矩阵
    :param data_flat: 展平的高光谱数据 (N_pixels, L)
    :param dist_func: 距离计算函数（mutual_info_distance或symmetric_kl_distance）
    :param n_jobs: 并行核心数，-1使用所有CPU核心
    :return: 对称距离矩阵 (L, L)
    """
    L = X_2D.shape[1]
    dist_matrix = np.zeros((L, L), dtype=np.float32)
    
    # 生成所有上三角波段对索引（避免重复计算）
    pairs = [(i, j) for i in range(L) for j in range(i + 1, L)]
    
    # 并行计算所有波段对的距离
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_compute_single_distance)(i, j, X_2D, dist_func, bins, eps)
        for i, j in pairs
    )
    
    # 填充对称距离矩阵
    for i, j, dist in results:
        dist_matrix[i, j] = dist
        dist_matrix[j, i] = dist
    
    return dist_matrix