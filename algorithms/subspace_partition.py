import numpy as np
import math
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import cdist,squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import pearsonr

def ocf_subspace_partition(X_2d: np.ndarray, cluster_num: int, 
                   sigma: float = None, 
                   custom_similarity: np.ndarray = None):
    """
    修正后的 Optimal Clustering Framework (OCF) 实现
    整合了数据输入、相似度计算、目标函数评估和动态规划求解

    Parameters
    ----------
    X_2d : np.ndarray
        高光谱数据矩阵，形状为 (L, N)，其中：
        L 是波段数 (Bands)，N 是像素数 (Pixels)
    cluster_num : int
        要划分的簇数 K
    sigma : float, optional
        构建高斯相似度矩阵的带宽参数 gamma 的倒数 (gamma=1/(2*sigma^2))
        如果为 None，将使用论文中的 Local Scaling 启发式方法
    custom_similarity : np.ndarray, optional
        自定义相似度矩阵 (L, L)，如果提供，则跳过内部计算
    Returns
    -------
    clusters : list[tuple[int, int]]
        最优簇划分结果，每个元素为 (起始索引, 结束索引)，0-based 闭区间
    critical_bands : np.ndarray
        临界波段索引（每个簇的结束索引）
    """
    N, L= X_2d.shape
    K = cluster_num

    if K < 1 or K > L:
        raise ValueError(f"簇数 K 必须满足 1 ≤ K ≤ L (当前 L={L})")

    # ==========================================
    # 步骤 1：构建相似度矩阵 W (对应论文 Section IV-A)
    # ==========================================
    if custom_similarity is not None:
        W = custom_similarity
    else:
        # 论文采用 Local Scaling 或 标准 RBF 核
        # 这里实现标准 RBF 核，并提供自动 sigma 估计
        if sigma is None:
            # 简单启发式：使用成对距离的中位数作为 sigma
            # 注：论文 Local Scaling (4.4节) 使用每个点的第7近邻，这里为简化使用全局估计
            from scipy.spatial.distance import pdist
            pairwise_dists = pdist(X_2d.T)
            sigma = np.median(pairwise_dists)
        
        gamma = 1.0 / (2 * sigma ** 2)
        W = rbf_kernel(X_2d, gamma=gamma)
        np.fill_diagonal(W, 0.0) # 对角线置0，避免自环影响

    # 预计算度矩阵 d (每行的和)，用于加速 NA 计算
    d = W.sum(axis=1)
    total_assoc_per_band = d # 论文中 sum_{k=1 to L} w_{jk}

    # ==========================================
    # 步骤 2：定义区间贡献评估函数 f_na
    # 对应论文公式 (14)：Normalized Association
    # ==========================================
    # 预计算前缀和矩阵，避免 DP 过程中重复计算，极大提升速度
    # prefix_W[i, j] = sum_{x=0 to i-1} sum_{y=0 to j-1} W[x, y]
    prefix_W = np.zeros((L + 1, L + 1), dtype=np.float64)
    for i in range(L):
        row_sum = 0
        for j in range(L):
            row_sum += W[i, j]
            prefix_W[i+1, j+1] = prefix_W[i, j+1] + row_sum

    def get_submatrix_sum(start, end):
        """计算 W[start:end+1, start:end+1] 的和"""
        return prefix_W[end+1, end+1] - prefix_W[start, end+1] - prefix_W[end+1, start] + prefix_W[start, start]
    
    def get_row_sum(start, end):
        """计算 sum_{j=start to end} total_assoc_per_band[j]"""
        return np.sum(total_assoc_per_band[start:end+1])

    # 这里的 f 对应论文公式 (1) 和 (14)
    # 注意：为了数值稳定性，我们不在 f 中除以 K，因为 K 是常数，不影响 argmax
    def eval_func(start_idx: int, end_idx: int):
        # 分子：sum_{within cluster} w_{jk}
        inner_assoc = get_submatrix_sum(start_idx, end_idx)
        # 分母：sum_{within cluster, all k} w_{jk}
        total_assoc = get_row_sum(start_idx, end_idx)
        
        if total_assoc < 1e-12:
            return 0.0
        return inner_assoc / total_assoc

    # ==========================================
    # 步骤 3：动态规划求解
    # ==========================================
    # 初始化 DP 表
    M = np.zeros((K + 1, L + 1), dtype=np.float64)
    Q = np.zeros((K + 1, L + 1), dtype=np.int32)

    # 边界条件：k=1
    for l in range(1, L + 1):
        M[1, l] = eval_func(0, l - 1)
        Q[1, l] = 0

    # 递推：k >= 2
    for k in range(2, K + 1):
        for l in range(k, L + 1):
            max_val = -np.inf
            best_p = 0
            # 遍历所有可能的分割点 p
            # p 是前 k-1 个簇的结束位置 (0-based index in array)
            # 对应论文中 s_{k-1}
            for p in range(k - 1, l):
                # 前 p 个点 (0~p-1) 分为 k-1 簇
                # p ~ l-1 为第 k 簇
                current_val = M[k - 1, p] + eval_func(p, l - 1)
                if current_val > max_val:
                    max_val = current_val
                    best_p = p
            
            M[k, l] = max_val
            Q[k, l] = best_p

    # ==========================================
    # 步骤 4：回溯路径
    # ==========================================
    critical_bands_1based = np.zeros(K + 1, dtype=np.int32)
    critical_bands_1based[K] = L
    
    for k in range(K, 0, -1):
        critical_bands_1based[k - 1] = Q[k, critical_bands_1based[k]]

    # 转换为 0-based 区间
    clusters = []
    for i in range(K):
        start = critical_bands_1based[i]
        end = critical_bands_1based[i + 1] - 1
        clusters.append(list(range(start, end + 1)))

    return clusters
def asps_subspace_partition(X_2d: np.ndarray, cluster_num: int, 
                            custom_similarity: np.ndarray = None):
    """
    implementation of ASPS (Adaptive Subspace Partition Strategy)
    paramters X_2d: np.ndarray, hypothetical 2D hyperspectral image
    paramters cluster_num: int, target number of subspaces
    paramters custom_similarity: np.ndarray, optional, custom similarity matrix
    return: list[list[int]], subspace partition result
    """

    
    N, L = X_2d.shape
    K = cluster_num

    if K < 1 or K > L:
        raise ValueError(f"subspace子空间数K必须满足1 ≤ K ≤ 总波段数L，当前K={K}, L={L}")

    # 波段向量化：(W, H, L) -> (L, W*H)，每一行对应一个波段的一维向量
    band_vectors = X_2d.T

    # --------------------------
    # 步骤2：计算波段间欧氏距离矩阵
    # --------------------------
    if custom_similarity is not None:
        if custom_similarity.shape != (L, L):
            raise ValueError(f"自定义相似度矩阵必须是(L, L)维度，当前为{custom_similarity.shape}")
        distance_matrix = custom_similarity
    else:
        # 计算欧氏距离矩阵，对齐论文公式(2)
        distance_matrix = cdist(band_vectors, band_vectors, metric='euclidean')

    # --------------------------
    # 步骤3：粗子空间划分（对齐论文公式1）
    # --------------------------
    # 计算每个初始子空间的波段数Z，处理无法整除的情况
    base_z = L // K
    remainder = L % K
    # 初始化分割点：0-based，split_points[i]是第i个子空间的结束索引+1
    split_points = [0]
    current = 0
    for i in range(K):
        # 前remainder个子空间多1个波段，保证总波段数一致
        step = base_z + 1 if i < remainder else base_z
        current += step
        split_points.append(current)
    # --------------------------
    # 步骤4：精子空间划分（核心，对齐论文公式4-8）
    # --------------------------
    # 依次处理每一对相邻子空间
    for i in range(K - 1):
        # 获取当前两个相邻子空间的波段范围
        start = split_points[i]
        mid = split_points[i + 1]
        end = split_points[i + 2]
        # 合并两个子空间的所有波段索引
        merged_band_idx = np.arange(start, end)
        n_merged = len(merged_band_idx)  # 合并后的总波段数，理论上为2Z

        # 提取合并波段对应的距离子矩阵
        sub_dist_matrix = distance_matrix[np.ix_(merged_band_idx, merged_band_idx)]

        # 遍历所有合法的分割点t，寻找最优解
        # t是分割点，前t个波段为第一个子空间，后n_merged-t个为第二个
        best_score = -np.inf
        best_t = mid - start  # 初始分割点为粗划分的位置

        for t in range(2, n_merged-1):
            # 计算类内距离 U1 + U2
            # U1: 前t个波段的平均类内距离
            
            u1 = np.sum(sub_dist_matrix[:t, :t]) / (t * (t - 1))

            # U2: 后n_merged-t个波段的平均类内距离
            u2 = np.sum(sub_dist_matrix[t:, t:]) / ((n_merged - t) * (n_merged - t - 1))

            d_intra = u1 + u2

            # 计算类间距离 D_inter：两个类之间的最大欧氏距离（论文公式5）
            inter_dist_submatrix = sub_dist_matrix[:t, t:]
            d_inter = np.max(inter_dist_submatrix) if inter_dist_submatrix.size > 0 else 0.0

            # 计算目标函数值，处理分母为0的边界情况
            if d_intra < 1e-12:
                score = np.inf if d_inter > 0 else 0.0
            else:
                score = d_inter / d_intra

            # 更新最优分割点
            if score > best_score:
                best_score = score
                best_t = t

        # 用最优分割点更新全局分割点
        new_mid = start + best_t
        split_points[i + 1] = new_mid

    # --------------------------
    # 步骤5：整理输出结果
    # --------------------------
    # 转换为(起始索引, 结束索引)的簇格式，0-based闭区间
    clusters = []
    for i in range(K):
        start = split_points[i]
        end = split_points[i + 1] - 1
        clusters.append(list(range(start, end + 1)))

    return clusters


def _calculate_sr_score(ssim_matrix, cluster_num, sr_eta = None):
    """
    SR函数：基于相似度矩阵计算波段的优先级得分（alpha/theta/eta），返回前k个波段索引、排序索引、eta得分
    注：k为外部循环变量（主函数中的选中波段数），此处为原代码隐式依赖逻辑
    Input:
        similarity_matrix (np.array): 相似度矩阵（SSIM/CC矩阵） (波段数×波段数)
    Output:
        top_k_indices (list): 按eta得分降序的前k个波段索引
        eta_sorted_indices (np.array): 所有波段按eta得分降序的索引
        eta_list (np.array): 每个波段的eta得分
    """
    
    L = ssim_matrix.shape[0]  # 矩阵大小=波段数
    # 提取相似度矩阵的上三角部分（不含对角线）
    ssim_values = ssim_matrix[np.triu_indices(L, k=1)]
    # 按相似度降序排序
    ssim_values_sorted = sorted(ssim_values, reverse = True)
    
    # 计算高相似度阈值：取前5%-10%相似度值的均值
    high_similarity_mean = np.mean(
        ssim_values_sorted[math.floor(len(ssim_values_sorted) * 0.05) - 1: math.floor(len(ssim_values_sorted) * 0.1) - 1])
    
    # 计算每个波段的alpha值：该波段与其他波段相似度>high_similarity_mean的均值
    alpha_list = []
    for i in range(L):
        current_band_similarity = ssim_matrix[i, :]
        high_similarity_values = current_band_similarity[current_band_similarity > high_similarity_mean]
        if len(high_similarity_values) == 0:
            alpha_list.append(0)
        else:
            alpha_list.append(np.mean(high_similarity_values))
    alpha_list = np.array(alpha_list)
    
    # 按alpha值降序获取波段索引
    alpha_sorted_indices = np.argsort(-alpha_list)
    
    # 计算varphi值：衡量波段的独立性
    varphi_list = np.zeros(L)
    varphi_list[alpha_sorted_indices[0]] = 1  # 第一个波段varphi=1
    for i in range(1, L):
        for j in range(i):
            # varphi取当前波段与已排序前j个波段的最大相似度
            if varphi_list[alpha_sorted_indices[i]] < ssim_matrix[alpha_sorted_indices[i], alpha_sorted_indices[j]]:
                varphi_list[alpha_sorted_indices[i]] = ssim_matrix[alpha_sorted_indices[i], alpha_sorted_indices[j]]
    varphi_list[alpha_sorted_indices[0]] = min(varphi_list)  # 第一个波段varphi设为最小值
    
    # 计算theta值：基于varphi的归一化得分
    theta_list = np.sqrt(1 - varphi_list ** 2)
    
    # 归一化alpha和theta，计算eta（alpha*theta）
    alpha_list = (alpha_list - min(alpha_list)) / (max(alpha_list) - min(alpha_list))
    theta_list = (theta_list - min(theta_list)) / (max(theta_list) - min(theta_list))
    eta_list = alpha_list * theta_list
    
    # 按eta得分降序排序波段索引
    eta_sorted_indices = np.argsort(-eta_list)
    # 取前k个波段索引（k为主函数的选中波段数，隐式依赖）
    top_k_indices = eta_sorted_indices[0:cluster_num].tolist()
    
    return top_k_indices, eta_sorted_indices, eta_list


def _find_best_tk(density, density_index,correlation_matrix):
    """
    计算最优划分点TK：基于划分代价（TKP）最小化，确定波段聚类的划分边界
    Input:
        density_index (int): density数组的索引
    Output:
        optimal_tk (int): 最优的划分点索引
    """
    # 初始最优划分点：density[i]和density[i+1]的均值向下取整
    optimal_tk = math.floor((density[density_index] + density[density_index + 1]) / 2)
    min_partition_cost = float('inf')  # 初始化最小划分代价为无穷大
    
    # 不同间隔的边界处理（原代码逻辑）
    if density[density_index] + 1 == density[density_index + 1]:
        return density[density_index] + 1
    elif density[density_index + 1] - density[density_index] == 2:
        return density[density_index] + 1
    elif density[density_index + 1] - density[density_index] == 3:
        return density[density_index] + 2
    else:
        # 遍历候选划分点，计算划分代价TKP，取最小值对应的tk
        for tk in range(density[density_index] + 2, density[density_index + 1]):
            # 跨聚类距离和：聚类0和聚类1之间的相关系数绝对值和
            cross_cluster_distance = 0
            for row in range(density[density_index], tk):
                for col in range(tk, density[density_index + 1] + 1):
                    cross_cluster_distance += abs(correlation_matrix[row, col])
            
            # 聚类0内部的相关系数绝对值和
            intra_cluster0_sum = 0
            for row in range(density[density_index], tk):
                for col in range(row, tk):
                    intra_cluster0_sum += abs(correlation_matrix[row, col])
            
            # 聚类1内部的相关系数绝对值和
            intra_cluster1_sum = 0
            for row in range(tk, density[density_index + 1] + 1):
                for col in range(row, density[density_index + 1] + 1):
                    intra_cluster1_sum += abs(correlation_matrix[row, col])
            
            # 计算划分代价：跨聚类距离 / (聚类0内和 × 聚类1内和)
            partition_cost = cross_cluster_distance / (intra_cluster0_sum * intra_cluster1_sum)
            
            # 更新最小代价和最优划分点
            if partition_cost < min_partition_cost:
                min_partition_cost = partition_cost
                optimal_tk = tk
        return optimal_tk

def _find_fine_best_tk(density, density_index, best_tk_list, correlation_matrix):
    """
    优化划分点TK：基于相邻划分点约束，精细调整最优划分点
    Input:
        density_index (int): density数组的索引
        best_tk_list (list): 当前最优划分点列表
    Output:
        optimal_tk (int): 精细调整后的最优划分点
    """
    optimal_tk = best_tk_list[density_index]
    min_partition_cost = float('inf')
    
    # 仅当当前density间隔>1时优化
    if density[density_index] - density[density_index - 1] > 1:
        # 遍历候选划分点（受相邻划分点约束）
        start_tk = max(best_tk_list[density_index - 1] + 2, density[density_index - 1] + 1)
        end_tk = min(density[density_index] + 1, best_tk_list[density_index + 1] - 1)
        for tk in range(start_tk, end_tk):
            # 计算跨聚类距离和
            cross_cluster_distance = 0
            for row in range(best_tk_list[density_index - 1], tk):
                for col in range(tk, best_tk_list[density_index + 1]):
                    cross_cluster_distance += abs(correlation_matrix[row, col])
            
            # 计算聚类0内部和
            intra_cluster0_sum = 0
            for row in range(best_tk_list[density_index - 1], tk):
                for col in range(row, tk):
                    intra_cluster0_sum += abs(correlation_matrix[row, col])
            
            # 计算聚类1内部和
            intra_cluster1_sum = 0
            for row in range(tk, best_tk_list[density_index + 1]):
                for col in range(row, best_tk_list[density_index + 1]):
                    intra_cluster1_sum += abs(correlation_matrix[row, col])
            
            # 计算划分代价并更新最优值
            partition_cost = cross_cluster_distance / (intra_cluster0_sum * intra_cluster1_sum)
            if partition_cost < min_partition_cost:
                min_partition_cost = partition_cost
                optimal_tk = tk
        return optimal_tk
    else:
        return optimal_tk

# -------------------------- GPC核心划分函数 --------------------------
def gpc_subspace_partition(X_3D: np.ndarray, cluster_num: int,
                           ssim_matrix: np.ndarray = None, correlation_matrix: np.ndarray = None):
    """
    实现论文Global Partition Clustering (GPC)的子空间划分核心功能
    
    Parameters
    ----------
    hsi_data : np.ndarray
        输入高光谱图像立方体，shape=(W, H, L)
        W: 图像宽度, H: 图像高度, L: 总波段数
    cluster_num : int
        目标子空间数N（即最终要选择的波段数）
    ssim_matrix : np.ndarray, optional
        预计算的波段间SSIM矩阵，shape=(L, L)，不提供则内部自动计算
    corr_matrix : np.ndarray, optional
        预计算的波段间皮尔逊相关系数矩阵，shape=(L, L)，不提供则内部自动计算
    max_iter : int, optional
        精划分的最大迭代次数，默认20次，防止无限迭代


    Returns
    -------
    clusters : list[tuple[int, int]]
        GPC最优子空间划分结果，每个元素为(起始波段索引, 结束波段索引)
        0-based闭区间，连续无重叠、无遗漏
    critical_bands : np.ndarray
        临界分割点索引，每个子空间的结束波段索引
    center_bands : np.ndarray
        每个子空间对应的中心波段索引
    """
    # -------------------------- 输入校验与预处理 --------------------------

    L = X_3D.shape[-1]
    K = cluster_num

    if K < 1 or K > L:
        raise ValueError(f"子空间数N必须满足1 ≤ N ≤ 总波段数L，当前K={K}, L={L}")
    
    density_indices, _, _ = _calculate_sr_score(ssim_matrix, K)
    density_indices = np.sort(density_indices)  # 排序density索引
    
    # 初始化划分点列表TK
    tk_partition_list = []
    tk_partition_list.append(0)  # 起始点
    for density_idx in range(len(density_indices) - 1):
        tk_partition_list.append(_find_best_tk(density_indices,density_idx, correlation_matrix))
    tk_partition_list.append(L)  # 终止点
    
    # 迭代优化划分点BTK
    best_tk_list = tk_partition_list.copy()
    for j in range(1, len(tk_partition_list) - 2):
        best_tk_list[j] = _find_fine_best_tk(density_indices,j, best_tk_list,correlation_matrix)
    # 循环优化直到划分点稳定
    while best_tk_list != tk_partition_list:
        tk_partition_list = best_tk_list.copy()
        for j in range(1, len(best_tk_list) - 2):
            best_tk_list[j] = _find_fine_best_tk(density_indices, j, best_tk_list, correlation_matrix)
    
    clusters = []
    for i in range(K):
        start = best_tk_list[i]
        end = best_tk_list[i + 1] - 1
        clusters.append(list(range(start, end + 1)))

    return clusters

def _ward_hierarchical_clustering(distance_matrix, n_clusters):
    """基于Ward链接的层次凝聚聚类（论文指定策略）"""
    condensed_dist = squareform(distance_matrix)
    Z = linkage(condensed_dist, method='ward')
    return fcluster(Z, t=n_clusters, criterion='maxclust')
def walumi_subspace_partition(X_2D: np.ndarray, cluster_num: int, 
                            custom_similarity: np.ndarray = None):
    # 参数校验
    if cluster_num >= X_2D.shape[1] or cluster_num <= 0:
        raise ValueError(f"目标波段数必须在 1~{X_2D.shape[1]-1} 之间")
    
    if custom_similarity is not None:
        dist_matrix = custom_similarity
    else:
        print("【WaLuMI】需要预先计算互信息距离矩阵...")
        return 

    cluster_labels = _ward_hierarchical_clustering(dist_matrix, cluster_num)
    
    clusters = []    
    for i in range(1,cluster_num+1):
        indices_tmp = np.where(cluster_labels == i)[0].tolist()
        clusters.append(indices_tmp)
    return clusters