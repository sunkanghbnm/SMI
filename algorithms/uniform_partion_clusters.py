import numpy as np

def uniform_band_partition(n_bands: int, k: int, random_state: int = 42) -> list:
    """
    将波段索引（0~n_bands-1）均匀划分为k个子集，若不能整除则随机选择若干子集使其元素数+1
    
    参数:
        n_bands: 总波段数（对应高光谱数据的L）
        k: 要划分的子集数目
        random_state: 随机种子，保证结果可复现
    
    返回:
        band_subsets: 列表，每个元素是一个子集的波段索引列表（如[[0,1], [2,3], [4]]）
    
    异常处理:
        若k > n_bands 或 k < 1，抛出ValueError
    """
    # 合法性检查
    if k < 1 or k > n_bands:
        raise ValueError(f"子集数目k需满足 1 ≤ k ≤ {n_bands}（总波段数），当前k={k}")
    
    # 计算基础子集大小和余数（需要额外+1的子集数量）
    base_size = n_bands // k
    remainder = n_bands % k  # 余数即为需要多1个元素的子集数量
    
    # 设置随机种子
    np.random.seed(random_state)
    
    # 生成k个子集的大小列表：先初始化为base_size，再随机选remainder个子集加1
    subset_sizes = [base_size] * k
    if remainder > 0:
        # 随机选择remainder个不重复的子集索引，使其大小+1
        add_indices = np.random.choice(k, size=remainder, replace=False)
        for idx in add_indices:
            subset_sizes[idx] += 1
    
    # 分配波段索引到各个子集
    band_indices = np.arange(n_bands)  # 总波段索引：0,1,...,n_bands-1
    band_subsets = []
    start_idx = 0
    for size in subset_sizes:
        # 截取对应大小的索引段
        subset = band_indices[start_idx:start_idx + size].tolist()
        band_subsets.append(subset)
        start_idx += size
    
    return band_subsets

def select_elements_from_disjoint_subsets(arr, subsets, num_selected_elements = None):
    """
    假设子集之间不相交（即每个元素只属于一个子集）
    """
    if num_selected_elements is None:
        num_selected_elements = len(subsets)
    # 步骤1: 创建一个映射，记录每个元素所属的子集索引
    element_to_subset = {}
    for subset_idx, subset in enumerate(subsets):
        for element in subset:
            element_to_subset[element] = subset_idx
    
    # 步骤2: 初始化
    selected_elements = []  # 存储选中的元素
    selected_subsets = set()  # 记录已选中的子集索引
    
    # 步骤3: 按照数组顺序遍历元素
    for element in arr:
        # 如果元素不属于任何子集，跳过
        if element not in element_to_subset:
            continue
            
        # 获取当前元素所属的子集索引
        subset_idx = element_to_subset[element]
        
        # 如果这个子集还没有被选中
        if subset_idx not in selected_subsets:
            selected_elements.append(element)
            selected_subsets.add(subset_idx)
        
        # 如果已选中足够的元素，提前结束
        if len(selected_elements) == num_selected_elements:
            break
    
    return np.array(selected_elements)

def select_elements(arr: list, subsets: list) -> list:
    """
    按规则筛选元素：数组顺序遍历 + 每个子集必选1个 + 首元素必选 + 同子集互斥
    
    参数：
        arr: 有序数组（所有元素必须在subsets中）
        subsets: 子集列表，每个子集是列表
    
    返回：
        按数组顺序选中的元素列表，长度 = 子集数量
    """
    # ============== 1. 构建【元素 → 所属子集】映射（快速查询）==============
    element_to_subset = {}
    for subset in subsets:
        for elem in subset:
            element_to_subset[elem] = subset

    # ============== 2. 初始化变量 ==============
    selected = []               # 最终选中的元素
    selected_subsets = set()    # 已经选过的子集（用集合存，去重+快速判断）
    total_subsets = len(subsets)  # 需要选的总数量

    # ============== 3. 按数组顺序遍历筛选 ==============
    for elem in arr:
        # 选满了，直接退出
        if len(selected) == total_subsets:
            break
        
        current_subset = element_to_subset[elem]  # 当前元素所属子集
        
        # 第一个元素：强制选中
        if len(selected) == 0:
            selected.append(elem)
            selected_subsets.add(tuple(current_subset))  # 列表转元组才能存集合
            continue
        
        # 后续元素：所属子集未被选过 → 选中；否则跳过
        if tuple(current_subset) not in selected_subsets:
            selected.append(elem)
            selected_subsets.add(tuple(current_subset))

    return selected