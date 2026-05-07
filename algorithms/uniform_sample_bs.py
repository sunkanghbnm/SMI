import numpy as np

def uniform_sample_bs(num_total_bands, num_selected_bands):
    """
    波段选择函数：按均匀抽样方式选择波段
    
    参数:
    n -- 波段总数目
    k -- 需要选择的波段数目
    
    返回:
    包含选择波段序号(从1开始)的列表，按从小到大排序
    """
    # 计算基本组大小和剩余波段数
    step = num_total_bands / num_selected_bands
    
    start_ = step / 2
    end_ = num_total_bands - 1 - step / 2
    selected_bands = np.round(np.linspace(start_, end_, num = num_selected_bands)).astype(int)
    
    '''
    base_size = n // k
    remainder = n % k
    
    # 计算每个组的大小，均匀分配剩余波段
    group_sizes = [base_size] * k
    # 均匀分配剩余波段
    for i in range(remainder):
        # 计算分配位置，使用均匀间隔
        index = int(i * k / remainder)
        group_sizes[index] += 1
    
    selected_bands = []
    start = 1  # 波段序号从1开始
    for size in group_sizes:
        # 计算当前组的结束位置
        end = start + size - 1
        
        # 计算当前组的中心位置
        center = (start + end) / 2.0
        
        # 选择最接近中心的波段(四舍五入取整)
        band_index = round(center)
        selected_bands.append(int(band_index))
        
        # 更新下一组的起始位置
        start = end + 1
    '''
    return np.array(selected_bands)


def uniform_random_bs(band_subsets):
    """

    """
    # 
    selected_indices = []
    for subset in band_subsets:
       selected_indices.append(np.random.choice(subset))    
    return np.array(selected_indices)