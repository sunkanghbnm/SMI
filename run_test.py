import time, os
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from algorithms.basic_utils import compute_entropy, euclidean_distance_matrix, compute_corrcoef_matrix, \
    compute_correlation_matrix_unsupervised, compute_covariance_matrix,   rbf_kernel_dismat,compute_variances,  compute_ssim_matrix
from algorithms.similarity_ranking_bs import compute_mssim_matrix, similarity_ranking_bs,similarity_ranking_smi_bs
from algorithms.eca_bs import eca_bs,eca_smi_bs
from algorithms.self_representation_bs import  self_representation_bs, self_representation_smi_bs
from algorithms.max_covmatrix_det_bs import max_covmatrix_det_bs,  max_covmatrix_det_smi_bs
from algorithms.mvpca_bs import mvpca_bs,  mvpca_smi_bs
from algorithms.uniform_sample_bs import uniform_sample_bs, uniform_random_bs
from algorithms.uniform_partion_clusters import uniform_band_partition
from algorithms.subspace_partition import ocf_subspace_partition, asps_subspace_partition,gpc_subspace_partition,walumi_subspace_partition
from smi import calc_cis, calc_mic_adj,calc_mic_unadj,calc_mic_vs_bandinterval,calc_smi,calc_band_fit_r2, calculate_adjacent_conditional_entropy
from data_loader import HSIDataLoader
from config import (
    DATASET_PATHS, BASE_DATA_DIR, 
    SMI_PARAMS, SVM_PARAMS, OUTPUT_DIR
)
from scipy.stats import ttest_rel
from typing import List, Optional, Dict, Tuple
from joblib import Parallel, delayed, cpu_count

def compute_and_cache_statistics(X, X_normlized, X_3D, 
                                  dataset_dir = '/mnt/f/pytorch/HSI/datasets/dataset_name',
                                  dataset_prefix = 'dataset_name',
                                  n_jobs = -1):
    """
    计算并缓存所有统计参数矩阵
    
    参数:
        X: 原始数据 (n_samples, n_features)
        X_normlized: 标准化后的数据
        X_3D: 3D高光谱数据
        X_train: 训练集数据
        X_test: 测试集数据
        y_train: 训练集标签
        y_test: 测试集标签
        dataset_dir: 缓存文件保存目录
        n_jobs: 并行计算的job数量
    
    返回:
        包含所有统计参数的字典
    """
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 定义缓存文件路径
    cache_paths = {
        'ssim': os.path.join(dataset_dir, f'{dataset_prefix}_ssim_matrix.npy'),
        'ssim_win3': os.path.join(dataset_dir, f'{dataset_prefix}_ssim_win3_matrix.npy'),
        'sr_eta': os.path.join(dataset_dir, f'{dataset_prefix}_sr_eta.npy'),
        'variances': os.path.join(dataset_dir, f'{dataset_prefix}_variances.npy'),
        'entropy': os.path.join(dataset_dir, f'{dataset_prefix}_entropy.npy'),
        'distance_matrix': os.path.join(dataset_dir, f'{dataset_prefix}_distance_matrix.npy'),
        'rbf_kernel': os.path.join(dataset_dir, f'{dataset_prefix}_rbf_kernel_matrix.npy'),
        'covariance_matrix': os.path.join(dataset_dir, f'{dataset_prefix}_covariance_matrix.npy'),
        'correlation_matrix': os.path.join(dataset_dir, f'{dataset_prefix}_correlation_matrix.npy'),
        'corrcoef_matrix': os.path.join(dataset_dir, f'{dataset_prefix}_corrcoef_matrix.npy'),
        'mic_adj': os.path.join(dataset_dir, f'{dataset_prefix}_mic_adj.npy'),
        'mic_unadj': os.path.join(dataset_dir, f'{dataset_prefix}_mic_unadj.npy'),
        'cis': os.path.join(dataset_dir, f'{dataset_prefix}_cis.npy'),
        'adj_entropy': os.path.join(dataset_dir, f'{dataset_prefix}_adj_entropy.npy'),
        'mic_vs_bandinterval': os.path.join(dataset_dir, f'{dataset_prefix}_mic_vs_bandinterval.npy'),
        'r2_array': os.path.join(dataset_dir, f'{dataset_prefix}_r2_array.npy'),
        'mic_adj_shuffled': os.path.join(dataset_dir, f'{dataset_prefix}_mic_adj_shuffled.npy'),
        'mic_unadj_shuffled': os.path.join(dataset_dir, f'{dataset_prefix}_mic_unadj_shuffled.npy'),
        'cis_shuffled': os.path.join(dataset_dir, f'{dataset_prefix}_cis_shuffled.npy'),
        'mic_vs_bandinterval_shuffled': os.path.join(dataset_dir, f'{dataset_prefix}_mic_vs_bandinterval_shuffled.npy'),
        'r2_array_shuffled': os.path.join(dataset_dir, f'{dataset_prefix}_r2_array_shuffled.npy'),
    }    
    stats = {}
    
    # SSIM矩阵
    if os.path.exists(cache_paths['ssim']):
        #print("检测到已有ssim矩阵文件，正在加载...")
        stats['ssim'] = np.load(cache_paths['ssim'])
    else:
        print("未检测到ssim矩阵文件，正在计算...")
        stats['ssim'] = compute_mssim_matrix(X_3D, window_size=11, sigma=1.5, L=255, n_jobs=n_jobs)
        np.save(cache_paths['ssim'], stats['ssim'])
    
    # SSIM矩阵 windows size =3
    if os.path.exists(cache_paths['ssim_win3']):
        #print("检测到已有ssi win3矩阵文件，正在加载...")
        stats['ssim_win3'] = np.load(cache_paths['ssim_win3'])
    else:
        print("未检测到ssim win3矩阵文件，正在计算...")
        stats['ssim_win3'] = compute_ssim_matrix(X_3D, window_size = 3)
        np.save(cache_paths['ssim_win3'], stats['ssim_win3'])
    
    # SR Eta
    if os.path.exists(cache_paths['sr_eta']):
        #print("检测到已有sr_eta矩阵文件，正在加载...")
        stats['sr_eta'] = np.load(cache_paths['sr_eta'])
    else:
        print("未检测到sr_eta矩阵文件，正在计算...")
        _, stats['sr_eta'] = similarity_ranking_bs(stats['ssim'], similarity_threshold=0.8, num_selected_bands=10)
        np.save(cache_paths['sr_eta'], stats['sr_eta'])
    
    # Variances
    if os.path.exists(cache_paths['variances']):
        #print("检测到已有variance矩阵文件，正在加载...")
        stats['variances'] = np.load(cache_paths['variances'])
    else:
        print("未检测到variance矩阵文件，正在计算...")
        stats['variances'] = compute_variances(X)
        np.save(cache_paths['variances'], stats['variances'])
    
    # Entropy
    if os.path.exists(cache_paths['entropy']):
        #print("检测到已有entropy文件，正在加载...")
        stats['entropy'] = np.load(cache_paths['entropy'])
    else:
        print("未检测到entropy文件，正在计算...")
        stats['entropy'] = compute_entropy(X_normlized)
        np.save(cache_paths['entropy'], stats['entropy'])
    
    # Distance Matrix
    if os.path.exists(cache_paths['distance_matrix']):
        #print("检测到已有distance矩阵文件，正在加载...")
        stats['distance_matrix'] = np.load(cache_paths['distance_matrix'])
    else:
        print("未检测到distance矩阵文件，正在计算...")
        stats['distance_matrix'] = euclidean_distance_matrix(X)
        np.save(cache_paths['distance_matrix'], stats['distance_matrix'])
    
    # RBF Kernel Distance
    if os.path.exists(cache_paths['rbf_kernel']):
        #print("检测到已有rbf distance矩阵文件，正在加载...")
        stats['rbf_kernel'] = np.load(cache_paths['rbf_kernel'])
    else:
        print("未检测到rbf distance矩阵文件，正在计算...")
        stats['rbf_kernel'] = rbf_kernel_dismat(stats['distance_matrix'], sigma=np.mean(stats['distance_matrix']) / 30)
        np.save(cache_paths['rbf_kernel'], stats['rbf_kernel'])
    
    # Covariance Matrix
    if os.path.exists(cache_paths['covariance_matrix']):
        #print("检测到已有covariance matrix矩阵文件，正在加载...")
        stats['covariance_matrix'] = np.load(cache_paths['covariance_matrix'])
    else:
        print("未检测到covariance matrix矩阵文件，正在计算...")
        stats['covariance_matrix'] = compute_covariance_matrix(X)
        np.save(cache_paths['covariance_matrix'], stats['covariance_matrix'])
    
    # Correlation Matrix (全量)
    if os.path.exists(cache_paths['correlation_matrix']):
        #print("检测到已有correlation matrix矩阵文件，正在加载...")
        stats['correlation_matrix'] = np.load(cache_paths['correlation_matrix'])
    else:
        print("未检测到correlation matrix矩阵文件，正在计算...")
        stats['correlation_matrix'] = compute_correlation_matrix_unsupervised(X)
        np.save(cache_paths['correlation_matrix'], stats['correlation_matrix'])
    
    
    # Correlation Coefficient Matrix (全量)
    if os.path.exists(cache_paths['corrcoef_matrix']):
        #print("检测到已有corrcoef矩阵文件，正在加载...")
        stats['corrcoef_matrix'] = np.load(cache_paths['corrcoef_matrix'])
    else:
        print("未检测到corrcoef矩阵文件，正在计算...")
        stats['corrcoef_matrix'] = compute_corrcoef_matrix(X_normlized)
        np.save(cache_paths['corrcoef_matrix'], stats['corrcoef_matrix'])
    
    # MIC Adjacent
    if os.path.exists(cache_paths['mic_adj']):
        #print("检测到已有mic adj文件，正在加载...")
        stats['mic_adj'] = np.load(cache_paths['mic_adj'])
    else:
        print("未检测到mic adj文件，正在计算...")
        stats['mic_adj'] = calc_mic_adj(X)
        np.save(cache_paths['mic_adj'], stats['mic_adj'])
    
    # MIC Unadjusted
    if os.path.exists(cache_paths['mic_unadj']):
        #print("检测到已有mic unadj文件，正在加载...")
        stats['mic_unadj'] = np.load(cache_paths['mic_unadj'])
    else:
        print("未检测到mic unadj文件，正在计算...")
        stats['mic_unadj'] = calc_mic_unadj(X)
        np.save(cache_paths['mic_unadj'], stats['mic_unadj'])
    
    # MIC vs bandinterval
    if os.path.exists(cache_paths['mic_vs_bandinterval']):
        #print("检测到已有mic vs bandinterval文件，正在加载...")
        stats['mic_vs_bandinterval'] = np.load(cache_paths['mic_vs_bandinterval'])
        base_path_mic_vs_bandinterval, _ = os.path.splitext(cache_paths['mic_vs_bandinterval'])
        csv_path_mic_vs_bandinterval= f"{base_path_mic_vs_bandinterval}{'.csv'}"
        if not os.path.exists(csv_path_mic_vs_bandinterval):
            np.savetxt(csv_path_mic_vs_bandinterval, stats['mic_vs_bandinterval'], delimiter=',')
    else:
        print("未检测到mic vs bandinterval文件，正在计算...")
        stats['mic_vs_bandinterval'] = calc_mic_vs_bandinterval(X, max_k=50)
        np.save(cache_paths['mic_vs_bandinterval'], stats['mic_vs_bandinterval'])
    
    # CIS
    if os.path.exists(cache_paths['cis']):
        #print("检测到已有cis文件，正在加载...")
        stats['cis'] = np.load(cache_paths['cis'])
    else:
        print("未检测到cis文件，正在计算...")
        stats['cis'] = calc_cis(X)
        np.save(cache_paths['cis'], stats['cis'])
        
    if os.path.exists(cache_paths['adj_entropy']):
        #print("检测到已有cis文件，正在加载...")
        stats['adj_entropy'] = np.load(cache_paths['adj_entropy'])
        base_path_adj_entropy, _ = os.path.splitext(cache_paths['adj_entropy'])
        csv_path_adj_entropy = f"{base_path_adj_entropy}{'.csv'}"
        if not os.path.exists(csv_path_adj_entropy):
            np.savetxt(csv_path_adj_entropy, stats['adj_entropy'], delimiter=',')
    else:
        print("未检测到adj_entropy文件，正在计算...")
        stats['adj_entropy'] = calculate_adjacent_conditional_entropy(X, n_bins = 1024, bin_method = 'quantile')
        np.save(cache_paths['adj_entropy'], stats['adj_entropy'])
    
    # R² Array
    if os.path.exists(cache_paths['r2_array']):
        #print("检测到已有r2 array文件，正在加载...")
        stats['r2_array'] = np.load(cache_paths['r2_array'])
    else:
        print("未检测到r2 array文件，正在计算...")
        stats['r2_array'] = calc_band_fit_r2(X, remove_outlier=True, use_adjusted_r2=True)
        np.save(cache_paths['r2_array'], stats['r2_array'])
        
        
    # MIC Adjacent shuffled
    if os.path.exists(cache_paths['mic_adj_shuffled']):
        #print("检测到已有mic adj shuffled文件，正在加载...")
        stats['mic_adj_shuffled'] = np.load(cache_paths['mic_adj_shuffled'])
    else:
        print("未检测到mic adj shuffled文件，正在计算...")
        stats['mic_adj_shuffled'] = calc_mic_adj(X, shuffled = True)
        np.save(cache_paths['mic_adj_shuffled'], stats['mic_adj_shuffled'])
    
    # MIC Unadjusted shuffled
    if os.path.exists(cache_paths['mic_unadj_shuffled']):
        #print("检测到已有mic unadj shuffled文件，正在加载...")
        stats['mic_unadj_shuffled'] = np.load(cache_paths['mic_unadj_shuffled'])
    else:
        print("未检测到mic unadj shuffled文件，正在计算...")
        stats['mic_unadj_shuffled'] = calc_mic_unadj(X, shuffled = True)
        np.save(cache_paths['mic_unadj_shuffled'], stats['mic_unadj_shuffled'])
    
    # MIC vs bandinterval shuffled
    if os.path.exists(cache_paths['mic_vs_bandinterval_shuffled']):
        #print("检测到已有mic vs bandinterval shuffled文件，正在加载...")
        stats['mic_vs_bandinterval_shuffled'] = np.load(cache_paths['mic_vs_bandinterval_shuffled'])
    else:
        print("未检测到mic vs bandinterval shuffled文件，正在计算...")
        stats['mic_vs_bandinterval_shuffled'] = calc_mic_vs_bandinterval(X, max_k=50,shuffled = True)
        np.save(cache_paths['mic_vs_bandinterval_shuffled'], stats['mic_vs_bandinterval_shuffled'])
    
    # CIS
    if os.path.exists(cache_paths['cis_shuffled']):
        #print("检测到已有cis shuffled文件，正在加载...")
        stats['cis_shuffled'] = np.load(cache_paths['cis_shuffled'])
    else:
        print("未检测到cis shuffled文件，正在计算...")
        stats['cis_shuffled'] = calc_cis(X, shuffled = True)
        np.save(cache_paths['cis_shuffled'], stats['cis_shuffled'])
    
    # R² Array
    if os.path.exists(cache_paths['r2_array_shuffled']):
        #print("检测到已有r2 array shuffled文件，正在加载...")
        stats['r2_array_shuffled'] = np.load(cache_paths['r2_array_shuffled'])
    else:
        print("未检测到r2 array shuffled文件，正在计算...")
        stats['r2_array_shuffled'] = calc_band_fit_r2(X, remove_outlier=True, use_adjusted_r2=True, shuffled=True)
        np.save(cache_paths['r2_array_shuffled'], stats['r2_array_shuffled'])
    
    return stats


def _single_run_evaluation(
    run_id: int,
    result_dir: str,
    dataset_name: str,
    n_selected_bands_list: List[int],
    X_normlized: np.ndarray,
    y_true: np.ndarray,
    band_selection_methods: List[str],
    test_size: float,
    model_random_state_base: int,
    is_stochastic_method: Dict[str, bool],
) -> List[Dict]:
    """
    单次独立重复实验的评估逻辑（封装为独立函数供joblib并行调用）
    
    注意：这是一个内部函数，不需要直接调用
    """
    current_random_state = model_random_state_base + run_id
    run_results = []
    
    try:
        # 每次run都重新划分训练/测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_normlized, y_true, test_size=test_size, 
            random_state=current_random_state, stratify=y_true
        )
        
        # 每次run都重新初始化分类模型
        model = LinearSVC(C=1, random_state = current_random_state, max_iter=10000)
        
        # 遍历选带数量 + 波段选择方法
        for n_bands in n_selected_bands_list:
            for method in band_selection_methods:
                try:
                    # 根据方法类型加载对应的波段选择结果
                    if is_stochastic_method.get(method, False):
                        band_idx_path = os.path.join(
                            result_dir,
                            f"{dataset_name}_{method}_k{n_bands}_run{run_id}.npy"
                        )
                    else:
                        band_idx_path = os.path.join(
                            result_dir,
                            f"{dataset_name}_{method}_k{n_bands}.npy"
                        )
                    
                    if not os.path.exists(band_idx_path):
                        continue
                    
                    # 加载并处理波段索引
                    selected_band_idx = np.load(band_idx_path).flatten()
                    valid_idx = selected_band_idx[
                        (selected_band_idx >= 0) & (selected_band_idx < X_train.shape[1])
                    ]
                    
                    if len(valid_idx) == 0:
                        continue
                    
                    # 提取数据、训练模型、评估
                    X_train_selected = X_train[:, valid_idx]
                    X_test_selected = X_test[:, valid_idx]
                    
                    model.fit(X_train_selected, y_train)
                    y_test_pred = model.predict(X_test_selected)
                    test_acc = accuracy_score(y_test, y_test_pred)
                    
                    # 记录结果
                    run_results.append({
                        "run_id": run_id,
                        "method": method,
                        "n_selected_bands": n_bands,
                        "test_accuracy": test_acc,
                        "selected_band_count": len(valid_idx),
                        "selected_band_idx": valid_idx.tolist()
                    })
                    
                except Exception as e:
                    print(f"⚠️ Run {run_id} | {method} | K={n_bands} 评估出错：{str(e)}")
                    continue
                    
    except Exception as e:
        print(f"❌ Run {run_id} 整体执行出错：{str(e)}")
    
    return run_results

def evaluate_band_selection_results(
    result_dir: str,
    dataset_name: str,
    n_selected_bands_list: List[int],
    X_normlized: np.ndarray,
    y_true: np.ndarray,
    band_selection_methods: List[str],
    baseline_methods: Optional[List[str]] = None,
    test_size: float = 0.2,
    n_runs: int = 10,
    model_random_state_base: int = 42,
    is_stochastic_method: Optional[Dict[str, bool]] = None,
    n_jobs: Optional[int] = None,  # 新增：并行进程数，默认使用所有核心-1
    verbose = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    独立重复评估所有波段选择结果的分类性能（支持joblib并行化，符合IEEE TGRS顶刊规范）
    
    核心功能：
    1. 使用joblib并行执行n_runs次独立重复实验，充分利用多核CPU
    2. 完整记录每次实验的原始OA值
    3. 自动计算均值、标准差、方差
    4. 基于配对t检验计算相对于基准方法的统计显著性p值
    
    参数：
        result_dir: 波段选择结果(.npy)的保存目录
        dataset_name: 数据集前缀（用于拼接结果文件名）
        n_selected_bands_list: 选带数量列表（如[5,10,15]）
        X_normalized: 标准化后的全量数据（注意：是全量，不是划分后的训练集）
        y_true: 全量标签
        band_selection_methods: 波段选择方法列表（如['mvpca', 'mvpca_smi', 'eca', 'eca_smi']）
        baseline_methods: 基准方法列表（用于计算显著性，如['mvpca', 'eca']），默认取列表前半部分
        test_size: 测试集比例，默认0.2
        n_runs: 独立重复实验次数，默认10次
        model_random_state_base: 随机种子基准值，每次run递增1
        is_stochastic_method: 标记哪些是随机性波段选择方法（如{'ocf': True, 'mvpca': False}）
            - 确定性方法：每次run加载同一个.npy文件（波段选择结果固定）
            - 随机性方法：每次run加载不同的.npy文件（文件名需包含run_id，如{dataset_name}_{method}_k{n_bands}_run{run_id}.npy）
        n_jobs: 并行进程数，默认使用所有CPU核心-1
        verbose: joblib并行日志详细程度，0-10，越大越详细
    
    返回：
        raw_result_df: 包含所有单次实验原始结果的DataFrame
        summary_result_df: 包含统计汇总（均值、标准差、方差、p值）的DataFrame
    """
    # ---------------------- 初始化参数 ----------------------
    if baseline_methods is None:
        baseline_methods = band_selection_methods[:len(band_selection_methods)//2]
    
    if is_stochastic_method is None:
        is_stochastic_method = {method: False for method in band_selection_methods}
    
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)  # 默认使用所有核心-1，保留1个核心给系统
    
    print(f"{'='*80}")
    print(f"========== 开始并行评估（使用 {n_jobs} 个CPU核心） ==========")
    print(f"{'='*80}")
    raw_csv_path = os.path.join(result_dir, f"{dataset_name}_classification_results_raw.csv")
    summary_csv_path = os.path.join(result_dir, f"{dataset_name}_classification_results_summary.csv")
    
    if os.path.exists(summary_csv_path):
        print(f'The result file already exists, skip {dataset_name}.')
        return
    # ---------------------- 使用joblib并行执行n_runs次实验 ----------------------
    all_run_results = Parallel(
        n_jobs=n_jobs,
        verbose = verbose,
        backend='loky'  # 使用loky后端，Windows和Linux都兼容，且支持共享内存
    )(
        delayed(_single_run_evaluation)(
            run_id=run_id,
            result_dir=result_dir,
            dataset_name=dataset_name,
            n_selected_bands_list=n_selected_bands_list,
            X_normlized=X_normlized,
            y_true=y_true,
            band_selection_methods=band_selection_methods,
            test_size=test_size,
            model_random_state_base=model_random_state_base,
            is_stochastic_method=is_stochastic_method,
        )
        for run_id in range(n_runs)
    )
    
    # ---------------------- 合并所有并行结果 ----------------------
    raw_evaluation_results = []
    for run_result in all_run_results:
        raw_evaluation_results.extend(run_result)
    
    if len(raw_evaluation_results) == 0:
        raise ValueError("没有成功评估任何结果，请检查波段选择文件是否存在！")
    
    # ---------------------- 生成原始结果DataFrame并保存 ----------------------
    raw_result_df = pd.DataFrame(raw_evaluation_results)
    os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
    raw_result_df.to_csv(raw_csv_path, index=False, encoding='utf-8')
    print(f"\n📊 原始单次实验结果已保存至：{raw_csv_path}")
    
    # ---------------------- 生成统计汇总DataFrame ----------------------
    print(f"\n{'='*80}")
    print(f"========== 计算统计汇总与显著性检验 ==========")
    print(f"{'='*80}")
    
    summary_results = []
    
    # 按方法和选带数量分组
    grouped = raw_result_df.groupby(["method", "n_selected_bands"])
    
    for (method, n_bands), group in grouped:
        # 计算统计指标
        acc_list = group["test_accuracy"].values
        mean_acc = np.mean(acc_list)
        std_acc = np.std(acc_list, ddof=1)  # 使用样本标准差（ddof=1）
        var_acc = np.var(acc_list, ddof=1)
        
        # 初始化p值
        p_value = np.nan
        significance_mark = ""
        
        # ---------------------- 计算配对t检验p值（仅针对非基准方法） ----------------------
        if method not in baseline_methods:
            # 找到对应的基准方法
            corresponding_baseline = None
            for baseline in baseline_methods:
                if baseline in method:
                    corresponding_baseline = baseline
                    break
            
            if corresponding_baseline is not None:
                # 获取基准方法在相同n_bands下的10次OA值
                baseline_group = raw_result_df[
                    (raw_result_df["method"] == corresponding_baseline) &
                    (raw_result_df["n_selected_bands"] == n_bands)
                ]
                
                if len(baseline_group) == n_runs:
                    baseline_acc_list = baseline_group["test_accuracy"].values
                    
                    # 执行双侧配对t检验（TGRS唯一认可的检验方式）
                    t_stat, p_value = ttest_rel(acc_list, baseline_acc_list, alternative='two-sided')
                    
                    # 生成显著性标记
                    if p_value < 0.01:
                        significance_mark = "**"
                    elif p_value < 0.05:
                        significance_mark = "*"
                    
                    print(f"✅ {method} vs {corresponding_baseline} (K={n_bands}): p={p_value:.4f} {significance_mark}")
        
        # 记录汇总结果
        summary_results.append({
            "method": method,
            "n_selected_bands": n_bands,
            "mean_test_accuracy": mean_acc,
            "std_test_accuracy": std_acc,
            "var_test_accuracy": var_acc,
            "p_value_vs_baseline": p_value,
            "significance_mark": significance_mark,
            "raw_accuracy_list": acc_list.tolist()
        })
    
    # ---------------------- 生成汇总结果DataFrame并保存 ----------------------
    summary_result_df = pd.DataFrame(summary_results)
    summary_result_df = summary_result_df.sort_values(["method", "n_selected_bands"]).reset_index(drop=True)
    
    summary_result_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
    print(f"\n📊 统计汇总结果已保存至：{summary_csv_path}")
    print(f"\n{'='*80}")
    print(f"========== 所有评估完成！ ==========")
    print(f"{'='*80}")
    
    return raw_result_df, summary_result_df

def run_band_selection(result_dir, dataset_name, total_band_number, n_selected_bands_list, 
                       band_selection_methods, X_2D, precomputed_stats):
    for n_selected_bands in n_selected_bands_list:
        band_subsets_smi = uniform_band_partition(total_band_number, n_selected_bands)
        band_subsets_ocf = ocf_subspace_partition(X_2D, n_selected_bands,custom_similarity=precomputed_stats['rbf_kernel'])
        band_subsets_asps = asps_subspace_partition(X_2D, n_selected_bands, custom_similarity=precomputed_stats['distance_matrix'])
        band_subsets_gpc = gpc_subspace_partition(X_2D, n_selected_bands, ssim_matrix=precomputed_stats['ssim_win3'],
                                                  correlation_matrix=precomputed_stats['correlation_matrix'])
        band_subets_walumi = walumi_subspace_partition(X_2D, n_selected_bands, custom_similarity=precomputed_stats['mutual_info_distance'])
        
        result_paths = {
            band_selection_method: os.path.join(result_dir, f'{dataset_name}_{band_selection_method}_k{n_selected_bands}.npy')
            for band_selection_method in band_selection_methods
            }
        ####################MVPCA##################
        if not os.path.exists(result_paths['mvpca']):
            print(f"未检测到{n_selected_bands}个波段MVPCA文件，正在计算...")
            mvpca_result = mvpca_bs(X_2D, num_selected_feature = n_selected_bands, variances = precomputed_stats['variances'])
            np.save(result_paths['mvpca'], mvpca_result)
            
        if not os.path.exists(result_paths['mvpca_smi']):
            print(f"未检测到{n_selected_bands}个波段MVPCA_SMI文件，正在计算...")
            mvpca_smi_result = mvpca_smi_bs(X_2D, num_selected_feature = n_selected_bands, 
                                            band_subsets= band_subsets_smi, variances = precomputed_stats['variances'])
            np.save(result_paths['mvpca_smi'], mvpca_smi_result)
        
        if not os.path.exists(result_paths['mvpca_ocf']):
            print(f"未检测到{n_selected_bands}个波段MVPCA_OCF文件，正在计算...")
            mvpca_ocf_result = mvpca_smi_bs(X_2D, num_selected_feature = n_selected_bands, 
                                            band_subsets= band_subsets_ocf, variances = precomputed_stats['variances'])
            np.save(result_paths['mvpca_ocf'], mvpca_ocf_result)
        
        if not os.path.exists(result_paths['mvpca_asps']):
            print(f"未检测到{n_selected_bands}个波段MVPCA_ASPS文件，正在计算...")
            mvpca_asps_result = mvpca_smi_bs(X_2D, num_selected_feature = n_selected_bands, 
                                            band_subsets= band_subsets_asps, variances = precomputed_stats['variances'])
            np.save(result_paths['mvpca_asps'], mvpca_asps_result)
        if not os.path.exists(result_paths['mvpca_gpc']):
            print(f"未检测到{n_selected_bands}个波段MVPCA_GPC文件，正在计算...")
            mvpca_gpc_result = mvpca_smi_bs(X_2D, num_selected_feature = n_selected_bands, 
                                            band_subsets= band_subsets_gpc, variances = precomputed_stats['variances'])
            np.save(result_paths['mvpca_gpc'], mvpca_gpc_result)
        if not os.path.exists(result_paths['mvpca_walumi']):
            print(f"未检测到{n_selected_bands}个波段MVPCA_WALUMI文件，正在计算...")
            mvpca_walumi_result = mvpca_smi_bs(X_2D, num_selected_feature = n_selected_bands, 
                                            band_subsets= band_subets_walumi, variances = precomputed_stats['variances'])
            np.save(result_paths['mvpca_walumi'], mvpca_walumi_result)
        
        ####################ECA##############
        if not os.path.exists(result_paths['eca']):
            print(f"未检测到{n_selected_bands}个波段ECA文件，正在计算...")
            eca_result,_ = eca_bs(X_2D, distance_matrix = precomputed_stats['distance_matrix'])
            eca_result = eca_result[:n_selected_bands]
            np.save(result_paths['eca'], eca_result)
            
        if not os.path.exists(result_paths['eca_smi']):        
            print(f"未检测到{n_selected_bands}个波段ECA_SMI文件，正在计算...")
            eca_smi_result = eca_smi_bs(X_2D, band_subsets=band_subsets_smi, distance_matrix=precomputed_stats['distance_matrix'])
            np.save(result_paths['eca_smi'], eca_smi_result)
        if not os.path.exists(result_paths['eca_ocf']):
            print(f"未检测到{n_selected_bands}个波段ECA_OCF文件，正在计算...")
            eca_ocf_result = eca_smi_bs(X_2D, band_subsets=band_subsets_ocf, distance_matrix=precomputed_stats['distance_matrix'])
            np.save(result_paths['eca_ocf'], eca_ocf_result)
        if not os.path.exists(result_paths['eca_asps']):
            print(f"未检测到{n_selected_bands}个波段ECA_ASPS文件，正在计算...")
            eca_asps_result = eca_smi_bs(X_2D, band_subsets=band_subsets_asps, distance_matrix=precomputed_stats['distance_matrix'])
            np.save(result_paths['eca_asps'], eca_asps_result)
        if not os.path.exists(result_paths['eca_gpc']):
            print(f"未检测到{n_selected_bands}个波段ECA_GPC文件，正在计算...")
            eca_gpc_result = eca_smi_bs(X_2D, band_subsets=band_subsets_gpc, distance_matrix=precomputed_stats['distance_matrix'])
            np.save(result_paths['eca_gpc'], eca_gpc_result)
        if not os.path.exists(result_paths['eca_walumi']):
            print(f"未检测到{n_selected_bands}个波段ECA_WALUMI文件，正在计算...")
            eca_walumi_result = eca_smi_bs(X_2D, band_subsets=band_subets_walumi, distance_matrix=precomputed_stats['distance_matrix'])
            np.save(result_paths['eca_walumi'], eca_walumi_result)
        
        ####################MCD##########
            
        if not os.path.exists(result_paths['mcd']):
            print(f"未检测到{n_selected_bands}个波段MCD文件，正在计算...")
            mcd_result = max_covmatrix_det_bs(X_2D, num_selected_bands = n_selected_bands, covariance_matrix = precomputed_stats['covariance_matrix'])
            np.save(result_paths['mcd'], mcd_result)
            
        if not os.path.exists(result_paths['mcd_smi']):         
            print(f"未检测到{n_selected_bands}个波段MCD_SMI文件，正在计算...")
            mcd_smi_result = max_covmatrix_det_smi_bs(X_2D, num_selected_bands=n_selected_bands, band_subsets = band_subsets_smi,
                                                        covariance_matrix = precomputed_stats['covariance_matrix'])
            np.save(result_paths['mcd_smi'], mcd_smi_result)
        if  not os.path.exists(result_paths['mcd_ocf']):
            print(f"未检测到{n_selected_bands}个波段MCD_OCF文件，正在计算...")
            mcd_ocf_result = max_covmatrix_det_smi_bs(X_2D, num_selected_bands=n_selected_bands, band_subsets = band_subsets_ocf,
                                                        covariance_matrix = precomputed_stats['covariance_matrix'])
            np.save(result_paths['mcd_ocf'], mcd_ocf_result)
        if not os.path.exists(result_paths['mcd_asps']):
            print(f"未检测到{n_selected_bands}个波段MCD_ASPS文件，正在计算...")
            mcd_asps_result = max_covmatrix_det_smi_bs(X_2D, num_selected_bands=n_selected_bands, band_subsets = band_subsets_asps,
                                                        covariance_matrix = precomputed_stats['covariance_matrix'])
            np.save(result_paths['mcd_asps'], mcd_asps_result)
        if not os.path.exists(result_paths['mcd_gpc']):
            print(f"未检测到{n_selected_bands}个波段MCD_GPC文件，正在计算...")
            mcd_gpc_result = max_covmatrix_det_smi_bs(X_2D, num_selected_bands=n_selected_bands, band_subsets = band_subsets_gpc,
                                                        covariance_matrix = precomputed_stats['covariance_matrix'])
            np.save(result_paths['mcd_gpc'], mcd_gpc_result)
        if not os.path.exists(result_paths['mcd_walumi']):
            print(f"未检测到{n_selected_bands}个波段MCD_WALUMI文件，正在计算...")
            mcd_walumi_result = max_covmatrix_det_smi_bs(X_2D, num_selected_bands=n_selected_bands, band_subsets = band_subets_walumi,
                                                        covariance_matrix = precomputed_stats['covariance_matrix'])
            np.save(result_paths['mcd_walumi'], mcd_walumi_result)
            
        ################SEFREP############    
        if not os.path.exists(result_paths['sefrep']):
            print(f"未检测到{n_selected_bands}个波段SEFREP文件，正在计算...")
            sefrep_result = self_representation_bs(X_2D, num_selected_bands=n_selected_bands,correlation_matrix=precomputed_stats['correlation_matrix'])
            np.save(result_paths['sefrep'], sefrep_result)
            
        if not os.path.exists(result_paths['sefrep_smi']):
            print(f"未检测到{n_selected_bands}个波段SEFREP_SMI文件，正在计算...")
            sefrep_smi_result = self_representation_smi_bs(X_2D, num_selected_bands=n_selected_bands, band_subsets = band_subsets_smi,
                                                            correlation_matrix=precomputed_stats['correlation_matrix'])
            np.save(result_paths['sefrep_smi'], sefrep_smi_result)
        if not os.path.exists(result_paths['sefrep_ocf']):
            print(f"未检测到{n_selected_bands}个波段SEFREP_OCF文件，正在计算...")
            sefrep_ocf_result = self_representation_smi_bs(X_2D, num_selected_bands=n_selected_bands, band_subsets = band_subsets_ocf,
                                                            correlation_matrix=precomputed_stats['correlation_matrix'])
            np.save(result_paths['sefrep_ocf'], sefrep_ocf_result)
            
        if not os.path.exists(result_paths['sefrep_asps']):
            print(f"未检测到{n_selected_bands}个波段SEFREP_ASPS文件，正在计算...")
            sefrep_asps_result = self_representation_smi_bs(X_2D, num_selected_bands=n_selected_bands, band_subsets = band_subsets_asps,
                                                            correlation_matrix=precomputed_stats['correlation_matrix'])
            np.save(result_paths['sefrep_asps'], sefrep_asps_result)
            
        if not os.path.exists(result_paths['sefrep_gpc']):
            print(f"未检测到{n_selected_bands}个波段SEFREP_GPC文件，正在计算...")
            sefrep_gpc_result = self_representation_smi_bs(X_2D, num_selected_bands=n_selected_bands, band_subsets = band_subsets_gpc,
                                                            correlation_matrix=precomputed_stats['correlation_matrix'])
            np.save(result_paths['sefrep_gpc'], sefrep_gpc_result)
            
        if not os.path.exists(result_paths['sefrep_walumi']):
            print(f"未检测到{n_selected_bands}个波段SEFREP_WALUMI文件，正在计算...")
            sefrep_walumi_result = self_representation_smi_bs(X_2D, num_selected_bands=n_selected_bands, band_subsets = band_subets_walumi,
                                                              correlation_matrix=precomputed_stats['correlation_matrix'])
            np.save(result_paths['sefrep_walumi'], sefrep_walumi_result)
        
        ####################simrnk############        
        if not os.path.exists(result_paths['simrnk']):
            print(f"未检测到{n_selected_bands}个波段SIMRNK文件，正在计算...")
            simrnk_result = np.argsort(precomputed_stats['sr_eta'])[::-1][:n_selected_bands]
            np.save(result_paths['simrnk'], simrnk_result)
            
        if not os.path.exists(result_paths['simrnk_smi']):
            print(f"未检测到{n_selected_bands}个波段SIMRNK_SMI文件，正在计算...")
            simrnk_smi_result = similarity_ranking_smi_bs(band_subsets_smi, precomputed_stats['sr_eta'])
            np.save(result_paths['simrnk_smi'], simrnk_smi_result)
        if not os.path.exists(result_paths['simrnk_ocf']):
            print(f"未检测到{n_selected_bands}个波段SIMRNK_OCF文件，正在计算...")
            simrnk_ocf_result = similarity_ranking_smi_bs(band_subsets_ocf, precomputed_stats['sr_eta'])
            np.save(result_paths['simrnk_ocf'], simrnk_ocf_result)
        if not os.path.exists(result_paths['simrnk_asps']):
            print(f"未检测到{n_selected_bands}个波段SIMRNK_ASPS文件，正在计算...")
            simrnk_asps_result = similarity_ranking_smi_bs(band_subsets_asps, precomputed_stats['sr_eta'])
            np.save(result_paths['simrnk_asps'], simrnk_asps_result)
        if not os.path.exists(result_paths['simrnk_gpc']):
            print(f"未检测到{n_selected_bands}个波段SIMRNK_GPC文件，正在计算...")
            simrnk_gpc_result = similarity_ranking_smi_bs(band_subsets_gpc, precomputed_stats['sr_eta'])
            np.save(result_paths['simrnk_gpc'], simrnk_gpc_result)
        if not os.path.exists(result_paths['simrnk_walumi']):
            print(f"未检测到{n_selected_bands}个波段SIMRNK_WALUMI文件，正在计算...")
            simrnk_walumi_result = similarity_ranking_smi_bs(band_subets_walumi, precomputed_stats['sr_eta'])
            np.save(result_paths['simrnk_walumi'], simrnk_walumi_result)
        
        
        if not os.path.exists(result_paths['unisam']):
            print(f"未检测到{n_selected_bands}个波段UNISAM文件，正在计算...")
            unisam_result = uniform_sample_bs(total_band_number, num_selected_bands=n_selected_bands) 
            np.save(result_paths['unisam'], unisam_result)
                  
        if not os.path.exists(result_paths['uni_random']):
            print(f"未检测到{n_selected_bands}个波段UNI_RANDOM文件，正在计算...")
            uni_random_result = uniform_random_bs(band_subsets_smi)
            np.save(result_paths['uni_random'], uni_random_result)
    

def process_single_dataset(dataset_name):
    """
    处理单个数据集的完整流程（核心新增函数）
    """
    # 1. 路径配置（基于config的统一路径管理）
    dataset_dir = os.path.join(BASE_DATA_DIR, dataset_name)
    result_dir = os.path.join(dataset_dir, "result")
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    is_run_smi = False
    is_run_bs = False
    is_run_eva = False
    # 2. 加载数据
    
    hsi_data_loader = HSIDataLoader(dataset_name)
    X, y_true, X_3D = hsi_data_loader.load()
    print(f"{dataset_name} 数据集 - 数据大小: {X_3D.shape}")
    
    
    n_samples, n_features = X.shape
    
    # 3. 数据标准化
    X_normlized = StandardScaler().fit_transform(X.copy())
    
    # 4. 划分训练测试集（使用config中的参数）
    X_train, X_test, y_train, y_test = train_test_split(
        X_normlized, y_true, 
        test_size=SVM_PARAMS["test_size"],
        random_state=SVM_PARAMS["random_state"]
    )
    print(f'{dataset_name} 数据集 - 训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}')
           
    try:
        precomputed_stats = compute_and_cache_statistics(
            X, X_normlized, X_3D, 
            dataset_dir=dataset_dir,
            dataset_prefix = dataset_name,
            n_jobs=SMI_PARAMS["n_jobs"]
            )
    except Exception as e:
        print(f"❌ 计算{dataset_name}数据集统计量失败: {str(e)}")
        return
    
    
    # 5. 计算并缓存统计量
    if is_run_smi:      
        
        # 6. 输出R²统计结果
        stats = {
            "mean_r2": np.mean(precomputed_stats['r2_array']),
            "median_r2": np.median(precomputed_stats['r2_array']),
            "std_r2": np.std(precomputed_stats['r2_array']),
            "min_r2": np.min(precomputed_stats['r2_array']),
            "max_r2": np.max(precomputed_stats['r2_array'])
        }
        print("="*50)
        print(f"{dataset_name} 相邻波段线性拟合R²统计结果")
        print(f"平均R²: {stats['mean_r2']:.4f}")
        
        
        stats_shuffled  = {
            "mean_r2": np.mean(precomputed_stats['r2_array_shuffled']),
            "median_r2": np.median(precomputed_stats['r2_array_shuffled']),
            "std_r2": np.std(precomputed_stats['r2_array_shuffled']),
            "min_r2": np.min(precomputed_stats['r2_array_shuffled']),
            "max_r2": np.max(precomputed_stats['r2_array_shuffled'])
        }
        print("="*50)
        print(f"{dataset_name} shuffled 相邻波段线性拟合R²统计结果")
        print(f"平均R²: {stats_shuffled['mean_r2']:.4f}")
        print("="*50)
    
    # 7. 计算SMI（使用config中的参数）
        try:
            smi, acc, cis = calc_smi(
                X,
                n_jobs=SMI_PARAMS["n_jobs"],
                n_samples=SMI_PARAMS["n_samples"],
                precomputed_cis=precomputed_stats['cis'],
                precomputed_mic_adj=precomputed_stats['mic_adj'],
                precomputed_mic_nonadj=precomputed_stats['mic_unadj']
            )
            print(f'{dataset_name} 数据集 - SMI: {smi:.4f}, ACC: {acc:.4f}, CIS: {cis:.4f}')
            smi, acc, cis = calc_smi(
                X,
                n_jobs=SMI_PARAMS["n_jobs"],
                n_samples=SMI_PARAMS["n_samples"],
                precomputed_cis=precomputed_stats['cis_shuffled'],
                precomputed_mic_adj=precomputed_stats['mic_adj_shuffled'],
                precomputed_mic_nonadj=precomputed_stats['mic_unadj_shuffled']
            )
            print(f'{dataset_name} 数据集 shuffled - SMI: {smi:.4f}, ACC: {acc:.4f}, CIS: {cis:.4f}')
        except Exception as e:
            print(f"❌ 计算{dataset_name}数据集SMI失败: {str(e)}")
            return
    
    # 8. 波段选择配置
    n_selected_bands_list = [5, 10, 15, 20, 25, 30]
    band_selection_methods = [
        'mvpca', 'mvpca_smi','mvpca_ocf','mvpca_asps','mvpca_gpc','mvpca_walumi',
        'eca', 'eca_smi', 'eca_ocf','eca_asps','eca_gpc','eca_walumi',
        'mcd', 'mcd_smi','mcd_ocf','mcd_asps','mcd_gpc','mcd_walumi',
        'sefrep', 'sefrep_smi','sefrep_ocf','sefrep_asps','sefrep_gpc','sefrep_walumi',
        'simrnk', 'simrnk_smi','simrnk_ocf','simrnk_asps','simrnk_gpc','simrnk_walumi',
        'unisam','uni_random'
    ]
    baseline_methods = [
        'mvpca', 
        'eca', 
        'mcd', 
        'sefrep', 
        'simrnk'
        ]
    
    # 9. 运行波段选择
    if is_run_bs:
        print(f"{dataset_name} 数据集 - 正在运行波段选择...")
        
        run_band_selection(
            result_dir = result_dir,
            dataset_name = dataset_name,
            total_band_number = X.shape[1],
            n_selected_bands_list = n_selected_bands_list,
            band_selection_methods = band_selection_methods,
            X_2D = X,
            precomputed_stats = precomputed_stats
        )
    
    # 10. 过滤非零标签（去除背景）
    non_zero_mask = y_true != 0    
    X_normlized_non_zero = X_normlized[non_zero_mask, :]
    y_true_non_zero = y_true[non_zero_mask]
        
    

    # 11. 评估波段选择结果
    if is_run_eva:
        try:
            # 初始化SVM模型（使用config中的参数）
            model = LinearSVC(
                C=SVM_PARAMS["C"],
                random_state=SVM_PARAMS["random_state"],
                max_iter=SVM_PARAMS["max_iter"]
            )
            cls_res = evaluate_band_selection_results(
                result_dir=result_dir,
                dataset_name=dataset_name,
                n_selected_bands_list=n_selected_bands_list,
                X_normlized=X_normlized_non_zero,
                y_true = y_true_non_zero,
                band_selection_methods = band_selection_methods,
                baseline_methods=baseline_methods,
                verbose = True,
                )
            print(f"\n{dataset_name} 数据集分类评估结果:")
            print(cls_res)
        except Exception as e:
            print(f"❌ 评估{dataset_name}数据集波段选择结果失败: {str(e)}")
            return
        
        # 12. 保存汇总结果到全局输出目录
    

if __name__ == "__main__":
    
    # 配置要处理的数据集（灵活控制：None处理所有，列表指定部分）
    # TARGET_DATASETS = ["indianpines"]  # 测试单个数据集
    TARGET_DATASETS = None  # 处理所有数据集
    
    # 获取待处理数据集列表
    if TARGET_DATASETS is None:
        datasets_to_process = list(DATASET_PATHS.keys())
    else:
        # 验证数据集配置有效性
        datasets_to_process = [ds for ds in TARGET_DATASETS if ds in DATASET_PATHS]
        missing_datasets = [ds for ds in TARGET_DATASETS if ds not in DATASET_PATHS]
        if missing_datasets:
            print(f"⚠️ 以下数据集在config中未配置: {missing_datasets}")
    
    # 批量处理入口
    print(f"\n🚀 开始批量处理数据集: {datasets_to_process}")
    print(f"📝 结果输出根目录: {OUTPUT_DIR}")
    
    for ds_name in datasets_to_process:
        print("\n" + "="*60)
        print(f"处理数据集: {ds_name.upper()}")
        print("="*60)
        process_single_dataset(ds_name)
    
    print("\n🎉 所有数据集处理完成！")