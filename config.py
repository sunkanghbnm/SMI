# config.py
import os

# ===================== 路径配置 =====================
BASE_DATA_DIR = "/mnt/f/pytorch/HSI/datasets"
DATASET_PATHS = {
    "indian_pines": {
        "hsi": os.path.join(BASE_DATA_DIR, "indian_pines/Indian_pines.mat"),
        "gt": os.path.join(BASE_DATA_DIR, "indian_pines/Indian_pines_gt.mat")
    },
    "pavia": {
        "hsi": os.path.join(BASE_DATA_DIR, "pavia/Pavia.mat"),
        "gt": os.path.join(BASE_DATA_DIR, "pavia/Pavia_gt.mat")
    },
    "botswana": {
        "hsi": os.path.join(BASE_DATA_DIR, "botswana/Botswana.mat"),
        "gt": os.path.join(BASE_DATA_DIR, "botswana/Botswana_gt.mat")
    },
    "KSC": {
        "hsi": os.path.join(BASE_DATA_DIR, "KSC/KSC.mat"),
        "gt": os.path.join(BASE_DATA_DIR, "KSC/KSC_gt.mat")
    },
    "salinas": {
        "hsi": os.path.join(BASE_DATA_DIR, "salinas/Salinas.mat"),
        "gt": os.path.join(BASE_DATA_DIR, "salinas/Salinas_gt.mat")
    }    
}

# ===================== 实验超参数 =====================
# SMSI计算参数
SMSI_PARAMS = {
    "n_samples": 1000,          # 像素抽样数
    "n_jobs": -1,               # 并行核心数
    "random_state": 42,         # 随机种子
    "nonadj_max_pairs": 1000,   # 非相邻波段对最大抽样数
    "max_k_mic": 30             # MIC间隔分析的最大k
}

# 波段选择参数
BAND_SELECTION_PARAMS = {
    "top_k": 30,                # 选择最优k个波段
    "mrmr_n_features": 30,      # mRMR选择波段数
    "f_score_k": 30,            # F-score选择波段数
    "mi_k": 30                  # 互信息选择波段数
}

# 波段分组参数
BAND_GROUP_PARAMS = {
    "group_strategy": "smsi_threshold",  # 分组策略：smsi_threshold/cluster/interval
    "smsi_threshold": 0.5,               # SMSI阈值分组
    "interval_step": 5,                  # 按间隔分组的步长
    "n_clusters": 10                     # 聚类分组的簇数
}

# SVM分类参数
SVM_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "max_iter": 1000,
    "test_size": 0.2,          # 测试集比例
    "random_state": 42
}

# ===================== 实验输出配置 =====================
OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)