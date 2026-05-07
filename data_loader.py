# data_loader.py
import scipy.io
import numpy as np
from config import DATASET_PATHS

class HSIDataLoader:
    """高光谱数据加载器，支持多数据集+可选预处理"""
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.hsi_path = DATASET_PATHS[self.dataset_name]["hsi"]
        self.gt_path = DATASET_PATHS[self.dataset_name]["gt"]
    
    def load(self):
        """加载数据并返回处理后的2D数据、标签、3D原始数据"""
        if self.dataset_name == "indian_pines":
            return self._load_indianpines()
        elif self.dataset_name == "pavia":
            return self._load_pavia()
        elif self.dataset_name == "botswana":
            return self._load_botswana()
        elif self.dataset_name == "KSC":
            return self._load_ksc()
        elif self.dataset_name == "salinas":
            return self._load_salinas()
        else:
            raise ValueError(f"不支持的数据集：{self.dataset_name}")

    
    def _load_indianpines(self):
        hsi_data = scipy.io.loadmat(self.hsi_path)
        X_3D = hsi_data['indian_pines_corrected'].astype(np.float64)
        X_2D = X_3D.reshape(-1, X_3D.shape[-1])
        
        gt_data = scipy.io.loadmat(self.gt_path)
        y_true = gt_data['indian_pines_gt'].reshape(-1).T
        return X_2D, y_true, X_3D
    
    def _load_pavia(self):
        hsi_data = scipy.io.loadmat(self.hsi_path)
        X_3D = hsi_data['pavia'].astype(np.float64)
        X_2D = X_3D.reshape(-1, X_3D.shape[-1])
        
        gt_data = scipy.io.loadmat(self.gt_path)
        y_true = gt_data['pavia_gt'].reshape(-1).T
        return X_2D, y_true, X_3D
    
    def _load_botswana(self):
        hsi_data = scipy.io.loadmat(self.hsi_path)
        X_3D = hsi_data['Botswana'].astype(np.float64)
        X_2D = X_3D.reshape(-1, X_3D.shape[-1])
        
        gt_data = scipy.io.loadmat(self.gt_path)
        y_true = gt_data['Botswana_gt'].reshape(-1).T.astype(np.int64)
        return X_2D, y_true, X_3D
    
    def _load_ksc(self):
        hsi_data = scipy.io.loadmat(self.hsi_path)
        X_3D = hsi_data['KSC'].astype(np.float64)
        X_2D = X_3D.reshape(-1, X_3D.shape[-1])
        
        gt_data = scipy.io.loadmat(self.gt_path)
        y_true = gt_data['KSC_gt'].reshape(-1).T
        return X_2D, y_true, X_3D
    
    def _load_salinas(self):
        hsi_data = scipy.io.loadmat(self.hsi_path)
        X_3D = hsi_data['salinas_corrected'].astype(np.float64)
        X_2D = X_3D.reshape(-1, X_3D.shape[-1])
        
        gt_data = scipy.io.loadmat(self.gt_path)
        y_true = gt_data['salinas_gt'].reshape(-1).T
        return X_2D, y_true, X_3D

# 便捷函数：快速加载数据集
def load_hsi_data(dataset_name, preprocess=True):
    loader = HSIDataLoader(dataset_name, preprocess)
    return loader.load()