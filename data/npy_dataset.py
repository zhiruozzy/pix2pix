import os
import numpy as np
import torch
from data.base_dataset import BaseDataset
import random

class NpyDataset(BaseDataset):
    """
    【ResNet 专用版】
    1. 强制归一化到 [-1, 1] (解决 NaN，保留 float32 高精度细节)
    2. 移除所有裁剪代码 (原样输出 320x320，适配 ResNet)
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        # 路径处理
        self.dir_A = os.path.join(opt.dataroot, opt.phase + '_A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + '_B')
        
        # 兼容性处理
        if not os.path.exists(self.dir_A):
            self.dir_A = os.path.join(opt.dataroot, 'train_A')
            self.dir_B = os.path.join(opt.dataroot, 'train_B')

        self.A_paths = sorted([os.path.join(self.dir_A, f) for f in os.listdir(self.dir_A) if f.endswith('.npy')])
        self.B_paths = sorted([os.path.join(self.dir_B, f) for f in os.listdir(self.dir_B) if f.endswith('.npy')])
        
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        print(f"Dataset Loaded: {self.A_size} pairs (No Crop mode)")

    def _normalize(self, data):
        """
        核心函数：将任意范围数据线性缩放到 [-1, 1]
        全程使用 float32，绝对不会丢失细节！
        """
        data = data.astype(np.float32)
        min_val = data.min()
        max_val = data.max()
        
        # 只有当数据范围不对时才处理
        if min_val < -1.1 or max_val > 1.1:
            if max_val - min_val > 1e-5:
                # 线性映射：(x - min) / (max - min) -> [0, 1]
                # 再映射：x * 2 - 1 -> [-1, 1]
                data = (data - min_val) / (max_val - min_val)
                data = (data - 0.5) * 2.0
            else:
                data = np.zeros_like(data)
        return data

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        # Pix2Pix 还是建议成对训练
        index_B = index % self.B_size 
        B_path = self.B_paths[index_B]

        # 1. 读取
        A_np = np.load(A_path)
        B_np = np.load(B_path)
        
        # 2. 【关键】归一化 (保留细节，防止报错)
        A_np = self._normalize(A_np)
        B_np = self._normalize(B_np)

        # 3. 转 Tensor
        A_tensor = torch.from_numpy(A_np).unsqueeze(0).float()
        B_tensor = torch.from_numpy(B_np).unsqueeze(0).float()

        # 4. 【已移除裁剪代码】
        # 直接返回原尺寸 (320x320)，ResNet 能吃得消

        return {'A': A_tensor, 'B': B_tensor, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)
