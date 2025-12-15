import os
import numpy as np
import torch
from data.base_dataset import BaseDataset
import random

class NumpyDataset(BaseDataset):
    """
    读取预处理好的 .npy 文件 (无损精度)
    """
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, 'train_A')
        self.dir_B = os.path.join(opt.dataroot, 'train_B')
        
        self.A_paths = sorted([os.path.join(self.dir_A, f) for f in os.listdir(self.dir_A) if f.endswith('.npy')])
        self.B_paths = sorted([os.path.join(self.dir_B, f) for f in os.listdir(self.dir_B) if f.endswith('.npy')])
        
        self.input_size = opt.load_size # 比如 340
        self.crop_size = opt.crop_size  # 比如 320

    def __getitem__(self, index):
        # 1. 极速读取
        A_np = np.load(self.A_paths[index]) # (H, W) float32 [-1, 1]
        B_np = np.load(self.B_paths[index]) 
        
        # 2. 转 Tensor
        A = torch.from_numpy(A_np).unsqueeze(0) # (1, H, W)
        B = torch.from_numpy(B_np).unsqueeze(0)
        
        # 3. 简单的 Resize (如果需要) & Random Crop
        # 这里假设你的 npy 已经是 320x320，如果需要数据增强：
        if self.opt.preprocess == 'resize_and_crop':
            # 简单的随机裁剪实现
            h, w = A.shape[1], A.shape[2]
            if h > self.crop_size and w > self.crop_size:
                x = random.randint(0, w - self.crop_size)
                y = random.randint(0, h - self.crop_size)
                A = A[:, y:y+self.crop_size, x:x+self.crop_size]
                B = B[:, y:y+self.crop_size, x:x+self.crop_size]

        return {'A': A, 'B': B, 'A_paths': self.A_paths[index], 'B_paths': self.B_paths[index]}

    def __len__(self):
        return len(self.A_paths)

