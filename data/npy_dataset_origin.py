import os
import numpy as np
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
import random

class NpyDataset(BaseDataset):
    """
    这是一个极速读取 .npy 切片的 Dataset。
    它保留了 float32 精度，且不需要反复读取 3D NIfTI 文件。
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        # 可以在这里设置默认参数
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + '_A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + '_B')

        # 获取所有 .npy 文件
        self.A_paths = sorted([os.path.join(self.dir_A, f) for f in os.listdir(self.dir_A) if f.endswith('.npy')])
        self.B_paths = sorted([os.path.join(self.dir_B, f) for f in os.listdir(self.dir_B) if f.endswith('.npy')])
        
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        # 打印一下数量，让你放心
        print(f"Dataset Loaded: MRI(A)={self.A_size}, CT(B)={self.B_size}")

    def __getitem__(self, index):
        # A 域数据 (MRI)
        A_path = self.A_paths[index % self.A_size]
        
        # B 域数据 (CT) - 随机采样 (Unpaired)
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        # 读取 .npy (极速)
        A_np = np.load(A_path) # [H, W] float32, range [-1, 1]
        B_np = np.load(B_path) # [H, W] float32, range [-1, 1]

        # 转 Tensor [1, H, W]
        A_tensor = torch.from_numpy(A_np).unsqueeze(0).float()
        B_tensor = torch.from_numpy(B_np).unsqueeze(0).float()
        
        # 如果需要 Resize (比如你的切片不是 320x320)
        # 这里你可以加 torch.nn.functional.interpolate
        # 但既然你切片前已经是 NIfTI，通常尺寸是固定的，这里假设已经是 320x320

        return {'A': A_tensor, 'B': B_tensor, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)
