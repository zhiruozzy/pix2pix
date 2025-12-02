import torch
import torch.utils.data as data
import nibabel as nib
import numpy as np
import os
import torch.nn.functional as F
from data.base_dataset import BaseDataset


class NiftiSliceDataset(BaseDataset):
    """
    修正版：优化了 MRI 和 CT 的归一化策略，专门针对骨骼生成任务。
    """
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(crop_size=320, load_size=320) 
        parser.add_argument(
            '--resample_mode', 
            type=str, 
            default='interpolate', 
            choices=['interpolate', 'crop'], 
            help='How to resize slices.'
        )
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        
        self.mr_dir = os.path.join(opt.dataroot, 'train_A')
        self.ct_dir = os.path.join(opt.dataroot, 'train_B')
        
        self.mr_files = sorted([os.path.join(self.mr_dir, f) for f in os.listdir(self.mr_dir) if f.endswith(('.nii', '.nii.gz'))])
        self.ct_files = sorted([os.path.join(self.ct_dir, f) for f in os.listdir(self.ct_dir) if f.endswith(('.nii', '.nii.gz'))])
        
        assert len(self.mr_files) == len(self.ct_files), "MR and CT file counts do not match!"
        
        self.input_size = 320
        self.resample_mode = opt.resample_mode

        # --- 修改点 1: 调整 CT 窗宽窗位 (针对脊柱骨骼优化) ---
        # 舍弃 > 1500 的高亮金属伪影，舍弃 < -1000 的空气背景
        # 这样可以让骨骼 (300~1000) 在 [-1, 1] 区间内占据更大的动态范围
        self.ct_min_val = -1000.0
        self.ct_max_val = 1500.0 

    def normalize_ct(self, data):
        """CT 使用固定阈值截断 (Clip) + 线性映射"""
        data = np.clip(data, self.ct_min_val, self.ct_max_val)
        # 映射到 [0, 1]
        data = (data - self.ct_min_val) / (self.ct_max_val - self.ct_min_val)
        # 映射到 [-1, 1]
        data = data * 2.0 - 1.0
        return data

    def normalize_mri(self, data):
        """
        --- 修改点 2: MRI 使用鲁棒归一化 ---
        MRI 绝对值无意义，使用 99% 分位数作为 Max，防止个别噪点拉低整体亮度
        """
        if data.size == 0: return data
        
        min_val = np.percentile(data, 1)  # 1% 分位数作为底 (去底噪)
        max_val = np.percentile(data, 99) # 99% 分位数作为顶 (去极值)
        
        # 防止除以 0
        if max_val - min_val < 1e-6:
            return np.zeros_like(data)
            
        data = np.clip(data, min_val, max_val)
        data = (data - min_val) / (max_val - min_val)
        data = data * 2.0 - 1.0
        return data

    def __getitem__(self, index):
        # 1. 加载 3D NIfTI 图像
        # 警告：这里有一个性能瓶颈。每次读取整个 3D 卷只为了取 1 张切片非常慢。
        # 但为了不改变你的架构，我们先保持这样。
        mr_obj = nib.load(self.mr_files[index])
        ct_obj = nib.load(self.ct_files[index])
        
        # 确保方向一致 (RAS) - 这是一个好习惯，防止有的图是倒着的
        mr_obj = nib.as_closest_canonical(mr_obj)
        ct_obj = nib.as_closest_canonical(ct_obj)
        
        mr_vol = mr_obj.get_fdata().astype(np.float32)
        ct_vol = ct_obj.get_fdata().astype(np.float32)

        # 2. 随机选择切片 (Z轴)
        D1, D2, D3 = mr_vol.shape 
        # 确保 CT 和 MRI 深度一致，如果不一致取最小值 (防止报错)
        D3_ct = ct_vol.shape[2]
        slice_idx = np.random.randint(0, min(D3, D3_ct))
        
        mr_slice = mr_vol[:, :, slice_idx] 
        ct_slice = ct_vol[:, :, slice_idx] 
        
        # 3. 归一化 (使用新的逻辑)
        mr_slice_norm = self.normalize_mri(mr_slice)
        ct_slice_norm = self.normalize_ct(ct_slice)
        
        # 4. 转换为 Tensor (C x H x W)
        mr_tensor = torch.from_numpy(mr_slice_norm).unsqueeze(0) 
        ct_tensor = torch.from_numpy(ct_slice_norm).unsqueeze(0)
        
        target_size = self.input_size
        _, H_orig, W_orig = mr_tensor.shape

        # 5. Resample logic (保持你原来的逻辑)
        if self.resample_mode == 'crop':
            if H_orig >= target_size and W_orig >= target_size:
                x_start = (W_orig - target_size) // 2
                y_start = (H_orig - target_size) // 2
                mr_tensor_out = mr_tensor[:, y_start:y_start + target_size, x_start:x_start + target_size]
                ct_tensor_out = ct_tensor[:, y_start:y_start + target_size, x_start:x_start + target_size]
            elif H_orig == target_size and W_orig == target_size:
                mr_tensor_out = mr_tensor
                ct_tensor_out = ct_tensor
            else:
                mr_tensor_out = F.interpolate(mr_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
                ct_tensor_out = F.interpolate(ct_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)

        elif self.resample_mode == 'interpolate':
            mr_tensor_out = F.interpolate(mr_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
            ct_tensor_out = F.interpolate(ct_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)

        return {
            'A': mr_tensor_out,    
            'B': ct_tensor_out,    
            'A_paths': self.mr_files[index], 
            'B_paths': self.ct_files[index]
        }

    def __len__(self):
        return len(self.mr_files)
