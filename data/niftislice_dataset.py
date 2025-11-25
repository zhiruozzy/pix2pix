import torch
import torch.utils.data as data
import nibabel as nib
import numpy as np
import os
import torch.nn.functional as F
from data.base_dataset import BaseDataset


class NiftiSliceDataset(BaseDataset):
    """
    用于训练阶段的数据集：从 NIfTI 卷中随机抽取一个 2D 切片，
    并根据 resample_mode 参数进行中心裁剪或双线性插值到 320x320。
    """
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """添加 dataset-specific 选项并设置默认值"""
        # 确保尺寸默认为 320
        parser.set_defaults(crop_size=320, load_size=320) 
        
        # *** 命令行选项 ***
        parser.add_argument(
            '--resample_mode', 
            type=str, 
            default='interpolate', 
            choices=['interpolate', 'crop'], 
            help='How to resize slices to 320x320: "interpolate" (bilinear) or "crop" (center crop).'
        )
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        
        self.mr_dir = os.path.join(opt.dataroot, 'train_A')  # 输入 (MR)
        self.ct_dir = os.path.join(opt.dataroot, 'train_B')  # 目标 (CT)
        
        # 确保文件列表已正确加载
        self.mr_files = sorted([os.path.join(self.mr_dir, f) for f in os.listdir(self.mr_dir) if f.endswith(('.nii', '.nii.gz'))])
        self.ct_files = sorted([os.path.join(self.ct_dir, f) for f in os.listdir(self.ct_dir) if f.endswith(('.nii', '.nii.gz'))])
        
        assert len(self.mr_files) == len(self.ct_files), "MR and CT file counts do not match!"
        
        self.input_size = 320 # 目标尺寸固定为 320
        self.resample_mode = opt.resample_mode # *** 存储选择的模式 ***

        # 定义归一化参数 (必须与训练时使用的参数保持一致)
        self.mr_min, self.mr_max = 0, 1000  
        self.ct_min, self.ct_max = -1024, 3072

    def normalize(self, data, min_val, max_val):
        """将数据归一化到 [-1, 1] 范围"""
        data = data.astype(np.float32)
        data = (data - min_val) / (max_val - min_val)
        data = data * 2.0 - 1.0
        return np.clip(data, -1.0, 1.0)

    def __getitem__(self, index):
        # 1. 加载 3D NIfTI 图像
        mr_vol = nib.load(self.mr_files[index]).get_fdata()
        ct_vol = nib.load(self.ct_files[index]).get_fdata()

        # 2. 随机选择切片 (训练阶段必须随机抽样)
        # 假设 D3 是切片轴 (深度/Z轴)
        D1, D2, D3 = mr_vol.shape 
        slice_idx = np.random.randint(0, D3) 
        
        mr_slice = mr_vol[:, :, slice_idx] 
        ct_slice = ct_vol[:, :, slice_idx] 
        
        # 3. 归一化
        mr_slice_norm = self.normalize(mr_slice, self.mr_min, self.mr_max)
        ct_slice_norm = self.normalize(ct_slice, self.ct_min, self.ct_max)
        
        # 4. 转换为 Tensor (C x H x W)
        mr_tensor = torch.from_numpy(mr_slice_norm).unsqueeze(0) # 1 x H_orig x W_orig
        ct_tensor = torch.from_numpy(ct_slice_norm).unsqueeze(0) # 1 x H_orig x W_orig
        
        target_size = self.input_size # 320
        _, H_orig, W_orig = mr_tensor.shape


        # 5. 【根据 resample_mode 执行尺寸调整】
        
        if self.resample_mode == 'crop':
            # === 中心裁剪逻辑 ===
            if H_orig >= target_size and W_orig >= target_size:
                # 仅在原始尺寸大于目标尺寸时进行裁剪
                x_start = (W_orig - target_size) // 2
                y_start = (H_orig - target_size) // 2
                
                mr_tensor_out = mr_tensor[:, y_start:y_start + target_size, x_start:x_start + target_size]
                ct_tensor_out = ct_tensor[:, y_start:y_start + target_size, x_start:x_start + target_size]
            elif H_orig == target_size and W_orig == target_size:
                # 尺寸相等，无需操作
                mr_tensor_out = mr_tensor
                ct_tensor_out = ct_tensor
            else:
                # 原始尺寸小于 320x320 (例如 300x300)。裁剪无法实现，使用插值作为回退。
                print(f"Warning: Slice size {H_orig}x{W_orig} is smaller than 320x320 for cropping. Falling back to interpolation.")
                mr_tensor_out = F.interpolate(mr_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
                ct_tensor_out = F.interpolate(ct_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)


        elif self.resample_mode == 'interpolate':
            # === 双线性插值逻辑 (默认) ===
            mr_tensor_out = F.interpolate(mr_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
            ct_tensor_out = F.interpolate(ct_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)

        # 6. 返回数据
        return {
            'A': mr_tensor_out,    # 输入 MR (320x320)
            'B': ct_tensor_out,    # 目标 CT (320x320)
            'A_paths': self.mr_files[index], 
            'B_paths': self.ct_files[index]
        }

    def __len__(self):
        # 训练集的长度是 NIfTI 文件的数量
        return len(self.mr_files)
