import torch
import torch.utils.data as data
import nibabel as nib
import numpy as np
import os
import torch.nn.functional as F
from data.base_dataset import BaseDataset


class NiftiTestDataset(BaseDataset):
    """
    专门用于测试阶段的数据集：按顺序加载 NIfTI 卷，并返回所有切片、
    原始尺寸和仿射矩阵，以便进行 3D 重建。
    """
    
    # 必须实现这个方法，因为 BaseOptions.py 需要它来添加数据集特有的参数
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """添加数据集特有的选项。"""
        # TestOptions 中一般不需要额外参数，但为了兼容框架必须实现
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        
        # 假设测试数据位于 opt.dataroot/test_A
        self.mr_dir = os.path.join(opt.dataroot, 'test_A') 
        self.mr_files = sorted([os.path.join(self.mr_dir, f) for f in os.listdir(self.mr_dir) if f.endswith(('.nii', '.nii.gz'))])
        
        # 目标输入尺寸 (256x256) 和原始输出尺寸 (512x512)
        # 这里的 opt.crop_size 应该是 256，与训练时一致
        self.input_size = opt.crop_size  
        self.output_size = 512 # 原始图像尺寸，用于反归一化和重建

        # 定义归一化参数 (!!! 必须与训练时 nifti_slice_dataset.py 中的参数保持一致 !!!)
        self.mr_min, self.mr_max = 0, 1000  
        self.ct_min, self.ct_max = -1024, 3072

    def normalize(self, data, min_val, max_val):
        """将数据归一化到 [-1, 1] 范围"""
        data = data.astype(np.float32)
        # 确保分母不是零，尽管在 HU 值范围中不太可能
        range_val = max_val - min_val
        if range_val == 0:
             return np.zeros_like(data)
             
        data = (data - min_val) / range_val
        data = data * 2.0 - 1.0 # 转换到 [-1, 1]
        return np.clip(data, -1.0, 1.0)
    
    # 注意：测试阶段的 __getitem__ 一次返回一个 NIfTI 卷的所有切片
    def __getitem__(self, index):
        path = self.mr_files[index]
        
        # 1. 加载 NIfTI 文件和元数据
        nifti_img = nib.load(path)
        mr_vol = nifti_img.get_fdata()
        
        # 提取关键元数据 (用于最终重建)
        affine = nifti_img.affine
        vol_shape = mr_vol.shape # (D1, D2, D3)
        
        # 2. 确定切片轴和数量 (假设 D3 是深度轴)
        D1, D2, D3 = vol_shape
        num_slices = D3

        input_slices = []
        
        # 3. 系统遍历所有切片 (Test Dataset 不进行随机抽样)
        for i in range(num_slices):
            # 提取 512 x 512 切片
            mr_slice = mr_vol[:, :, i] 
            
            # 4. 归一化 (512x512)
            mr_slice_norm = self.normalize(mr_slice, self.mr_min, self.mr_max)
            
            # 5. 转换为 Tensor (C x H x W)
            mr_tensor = torch.from_numpy(mr_slice_norm).unsqueeze(0) # 1 x 512 x 512
            
            # 6. 【下采样到 256x256】 (必须与训练时一致)
            # F.interpolate 期望 B x C x H x W
            mr_tensor_256 = F.interpolate(mr_tensor.unsqueeze(0), 
                                          size=self.input_size, 
                                          mode='bilinear', 
                                          align_corners=False).squeeze(0)
            
            input_slices.append(mr_tensor_256.float()) # 确保是 float 类型
        
        # 7. 返回整个卷的切片列表和元数据
        return {
            'A_slices': input_slices,  # N个 1 x 256 x 256 Tensor
            'A_paths': path,
            'affine': affine,
            'vol_shape': vol_shape
        }

    def __len__(self):
        # 测试集的长度是 NIfTI 文件的数量 (即病人的数量)
        return len(self.mr_files)

