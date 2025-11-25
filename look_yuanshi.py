import nibabel as nib
import numpy as np
import os
from glob import glob

# 1. 设置要检查的目录路径 (请确保路径正确)
# 假设您在 pytorch-CycleGAN-and-pix2pix/ 目录下运行
data_dir = 'datasets/CT2MR_SPLIT/train_A'

# 2. 查找所有 NIfTI 文件 (.nii 和 .nii.gz)
nifti_files = glob(os.path.join(data_dir, '*.nii')) + glob(os.path.join(data_dir, '*.nii.gz'))

print(f"--- 检查目录: {data_dir} 中的 {len(nifti_files)} 个文件 ---")

# 3. 循环遍历文件并检查形状
for file_path in nifti_files:
    try:
        # 加载 NIfTI 文件
        nifti_img = nib.load(file_path)
        
        # 获取 NumPy 数组数据
        mr_vol = nifti_img.get_fdata()
        
        # 获取形状
        shape = mr_vol.shape
        
        # 打印关键信息
        print("-" * 40)
        print(f"文件名: {os.path.basename(file_path)}")
        print(f"原始形状 (Dim0, Dim1, Dim2): {shape}")
        
        if len(shape) < 3:
            print("警告: 形状维度小于 3，可能不是 3D 卷。")
            continue
            
        # 假设图像 W/H 维度是 512x512，深度 D 维度是较小的数 (例如 80, 100等)
        
        # 找出切片轴（通常是维度中最小的那个数）
        # 如果您的 W/H 都是 512，且深度是 80，那么 80 对应的索引就是切片轴
        # 让我们找出与 512 不同的那个维度
        wh_dims = [d for d in shape if d != 512]
        
        if len(wh_dims) == 1:
            slice_dim = wh_dims[0]
            slice_dim_index = shape.index(slice_dim)
        else:
            # 如果 W/H 不是 512，或者有多个不同维度，则默认假设 D 在最后
            slice_dim = shape[-1]
            slice_dim_index = 2

        print(f"推断切片数 (D): {slice_dim}")
        print(f"推断切片轴索引: {slice_dim_index}")
                
    except Exception as e:
        print(f"--- 错误处理文件 {os.path.basename(file_path)}: {e} ---")
        
print("--- 检查完成 ---")
