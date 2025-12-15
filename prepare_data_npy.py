import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 这里填你存放原始 NIfTI 文件的文件夹路径
#    结构应该是:
#    final_data/
#       train_A/ (放 MR 的 .nii)
#       train_B/ (放 CT 的 .nii)
RAW_DATAROOT = '/home/myp/Documents/DICOM/data/final_data' 

# 2. 这里填你想把切好的 .npy 存到哪里 (脚本会自动创建)
SAVE_DATAROOT = './datasets/ct_mr_npy' 

# ================= 核心处理逻辑 =================

def normalize_ct(data):
    """
    CT 归一化策略 (针对骨骼优化):
    1. 截断: 保留 [-1000, 1500] 范围。
       -1000 是空气，1500 是高亮骨骼/金属，再高就是伪影了。
    2. 映射: 线性映射到 [-1, 1] 区间，这是 Pix2Pix 的标准输入范围。
    """
    min_val, max_val = -1000.0, 1500.0
    data = np.clip(data, min_val, max_val)
    # 归一化公式: (x - min) / (max - min) -> [0, 1]
    data = (data - min_val) / (max_val - min_val)
    # 映射到 [-1, 1]: x * 2 - 1
    data = data * 2.0 - 1.0 
    return data.astype(np.float32)

def normalize_mri(data):
    """
    MRI 归一化策略 (鲁棒性):
    使用 1% 和 99% 分位数去除非解剖结构的极值噪点。
    """
    if data.size == 0: return data
    
    # 计算分位数
    min_val = np.percentile(data, 1)
    max_val = np.percentile(data, 99)
    
    # 防止分母为 0 (假如一张图全是黑的)
    if max_val - min_val < 1e-6:
        return np.zeros_like(data).astype(np.float32)
        
    data = np.clip(data, min_val, max_val)
    data = (data - min_val) / (max_val - min_val)
    data = data * 2.0 - 1.0
    return data.astype(np.float32)

def process_and_save(phase='train'):
    """
    核心循环: 读取 -> 切片 -> 对应 -> 保存
    """
    # 定义输入路径 (根据你说的文件夹名 trainA/trainB)
    # 如果你的文件夹叫 train_A, 这里就改 train_A，这里假设是 trainA
    # 根据你的描述，Pix2Pix 标准通常是 trainA, trainB 或 train_A, train_B
    # 这里我做了自动适配，尝试两种常见命名
    
    src_mr_dir = os.path.join(RAW_DATAROOT, f'{phase}A') # 比如 trainA
    src_ct_dir = os.path.join(RAW_DATAROOT, f'{phase}B') # 比如 trainB
    
    # 如果找不到 trainA，尝试找 train_A (容错处理)
    if not os.path.exists(src_mr_dir):
        src_mr_dir = os.path.join(RAW_DATAROOT, f'{phase}_A')
        src_ct_dir = os.path.join(RAW_DATAROOT, f'{phase}_B')

    if not os.path.exists(src_mr_dir) or not os.path.exists(src_ct_dir):
        print(f"❌ 错误: 找不到输入文件夹! 请检查路径: {src_mr_dir}")
        return

    # 定义输出路径
    dst_mr_dir = os.path.join(SAVE_DATAROOT, f'{phase}_A')
    dst_ct_dir = os.path.join(SAVE_DATAROOT, f'{phase}_B')
    os.makedirs(dst_mr_dir, exist_ok=True)
    os.makedirs(dst_ct_dir, exist_ok=True)
    
    # 获取文件名列表 (只取 .nii 或 .nii.gz)
    filenames = sorted([f for f in os.listdir(src_mr_dir) if f.endswith(('.nii', '.nii.gz'))])
    
    print(f"🔄 开始处理 {phase} 集，共找到 {len(filenames)} 个 3D 卷...")

    count_slices = 0
    
    for fname in tqdm(filenames):
        # 1. 构造完整路径
        mr_path = os.path.join(src_mr_dir, fname)
        ct_path = os.path.join(src_ct_dir, fname)
        
        # 2. 检查配对: 如果 CT 文件夹里没有同名文件，就跳过
        if not os.path.exists(ct_path):
            print(f"⚠️ 跳过不匹配文件 (CT缺失): {fname}")
            continue

        try:
            # 3. 读取 NIfTI 数据
            mr_obj = nib.load(mr_path)
            ct_obj = nib.load(ct_path)
            
            # 确保方向一致 (RAS)
            mr_obj = nib.as_closest_canonical(mr_obj)
            ct_obj = nib.as_closest_canonical(ct_obj)
            
            mr_vol = mr_obj.get_fdata().astype(np.float32)
            ct_vol = ct_obj.get_fdata().astype(np.float32)

            # 4. 确定切片数量 (取两者最小值，防止溢出)
            D3 = min(mr_vol.shape[2], ct_vol.shape[2])
            
            # 提取基础文件名 (去掉 .nii.gz 后缀)
            base_name = fname.replace('.nii.gz', '').replace('.nii', '')

            # 5. 逐层切片并保存
            for i in range(D3):
                # 取切片
                mr_slice = mr_vol[:, :, i]
                ct_slice = ct_vol[:, :, i]

                # 归一化
                mr_norm = normalize_mri(mr_slice)
                ct_norm = normalize_ct(ct_slice)

                # 构造保存文件名: 原文件名_层号.npy
                # 关键点: A和B使用完全相同的文件名，保证了一一对应
                save_name = f"{base_name}_{i:03d}.npy"
                
                np.save(os.path.join(dst_mr_dir, save_name), mr_norm)
                np.save(os.path.join(dst_ct_dir, save_name), ct_norm)
                
                count_slices += 1

        except Exception as e:
            print(f"❌ 处理文件 {fname} 时出错: {e}")

    print(f"✅ 处理完成! 共生成了 {count_slices} 对 .npy 切片文件。")
    print(f"📂 数据保存在: {SAVE_DATAROOT}")

if __name__ == '__main__':
    # 只需要跑这一行即可
    process_and_save('train') 
    
    # 如果你有 testA 和 testB，可以把下面这行注释取消掉
    # process_and_save('test')
