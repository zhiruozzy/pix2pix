import os
from pathlib import Path

def batch_rename_files(root_dir, suffix_to_remove):
    """
    批量重命名文件夹（含子文件夹）中的文件，删除指定后缀
    :param root_dir: 根目录（A 或 B 文件夹路径）
    :param suffix_to_remove: 要删除的后缀（如 "_MR"、"_CT"）
    """
    # 遍历根目录下的所有文件（含子文件夹）
    for file_path in Path(root_dir).rglob("*.*"):
        if file_path.is_file():  # 只处理文件，跳过文件夹
            # 获取文件名（含扩展名）和文件目录
            file_name = file_path.name
            file_dir = file_path.parent
            
            # 检查文件名是否包含要删除的后缀（且后缀在扩展名前）
            if suffix_to_remove in file_name:
                # 分割文件名和扩展名（如 "xxx_MR.png" → ("xxx_MR", ".png")）
                name_without_ext, ext = os.path.splitext(file_name)
                # 删除后缀（如 "xxx_MR" → "xxx"）
                new_name_without_ext = name_without_ext.replace(suffix_to_remove, "")
                # 拼接新文件名（如 "xxx.png"）
                new_file_name = new_name_without_ext + ext
                # 构建新文件路径
                new_file_path = file_dir / new_file_name
                
                # 重命名文件（如果新文件名不存在）
                if not new_file_path.exists():
                    file_path.rename(new_file_path)
                    print(f"重命名成功：{file_name} → {new_file_name}")
                else:
                    print(f"跳过：新文件名已存在 → {new_file_name}")

# -------------------------- 配置参数 --------------------------
A_DIR = "datasets/pix2pix_data_orgnized/A"  # 你的 A 文件夹路径
B_DIR = "datasets/pix2pix_data_orgnized/B"  # 你的 B 文件夹路径
A_SUFFIX = "_MR"  # A 文件夹要删除的后缀
B_SUFFIX = "_CT"  # B 文件夹要删除的后缀
# --------------------------------------------------------------

if __name__ == "__main__":
    print("开始重命名 A 文件夹文件...")
    batch_rename_files(A_DIR, A_SUFFIX)
    
    print("\n开始重命名 B 文件夹文件...")
    batch_rename_files(B_DIR, B_SUFFIX)
    
    print("\n重命名完成！")
