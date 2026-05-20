import os
import shutil
import random

def random_copy_mp4_recursive(source_dir, target_dir, count):
    # 1. 检查源路径
    if not os.path.exists(source_dir):
        print(f"错误：源路径 '{source_dir}' 不存在。")
        return

    # 2. 准备目标路径
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 3. 递归获取所有子文件夹下的 .mp4 文件
    mp4_files_paths = []
    print("正在扫描文件夹及子文件夹，请稍候...")
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.mp4'):
                # 存储完整路径，防止不同文件夹下有同名文件导致混淆
                full_path = os.path.join(root, file)
                mp4_files_paths.append(full_path)

    # 4. 检查文件数量
    total_found = len(mp4_files_paths)
    print(f"共发现 {total_found} 个视频文件。")
    
    if total_found == 0:
        return

    if count > total_found:
        print(f"提示：请求数量超过现有总数，将复制全部文件。")
        count = total_found

    # 5. 随机抽取
    selected_paths = random.sample(mp4_files_paths, count)

    # 6. 执行复制
    print(f"正在随机复制 {count} 个文件...")
    for src_path in selected_paths:
        file_name = os.path.basename(src_path)
        dst_path = os.path.join(target_dir, file_name)
        
        # 处理重名风险：如果目标文件夹已有同名文件，自动加上数字后缀
        if os.path.exists(dst_path):
            name, ext = os.path.splitext(file_name)
            dst_path = os.path.join(target_dir, f"{name}_{random.randint(1, 9999)}{ext}")

        shutil.copy2(src_path, dst_path)
        print(f"已从 {os.path.dirname(src_path)} 复制: {file_name}")

    print("\n任务完成！")

# --- 配置区 ---
if __name__ == "__main__":
    # 请在这里修改你的路径和需要复制的数量
    SOURCE = '../../data3/VLA_LLM_DATA'      # 源文件夹路径
    TARGET = './VLM/dataset'      # 目标文件夹路径
    NUM_TO_COPY = 5                   # 随机复制的数量

    random_copy_mp4_recursive(SOURCE, TARGET, NUM_TO_COPY)