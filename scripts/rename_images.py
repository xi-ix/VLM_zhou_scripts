import os
import sys

def rename_images(directory, prefix="img_"):
    # 定义要处理的图片后缀
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    
    # 获取目录下的所有文件并过滤出图片，按文件名排序保证一致性
    files = os.listdir(directory)
    images = sorted([f for f in files if os.path.splitext(f)[1].lower() in valid_extensions])
    
    total_images = len(images)
    if total_images == 0:
        print("未在指定目录找到支持的图片文件。")
        return

    print(f"共找到 {total_images} 张图片，开始重命名...")
    
    # 动态计算序号的位数（至少3位，如 001, 002）
    pad_length = max(3, len(str(total_images)))

    for index, filename in enumerate(images, start=1):
        old_path = os.path.join(directory, filename)
        
        # 提取原文件后缀并保留
        ext = os.path.splitext(filename)[1].lower()
        # 生成新文件名，例如 img_001.jpg
        new_name = f"{prefix}{str(index).zfill(pad_length)}{ext}"
        new_path = os.path.join(directory, new_name)

        # 如果新名字刚好和旧名字一样，跳过
        if old_path == new_path:
            continue
            
        # 防止覆盖已存在的文件（非常重要）
        if os.path.exists(new_path):
            print(f"⚠️ 警告: 文件名冲突，跳过 {filename} (目标 {new_name} 已存在)")
            continue

        os.rename(old_path, new_path)
        print(f"✅ {filename} -> {new_name}")

if __name__ == "__main__":
    # 接收目标目录参数，默认当前目录
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    if not os.path.isdir(target_dir):
        print(f"错误：目录 '{target_dir}' 不存在。")
        sys.exit(1)

    rename_images(target_dir)
    print("🎉 重命名完成！")