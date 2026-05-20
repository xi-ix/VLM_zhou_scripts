import os
import sys

def count_images(directory):
    # 定义你要检测的常见图片扩展名（全部小写）
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.tiff'}
    count = 0

    print(f"正在扫描目录: {os.path.abspath(directory)}")

    # os.walk 会自动递归遍历目录及所有子目录
    for root, _, files in os.walk(directory):
        for file in files:
            # 提取文件后缀并转换为小写进行比对
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                count += 1

    return count

if __name__ == "__main__":
    # 如果通过命令行传入了路径则使用传入路径，否则默认当前目录
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    if not os.path.isdir(target_dir):
        print(f"错误：指定的目录 '{target_dir}' 不存在！")
        sys.exit(1)

    total_images = count_images(target_dir)
    print(f"找到的图片总数: {total_images}")