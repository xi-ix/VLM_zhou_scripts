import os
import sys
from PIL import Image

def convert_and_delete_originals(directory, target_format="jpg"):
    # 支持识别并处理的非 JPG 格式
    valid_extensions = {'.png', '.jpeg', '.bmp', '.webp', '.tiff', '.gif'}
    target_format = target_format.lower()
    target_ext = f".{target_format}"

    print(f"正在扫描目录及子目录: {os.path.abspath(directory)}")
    print(f"目标统一格式: {target_ext.upper()}")
    print("⚠️  警告：图片成功转换后，原非JPG文件将被【自动删除】！\n" + "-" * 40)

    success_count = 0
    fail_count = 0
    deleted_count = 0

    for root, dirs, files in os.walk(directory):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            
            # 如果是支持的图片格式，并且不是我们要的最终格式(.jpg)
            if ext in valid_extensions and ext != target_ext:
                old_path = os.path.join(root, filename)
                new_name = os.path.splitext(filename)[0] + target_ext
                new_path = os.path.join(root, new_name)

                try:
                    # 使用 with 确保图片处理完后，文件句柄被正常关闭
                    with Image.open(old_path) as img:
                        # 解决动态图报错：如果是动图(GIF/WEBP)，提取第一帧
                        if getattr(img, "is_animated", False):
                            img.seek(0)
                            img = img.copy()

                        # 转换色彩模式（处理透明背景变黑的问题）
                        if img.mode in ('RGBA', 'LA', 'P'):
                            if img.mode == 'P':
                                img = img.convert('RGBA')
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            if len(img.split()) == 4: 
                                background.paste(img, mask=img.split()[3])
                            elif len(img.split()) == 2: 
                                background.paste(img, mask=img.split()[1])
                            else:
                                background.paste(img)
                            img = background
                        else:
                            img = img.convert('RGB')

                        # 保存为新 JPG 文件
                        img.save(new_path, format='JPEG', quality=95)
                        print(f"✅ [转换成功] {filename} -> {new_name}")
                        success_count += 1
                    
                    # 【核心修改】文件关闭后，安全删除原图
                    os.remove(old_path)
                    print(f"🗑️ [原图已删] {old_path}")
                    deleted_count += 1
                        
                except Exception as e:
                    print(f"❌ [处理失败] 文件: {old_path} | 原因: {e}")
                    fail_count += 1

    print("-" * 40)
    print(f"🎉 任务完成！成功转换: {success_count} 张 | 自动删除原图: {deleted_count} 张 | 失败: {fail_count} 张。")

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    if not os.path.isdir(target_dir):
        print(f"错误：指定的目录 '{target_dir}' 不存在！")
        sys.exit(1)

    convert_and_delete_originals(target_dir)