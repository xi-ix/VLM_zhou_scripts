from PIL import Image, ImageDraw
import numpy as np

def draw_bounding_box(image_path, bbox_coords, output_path="output.jpg"):
    """
    在图片上绘制边界框
    Args:
        image_path: 输入图片路径
        bbox_coords: 归一化的边界框坐标 [x_min, y_min, x_max, y_max]
        output_path: 输出图片路径
    """
    # 加载图片
    image = Image.open(image_path)
    width, height = image.size
    
    # 将归一化坐标转换为实际像素坐标
    x_min = int(bbox_coords[0] * width)
    y_min = int(bbox_coords[1] * height)
    x_max = int(bbox_coords[2] * width)
    y_max = int(bbox_coords[3] * height)
    
    # 创建绘图对象
    draw = ImageDraw.Draw(image)
    
    # 绘制红色边界框
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
    
    # 保存结果
    image.save(output_path)
    print(f"已保存带有边界框的图片到: {output_path}")

if __name__ == "__main__":
    file_name = "test"
    image_path = f"images/test/{file_name}.jpg"
    output_path=f"images/outputs/{file_name}_output.jpg"
    # 边界框坐标（归一化）
    bbox_coords = [0.35, 0.2, 0.42, 0.5] # [x_min, y_min, x_max, y_max]
    
    draw_bounding_box(image_path, bbox_coords, output_path)
