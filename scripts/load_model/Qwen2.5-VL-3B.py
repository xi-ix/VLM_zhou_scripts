import torch
import os
import sys
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= 1. 离线环境强制设置 =================
# 必须在导入 transformers 之后尽快设置，确保不进行任何联网检查
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

def get_absolute_path():
    """
    自动获取模型快照的绝对路径，避免手动输入长字符串文件夹名
    """
    base_dir = r"model_weights/Qwen2.5-VL-3B/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots"
    if not os.path.exists(base_dir):
        print(f"错误：找不到基础路径 {base_dir}")
        return None
    snapshots = os.listdir(base_dir)
    if not snapshots:
        print("错误：snapshots 文件夹下没有内容")
        return None
    full_path = os.path.join(base_dir, snapshots[0])
    return full_path

def load_model(model_path):
    print(f"正在从本地加载模型: {model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16, # 如果是老显卡请改为 torch.float16
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True 
    )
    processor = AutoProcessor.from_pretrained(
        model_path, 
        min_pixels=256*28*28, 
        max_pixels=1280*28*28,
        trust_remote_code=True,
        local_files_only=True 
    )
    print("模型加载成功！")
    return model, processor

def process(model, processor, image_path, prompt_text):
    abs_image_path = os.path.abspath(image_path)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{abs_image_path}"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=256,
        do_sample=False,  # 贪婪搜索，保证结果一致性
    )
    
    full_output = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    output_text = full_output.split("assistant")[-1].strip()
    return output_text

if __name__ == '__main__':
    target_path = get_absolute_path()
    
    if target_path:
        model, processor = load_model(target_path)
        
        prompt_text = '''
        1. 首先计算画面中可见车辆的数量和汽车占道路比例。
        2. 然后判断道路是否拥堵。（可以通过汽车数量、汽车之间的距离、汽车占据道路的比例来判断）。
        请严格按此格式输出：
        车辆数量：[数字]
        汽车占道路比例：[数字]
        是否拥堵：[True/False]'''
        
        print(f"Prompt 设置完毕，准备开始测试...\n")
        
        for num in range(0, 1):
            print("-" * 30)
            print(f"第 {num} 次测试")
            img_rel_path = f"dataset/images/test/{num}.png"
            
            if not os.path.exists(img_rel_path):
                print(f"跳过：找不到图片 {img_rel_path}")
                continue
                
            try:
                output = process(model, processor, img_rel_path, prompt_text)
                print(f"输出结果：\n{output}\n")
            except Exception as e:
                print(f"处理图片 {num} 时出错: {e}")
    else:
        print("模型路径初始化失败，请检查文件夹结构。")