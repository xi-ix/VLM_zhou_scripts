import torch
import os
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

CUSTOM_SAVE_PATH = r"VLM/model_weights/Qwen3-VL-4B-Instruct"

def get_local_snapshot_path(base_cache_path):
    snapshot_base = os.path.join(base_cache_path, "models--Qwen--Qwen3-VL-4B-Instruct", "snapshots")
    if not os.path.exists(snapshot_base):
        return None
    snapshots = os.listdir(snapshot_base)
    if not snapshots:
        return None
    return os.path.join(snapshot_base, snapshots[0])

def load_model_and_processor(model_path):
    processor = AutoProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True,
        local_files_only=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True
    ).eval()
    return model, processor

def process_image_inputs(processor, image_path, prompt_text, device):
    abs_image_path = os.path.abspath(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{abs_image_path}"},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    return inputs

def get_model_response(model, processor, inputs):
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=512,  # 调大这个值
            do_sample=True,      # 开启采样
            temperature=0.7,     # 增加一点随机性，避免死循环
            top_p=0.9
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response

if __name__ == "__main__":
    local_path = get_local_snapshot_path(CUSTOM_SAVE_PATH)
    if local_path:
        model, processor = load_model_and_processor(local_path)
        
        p1 = '''1. 首先计算画面中可见车辆的数量和汽车占道路比例。
                2. 然后判断道路是否拥堵。
                请严格按此格式输出：
                车辆数量：[数字]
                汽车占道路比例：[数字]
                是否拥堵：[True/False]'''
                
        p2 = '''# 角色
                你是一个资深的交通分析专家。
                # 任务
                请观察这张十字路口监控图片，完成以下分析：
                1. **拥堵判定**：明确回答当前是否存在拥堵（是/否）。（根据汽车数目和车道占用情况）
                2. **场景描述**：简要描述当前的交通流状态（各车道的车辆密度）。
                3. **根本原因分析**：如果存在拥堵，请结合图中细节给出具体原因。
                - 重点观察：是否有交通事故、道路施工、违章停车、红绿灯时长不合理、恶劣天气或单纯的流量饱和。
                4. **堵点定位**：指出拥堵最严重的具体方位（如：由东向西左转车道）。
                # 输出格式
                - 拥堵状态：
                - 情况详述：
                - 原因推断：'''
        prompt_text = p2
        image_path = "VLM/images/test/0.png"
        if os.path.exists(image_path):
            inputs = process_image_inputs(processor, image_path, prompt_text, model.device)
            result = get_model_response(model, processor, inputs)
            print(image_path)
            print(result)