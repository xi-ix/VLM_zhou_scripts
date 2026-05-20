import torch
import os  
import re
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

CUSTOM_SAVE_PATH = r"model_weights/Qwen3-VL-4B-Instruct"

def get_local_snapshot_path(base_cache_path):
    snapshot_base = os.path.join(base_cache_path, "models--Qwen--Qwen3-VL-4B-Instruct", "snapshots")
    if not os.path.exists(snapshot_base):
        return None
    snapshots = os.listdir(snapshot_base)
    if not snapshots:
        return None
    return os.path.join(snapshot_base, snapshots[0])

def load_model_and_processor(model_path):
    print("loading model")
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
    print("successfully loaded")
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
        generated_ids = model.generate(**inputs, max_new_tokens=128)
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
        VIDEO = "D10_20260115193540_20260115195256.mp4"
        IMAGE_FOLDER = f"images/{VIDEO}"
        LOG_FILE = f"log/{VIDEO}.txt"
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        prompt_text = '''1. 首先计算画面中可见车辆的数量和汽车占道路比例。
                        2. 然后判断道路是否拥堵。
                        请严格按此格式输出：
                        车辆数量：[数字]
                        汽车占道路比例：[数字]
                        是否拥堵：[True/False]'''
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        if os.path.exists(IMAGE_FOLDER):
            image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(valid_extensions)]
            print(f"图片总数：{image_files.Len()}")
            # image_files.sort()
            image_files.sort(key=lambda f: [float(c) if c.replace('.', '', 1).isdigit() else c for c in re.split(r'(\d+\.?\d*)', f)])
            for filename in image_files:
                image_path = os.path.join(IMAGE_FOLDER, filename)
                print(f"--- 正在处理: {filename} ---")
                try:
                    inputs = process_image_inputs(processor, image_path, prompt_text, model.device)
                    result = get_model_response(model, processor, inputs)
                    print(result)
                    if "True" in result:
                        try:
                            with open(LOG_FILE, "a", encoding="utf-8") as f:
                                filename , _ = os.path.splitext(filename)
                                f.write(filename + "\n")
                            print(f"--> [检测到 True] 文件名已写入 {LOG_FILE}")
                        except Exception as e:
                            print(f"写入日志失败: {e}")
                except Exception as e:
                    print(f"处理失败 {filename}: {e}")
                finally:
                    torch.cuda.empty_cache()
                print("-" * 30)
            print(f"{image_files.Len()}张图片已处理，文件{LOG_FILE}处理完成")