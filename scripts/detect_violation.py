import torch
import os
import re
import shutil
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
        generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response

if __name__ == "__main__":
    # ======================== 配置区域 ========================
    # 输入图片文件夹路径
    IMAGE_FOLDER = "images/violation_input"
    # 违章图片输出文件夹路径
    OUTPUT_FOLDER = "images/violation_output"
    # 日志文件路径
    LOG_FILE = "log/violation_log.txt"
    # =========================================================

    local_path = get_local_snapshot_path(CUSTOM_SAVE_PATH)
    if not local_path:
        print(f"错误：未找到模型快照路径，请检查 {CUSTOM_SAVE_PATH}")
        exit(1)

    model, processor = load_model_and_processor(local_path)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    prompt_text = '''请仔细观察这张交通监控图片，判断是否存在交通违章行为。
常见的交通违章包括但不限于：闯红灯、违停、逆行、压实线变道、占用应急车道、不按车道行驶等。
请严格按此格式输出：
是否存在违章：[是/否]
违章类型：[具体违章类型，如无则填"无"]
简要描述：[一句话描述违章情况，如无则填"无"]'''

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

    if not os.path.exists(IMAGE_FOLDER):
        print(f"错误：输入文件夹不存在 {IMAGE_FOLDER}")
        exit(1)

    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(valid_extensions)]
    image_files.sort(key=lambda f: [float(c) if c.replace('.', '', 1).isdigit() else c for c in re.split(r'(\d+\.?\d*)', f)])

    total = len(image_files)
    print(f"图片总数：{total}")

    violation_count = 0

    for idx, filename in enumerate(image_files, 1):
        image_path = os.path.join(IMAGE_FOLDER, filename)
        print(f"[{idx}/{total}] 正在处理: {filename}")

        try:
            inputs = process_image_inputs(processor, image_path, prompt_text, model.device)
            result = get_model_response(model, processor, inputs)
            print(f"  模型输出: {result}")

            if "是" in result and "违章" in result:
                violation_count += 1
                # 复制图片到输出文件夹
                src_path = os.path.join(IMAGE_FOLDER, filename)
                dst_path = os.path.join(OUTPUT_FOLDER, filename)
                shutil.copy2(src_path, dst_path)
                print(f"  --> [检测到违章] 图片已复制到 {OUTPUT_FOLDER}")

                # 写入日志
                try:
                    with open(LOG_FILE, "a", encoding="utf-8") as f:
                        name_without_ext, _ = os.path.splitext(filename)
                        f.write(f"{name_without_ext}\t{result.strip()}\n")
                except Exception as e:
                    print(f"  写入日志失败: {e}")

        except Exception as e:
            print(f"  处理失败 {filename}: {e}")
        finally:
            torch.cuda.empty_cache()

        print("-" * 50)

    print(f"处理完成！共 {total} 张图片，检测到 {violation_count} 张违章图片")
    print(f"违章图片已保存到: {OUTPUT_FOLDER}")
    print(f"日志文件: {LOG_FILE}")
