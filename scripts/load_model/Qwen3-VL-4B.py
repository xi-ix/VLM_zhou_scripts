import argparse
import os
from types import MethodType

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

os.environ["TRANSFORMERS_DISABLE_TORCH_CHECK"] = "1"

DEFAULT_MODEL_DIR = "model_weights/Qwen3-VL-4B-Instruct"
DEFAULT_IMAGE = "dataset/images/test/0.png"
MODEL_CACHE_NAME = "models--Qwen--Qwen3-VL-4B-Instruct"
DEFAULT_PROMPT = "请描述这张图片，并回答图片中有什么重要信息。"


def find_model_path(model_dir):
    """Return the local snapshot path when model_dir is a Hugging Face cache dir."""
    if os.path.isfile(os.path.join(model_dir, "config.json")):
        return model_dir

    snapshot_base = os.path.join(model_dir, MODEL_CACHE_NAME, "snapshots")
    if not os.path.isdir(snapshot_base):
        raise FileNotFoundError(
            f"找不到模型目录: {model_dir}，也找不到 snapshots: {snapshot_base}"
        )

    snapshots = sorted(os.listdir(snapshot_base))
    if not snapshots:
        raise FileNotFoundError(f"snapshots 目录为空: {snapshot_base}")

    return os.path.join(snapshot_base, snapshots[-1])


def load_model_and_processor(model_path):
    print(f"正在加载模型: {model_path}")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    patch_qwen3_vl_placeholder_mask(model)
    print("模型加载完成")
    return model, processor


def patch_qwen3_vl_placeholder_mask(model):
    original_get_placeholder_mask = model.model.get_placeholder_mask

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
        try:
            return original_get_placeholder_mask(
                input_ids,
                inputs_embeds,
                image_features=image_features,
                video_features=video_features,
            )
        except RuntimeError:
            raise
        except Exception:
            raise

    def get_mask_from_token_types(self, inputs_embeds, modality_id, features):
        token_types = getattr(self, "_qwen3_vl_mm_token_type_ids", None)
        if token_types is None:
            return None

        token_types = token_types.to(inputs_embeds.device)
        mask = token_types == modality_id
        if mask.shape[-1] != inputs_embeds.shape[1]:
            mask = mask[:, : inputs_embeds.shape[1]]
        if features is not None and mask.sum().item() != features.shape[0]:
            return None
        return mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

    def patched_get_placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
        image_mask = get_mask_from_token_types(self, inputs_embeds, 1, image_features)
        video_mask = get_mask_from_token_types(self, inputs_embeds, 2, video_features)

        if image_mask is not None or video_mask is not None:
            if image_mask is None:
                image_mask = torch.zeros_like(inputs_embeds, dtype=torch.bool)
            if video_mask is None:
                video_mask = torch.zeros_like(inputs_embeds, dtype=torch.bool)
            return image_mask, video_mask

        return original_get_placeholder_mask(
            input_ids,
            inputs_embeds,
            image_features=image_features,
            video_features=video_features,
        )

    model.model.get_placeholder_mask = MethodType(patched_get_placeholder_mask, model.model)


def build_inputs(processor, image_path, prompt_text, device):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片: {image_path}")

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs.to(device)


def generate_response(model, processor, inputs, max_new_tokens):
    if "mm_token_type_ids" in inputs:
        model.model._qwen3_vl_mm_token_type_ids = inputs["mm_token_type_ids"]
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def ask_once(model, processor, image_path, prompt_text, max_new_tokens):
    inputs = build_inputs(processor, image_path, prompt_text, model.device)
    return generate_response(model, processor, inputs, max_new_tokens)


def run_interactive(model, processor, max_new_tokens):
    print("\n进入交互模式。输入 q 退出。")
    while True:
        image_path = input("\n图片路径: ").strip()
        if image_path.lower() in {"q", "quit", "exit"}:
            break

        prompt_text = input("问题/提示词: ").strip()
        if not prompt_text:
            prompt_text = DEFAULT_PROMPT

        try:
            response = ask_once(model, processor, image_path, prompt_text, max_new_tokens)
            print("\n模型回答:")
            print(response)
        except Exception as exc:
            print(f"处理失败: {exc}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="本地加载 Qwen3-VL-4B-Instruct，输入图片和文本进行问答。"
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"模型缓存目录或 snapshot 目录，默认: {DEFAULT_MODEL_DIR}",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="文件路径，例如: dataset/images/test/0.png")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="文本问题/提示词")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="最大生成长度")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="进入交互模式，连续输入图片路径和问题",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = find_model_path(args.model_dir)
    model, processor = load_model_and_processor(model_path)

    if args.interactive or not args.image:
        run_interactive(model, processor, args.max_new_tokens)
        return

    response = ask_once(
        model=model,
        processor=processor,
        image_path=args.image,
        prompt_text=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )
    print(response)


if __name__ == "__main__":
    main()
