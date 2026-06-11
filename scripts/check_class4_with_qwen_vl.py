#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from types import MethodType

import cv2
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None


os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_DISABLE_TORCH_CHECK"] = "1"

DEFAULT_MODEL_DIR = "/home/wangzhe/VLM/model_weights/Qwen3-VL-4B-Instruct"
DEFAULT_QWEN25_MODEL_DIR = "/home/wangzhe/VLM/model_weights/Qwen2.5-VL-3B"
MODEL_CACHE_NAME = "models--Qwen--Qwen3-VL-4B-Instruct"
QWEN25_CACHE_NAME = "models--Qwen--Qwen2.5-VL-3B-Instruct"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


PROMPT = """你是交通图像标注质检助手。
图中红色框标出了一个原标签为4的目标，目标是电动车/非机动车。
请只判断红框中的这个目标属于哪一种情况：
4 = 电动车在机动车道行驶
5 = 电动车在斑马线上行驶

判断依据：
- 如果红框目标位于机动车车道、汽车行驶车道、机动车道区域，输出4。
- 如果红框目标位于斑马线/人行横道白色条纹区域上，输出5。
- 如果看不清或无法确定，选择更可能的一个，并降低confidence。

请严格只输出JSON，不要输出其他文字：
{"label": 4或5, "confidence": 0到1之间的小数, "reason": "简短中文理由"}"""


def find_model_path(model_dir: str, cache_name: str) -> str:
    if os.path.isfile(os.path.join(model_dir, "config.json")):
        return model_dir

    snapshot_base = os.path.join(model_dir, cache_name, "snapshots")
    if not os.path.isdir(snapshot_base):
        raise FileNotFoundError(f"找不到模型目录: {model_dir}")

    snapshots = sorted(os.listdir(snapshot_base))
    if not snapshots:
        raise FileNotFoundError(f"snapshots 目录为空: {snapshot_base}")

    return os.path.join(snapshot_base, snapshots[-1])


def warn_transformers_version(model_path: str) -> None:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return

    try:
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return

    expected = model_config.get("transformers_version")
    actual = transformers.__version__
    if expected and expected != actual:
        print(
            "Warning: transformers version mismatch. "
            f"model config expects {expected}, current environment has {actual}. "
            "Qwen3-VL may fail if processor/model/generation APIs are incompatible.",
            flush=True,
        )


def load_qwen3_model_and_processor(model_dir: str):
    model_path = find_model_path(model_dir, MODEL_CACHE_NAME)
    print(f"Loading model: {model_path}", flush=True)
    warn_transformers_version(model_path)
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
    patch_qwen3_placeholder_mask_check(model)
    print("Model loaded", flush=True)
    return model, processor


def patch_qwen3_placeholder_mask_check(model) -> None:
    original_get_placeholder_mask = model.model.get_placeholder_mask

    def patched_get_placeholder_mask(
        self, input_ids, inputs_embeds, image_features=None, video_features=None
    ):
        if input_ids is not None:
            image_mask = input_ids == self.config.image_token_id
            video_mask = input_ids == self.config.video_token_id

            if image_features is not None and image_mask.sum().item() != image_features.shape[0]:
                raise ValueError(
                    "Image features and image tokens do not match: "
                    f"tokens: {image_mask.sum().item()}, features: {image_features.shape[0]}"
                )
            if video_features is not None and video_mask.sum().item() != video_features.shape[0]:
                raise ValueError(
                    "Video features and video tokens do not match: "
                    f"tokens: {video_mask.sum().item()}, features: {video_features.shape[0]}"
                )

            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            return image_mask, video_mask

        return original_get_placeholder_mask(
            input_ids,
            inputs_embeds,
            image_features=image_features,
            video_features=video_features,
        )

    model.model.get_placeholder_mask = MethodType(patched_get_placeholder_mask, model.model)


def load_qwen25_model_and_processor(model_dir: str):
    model_path = find_model_path(model_dir, QWEN25_CACHE_NAME)
    print(f"Loading model: {model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=1024 * 28 * 28,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    print("Model loaded", flush=True)
    return model, processor


def check_cuda_or_exit(require_cuda: bool) -> None:
    if not require_cuda:
        return

    try:
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
    except Exception as exc:
        raise SystemExit(f"CUDA 检查失败，停止加载模型: {exc}") from exc

    if not cuda_available or device_count == 0:
        raise SystemExit(
            "CUDA/GPU 在当前运行环境不可见，停止加载模型。"
            "请先确认当前终端运行 nvidia-smi 正常，或加 --no-require-cuda 强制 CPU/其它设备运行。"
        )

    device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
    print(f"CUDA available: {device_count} device(s): {device_names}", flush=True)


def yolo_to_xyxy(parts: list[str], image_w: int, image_h: int) -> tuple[int, int, int, int]:
    x_center, y_center, width, height = map(float, parts[1:5])
    x1 = int(round((x_center - width / 2) * image_w))
    y1 = int(round((y_center - height / 2) * image_h))
    x2 = int(round((x_center + width / 2) * image_w))
    y2 = int(round((y_center + height / 2) * image_h))
    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(0, min(image_w - 1, x2))
    y2 = max(0, min(image_h - 1, y2))
    return x1, y1, x2, y2


def expand_box(
    box: tuple[int, int, int, int], image_w: int, image_h: int, scale: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    width = max(1, (x2 - x1) * scale)
    height = max(1, (y2 - y1) * scale)
    x1 = int(round(cx - width / 2))
    y1 = int(round(cy - height / 2))
    x2 = int(round(cx + width / 2))
    y2 = int(round(cy + height / 2))
    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(0, min(image_w - 1, x2))
    y2 = max(0, min(image_h - 1, y2))
    return x1, y1, x2, y2


def make_query_image(
    image_path: Path, box: tuple[int, int, int, int], output_path: Path, crop_scale: float
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"读取图片失败: {image_path}")

    x1, y1, x2, y2 = box
    if crop_scale > 1:
        crop_x1, crop_y1, crop_x2, crop_y2 = expand_box(
            box, image.shape[1], image.shape[0], crop_scale
        )
        image = image[crop_y1 : crop_y2 + 1, crop_x1 : crop_x2 + 1].copy()
        x1 -= crop_x1
        x2 -= crop_x1
        y1 -= crop_y1
        y2 -= crop_y1

    thickness = max(3, round(min(image.shape[:2]) / 250))
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness)
    cv2.putText(
        image,
        "target",
        (x1, max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def ask_qwen3_model(model, processor, image_path: Path, max_new_tokens: int) -> str:
    if process_vision_info is None:
        raise RuntimeError("qwen_vl_utils is not installed, cannot run Qwen3-VL safely.")

    abs_image_path = image_path.resolve()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{abs_image_path}"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

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


def ensure_qwen3_mm_token_type_ids(model, processor, inputs) -> None:
    if "mm_token_type_ids" in inputs:
        return

    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token = getattr(processor, "image_token", "<|image_pad|>")
        image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token)
    if image_token_id is None or image_token_id < 0:
        image_token_id = getattr(model.config, "image_token_id", None)

    video_token_id = getattr(processor, "video_token_id", None)
    if video_token_id is None:
        video_token = getattr(processor, "video_token", "<|video_pad|>")
        video_token_id = processor.tokenizer.convert_tokens_to_ids(video_token)
    if video_token_id is None or video_token_id < 0:
        video_token_id = getattr(model.config, "video_token_id", None)

    mm_token_type_ids = torch.zeros_like(inputs["input_ids"])
    if image_token_id is not None:
        mm_token_type_ids[inputs["input_ids"] == image_token_id] = 1
    if video_token_id is not None:
        mm_token_type_ids[inputs["input_ids"] == video_token_id] = 2

    inputs["mm_token_type_ids"] = mm_token_type_ids


def greedy_decode_qwen3(model, inputs, max_new_tokens: int) -> torch.Tensor:
    generated = []
    attention_mask = inputs["attention_mask"]
    eos_token_id = model.generation_config.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_ids = set(eos_token_id)
    elif eos_token_id is None:
        eos_token_ids = set()
    else:
        eos_token_ids = {eos_token_id}

    first_kwargs = dict(inputs)
    with torch.no_grad():
        outputs = model(
            **first_kwargs,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        past_key_values = outputs.past_key_values

        for _ in range(max_new_tokens - 1):
            if next_token.item() in eos_token_ids:
                break

            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token, device=attention_mask.device)],
                dim=1,
            )
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            past_key_values = outputs.past_key_values

    return torch.cat(generated, dim=1)


def ask_qwen25_model(model, processor, image_path: Path, max_new_tokens: int) -> str:
    if process_vision_info is None:
        raise RuntimeError("qwen_vl_utils is not installed, cannot run Qwen2.5-VL.")

    abs_image_path = image_path.resolve()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{abs_image_path}"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

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


def parse_prediction(text: str) -> tuple[str, str, str]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return "", "", text

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "", "", text

    return str(data.get("label", "")), str(data.get("confidence", "")), str(data.get("reason", ""))


def apply_label_updates(rows: list[dict], backup_dir: Path) -> int:
    rows_to_update = [row for row in rows if row.get("predicted_label") == "5"]
    if not rows_to_update:
        return 0

    backup_dir.mkdir(parents=True, exist_ok=True)
    by_label_file = {}
    for row in rows_to_update:
        by_label_file.setdefault(Path(row["label_file"]), []).append(row)

    changed = 0
    for label_file, file_rows in sorted(by_label_file.items()):
        backup_file = backup_dir / label_file.name
        if not backup_file.exists():
            shutil.copy2(label_file, backup_file)

        lines = label_file.read_text(encoding="utf-8").splitlines()
        for row in file_rows:
            line_index = int(row["line_no"]) - 1
            if line_index < 0 or line_index >= len(lines):
                print(f"Warning: line out of range, skip {label_file}:{row['line_no']}")
                continue

            parts = lines[line_index].split()
            if not parts:
                print(f"Warning: empty line, skip {label_file}:{row['line_no']}")
                continue
            if parts[0] != "4":
                print(
                    f"Warning: expected class 4, got {parts[0]}, skip {label_file}:{row['line_no']}"
                )
                continue

            parts[0] = "5"
            lines[line_index] = " ".join(parts)
            changed += 1

        label_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return changed


def collect_class4_targets(input_dir: Path) -> list[dict]:
    targets = []
    for label_path in sorted(input_dir.rglob("*.txt")):
        image_path = None
        for suffix in IMAGE_SUFFIXES:
            candidate = label_path.with_suffix(suffix)
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image_h, image_w = image.shape[:2]

        for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.split()
            if len(parts) < 5 or parts[0] != "4":
                continue
            targets.append(
                {
                    "image_path": image_path,
                    "label_path": label_path,
                    "line_no": line_no,
                    "label_line": line,
                    "box": yolo_to_xyxy(parts, image_w, image_h),
                }
            )
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use local Qwen3-VL to check class 4 targets and visualize predictions."
    )
    parser.add_argument(
        "--input",
        default="/home/wangzhe/VLM/dataset/class_samples",
        help="Input dataset directory",
    )
    parser.add_argument(
        "--output",
        default="/home/wangzhe/VLM/dataset/class4_qwen_check",
        help="Output directory for visualizations and CSV",
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Local Qwen3-VL model dir")
    parser.add_argument(
        "--model-family",
        choices=["qwen3-4b", "qwen2.5-3b"],
        default="qwen2.5-3b",
        help="Local vision model to use",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="Only process first N targets, 0 means all")
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=8.0,
        help="Crop around target box before asking model. Use 0 or 1 for full image.",
    )
    parser.add_argument(
        "--no-require-cuda",
        action="store_true",
        help="Allow running even when CUDA is not visible. This may be very slow.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Modify txt labels: change predicted class 5 targets from 4 to 5 after writing CSV.",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Do not keep query visualization images. Temporary marked images are deleted after inference.",
    )
    parser.add_argument(
        "--backup-dir",
        help="Directory to save original txt backups before --apply changes.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    query_dir = output_dir / "query_images"
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    targets = collect_class4_targets(input_dir)
    if args.limit > 0:
        targets = targets[: args.limit]
    if not targets:
        raise SystemExit("No class 4 targets found.")

    check_cuda_or_exit(require_cuda=not args.no_require_cuda)

    if args.model_family == "qwen2.5-3b" and args.model_dir == DEFAULT_MODEL_DIR:
        args.model_dir = DEFAULT_QWEN25_MODEL_DIR

    if args.model_family == "qwen2.5-3b":
        model, processor = load_qwen25_model_and_processor(args.model_dir)
        ask_model = ask_qwen25_model
    else:
        model, processor = load_qwen3_model_and_processor(args.model_dir)
        ask_model = ask_qwen3_model

    rows = []
    temp_dir_obj = tempfile.TemporaryDirectory() if args.no_visualization else None
    temp_query_dir = Path(temp_dir_obj.name) if temp_dir_obj else None
    for index, target in enumerate(targets, start=1):
        rel_image = target["image_path"].relative_to(input_dir)
        stem = f"{index:04d}_{rel_image.parent.name}_{target['image_path'].stem}_line{target['line_no']}"
        query_path = (temp_query_dir if temp_query_dir else query_dir) / f"{stem}.jpg"
        make_query_image(target["image_path"], target["box"], query_path, args.crop_scale)

        print(f"[{index}/{len(targets)}] {rel_image} line {target['line_no']}", flush=True)
        raw_response = ask_model(model, processor, query_path, args.max_new_tokens)
        pred_label, confidence, reason = parse_prediction(raw_response)
        rows.append(
            {
                "index": index,
                "image": str(target["image_path"]),
                "label_file": str(target["label_path"]),
                "line_no": target["line_no"],
                "original_label": 4,
                "predicted_label": pred_label,
                "confidence": confidence,
                "reason": reason,
                "box_xyxy": " ".join(map(str, target["box"])),
                "query_image": "" if args.no_visualization else str(query_path),
                "raw_response": raw_response,
            }
        )
        print(f"  -> label={pred_label}, confidence={confidence}, reason={reason}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "class4_qwen_predictions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if args.apply:
        backup_dir = Path(args.backup_dir) if args.backup_dir else output_dir / "label_backup_before_apply"
        changed = apply_label_updates(rows, backup_dir)
        print(f"Applied label updates: {changed}")
        print(f"Original txt backups: {backup_dir}")

    print(f"Done. CSV: {csv_path}", flush=True)
    if args.no_visualization:
        temp_dir_obj.cleanup()
        print("Visualizations: disabled", flush=True)
    else:
        print(f"Visualizations: {query_dir}", flush=True)


if __name__ == "__main__":
    main()
