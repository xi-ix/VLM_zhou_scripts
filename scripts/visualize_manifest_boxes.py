#!/usr/bin/env python3
"""Draw detection boxes from manifest.jsonl onto images.

Example:
  /home/wangzhe/miniconda3/envs/qwen_vl/bin/python \
    /home/wangzhe/VLM/scripts/visualize_manifest_boxes.py \
    --manifest /data3/wangzhe/data/manifest.jsonl \
    --output-dir /data3/wangzhe/data/vis_labels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

COLOR_MAP = {
    "person": (60, 220, 60),
    "bicycle": (60, 180, 255),
    "motorcycle": (255, 170, 60),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize manifest boxes on images.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=0, help="0 means all.")
    parser.add_argument("--thickness", type=int, default=2)
    parser.add_argument("--font-scale", type=float, default=0.6)
    return parser.parse_args()


def draw_record(record: dict, output_dir: Path, thickness: int, font_scale: float) -> bool:
    image_path = Path(record.get("image_path", ""))
    if not image_path.exists():
        return False

    image = cv2.imread(str(image_path))
    if image is None:
        return False

    boxes = record.get("boxes_xyxy", [])
    classes = record.get("classes", [])
    confs = record.get("confs", [])

    for i, box in enumerate(boxes):
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        cls_name = classes[i] if i < len(classes) else "obj"
        conf = confs[i] if i < len(confs) else None
        color = COLOR_MAP.get(cls_name, (220, 220, 60))

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        if conf is None:
            label = cls_name
        else:
            label = f"{cls_name} {float(conf):.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        y_text_top = max(0, y1 - th - 6)
        cv2.rectangle(image, (x1, y_text_top), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 3, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    rel_name = image_path.name
    inter = record.get("intersection", "unknown")
    cam = record.get("camera", "unknown")
    out_path = output_dir / str(inter) / str(cam) / rel_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), image))


def main() -> None:
    args = parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    ok_count = 0

    with args.manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            if args.max_images > 0 and total > args.max_images:
                break

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if draw_record(rec, args.output_dir, args.thickness, args.font_scale):
                ok_count += 1

    print(f"Done. visualized={ok_count}, read={total}, output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
