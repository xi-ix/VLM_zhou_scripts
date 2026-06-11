#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = {
    "0": (0, 0, 255),
    "1": (255, 128, 0),
    "2": (0, 180, 0),
    "3": (255, 0, 255),
    "4": (255, 0, 0),
    "5": (0, 180, 255),
}


def yolo_to_xyxy(
    x_center: float, y_center: float, width: float, height: float, image_w: int, image_h: int
) -> tuple[int, int, int, int]:
    x1 = int(round((x_center - width / 2) * image_w))
    y1 = int(round((y_center - height / 2) * image_h))
    x2 = int(round((x_center + width / 2) * image_w))
    y2 = int(round((y_center + height / 2) * image_h))

    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(0, min(image_w - 1, x2))
    y2 = max(0, min(image_h - 1, y2))
    return x1, y1, x2, y2


def draw_label(image, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    text_w, text_h = cv2.getTextSize(text, font, scale, thickness)[0]
    y_top = max(0, y - text_h - 8)
    cv2.rectangle(image, (x, y_top), (x + text_w + 8, y_top + text_h + 8), color, -1)
    cv2.putText(
        image,
        text,
        (x + 4, y_top + text_h + 4),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_boxes(image_path: Path, label_path: Path, output_path: Path) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: failed to read image: {image_path}")
        return False

    image_h, image_w = image.shape[:2]
    if label_path.exists():
        with label_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) < 5:
                    print(f"Warning: bad label line {label_path}:{line_no}: {line.strip()}")
                    continue

                class_id = parts[0]
                try:
                    x_center, y_center, width, height = map(float, parts[1:5])
                except ValueError:
                    print(f"Warning: bad label line {label_path}:{line_no}: {line.strip()}")
                    continue

                x1, y1, x2, y2 = yolo_to_xyxy(
                    x_center, y_center, width, height, image_w, image_h
                )
                color = COLORS.get(class_id, (255, 255, 255))
                line_width = max(2, round(min(image_w, image_h) / 400))
                cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)
                draw_label(image, class_id, x1, y1, color)
    else:
        print(f"Warning: missing label: {label_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(output_path), image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw YOLO txt boxes on images.")
    parser.add_argument(
        "--input",
        default="/home/wangzhe/VLM/dataset/class_samples",
        help="Input image/label directory",
    )
    parser.add_argument(
        "--output",
        default="/home/wangzhe/VLM/dataset/class_samples_vis",
        help="Output directory for visualized images",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    image_files = sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_files:
        raise SystemExit(f"No images found in: {input_dir}")

    ok_count = 0
    for image_path in image_files:
        relative_path = image_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        label_path = image_path.with_suffix(".txt")
        if draw_boxes(image_path, label_path, output_path):
            ok_count += 1

    print(f"Input images: {len(image_files)}")
    print(f"Visualized images: {ok_count}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
