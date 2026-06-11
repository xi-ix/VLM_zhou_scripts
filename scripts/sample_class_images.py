#!/usr/bin/env python3
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


def read_classes(label_file: Path) -> set[str]:
    classes = set()
    with label_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_id = line.split()[0]
            try:
                int(class_id)
            except ValueError:
                continue
            classes.add(class_id)
    return classes


def find_image(label_file: Path) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png", ".bmp"):
        image_file = label_file.with_suffix(suffix)
        if image_file.exists():
            return image_file
    return None


def collect_by_class(source_dir: Path) -> dict[str, list[tuple[Path, Path]]]:
    by_class = defaultdict(list)

    for label_file in sorted(source_dir.glob("*.txt")):
        image_file = find_image(label_file)
        if image_file is None:
            continue

        for class_id in read_classes(label_file):
            by_class[class_id].append((image_file, label_file))

    return by_class


def copy_samples(
    by_class: dict[str, list[tuple[Path, Path]]],
    output_dir: Path,
    samples_per_class: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_id in sorted(by_class, key=int):
        class_output = output_dir / f"class_{class_id}"
        class_output.mkdir(parents=True, exist_ok=True)

        candidates = by_class[class_id]
        selected = rng.sample(candidates, min(samples_per_class, len(candidates)))
        for image_file, label_file in selected:
            shutil.copy2(image_file, class_output / image_file.name)
            shutil.copy2(label_file, class_output / label_file.name)

        print(f"class {class_id}: copied {len(selected)} / {len(candidates)} images")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample images for each YOLO class and copy image+label pairs."
    )
    parser.add_argument("--source", default="/data3/VLA/set", help="Source dataset dir")
    parser.add_argument(
        "--output",
        default="/home/wangzhe/VLM/dataset/class_samples",
        help="Output directory",
    )
    parser.add_argument("--num", type=int, default=10, help="Samples per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)
    if not source_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    by_class = collect_by_class(source_dir)
    if not by_class:
        raise SystemExit(f"No valid labels found in: {source_dir}")

    copy_samples(by_class, output_dir, args.num, args.seed)
    print(f"Done: {output_dir}")


if __name__ == "__main__":
    main()
