#!/usr/bin/env python3
import argparse
from collections import Counter
from pathlib import Path

# 查看数据分布
def count_classes(label_dir: Path) -> tuple[Counter, int, int, list[str]]:
    counts = Counter()
    txt_files = sorted(label_dir.rglob("*.txt"))
    bad_lines = []
    total_objects = 0

    for txt_file in txt_files:
        with txt_file.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if not parts:
                    continue

                class_id = parts[0]
                try:
                    int(class_id)
                except ValueError:
                    bad_lines.append(f"{txt_file}:{line_no}: {line}")
                    continue

                counts[class_id] += 1
                total_objects += 1

    return counts, len(txt_files), total_objects, bad_lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count YOLO label classes from txt files."
    )
    parser.add_argument(
        "label_dir",
        nargs="?",
        default="/data3/VLA/set",
        help="Directory containing label txt files. Default: /data3/VLA/set",
    )
    args = parser.parse_args()

    label_dir = Path(args.label_dir)
    if not label_dir.exists():
        raise SystemExit(f"Path does not exist: {label_dir}")
    if not label_dir.is_dir():
        raise SystemExit(f"Path is not a directory: {label_dir}")

    counts, file_count, total_objects, bad_lines = count_classes(label_dir)

    print(f"Label directory: {label_dir}")
    print(f"TXT files: {file_count}")
    print(f"Total objects: {total_objects}")
    print(f"Classes: {len(counts)}")
    print()
    print("class_id\tcount")
    for class_id, count in sorted(counts.items(), key=lambda item: int(item[0])):
        print(f"{class_id}\t{count}")

    if bad_lines:
        print()
        print(f"Warning: skipped {len(bad_lines)} bad label lines.")
        for bad_line in bad_lines[:20]:
            print(bad_line)
        if len(bad_lines) > 20:
            print(f"... and {len(bad_lines) - 20} more")


if __name__ == "__main__":
    main()
