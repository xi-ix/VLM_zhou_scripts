#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def label_path_for(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def read_label_lines(label_file: Path) -> list[str]:
    if not label_file.exists():
        return []

    lines = []
    with label_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def merged_label_lines(files: list[Path]) -> list[str]:
    seen = set()
    merged = []
    for image_file in sorted(files):
        for line in read_label_lines(label_path_for(image_file)):
            if line in seen:
                continue
            seen.add(line)
            merged.append(line)
    return merged


def choose_keep_file(files: list[Path]) -> Path:
    # Keep the lexicographically first file for stable, repeatable results.
    return sorted(files)[0]


def scan_duplicates(source_dir: Path) -> dict[str, list[Path]]:
    groups = defaultdict(list)
    image_files = sorted(
        p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )

    for image_file in image_files:
        groups[file_sha256(image_file)].append(image_file)

    return {hash_value: files for hash_value, files in groups.items() if len(files) > 1}


def write_reports(duplicate_groups: dict[str, list[Path]], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    json_data = []
    csv_rows = []
    for group_id, (hash_value, files) in enumerate(
        sorted(duplicate_groups.items(), key=lambda item: str(choose_keep_file(item[1]))),
        start=1,
    ):
        keep_file = choose_keep_file(files)
        duplicate_files = [p for p in sorted(files) if p != keep_file]

        json_data.append(
            {
                "group_id": group_id,
                "sha256": hash_value,
                "keep": str(keep_file),
                "duplicates": [str(p) for p in duplicate_files],
            }
        )

        for duplicate_file in duplicate_files:
            csv_rows.append(
                {
                    "group_id": group_id,
                    "sha256": hash_value,
                    "keep_image": str(keep_file),
                    "duplicate_image": str(duplicate_file),
                    "keep_label": str(label_path_for(keep_file)),
                    "duplicate_label": str(label_path_for(duplicate_file)),
                }
            )

    with (report_dir / "duplicate_groups.json").open("w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    with (report_dir / "duplicate_pairs.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group_id",
                "sha256",
                "keep_image",
                "duplicate_image",
                "keep_label",
                "duplicate_label",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    label_diff_rows = []
    for group_id, (_, files) in enumerate(
        sorted(duplicate_groups.items(), key=lambda item: str(choose_keep_file(item[1]))),
        start=1,
    ):
        label_texts = {
            str(label_path_for(image_file)): "\n".join(read_label_lines(label_path_for(image_file)))
            for image_file in sorted(files)
        }
        if len(set(label_texts.values())) <= 1:
            continue

        keep_file = choose_keep_file(files)
        label_diff_rows.append(
            {
                "group_id": group_id,
                "keep_image": str(keep_file),
                "files_in_group": len(files),
                "unique_label_versions": len(set(label_texts.values())),
                "merged_label_count": len(merged_label_lines(files)),
            }
        )

    with (report_dir / "label_diff_groups.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group_id",
                "keep_image",
                "files_in_group",
                "unique_label_versions",
                "merged_label_count",
            ],
        )
        writer.writeheader()
        writer.writerows(label_diff_rows)


def move_duplicates(
    duplicate_groups: dict[str, list[Path]], trash_dir: Path, merge_labels: bool
) -> int:
    moved = 0
    trash_dir.mkdir(parents=True, exist_ok=True)

    for files in duplicate_groups.values():
        keep_file = choose_keep_file(files)
        if merge_labels:
            keep_label = label_path_for(keep_file)
            lines = merged_label_lines(files)
            keep_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        for duplicate_file in sorted(files):
            if duplicate_file == keep_file:
                continue

            target_image = trash_dir / duplicate_file.name
            shutil.move(str(duplicate_file), str(target_image))
            moved += 1

            duplicate_label = label_path_for(duplicate_file)
            if duplicate_label.exists():
                target_label = trash_dir / duplicate_label.name
                shutil.move(str(duplicate_label), str(target_label))

    return moved


def copy_duplicates(duplicate_groups: dict[str, list[Path]], trash_dir: Path) -> int:
    copied = 0
    trash_dir.mkdir(parents=True, exist_ok=True)

    for files in duplicate_groups.values():
        keep_file = choose_keep_file(files)
        for duplicate_file in sorted(files):
            if duplicate_file == keep_file:
                continue

            shutil.copy2(duplicate_file, trash_dir / duplicate_file.name)
            copied += 1

            duplicate_label = label_path_for(duplicate_file)
            if duplicate_label.exists():
                shutil.copy2(duplicate_label, trash_dir / duplicate_label.name)

    return copied


def copy_dedup_dataset(
    source_dir: Path, duplicate_groups: dict[str, list[Path]], output_dir: Path
) -> int:
    duplicate_files_to_skip = set()
    for files in duplicate_groups.values():
        keep_file = choose_keep_file(files)
        duplicate_files_to_skip.update(p for p in files if p != keep_file)

    output_dir.mkdir(parents=True, exist_ok=True)
    copied_images = 0
    for image_file in sorted(
        p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ):
        if image_file in duplicate_files_to_skip:
            continue

        shutil.copy2(image_file, output_dir / image_file.name)
        copied_images += 1

        label_file = label_path_for(image_file)
        if label_file.exists():
            shutil.copy2(label_file, output_dir / label_file.name)

    return copied_images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find exact duplicate images by SHA256 and optionally move duplicates."
    )
    parser.add_argument("--source", default="/data3/VLA/set", help="Dataset directory")
    parser.add_argument(
        "--report-dir",
        default="/home/wangzhe/VLM/dataset/duplicate_report",
        help="Directory to save duplicate reports",
    )
    parser.add_argument(
        "--move-duplicates",
        action="store_true",
        help="Move duplicate image+label files to --trash-dir after writing reports",
    )
    parser.add_argument(
        "--merge-labels",
        action="store_true",
        help="Before moving duplicates, merge all labels in each duplicate group into the kept txt file",
    )
    parser.add_argument(
        "--trash-dir",
        default="/home/wangzhe/VLM/dataset/duplicate_removed",
        help="Directory for moved duplicate files",
    )
    parser.add_argument(
        "--copy-duplicates",
        action="store_true",
        help="Copy duplicate image+label files to --trash-dir without changing source",
    )
    parser.add_argument(
        "--dedup-output",
        help="Copy a deduplicated dataset to this directory without changing source",
    )
    args = parser.parse_args()

    source_dir = Path(args.source)
    report_dir = Path(args.report_dir)
    trash_dir = Path(args.trash_dir)

    if not source_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    duplicate_groups = scan_duplicates(source_dir)
    duplicate_image_count = sum(len(files) - 1 for files in duplicate_groups.values())
    write_reports(duplicate_groups, report_dir)

    print(f"Source: {source_dir}")
    print(f"Duplicate groups: {len(duplicate_groups)}")
    print(f"Duplicate images that can be removed: {duplicate_image_count}")
    print(f"Reports saved to: {report_dir}")

    if args.move_duplicates:
        moved = move_duplicates(duplicate_groups, trash_dir, args.merge_labels)
        print(f"Moved duplicate images: {moved}")
        print(f"Moved files saved to: {trash_dir}")
        if args.merge_labels:
            print("Merged labels into each kept txt file before moving duplicates.")
    elif args.copy_duplicates:
        copied = copy_duplicates(duplicate_groups, trash_dir)
        print(f"Copied duplicate images: {copied}")
        print(f"Copied duplicate files saved to: {trash_dir}")
    else:
        print("No files were changed. Add --move-duplicates to move duplicate files.")

    if args.dedup_output:
        copied_images = copy_dedup_dataset(source_dir, duplicate_groups, Path(args.dedup_output))
        print(f"Copied deduplicated dataset images: {copied_images}")
        print(f"Deduplicated dataset saved to: {args.dedup_output}")


if __name__ == "__main__":
    main()
