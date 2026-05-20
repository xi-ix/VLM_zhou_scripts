from __future__ import annotations

import csv
import subprocess
from pathlib import Path


root_path = Path("/data3/VLA_LLM_DATA")
path_list_file = Path("/home/wangzhe/VLM/scripts/video_path_list.txt")
output_csv_file = Path("/home/wangzhe/VLM/scripts/video_duration_summary.csv")
video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".dav", ".flv", ".ts", ".m4v"}


def iter_video_files(root: Path):
    for file_path in root.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            yield file_path


def get_video_duration_seconds(video_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"failed to read duration: {video_path}")

    output = result.stdout.strip()
    if not output:
        raise RuntimeError(f"empty duration returned by ffprobe: {video_path}")
    return float(output)


def format_seconds(total_seconds: float) -> str:
    total_seconds_int = int(round(total_seconds))
    hours, remainder = divmod(total_seconds_int, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_target_paths(list_file: Path) -> list[Path]:
    if not list_file.exists():
        raise FileNotFoundError(f"path list file not found: {list_file}")

    targets: list[Path] = []
    for raw_line in list_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        targets.append(candidate if candidate.is_absolute() else root_path / candidate)
    return targets


def stat_single_path(target_path: Path) -> tuple[int, float, list[Path]]:
    if not target_path.exists():
        raise FileNotFoundError(f"path not found: {target_path}")

    video_files = sorted(iter_video_files(target_path))
    total_duration_seconds = 0.0
    failed_files: list[Path] = []

    for video_file in video_files:
        try:
            total_duration_seconds += get_video_duration_seconds(video_file)
        except Exception:
            failed_files.append(video_file)

    return len(video_files), total_duration_seconds, failed_files


def main() -> None:
    target_paths = load_target_paths(path_list_file)
    if not target_paths:
        raise ValueError(f"no target paths found in {path_list_file}")

    csv_rows: list[dict[str, str]] = []
    total_video_count = 0
    total_duration_seconds = 0.0
    total_failed_count = 0

    for index, target_path in enumerate(target_paths, start=1):
        video_count, duration_seconds, failed_files = stat_single_path(target_path)
        total_video_count += video_count
        total_duration_seconds += duration_seconds
        total_failed_count += len(failed_files)

        csv_rows.append(
            {
                "index": str(index),
                "path": str(target_path),
                "video_count": str(video_count),
                "total_duration_seconds": f"{duration_seconds:.3f}",
                "total_duration_hms": format_seconds(duration_seconds),
                "failed_count": str(len(failed_files)),
                "failed_files": " | ".join(str(failed_file) for failed_file in failed_files),
            }
        )

        print(f"[{index}/{len(target_paths)}] path: {target_path}")
        print(f"video_count: {video_count}")
        print(f"total_duration_hms: {format_seconds(duration_seconds)}")
        if failed_files:
            print(f"failed_count: {len(failed_files)}")
            print("failed_files:")
            for failed_file in failed_files[:20]:
                print(f"  {failed_file}")
        print()

    print("summary:")
    print(f"path_list_file: {path_list_file}")
    print(f"target_count: {len(target_paths)}")
    print(f"total_video_count: {total_video_count}")
    print(f"total_duration_hms: {format_seconds(total_duration_seconds)}")
    print(f"total_failed_count: {total_failed_count}")

    with output_csv_file.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "index",
                "path",
                "video_count",
                "total_duration_seconds",
                "total_duration_hms",
                "failed_count",
                "failed_files",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

        writer.writerow(
            {
                "index": "summary",
                "path": str(path_list_file),
                "video_count": str(total_video_count),
                "total_duration_seconds": f"{total_duration_seconds:.3f}",
                "total_duration_hms": format_seconds(total_duration_seconds),
                "failed_count": str(total_failed_count),
                "failed_files": "",
            }
        )

    print(f"csv_output: {output_csv_file}")


if __name__ == "__main__":
    main()