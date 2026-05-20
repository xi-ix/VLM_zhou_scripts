#!/usr/bin/env python3
"""Select ~10k traffic frames using YOLO from /data3/VLA_LLM_DATA.

Design goals:
- Pick about 10 intersections, 4 camera positions each.
- Must include video_of_intersection_footage2025-03-25.zip as one intersection.
- Prefer clearer intersections/cameras with less occlusion.
- Save outputs outside workspace (default /data3) to avoid root disk pressure.
- Avoid full zip extraction: extract one video at a time to temp and clean up.

Example:
  /home/wangzhe/miniconda3/envs/qwen_vl/bin/python \
        /home/wangzhe/VLM/scripts/tools/select_yolo_intersection_frames.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
import sys
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from ultralytics import YOLO

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".ts", ".mts", ".m2ts"}
TARGET_NAMES = {"person", "bicycle", "motorcycle"}
RIDE_VEHICLE_NAMES = {"bicycle", "motorcycle"}
MOTOR_VEHICLE_NAMES = {"car", "bus", "truck"}
RAW_KEEP_NAMES = TARGET_NAMES | MOTOR_VEHICLE_NAMES


# =========================
# Edit these defaults only.
# =========================
DEFAULT_SOURCE_ROOT = Path("/data3/VLA_LLM_DATA")
DEFAULT_INCLUDE_ZIP = Path("/data3/VLA_LLM_DATA/video_of_intersection_footage2025-03-25.zip")
DEFAULT_OUTPUT_ROOT = Path(f"/data3/wangzhe/yolo_frame_pick_{datetime.now().strftime('%Y%m%d')}")
DEFAULT_MODEL_PATH = "/home/wangzhe/VLM/model_weights/yolov7.pt"
DEFAULT_YOLOV7_REPO = Path("/data3/tools/yolov7")
DEFAULT_TMP_DIR = Path("/data3/.tmp_yolo_pick")

DEFAULT_TARGET_IMAGES = 100
DEFAULT_TARGET_INTERSECTIONS = 10
DEFAULT_CAMERAS_PER_INTERSECTION = 4
DEFAULT_ZIP_IMAGE_RATIO = 0.12
DEFAULT_MIN_IMAGES_PER_CAMERA = 1
DEFAULT_FAST_MODE = True
DEFAULT_DEVICE = "cuda:0"
DEFAULT_MAX_CHECKS_PER_VIDEO = 16
DEFAULT_STRICT_CAMERA_MIN = False
DEFAULT_PROGRESS_INTERVAL_SEC = 3.0
DEFAULT_MAX_SECONDS_PER_CAMERA = 120.0


@dataclass
class VideoRef:
    intersection: str
    camera: str
    source_type: str  # "file" | "zip"
    path: Path
    member: str | None = None


@dataclass
class CandidateFrame:
    score: float
    video: VideoRef
    frame_idx: int
    image: np.ndarray
    boxes_xyxy: list[list[float]]
    classes: list[str]
    confs: list[float]


class TerminalProgress:
    """终端轻量进度条，避免额外依赖。"""

    def __init__(self, total: int, desc: str):
        self.total = max(1, int(total))
        self.desc = desc
        self.current = 0
        self.width = 28
        self.start = time.time()

    def update(self, n: int = 1) -> None:
        self.current = min(self.total, self.current + n)
        ratio = self.current / self.total
        fill = int(self.width * ratio)
        bar = "#" * fill + "-" * (self.width - fill)
        elapsed = time.time() - self.start
        msg = f"\r[{self.desc}] {self.current}/{self.total} [{bar}] {ratio*100:5.1f}% {elapsed:5.1f}s"
        sys.stdout.write(msg)
        sys.stdout.flush()

    def close(self) -> None:
        self.update(0)
        sys.stdout.write("\n")
        sys.stdout.flush()


class LegacyYoloV7Detector:
    """YOLOv7 detector backed by local yolov7 repo code."""

    def __init__(self, weights: Path, repo_path: Path, img_size: int = 640, device: str = ""):
        """初始化 YOLOv7 本地推理后端。"""
        if not repo_path.exists():
            raise SystemExit(
                f"YOLOv7 repo not found: {repo_path}. Clone it first or pass --yolov7-repo."
            )
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        from utils.datasets import letterbox  # type: ignore
        from utils.general import check_img_size, non_max_suppression, scale_coords  # type: ignore
        from utils.torch_utils import select_device  # type: ignore

        self.letterbox = letterbox
        self.non_max_suppression = non_max_suppression
        self.scale_coords = scale_coords

        self.device = select_device(device)
        ckpt = torch.load(str(weights), map_location=self.device, weights_only=False)
        ckpt_model = ckpt["ema" if isinstance(ckpt, dict) and ckpt.get("ema") else "model"]
        self.model = ckpt_model.float().fuse().eval()
        self.model.to(self.device)
        self.stride = int(self.model.stride.max())
        self.img_size = int(check_img_size(img_size, s=self.stride))
        self.half = self.device.type != "cpu"
        if self.half:
            self.model.half()
        self.model.eval()
        self.names = self.model.module.names if hasattr(self.model, "module") else self.model.names

    def infer(self, image: np.ndarray, min_conf: float) -> tuple[np.ndarray, list[str], list[float]]:
        """执行单帧推理并返回框、类别和置信度。"""
        im0 = image
        img = self.letterbox(im0, self.img_size, stride=self.stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        tensor = torch.from_numpy(img).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        tensor /= 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(tensor, augment=False)[0]

        pred = self.non_max_suppression(pred, min_conf, 0.45, classes=None, agnostic=False)[0]
        if pred is None or len(pred) == 0:
            return np.empty((0, 4), dtype=np.float32), [], []

        pred[:, :4] = self.scale_coords(tensor.shape[2:], pred[:, :4], im0.shape).round()
        boxes = pred[:, :4].detach().cpu().numpy().astype(np.float32)
        confs = pred[:, 4].detach().cpu().numpy().astype(float).tolist()
        cls_ids = pred[:, 5].detach().cpu().numpy().astype(int)
        classes = [self.names[int(c)] if int(c) < len(self.names) else str(int(c)) for c in cls_ids]
        return boxes, classes, confs


def parse_args() -> argparse.Namespace:
    """解析命令行参数与默认配置。"""
    parser = argparse.ArgumentParser(description="YOLO-based frame picker for intersections.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--include-zip",
        type=Path,
        default=DEFAULT_INCLUDE_ZIP,
        help="The must-include zip intersection path.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Local YOLO weight path. Default uses local yolov7.pt to avoid downloads.",
    )
    parser.add_argument("--target-images", type=int, default=DEFAULT_TARGET_IMAGES)
    parser.add_argument("--target-intersections", type=int, default=DEFAULT_TARGET_INTERSECTIONS)
    parser.add_argument("--cameras-per-intersection", type=int, default=DEFAULT_CAMERAS_PER_INTERSECTION)
    parser.add_argument(
        "--zip-image-ratio",
        type=float,
        default=DEFAULT_ZIP_IMAGE_RATIO,
        help="Fraction of target images allocated to include-zip intersection if selected.",
    )
    parser.add_argument("--eval-frames-per-camera", type=int, default=24)
    parser.add_argument("--sample-stride", type=int, default=45, help="Frame stride for candidate extraction.")
    parser.add_argument("--min-conf", type=float, default=0.25)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--seed", type=int, default=20260401)
    parser.add_argument("--max-videos-per-camera", type=int, default=18)
    parser.add_argument(
        "--max-checks-per-video",
        type=int,
        default=DEFAULT_MAX_CHECKS_PER_VIDEO,
        help="Max sampled frames to test per video in fast collection.",
    )
    parser.add_argument(
        "--progress-interval-sec",
        type=float,
        default=DEFAULT_PROGRESS_INTERVAL_SEC,
        help="How often to print per-camera progress logs.",
    )
    parser.add_argument(
        "--max-seconds-per-camera",
        type=float,
        default=DEFAULT_MAX_SECONDS_PER_CAMERA,
        help="Max processing time per camera before skipping to next one.",
    )
    parser.add_argument("--min-images-per-camera", type=int, default=DEFAULT_MIN_IMAGES_PER_CAMERA)
    parser.add_argument(
        "--strict-camera-min",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_STRICT_CAMERA_MIN,
        help="Fail job when a camera cannot provide min images.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Inference device, e.g. cuda:0 / 0 / cpu.",
    )
    parser.add_argument(
        "--fast-mode",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FAST_MODE,
        help="Use faster intersection selection and lighter sampling.",
    )
    parser.add_argument("--tmp-dir", type=Path, default=DEFAULT_TMP_DIR)
    parser.add_argument(
        "--max-eval-intersections",
        type=int,
        default=0,
        help="Limit number of directory intersections for occlusion evaluation. 0 means no limit.",
    )
    parser.add_argument(
        "--yolov7-repo",
        type=Path,
        default=DEFAULT_YOLOV7_REPO,
        help="Local YOLOv7 repo path used when loading yolov7*.pt weights.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--start-index",
        type=int,
        default=-1,
        help="Starting filename index; -1 means auto-resume from existing images.",
    )
    return parser.parse_args()


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个框的 IoU 重叠比。"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def compute_occlusion_score(boxes: np.ndarray, classes: list[str]) -> float:
    """计算画面清晰度分数：目标多且重叠少时分数更高。"""
    if boxes.shape[0] == 0:
        return 0.0
    target_idx = [i for i, c in enumerate(classes) if c in TARGET_NAMES]
    if not target_idx:
        return 0.0

    target_boxes = boxes[target_idx]
    n = target_boxes.shape[0]

    overlap_penalty = 0.0
    overlap_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            iou = iou_xyxy(target_boxes[i], target_boxes[j])
            if iou > 0.1:
                overlap_pairs += 1
                overlap_penalty += iou

    avg_area = float(np.mean([max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1])) + 1e-6 for b in target_boxes]))

    # 边缘附近目标更容易被树木/灯杆等遮挡，用边缘惩罚降低优先级。
    x_min = np.min(target_boxes[:, 0])
    y_min = np.min(target_boxes[:, 1])
    x_max = np.max(target_boxes[:, 2])
    y_max = np.max(target_boxes[:, 3])
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    edge_touch = 0
    for b in target_boxes:
        if b[0] <= x_min + 0.02 * w or b[1] <= y_min + 0.02 * h or b[2] >= x_max - 0.02 * w or b[3] >= y_max - 0.02 * h:
            edge_touch += 1

    density = float(n)
    area_bonus = min(2.0, math.log1p(avg_area / 2500.0))
    edge_penalty = edge_touch / max(1.0, float(n))

    score = density + area_bonus - 2.5 * overlap_pairs - 4.0 * overlap_penalty - 1.5 * edge_penalty
    return float(max(0.0, score))


def postprocess_target_detections(
    boxes: np.ndarray,
    classes: list[str],
    confs: list[float],
) -> tuple[np.ndarray, list[str], list[float]]:
    """后处理检测结果，尽量保留道路参与者有效目标。"""
    if boxes.shape[0] == 0:
        return boxes, classes, confs

    keep_mask = [True] * len(classes)

    person_idxs = [i for i, c in enumerate(classes) if c == "person"]
    ride_idxs = [i for i, c in enumerate(classes) if c in RIDE_VEHICLE_NAMES]
    motor_idxs = [i for i, c in enumerate(classes) if c in MOTOR_VEHICLE_NAMES]

    for pi in person_idxs:
        p_box = boxes[pi]
        p_area = max(0.0, float(p_box[2] - p_box[0])) * max(0.0, float(p_box[3] - p_box[1]))
        if p_area <= 1.0:
            continue

        p_cx = (float(p_box[0]) + float(p_box[2])) / 2.0
        p_cy = (float(p_box[1]) + float(p_box[3])) / 2.0

        # Rule 1: rider suppression. If person strongly overlaps bike/motorcycle,
        # drop person and keep the vehicle class as the representative label.
        is_rider = False
        for vi in ride_idxs:
            v_box = boxes[vi]
            x1 = max(float(p_box[0]), float(v_box[0]))
            y1 = max(float(p_box[1]), float(v_box[1]))
            x2 = min(float(p_box[2]), float(v_box[2]))
            y2 = min(float(p_box[3]), float(v_box[3]))
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            overlap_ratio_on_person = inter / max(1.0, p_area)
            if overlap_ratio_on_person >= 0.25 or iou_xyxy(p_box, v_box) >= 0.15:
                is_rider = True
                break

        if is_rider:
            keep_mask[pi] = False
            continue

        # Rule 2: in-vehicle suppression for car/bus/truck passengers.
        for vi in motor_idxs:
            v_box = boxes[vi]
            x1 = max(float(p_box[0]), float(v_box[0]))
            y1 = max(float(p_box[1]), float(v_box[1]))
            x2 = min(float(p_box[2]), float(v_box[2]))
            y2 = min(float(p_box[3]), float(v_box[3]))
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            overlap_ratio_on_person = inter / max(1.0, p_area)
            if float(v_box[0]) <= p_cx <= float(v_box[2]) and float(v_box[1]) <= p_cy <= float(v_box[3]) and overlap_ratio_on_person >= 0.5:
                keep_mask[pi] = False
                break

    filtered_boxes: list[np.ndarray] = []
    filtered_classes: list[str] = []
    filtered_confs: list[float] = []

    for i, keep in enumerate(keep_mask):
        if not keep:
            continue
        cls_name = classes[i]
        if cls_name not in TARGET_NAMES:
            continue
        filtered_boxes.append(boxes[i])
        filtered_classes.append(cls_name)
        filtered_confs.append(confs[i])

    if not filtered_boxes:
        return np.empty((0, 4), dtype=np.float32), [], []

    return np.stack(filtered_boxes).astype(np.float32), filtered_classes, filtered_confs


def is_video_file(p: Path) -> bool:
    """判断文件是否为支持的视频格式。"""
    return p.is_file() and p.suffix.lower() in VIDEO_EXTS


def camera_key_from_name(name: str) -> str:
    """从文件名提取机位编号键。"""
    # Prefer D1/D2... prefix style.
    head = Path(name).name.split("_")[0]
    return head


def discover_dir_intersections(source_root: Path) -> dict[str, dict[str, list[VideoRef]]]:
    """扫描目录型路口数据并按机位聚合视频。"""
    out: dict[str, dict[str, list[VideoRef]]] = {}
    for inter_dir in sorted([p for p in source_root.iterdir() if p.is_dir()]):
        cam_map: dict[str, list[VideoRef]] = defaultdict(list)
        cam_dirs = [p for p in inter_dir.iterdir() if p.is_dir()]
        for cam_dir in cam_dirs:
            cam_name = cam_dir.name
            for v in cam_dir.rglob("*"):
                if is_video_file(v):
                    cam_map[cam_name].append(
                        VideoRef(intersection=inter_dir.name, camera=cam_name, source_type="file", path=v)
                    )
        cam_map = {k: sorted(v, key=lambda x: str(x.path)) for k, v in cam_map.items() if v}
        if cam_map:
            out[inter_dir.name] = cam_map
    return out


def discover_zip_intersection(zip_path: Path) -> tuple[str, dict[str, list[VideoRef]]]:
    """扫描 zip 型路口数据并按机位聚合视频。"""
    inter_name = zip_path.stem
    cam_map: dict[str, list[VideoRef]] = defaultdict(list)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            member = info.filename
            if Path(member).suffix.lower() not in VIDEO_EXTS:
                continue
            cam = camera_key_from_name(member)
            cam_map[cam].append(
                VideoRef(intersection=inter_name, camera=cam, source_type="zip", path=zip_path, member=member)
            )
    cam_map = {k: sorted(v, key=lambda x: x.member or "") for k, v in cam_map.items() if v}
    return inter_name, cam_map


def open_video_capture(vref: VideoRef, tmp_dir: Path) -> tuple[cv2.VideoCapture, Path | None]:
    """打开视频源；zip 内视频会先临时解压再读取。"""
    if vref.source_type == "file":
        cap = cv2.VideoCapture(str(vref.path))
        return cap, None

    if vref.member is None:
        raise RuntimeError("zip video missing member")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(vref.member).suffix or ".mp4"
    tmp_file = Path(tempfile.mkstemp(prefix="zip_video_", suffix=suffix, dir=str(tmp_dir))[1])

    with zipfile.ZipFile(vref.path) as zf, zf.open(vref.member) as src, open(tmp_file, "wb") as dst:
        shutil.copyfileobj(src, dst)

    cap = cv2.VideoCapture(str(tmp_file))
    return cap, tmp_file


def yolo_detect(model: object, image: np.ndarray, min_conf: float) -> tuple[np.ndarray, list[str], list[float]]:
    """执行检测并过滤为行人/骑行相关目标。"""
    if hasattr(model, "infer"):
        boxes, classes, confs = model.infer(image, min_conf)  # type: ignore[attr-defined]
    else:
        run_device = getattr(model, "_run_device", None)  # type: ignore[attr-defined]
        kwargs = {"device": run_device} if run_device else {}
        result = model.predict(image, conf=min_conf, verbose=False, **kwargs)[0]  # type: ignore[attr-defined]
        if result.boxes is None or len(result.boxes) == 0:
            return np.empty((0, 4), dtype=np.float32), [], []

        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy().astype(float).tolist()
        names = model.names  # type: ignore[attr-defined]
        classes = [names.get(int(c), str(c)) if hasattr(names, "get") else str(int(c)) for c in cls_ids]

    keep = [i for i, name in enumerate(classes) if name in RAW_KEEP_NAMES]
    if not keep:
        return np.empty((0, 4), dtype=np.float32), [], []

    boxes = boxes[keep]
    classes = [classes[i] for i in keep]
    confs = [confs[i] for i in keep]
    return postprocess_target_detections(boxes, classes, confs)


def evaluate_camera_occlusion(
    model: object,
    videos: list[VideoRef],
    min_conf: float,
    eval_frames_per_camera: int,
    tmp_dir: Path,
    rng: random.Random,
) -> float:
    """评估机位质量分数，分数越高表示画面更清晰。"""
    if not videos:
        return 0.0

    picked = videos[:]
    rng.shuffle(picked)
    picked = picked[: min(3, len(picked))]

    scores: list[float] = []
    per_video_samples = max(4, math.ceil(eval_frames_per_camera / max(1, len(picked))))

    for vref in picked:
        cap = None
        tmp_file = None
        try:
            cap, tmp_file = open_video_capture(vref, tmp_dir)
            if not cap.isOpened():
                continue
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                continue
            indices = list(range(total)) if total <= per_video_samples else [min(total - 1, int(i * total / per_video_samples)) for i in range(per_video_samples)]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                boxes, classes, _ = yolo_detect(model, frame, min_conf)
                if boxes.shape[0] == 0:
                    continue
                scores.append(compute_occlusion_score(boxes, classes))
        except Exception:
            continue
        finally:
            if cap is not None:
                cap.release()
            if tmp_file is not None and tmp_file.exists():
                tmp_file.unlink(missing_ok=True)

    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def select_intersections(
    model: object,
    dir_intersections: dict[str, dict[str, list[VideoRef]]],
    zip_intersection: tuple[str, dict[str, list[VideoRef]]] | None,
    cameras_per_intersection: int,
    target_intersections: int,
    min_conf: float,
    eval_frames_per_camera: int,
    tmp_dir: Path,
    rng: random.Random,
    max_eval_intersections: int,
    fast_mode: bool,
) -> dict[str, dict[str, list[VideoRef]]]:
    """按机位质量选择路口，并尽量保证每个路口有4个可用机位。"""
    scored: list[tuple[float, str, dict[str, list[VideoRef]]]] = []

    # Evaluate directory intersections.
    dir_items = list(dir_intersections.items())
    if max_eval_intersections > 0:
        dir_items = dir_items[:max_eval_intersections]

    if fast_mode:
        # 快速模式：按视频量近似筛选，跳过逐帧评估。
        for inter_name, cams in dir_items:
            if len(cams) < cameras_per_intersection:
                continue
            cam_counts = sorted(((len(vs), cam) for cam, vs in cams.items()), reverse=True)
            picked = cam_counts[:cameras_per_intersection]
            if len(picked) < cameras_per_intersection:
                continue
            inter_score = float(sum(c for c, _ in picked))
            picked_cams = {cam: cams[cam] for _, cam in picked}
            scored.append((inter_score, inter_name, picked_cams))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected: dict[str, dict[str, list[VideoRef]]] = {}
        if zip_intersection is not None:
            z_name, z_cams = zip_intersection
            if len(z_cams) >= cameras_per_intersection:
                z_counts = sorted(((len(vs), cam) for cam, vs in z_cams.items()), reverse=True)
                z_names = [cam for _, cam in z_counts[:cameras_per_intersection]]
                selected[z_name] = {k: z_cams[k] for k in z_names}

        for _, inter_name, cams in scored:
            if len(selected) >= target_intersections:
                break
            if inter_name in selected:
                continue
            selected[inter_name] = cams
        return selected

    progress = TerminalProgress(total=max(1, len(dir_items)), desc="SelectIntersections")

    for inter_name, cams in dir_items:
        progress.update(1)
        if len(cams) < cameras_per_intersection:
            continue
        cam_scores: list[tuple[float, str]] = []
        for cam in sorted(cams.keys()):
            cam_score = evaluate_camera_occlusion(
                model, cams[cam], min_conf, eval_frames_per_camera, tmp_dir, rng
            )
            if cam_score > 0:
                cam_scores.append((cam_score, cam))

        if len(cam_scores) < cameras_per_intersection:
            continue

        cam_scores.sort(key=lambda x: x[0], reverse=True)
        picked = cam_scores[:cameras_per_intersection]
        inter_score = float(sum(s for s, _ in picked) / len(picked))
        picked_cams = {cam: cams[cam] for _, cam in picked}
        scored.append((inter_score, inter_name, picked_cams))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected: dict[str, dict[str, list[VideoRef]]] = {}

    if zip_intersection is not None:
        z_name, z_cams = zip_intersection
        if len(z_cams) >= cameras_per_intersection:
            z_cam_scores: list[tuple[float, str]] = []
            for cam in sorted(z_cams.keys()):
                cam_score = evaluate_camera_occlusion(
                    model, z_cams[cam], min_conf, eval_frames_per_camera, tmp_dir, rng
                )
                if cam_score > 0:
                    z_cam_scores.append((cam_score, cam))

            if len(z_cam_scores) >= cameras_per_intersection:
                z_cam_scores.sort(key=lambda x: x[0], reverse=True)
                z_cam_names = [cam for _, cam in z_cam_scores[:cameras_per_intersection]]
                selected[z_name] = {k: z_cams[k] for k in z_cam_names}

    progress.close()

    for _, inter_name, cams in scored:
        if len(selected) >= target_intersections:
            break
        if inter_name in selected:
            continue
        selected[inter_name] = cams

    return selected


def collect_frames_for_camera(
    model: object,
    videos: list[VideoRef],
    target_n: int,
    sample_stride: int,
    min_conf: float,
    max_videos_per_camera: int,
    tmp_dir: Path,
    rng: random.Random,
    min_required: int = DEFAULT_MIN_IMAGES_PER_CAMERA,
    max_checks_per_video: int = DEFAULT_MAX_CHECKS_PER_VIDEO,
    fast_mode: bool = True,
    camera_tag: str = "",
    progress_interval_sec: float = DEFAULT_PROGRESS_INTERVAL_SEC,
    max_seconds_per_camera: float = DEFAULT_MAX_SECONDS_PER_CAMERA,
) -> list[CandidateFrame]:
    """按清晰优先收集机位帧，不足时自动放宽阈值补样本。"""
    candidates: list[CandidateFrame] = []
    seen_keys: set[tuple[str, str | None, int]] = set()
    checked = 0
    started = time.time()
    last_log = started

    video_list = videos[:]
    rng.shuffle(video_list)
    video_list = video_list[: min(max_videos_per_camera, len(video_list))]

    if fast_mode:
        passes = [(max(1, sample_stride), max(0.16, min_conf - 0.06))]
    else:
        passes = [
            (sample_stride, min_conf),
            (max(10, sample_stride // 2), max(0.18, min_conf - 0.08)),
            (max(6, sample_stride // 3), max(0.12, min_conf - 0.12)),
        ]

    for cur_stride, cur_conf in passes:
        for vref in video_list:
            if time.time() - started >= max(5.0, max_seconds_per_camera):
                break
            cap = None
            tmp_file = None
            try:
                cap, tmp_file = open_video_capture(vref, tmp_dir)
                if not cap.isOpened():
                    continue
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    continue

                if fast_mode:
                    checks = min(max_checks_per_video, total)
                    if checks <= 0:
                        continue
                    step = max(1, total // checks)
                    indices = [min(total - 1, i * step) for i in range(checks)]
                else:
                    indices = list(range(0, total, cur_stride))

                if fast_mode:
                    for idx in indices:
                        if time.time() - started >= max(5.0, max_seconds_per_camera):
                            break
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            continue

                        key = (str(vref.path), vref.member, idx)
                        if key not in seen_keys:
                            seen_keys.add(key)
                            boxes, classes, confs = yolo_detect(model, frame, cur_conf)
                            checked += 1
                            if boxes.shape[0] != 0:
                                score = compute_occlusion_score(boxes, classes)
                                if score > 0:
                                    candidates.append(
                                        CandidateFrame(
                                            score=score,
                                            video=vref,
                                            frame_idx=idx,
                                            image=frame,
                                            boxes_xyxy=boxes.tolist(),
                                            classes=classes,
                                            confs=confs,
                                        )
                                    )

                        now = time.time()
                        if now - last_log >= max(0.5, progress_interval_sec):
                            tag = camera_tag or "camera"
                            print(
                                f"[PROGRESS] {tag}: checked={checked}, selected={len(candidates)}/{target_n}, elapsed={now-started:.1f}s"
                            )
                            last_log = now
                        if len(candidates) >= target_n:
                            break
                else:
                    for idx in indices:
                        if time.time() - started >= max(5.0, max_seconds_per_camera):
                            break
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            break

                        key = (str(vref.path), vref.member, idx)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)

                        boxes, classes, confs = yolo_detect(model, frame, cur_conf)
                        checked += 1
                        if boxes.shape[0] == 0:
                            continue
                        score = compute_occlusion_score(boxes, classes)
                        if score <= 0:
                            continue
                        candidates.append(
                            CandidateFrame(
                                score=score,
                                video=vref,
                                frame_idx=idx,
                                image=frame,
                                boxes_xyxy=boxes.tolist(),
                                classes=classes,
                                confs=confs,
                            )
                        )
                        if len(candidates) >= target_n:
                            break
            except Exception:
                continue
            finally:
                if cap is not None:
                    cap.release()
                if tmp_file is not None and tmp_file.exists():
                    tmp_file.unlink(missing_ok=True)

            if len(candidates) >= target_n:
                break
            if time.time() - started >= max(5.0, max_seconds_per_camera):
                break

        if len(candidates) >= max(target_n, min_required):
            break
        if time.time() - started >= max(5.0, max_seconds_per_camera):
            break

    if not fast_mode:
        candidates.sort(key=lambda x: x.score, reverse=True)
    tag = camera_tag or "camera"
    if time.time() - started >= max(5.0, max_seconds_per_camera):
        print(f"[WARN] {tag}: timeout reached ({max_seconds_per_camera:.0f}s), skip with {len(candidates)} selected")
    print(f"[PROGRESS] {tag}: done, checked={checked}, selected={len(candidates)}/{target_n}, elapsed={time.time()-started:.1f}s")
    return candidates[:target_n]


def write_frame(
    c: CandidateFrame,
    out_img_path: Path,
    jpeg_quality: int,
) -> bool:
    """将候选帧按 JPEG 写入磁盘。"""
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_img_path), c.image, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    return bool(ok)


def main() -> None:
    """主流程：选路口、抽帧、写出图片和清单。"""
    args = parse_args()
    rng = random.Random(args.seed)

    if args.target_images <= 0:
        raise SystemExit("--target-images must be positive")
    if args.min_images_per_camera <= 0:
        raise SystemExit("--min-images-per-camera must be positive")
    if not (0.0 <= args.zip_image_ratio <= 0.8):
        raise SystemExit("--zip-image-ratio must be in [0, 0.8]")

    if not args.source_root.exists():
        raise SystemExit(f"source root not found: {args.source_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"Model file not found: {model_path}. Please pass a local weight path via --model."
        )

    print(f"[INFO] Loading YOLO model: {model_path}")
    if "yolov7" in model_path.name.lower():
        print(f"[INFO] Using YOLOv7 legacy backend from: {args.yolov7_repo}")
        legacy_device = args.device.strip().lower()
        if legacy_device.startswith("cuda:"):
            legacy_device = legacy_device.split(":", 1)[1]
        if legacy_device == "cuda":
            legacy_device = "0"
        model = LegacyYoloV7Detector(weights=model_path, repo_path=args.yolov7_repo, device=legacy_device)
        print(f"[INFO] Inference device: {model.device} (requested={args.device})")
    else:
        model = YOLO(str(model_path))
        run_device = args.device
        if run_device.startswith("cuda") and not torch.cuda.is_available():
            run_device = "cpu"
        setattr(model, "_run_device", run_device)
        print(f"[INFO] Inference device: {run_device}")

    print("[INFO] Discovering intersections...")
    dir_intersections = discover_dir_intersections(args.source_root)
    zip_intersection = None
    if args.include_zip.exists():
        try:
            zip_intersection = discover_zip_intersection(args.include_zip)
            print(f"[INFO] Include zip intersection: {zip_intersection[0]}, cameras={len(zip_intersection[1])}")
        except Exception as e:
            print(f"[WARN] Failed to parse include zip {args.include_zip}: {e}")
    else:
        print(f"[WARN] include zip not found: {args.include_zip}")

    print("[INFO] Evaluating clarity and selecting intersections...")
    selected = select_intersections(
        model=model,
        dir_intersections=dir_intersections,
        zip_intersection=zip_intersection,
        cameras_per_intersection=args.cameras_per_intersection,
        target_intersections=args.target_intersections,
        min_conf=args.min_conf,
        eval_frames_per_camera=args.eval_frames_per_camera,
        tmp_dir=args.tmp_dir,
        rng=rng,
        max_eval_intersections=args.max_eval_intersections,
        fast_mode=args.fast_mode,
    )

    if not selected:
        raise SystemExit("No intersections selected. Please relax constraints.")

    inter_names = sorted(selected.keys())
    n_inter = len(inter_names)
    zip_inter_name = args.include_zip.stem if args.include_zip.exists() else None

    intersection_targets: dict[str, int] = {}
    if zip_inter_name is not None and zip_inter_name in inter_names and n_inter > 1:
        zip_target = max(1, int(round(args.target_images * args.zip_image_ratio)))
        zip_target = min(zip_target, args.target_images - (n_inter - 1))
        remain = args.target_images - zip_target
        others = [name for name in inter_names if name != zip_inter_name]
        base_other = max(1, remain // max(1, len(others)))

        for name in others:
            intersection_targets[name] = base_other
        used_other = base_other * len(others)
        leftover = remain - used_other
        for name in others[:leftover]:
            intersection_targets[name] += 1
        intersection_targets[zip_inter_name] = zip_target
    else:
        base = max(1, args.target_images // max(1, n_inter))
        for name in inter_names:
            intersection_targets[name] = base
        used = base * n_inter
        leftover = args.target_images - used
        for name in inter_names[:max(0, leftover)]:
            intersection_targets[name] += 1

    camera_targets: dict[tuple[str, str], int] = {}
    for inter in inter_names:
        cams = selected[inter]
        cam_names = sorted(cams.keys())[: args.cameras_per_intersection]
        inter_target = intersection_targets[inter]
        base_cam = max(1, inter_target // max(1, len(cam_names)))
        for cam in cam_names:
            camera_targets[(inter, cam)] = base_cam
        used_cam = base_cam * len(cam_names)
        cam_leftover = inter_target - used_cam
        for cam in cam_names[:max(0, cam_leftover)]:
            camera_targets[(inter, cam)] += 1

    total_target = int(sum(camera_targets.values()))

    print(f"[INFO] Selected intersections ({n_inter}): {inter_names}")
    if zip_inter_name is not None and zip_inter_name in intersection_targets:
        print(
            f"[INFO] include-zip target: {intersection_targets[zip_inter_name]} "
            f"({args.zip_image_ratio:.0%} requested ratio)"
        )
    print(f"[INFO] planned total: {total_target}")

    summary = {
        "config": {
            "source_root": str(args.source_root),
            "include_zip": str(args.include_zip),
            "output_root": str(args.output_root),
            "model": args.model,
            "target_images": args.target_images,
            "zip_image_ratio": args.zip_image_ratio,
            "min_conf": args.min_conf,
            "sample_stride": args.sample_stride,
            "seed": args.seed,
        },
        "selected_intersections": inter_names,
        "intersection_targets": intersection_targets,
        "camera_targets": [
            {"intersection": k[0], "camera": k[1], "target": v}
            for k, v in sorted(camera_targets.items(), key=lambda x: (x[0][0], x[0][1]))
        ],
        "records": [],
    }

    manifest_path = args.output_root / "manifest.jsonl"
    summary_path = args.output_root / "summary.json"
    images_root = args.output_root / "images"

    # Auto resume numbering from existing flat image filenames like 000123.jpg.
    start_index = args.start_index
    if start_index < 0:
        existing_max = -1
        if images_root.exists():
            for p in images_root.glob("*.jpg"):
                stem = p.stem
                if stem.isdigit():
                    existing_max = max(existing_max, int(stem))
        start_index = existing_max + 1

    if args.dry_run:
        summary["dry_run"] = True
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Dry run only. Summary written to: {summary_path}")
        return

    written = start_index
    run_written = 0
    total_cams = sum(len(selected[i]) for i in inter_names)
    cam_progress = TerminalProgress(total=max(1, total_cams), desc="CollectFrames")
    manifest_mode = "a" if manifest_path.exists() and start_index > 0 else "w"
    with open(manifest_path, manifest_mode, encoding="utf-8") as mf:
        for inter in inter_names:
            cams = selected[inter]
            cam_names = sorted(cams.keys())[: args.cameras_per_intersection]
            for cam in cam_names:
                cam_progress.update(1)
                cam_target = camera_targets.get((inter, cam), 0)
                if cam_target <= 0:
                    continue
                print(f"[INFO] Collecting {inter}/{cam}...")
                candidates = collect_frames_for_camera(
                    model=model,
                    videos=cams[cam],
                    target_n=cam_target,
                    sample_stride=args.sample_stride,
                    min_conf=args.min_conf,
                    max_videos_per_camera=args.max_videos_per_camera,
                    tmp_dir=args.tmp_dir,
                    rng=rng,
                    min_required=args.min_images_per_camera,
                    max_checks_per_video=args.max_checks_per_video,
                    fast_mode=args.fast_mode,
                    camera_tag=f"{inter}/{cam}",
                    progress_interval_sec=args.progress_interval_sec,
                    max_seconds_per_camera=args.max_seconds_per_camera,
                )
                if len(candidates) < args.min_images_per_camera and args.strict_camera_min:
                    raise SystemExit(
                        f"Camera has insufficient valid frames: {inter}/{cam}, got={len(candidates)}, "
                        f"required={args.min_images_per_camera}."
                    )

                for i, c in enumerate(candidates):
                    file_name = f"{written:06d}.jpg"
                    out_img_path = args.output_root / "images" / file_name
                    ok = write_frame(c, out_img_path, args.jpeg_quality)
                    if not ok:
                        continue

                    rec = {
                        "image_path": str(out_img_path),
                        "intersection": inter,
                        "camera": cam,
                        "score": c.score,
                        "source_type": c.video.source_type,
                        "source_path": str(c.video.path),
                        "source_member": c.video.member,
                        "source_frame_idx": c.frame_idx,
                        "classes": c.classes,
                        "confs": c.confs,
                        "boxes_xyxy": c.boxes_xyxy,
                    }
                    mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    summary["records"].append(
                        {
                            "image_path": str(out_img_path),
                            "intersection": inter,
                            "camera": cam,
                            "score": c.score,
                        }
                    )
                    written += 1
                    run_written += 1

    cam_progress.close()

    summary["start_index"] = start_index
    summary["written_images_this_run"] = run_written
    summary["written_images_total"] = written
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] written images this run: {run_written}")
    print(f"[DONE] written images total: {written}")
    print(f"[DONE] manifest: {manifest_path}")
    print(f"[DONE] summary: {summary_path}")


if __name__ == "__main__":
    main()
