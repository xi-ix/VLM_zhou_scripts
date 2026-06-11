# VLM/scripts 脚本说明

## 主目录脚本

### convert_images.py
将指定目录及子目录下的所有图片格式统一转换为 JPG，转换成功后自动删除原图。支持 PNG/JPEG/BMP/WEBP/TIFF/GIF 等格式，自动处理透明背景和动态图（取第一帧）。

### count_images.py
递归统计指定目录下的图片文件数量。支持 JPG/JPEG/PNG/GIF/BMP/WEBP/SVG/TIFF 等常见图片格式。

### count_classes.py
统计 YOLO 格式 txt 标签中的类别分布。默认读取 `/data3/VLA/set`，每行标签的第一个数字作为类别 ID，输出标签文件数量、目标框总数、类别数量以及每个类别的目标框数量。

用法示例：

```bash
python3 /home/wangzhe/VLM/scripts/count_classes.py /data3/VLA/set
```

### check_class4_with_qwen_vl.py
调用本地 Qwen 视觉模型检查 YOLO 标签中类别 4 的目标，用于将“电动车不在规定车道行驶”细分为：

- `4`：电动车在机动车道行驶
- `5`：电动车在斑马线上行驶

脚本会逐个读取 class 4 目标，在图片上临时画红框后交给模型判断，并输出预测 CSV。默认不修改标签；加 `--apply` 后会将预测为 `5` 的目标对应 txt 行首从 `4` 改为 `5`，并自动备份原始 txt。加 `--no-visualization` 后不保留可视化图片，只保留 CSV 和标签备份。

测试用法：

```bash
conda run -n qwenVL python /home/wangzhe/VLM/scripts/check_class4_with_qwen_vl.py \
  --input /home/wangzhe/VLM/dataset/class_samples \
  --output /home/wangzhe/VLM/dataset/class4_qwen_check \
  --model-family qwen3-4b \
  --model-dir /home/wangzhe/VLM/model_weights/Qwen3-VL-4B-Instruct
```

正式修改标签：

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n qwenVL python /home/wangzhe/VLM/scripts/check_class4_with_qwen_vl.py \
  --input /home/wangzhe/VLM/dataset/set_dedup \
  --output /home/wangzhe/VLM/dataset/class4_qwen_check_set_dedup \
  --model-family qwen3-4b \
  --model-dir /home/wangzhe/VLM/model_weights/Qwen3-VL-4B-Instruct \
  --apply \
  --no-visualization
```

### draw_yolo_boxes.py
读取图片及其同名 YOLO txt 标签，将归一化的 `[class_id, x_center, y_center, width, height]` 框转换为像素坐标并绘制到图片上。默认输入 `/home/wangzhe/VLM/dataset/class_samples`，输出到 `/home/wangzhe/VLM/dataset/class_samples_vis`，保留原目录结构且不覆盖原图。

用法示例：

```bash
python3 /home/wangzhe/VLM/scripts/draw_yolo_boxes.py \
  --input /home/wangzhe/VLM/dataset/class_samples \
  --output /home/wangzhe/VLM/dataset/class_samples_vis
```

### find_duplicate_images.py
按图片文件内容计算 SHA256，用于查找完全重复的图片。默认只生成报告，不改动源数据；报告包括重复组 JSON、重复图片配对 CSV，以及重复图片标签差异统计。支持将重复图片及同名标签移动到暂存目录，也支持在没有源目录删除权限时复制重复项和生成去重后的数据副本。

常用用法：

```bash
# 只生成重复报告，不修改源目录
python3 /home/wangzhe/VLM/scripts/find_duplicate_images.py \
  --source /data3/VLA/set \
  --report-dir /home/wangzhe/VLM/dataset/duplicate_report

# 复制重复项到暂存目录，并生成去重后的数据副本，不修改源目录
python3 /home/wangzhe/VLM/scripts/find_duplicate_images.py \
  --source /data3/VLA/set \
  --report-dir /home/wangzhe/VLM/dataset/duplicate_report \
  --copy-duplicates \
  --trash-dir /home/wangzhe/VLM/dataset/duplicate_removed \
  --dedup-output /home/wangzhe/VLM/dataset/set_dedup

# 在源目录有写权限时，将重复项移动到暂存目录
python3 /home/wangzhe/VLM/scripts/find_duplicate_images.py \
  --source /data3/VLA/set \
  --report-dir /home/wangzhe/VLM/dataset/duplicate_report \
  --move-duplicates \
  --trash-dir /home/wangzhe/VLM/dataset/duplicate_removed
```

### rename_images.py
将指定目录下的图片文件按序号批量重命名（如 img_001.jpg, img_002.jpg），支持自定义前缀，自动计算序号位数，防止文件名冲突和覆盖。

### sample_class_images.py
从 YOLO 数据集中按类别抽样图片。默认读取 `/data3/VLA/set`，每个类别随机抽取 10 张包含该类别的图片，并将图片和同名 txt 标签复制到 `/home/wangzhe/VLM/dataset/class_samples/class_类别ID`。支持通过随机种子复现抽样结果。

用法示例：

```bash
python3 /home/wangzhe/VLM/scripts/sample_class_images.py \
  --source /data3/VLA/set \
  --output /home/wangzhe/VLM/dataset/class_samples \
  --num 10 \
  --seed 42
```

### stat_video_duration.py
统计指定路径下所有视频文件的时长。通过 ffprobe 获取视频时长，支持从配置文件读取多个目标路径，结果输出为 CSV 汇总表（含视频数量、总时长、失败文件等）。

### visualize_manifest_boxes.py
读取 manifest.jsonl 中的检测框标注信息，在对应图片上绘制边界框和标签（支持 person/bicycle/motorcycle 等类别着色），输出可视化结果图片。

---

## load_model/ — 模型加载与推理

### Qwen2.5-VL-3B.py
加载 Qwen2.5-VL-3B-Instruct 模型，对单张图片进行交通拥堵分析推理（车辆数量、占道比例、是否拥堵）。使用 bfloat16 精度，支持离线加载。

### Qwen3-VL-4B.py
加载 Qwen3-VL-4B-Instruct 模型，封装了模型加载、图片输入处理、推理响应三个核心函数。支持交通拥堵判定和拥堵原因分析两种 prompt，使用 float16 精度，开启采样模式。

---

## tools/ — 工具脚本

### draw_bbox_01.py
在图片上绘制归一化坐标的边界框。输入归一化的 [x_min, y_min, x_max, y_max] 坐标，自动转换为像素坐标，绘制红色矩形框并保存。

### fine_tune.py
使用 HuggingFace Trainer 对视觉语言模型（如 SmolVLM-256M-Instruct）进行微调。封装了 VisionTextDataset 数据集类，支持从 annotations.json 加载图文对数据，配置了完整的训练参数。

### move_video.py
从源文件夹递归扫描所有 MP4 视频文件，随机抽取指定数量的视频复制到目标文件夹。自动处理同名文件冲突，适用于从大数据集中随机选取样本。

### read_log.py
读取模型推理生成的日志文件（如拥堵帧记录），提取时间戳并合并连续区间，以表格形式输出拥堵的开始时间、结束时间和持续时长。

### select_yolo_intersection_frames.py
使用 YOLO 模型从视频数据中智能筛选交通路口帧。按路口和机位评估画面清晰度，优先选择遮挡少、目标清晰的帧，输出图片和 manifest.jsonl 标注清单。支持 YOLOv7 和 ultralytics 两种推理后端，支持 zip 内视频直接读取。

### video_to_frames.py.py
视频抽帧工具。按指定时间间隔（秒）从视频中提取帧，保存为以时间戳命名的 JPG 图片。支持批量处理文件夹下的所有视频。
