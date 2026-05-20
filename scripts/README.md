# VLM/scripts 脚本说明

## 主目录脚本

### convert_images.py
将指定目录及子目录下的所有图片格式统一转换为 JPG，转换成功后自动删除原图。支持 PNG/JPEG/BMP/WEBP/TIFF/GIF 等格式，自动处理透明背景和动态图（取第一帧）。

### count_images.py
递归统计指定目录下的图片文件数量。支持 JPG/JPEG/PNG/GIF/BMP/WEBP/SVG/TIFF 等常见图片格式。

### detect_violation.py
调用 Qwen3-VL-4B 模型，读取文件夹中的图片，识别交通违章行为（闯红灯、违停、逆行、压实线变道、占用应急车道等），将检测到违章的图片复制到输出文件夹，并记录日志。

### Qwen3-VL-4B.py
调用 Qwen3-VL-4B 模型，对视频抽帧后的图片进行交通拥堵识别。分析每帧中车辆数量、汽车占道路比例，判断是否拥堵，将拥堵帧的文件名写入日志。

### rename_images.py
将指定目录下的图片文件按序号批量重命名（如 img_001.jpg, img_002.jpg），支持自定义前缀，自动计算序号位数，防止文件名冲突和覆盖。

### spider.py
从 Bing 和百度搜索引擎批量爬取图片。内置关键词矩阵（电动车违章、行人闯红灯等交通场景），支持中英文关键词，按关键词分目录保存。

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
