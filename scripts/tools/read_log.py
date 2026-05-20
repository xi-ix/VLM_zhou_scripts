import re
import os

def format_time(total_seconds):
    """将秒数转换为 MM:SS 格式"""
    minutes, seconds = divmod(int(total_seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"

def process_traffic_file(file_path, frame_interval=2.0):
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return

    raw_seconds = []
    
    # 1. 加载并提取数据
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # 提取文件名中的秒数 (例如 00016.00s.jpg -> 16.0)
            match = re.search(r'(\d+\.?\d*)s\.jpg', line)
            if match:
                raw_seconds.append(float(match.group(1)))

    if not raw_seconds:
        print("日志中未发现有效数据。")
        return

    # 排序确保逻辑正确
    raw_seconds.sort()

    # 2. 合并区间逻辑
    intervals = []
    start_t = raw_seconds[0]
    prev_t = raw_seconds[0]
    tolerance = frame_interval * 1.1 # 容错处理

    for i in range(1, len(raw_seconds)):
        curr_t = raw_seconds[i]
        if curr_t - prev_t <= tolerance:
            prev_t = curr_t
        else:
            intervals.append((start_t, prev_t))
            start_t = curr_t
            prev_t = curr_t
    
    intervals.append((start_t, prev_t))
    intervals.sort(key=lambda x: x[1] - x[0], reverse=True)
    # 3. 格式化输出
    print(f"{'开始时间':^8} | {'结束时间':^8} | {'持续时长'}")
    print("-" * 35)
    for s, e in intervals:
        duration = int(e - s)
        
        # 使用自定义的 MM:SS 格式
        print(f"{format_time(s):^12} | {format_time(e):^12} | {format_time(duration):>3}")

# 调用
process_traffic_file('log/01010000111000000.mp4.txt')