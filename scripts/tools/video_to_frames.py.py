import cv2
import os
# 视频抽帧
def extract_frames(video_path, output_folder, interval_seconds=1):
    print(f"viode:{video_path}")
    print(f"output:{output_folder}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("错误：无法打开视频文件。请检查路径是否正确。")
        print(video_path)
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)
    print(f"视频帧率: {fps:.2f} FPS")
    print(f"截取间隔: 每 {interval_seconds} 秒一张图 (每 {frame_interval} 帧)")
    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            current_time = frame_count / fps
            filename = f"{current_time:08.2f}s.jpg"
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, frame)
            saved_count += 1
            print(f"已保存: {filename}")
        frame_count += 1
    cap.release()
    print(f"\n处理完成！共保存了 {saved_count} 张图片。")

if __name__ == "__main__":
    folder_path = 'VLM/dataset'
    video_extensions = ('.mp4') 
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(video_extensions)]
    step = 2 
    for file_name in video_files:
        target_video =f"./VLM/dataset/{file_name}"
        output_path = f"./VLM/images/{file_name}"  
        extract_frames(target_video, output_path, step)