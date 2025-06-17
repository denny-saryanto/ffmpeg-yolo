import cv2
import subprocess
from ultralytics import YOLO

# === [Konfigurasi] ===
source = 0  # Gunakan 0 untuk webcam lokal, atau "rtsp://..." untuk IP cam
model_path = "yolov8n.pt"
output_hls_path = "stream/output.m3u8"

# === [Inisialisasi Model] ===
model = YOLO(model_path)

# === [Inisialisasi Kamera] ===
cap = cv2.VideoCapture(source)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

# === [Command FFMPEG untuk HLS] ===
ffmpeg_cmd = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f'{width}x{height}',
    '-r', str(fps),
    '-i', '-',
    '-c:v', 'libx264',
    '-preset', 'veryfast',
    '-f', 'hls',
    '-hls_time', '1',
    '-hls_list_size', '5',
    '-hls_flags', 'delete_segments',
    output_hls_path
]

# === [Start FFMPEG Process] ===
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# === [Loop Deteksi dan Streaming] ===
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi menggunakan YOLOv8
        results = model(frame, verbose=False)[0]

        # Gambar bounding box
        annotated_frame = results.plot()

        # Kirim ke FFMPEG
        ffmpeg_proc.stdin.write(annotated_frame.tobytes())

except KeyboardInterrupt:
    print("Interrupted.")

finally:
    cap.release()
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
