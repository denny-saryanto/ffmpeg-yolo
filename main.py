import os
import cv2
import subprocess
import threading
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

app = FastAPI()

# === [Konfigurasi] ===
source = 0  # atau 'rtsp://...'
model_path = "model/yolo11n.pt"
hls_output_path = "stream/stream.m3u8"

# === [Setup Folder Output] ===
os.makedirs("stream", exist_ok=True)
os.makedirs("model", exist_ok=True)

# === [Inisialisasi Model YOLO] ===
model = YOLO(model_path)

# === [Fungsi Streaming Deteksi] ===
def start_yolo_stream():
    cap = cv2.VideoCapture(source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    expected_bytes = width * height * 3

    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}',
        '-r', '25',
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-f', 'hls',
        '-hls_time', '1',
        '-hls_list_size', '5',
        '-hls_flags', 'delete_segments',
        hls_output_path
    ]

    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            if frame.shape[0] != height or frame.shape[1] != width:
                continue

            results = model(frame, verbose=False)[0]
            annotated = results.plot()

            if annotated.tobytes().__len__() == expected_bytes:
                ffmpeg_proc.stdin.write(annotated.tobytes())
    except Exception as e:
        print("Streaming stopped:", e)
    finally:
        cap.release()
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()

# === [Start YOLO Streaming di Background Thread] ===
threading.Thread(target=start_yolo_stream, daemon=True).start()

# === [Static Files untuk HLS Output] ===
app.mount("/stream", StaticFiles(directory="stream"), name="stream")

# === [Web Route] ===
@app.get("/")
def index():
    return {
        "message": "YOLOv8 HLS Streaming Ready",
        "hls_url": "/stream/stream.m3u8"
    }

@app.get("/player")
def player():
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLOv8 Live Stream</title>
    </head>
    <body>
        <h1>Live Stream</h1>
        <video width="640" height="480" controls autoplay>
            <source src="/stream/stream.m3u8" type="application/x-mpegURL">
        </video>
    </body>
    </html>
    """
    return html_content