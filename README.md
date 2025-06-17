# YOLOv8 Real-Time Object Detection + FFMPEG + HLS Streaming

Sistem deteksi objek real-time menggunakan [YOLOv8](https://github.com/ultralytics/ultralytics) dengan hasil streaming berbasis HLS (`.m3u8`) yang dapat diakses melalui browser atau media player seperti `ffplay`. Backend dikembangkan dengan [FastAPI](https://fastapi.tiangolo.com/) untuk penyajian stream dan API.

## Fitur Utama

- Deteksi objek secara real-time menggunakan YOLOv8
- Video hasil deteksi disalurkan ke FFmpeg untuk dikodekan ke format HLS
- HLS stream dapat diakses via browser atau `ffplay`
- Backend ringan dan cepat menggunakan FastAPI

---

## Struktur Proyek

```
ffmpeg-yolo/
├── model/ # Model YOLOv8 (.pt)
│ └── yolov8n.pt
├── stream/ # Folder output HLS (.ts dan .m3u8)
│ └── stream.m3u8
├── main.py # Script utama (YOLO + FFmpeg + FastAPI)
├── requirements.txt
└── README.md
```

---

## Instalasi

1. **Clone repo ini:**

```bash
git clone https://github.com/denny-saryanto/ffmpeg-yolo.git
cd ffmpeg-yolo
```

2. **Buat environment (opsional tapi disarankan):**

```bash
python -m venv env
.\env\Scripts\activate      # Windows
# source env/bin/activate  # Linux/Mac
```

3. **Install dependency:**

```bash
pip install -r requirements.txt
```

4. **Download model YOLOv8n:**

```bash
# Opsional: Jika belum ada
from ultralytics import YOLO
YOLO('yolov8n.pt').save('model/yolov8n.pt')
```

---

## Menjalankan Aplikasi

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Kemudian akses:

- Stream Page (via browser): [http://localhost:8000/player](http://localhost:8000/player)
- HLS `.m3u8`: [http://localhost:8000/stream/stream.m3u8](http://localhost:8000/stream/stream.m3u8)
- FFplay (optional):
  
```bash
ffplay http://localhost:8000/stream/stream.m3u8
```

---

## Konfigurasi Sumber Video

Di dalam `main.py`, ubah variabel berikut untuk sumber kamera:

```python
source = 0  # Gunakan 0 untuk webcam lokal
# atau source = "rtsp://user:pass@ip:port/stream"
```

---

## Catatan

- Format streaming: HLS (HTTP Live Streaming, berbasis `.m3u8`)
- Jalankan dengan GPU (`device='cuda'`) untuk performa optimal
- Folder `stream/` otomatis menyimpan `.ts` segmen video dan playlist HLS
- Jika buffer macet, pastikan deteksi tidak lambat atau coba model lebih ringan (mis. `yolov8n.pt`)

---