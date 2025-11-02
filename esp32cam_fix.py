import os
import time
from datetime import datetime
from supabase import create_client, Client
import requests
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter

# --- KONFIGURASI UTAMA ---
SUPABASE_URL = "https://wxkoqtcvkwduzfejtxib.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind4a29xdGN2a3dkdXpmZWp0eGliIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MDIzNzcyNCwiZXhwIjoyMDc1ODEzNzI0fQ.ZAJBuiPV_FsNIQwiK40wZQ1vWLXzs-fMxFTEmYJUsqc"
BUCKET_NAME = "gambar-hasil-deteksi"
YOLO_MODEL_PATH = "best.pt"

# --- KONFIGURASI PERANGKAT ---
WEBCAM_INDEX = 0
CHECK_INTERVAL = 5  
DEFAULT_DISTANCE_M = 1.0  # ✅ jarak default untuk uji (ubah saat pengujian)

# --- WARNA UNTUK KELAS DETEKSI ---
COLOR_MAP = {
    "daun_sehat": (0, 255, 0),
    "daun_kuning": (0, 255, 255),
    "kerusakan_hama": (0, 0, 255),
    "jamur_putih": (255, 0, 255),
    "bercak_daun": (255, 255, 0)
}


# === FUNGSI SUPABASE ===
def initialize_supabase():
    try:
        client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print("✅ Terhubung ke Supabase.")
        return client
    except Exception as e:
        print(f"❌ Gagal terhubung ke Supabase: {e}")
        return None


def delete_command(supabase, command_id):
    try:
        supabase.table("commands").delete().eq("id", command_id).execute()
        print(f"🗑️ Perintah ID {command_id} dihapus dari database.")
    except Exception as e:
        print(f"⚠️ Gagal menghapus perintah: {e}")


def get_pending_command(supabase):
    try:
        resp = (
            supabase.table("commands")
            .select("id, action, user_id, created_at")
            .order("created_at", desc=False)
            .limit(1)
            .execute()
        )
        if resp.data:
            cmd = resp.data[0]
            print(f"\n📩 Perintah baru ditemukan: {cmd}")
            return cmd
        return None
    except Exception as e:
        print(f"⚠️ Gagal membaca perintah: {e}")
        return None


# === SET FOKUS BERDASARKAN JARAK ===
def set_focus_for_distance(cap, distance_m):
    if distance_m <= 0.8:
        focus_val = 10
    elif distance_m <= 1.5:
        focus_val = 30
    elif distance_m <= 2.5:
        focus_val = 45
    else:
        focus_val = 60  # jarak jauh, daun kecil

    cap.set(cv2.CAP_PROP_FOCUS, focus_val)
    print(f"🎯 Fokus diset {focus_val} untuk jarak {distance_m}m")


# === FUNGSI KAMERA ===
def capture_image_from_webcam(index=0, width=1280, height=720, distance_m=DEFAULT_DISTANCE_M):
    print(f"🎥 Membuka kamera index {index} ...")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ Webcam tidak terdeteksi.")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    set_focus_for_distance(cap, distance_m)

    warmup_frames = 30
    print(f"⏳ Menstabilkan kamera ({warmup_frames} frame)...")
    for _ in range(warmup_frames):
        cap.read()
        time.sleep(0.02)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Gagal mengambil gambar.")
        return None

    print("📸 Gambar berhasil diambil (stabil dan fokus).")
    return frame


# === FUNGSI YOLO ===
def load_yolo_model():
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print("✅ Model YOLO siap.")
        return model
    except Exception as e:
        print(f"❌ Model gagal dimuat: {e}")
        exit()


def process_detection(model, image_np):
    print("🧠 Deteksi dimulai...")
    start_yolo = datetime.now()
    results = model.predict(image_np, conf=0.5, verbose=False)
    end_yolo = datetime.now()

    detected_objects = []
    annotated = image_np.copy()

    if results and results[0].boxes:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names.get(class_id, "unknown")
            confidence = float(box.conf[0])
            detected_objects.append({"nama": class_name, "akurasi": confidence})

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = COLOR_MAP.get(class_name, (255, 255, 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} ({confidence*100:.1f}%)"
            cv2.putText(annotated, label, (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    print(f"✅ Deteksi selesai dalam {(end_yolo - start_yolo).total_seconds():.2f}s")
    return detected_objects, annotated, (start_yolo, end_yolo)


# === UPLOAD DAN SIMPAN DATA ===
def upload_and_save_results(supabase, image, annotated, detected_objects, user_id, time_log):
    start_upload = datetime.now()
    timestamp = datetime.now()
    filename = timestamp.strftime("%Y%m%d_%H%M%S")

    _, buf_orig = cv2.imencode(".jpg", image)
    _, buf_anno = cv2.imencode(".jpg", annotated)

    try:
        supabase.storage.from_(BUCKET_NAME).upload(f"original/{filename}.jpg", buf_orig.tobytes())
        supabase.storage.from_(BUCKET_NAME).upload(f"hasil_deteksi/{filename}.jpg", buf_anno.tobytes())

        url_orig = supabase.storage.from_(BUCKET_NAME).get_public_url(f"original/{filename}.jpg")
        url_anno = supabase.storage.from_(BUCKET_NAME).get_public_url(f"hasil_deteksi/{filename}.jpg")

        print("✅ Upload gambar berhasil.")
    except Exception as e:
        print("⚠ Upload gagal:", e)
        return False

    detection_counts = Counter(obj["nama"] for obj in detected_objects)

    payload = {
        "timestamp": timestamp.isoformat(),
        "image_url": url_orig,
        "url_hasil_deteksi": url_anno,
        "accuracy": [round(obj["akurasi"] * 100, 2) for obj in detected_objects],
        "hama_deteksi": dict(detection_counts),
        "status": "selesai",
        "user_id": user_id,
        "waktu_proses": {
            "ambil_perintah": time_log["ambil_perintah"].isoformat(),
            "mulai_deteksi": time_log["mulai_deteksi"].isoformat(),
            "selesai_deteksi": time_log["selesai_deteksi"].isoformat(),
            "mulai_upload": start_upload.isoformat(),
            "selesai_upload": datetime.now().isoformat()
        }
    }

    supabase.table("gambar_hama").insert(payload).execute()
    print("📤 Data deteksi disimpan.")
    return True


# === LOOP ===
def main_loop(model):
    print("\n🔄 Sistem Deteksi Siap Berjalan 🔄")
    supabase = initialize_supabase()

    while True:
        try:
            cmd = get_pending_command(supabase)
            if cmd and cmd.get("action") == "capture":
                user_id = cmd["user_id"]
                cmd_id = cmd["id"]

                print(f"\n🚀 Mulai proses user {user_id}")

                time_log = {"ambil_perintah": datetime.now()}

                frame = capture_image_from_webcam(WEBCAM_INDEX, distance_m=DEFAULT_DISTANCE_M)
                if frame is None:
                    continue

                objs, annotated, det_time = process_detection(model, frame)
                time_log["mulai_deteksi"] = det_time[0]
                time_log["selesai_deteksi"] = det_time[1]

                if upload_and_save_results(supabase, frame, annotated, objs, user_id, time_log):
                    delete_command(supabase, cmd_id)

            else:
                print(".", end="", flush=True)

            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            print("\n⚠ Error:", e)
            time.sleep(5)


if __name__ == "__main__":
    model = load_yolo_model()
    main_loop(model)