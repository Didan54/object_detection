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
WEBCAM_INDEX = 0  # index kamera
CHECK_INTERVAL = 5  # cek perintah tiap 5 detik

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


# === FUNGSI KAMERA ===
def capture_image_from_webcam(index=0):
    print(f"🎥 Membuka kamera index {index} ...")
    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        print("❌ Webcam tidak terdeteksi.")
        return None

    # ✅ Stabilkan kamera dengan membuang beberapa frame pertama
    warmup_frames = 20
    print(f"⏳ Menstabilkan kamera ({warmup_frames} frame)...")
    for _ in range(warmup_frames):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("❌ Gagal mengambil gambar dari webcam.")
        return None

    print("📸 Gambar berhasil diambil dari webcam (sudah stabil).")
    return frame


# === FUNGSI YOLO ===
def load_yolo_model():
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print(f"✅ Model YOLO '{YOLO_MODEL_PATH}' dimuat.")
        print(f"   Kelas: {model.names}")
        return model
    except Exception as e:
        print(f"❌ Gagal memuat model YOLO: {e}")
        exit()


def process_detection(model, image_np):
    print("🧠 Memulai proses deteksi YOLOv11...")
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
    print(f"✅ Deteksi selesai dalam {(end_yolo - start_yolo).total_seconds():.2f} detik.")
    return detected_objects, annotated, (start_yolo, end_yolo)


# === UPLOAD HASIL ===
def upload_and_save_results(supabase, image, annotated, detected_objects, user_id, time_log):
    start_upload = datetime.now()
    timestamp = datetime.now()
    filename_base = timestamp.strftime("%Y%m%d_%H%M%S")

    _, buf_orig = cv2.imencode(".jpg", image)
    _, buf_anno = cv2.imencode(".jpg", annotated)

    public_original = None
    public_annotated = None

    try:
        supabase.storage.from_(BUCKET_NAME).upload(
            f"original/{filename_base}.jpg",
            buf_orig.tobytes(),
            file_options={"content-type": "image/jpeg"}
        )
        public_original = supabase.storage.from_(BUCKET_NAME).get_public_url(
            f"original/{filename_base}.jpg"
        )

        supabase.storage.from_(BUCKET_NAME).upload(
            f"hasil_deteksi/{filename_base}.jpg",
            buf_anno.tobytes(),
            file_options={"content-type": "image/jpeg"}
        )
        public_annotated = supabase.storage.from_(BUCKET_NAME).get_public_url(
            f"hasil_deteksi/{filename_base}.jpg"
        )

        print("✅ Gambar berhasil diupload ke Supabase.")
    except Exception as e:
        print(f"⚠️ Gagal upload gambar ke storage: {e}")

    detection_counts = Counter(obj['nama'] for obj in detected_objects)

    payload = {
        "timestamp": timestamp.isoformat(),
        "image_url": public_original,
        "status": "selesai",
        "url_hasil_deteksi": public_annotated,
        "accuracy": [round(obj["akurasi"] * 100, 2) for obj in detected_objects],
        "hama_deteksi": dict(detection_counts),
        "user_id": user_id,
        "log_waktu": {
            "ambil_perintah": time_log["ambil_perintah"].isoformat(),
            "mulai_deteksi": time_log["mulai_deteksi"].isoformat(),
            "selesai_deteksi": time_log["selesai_deteksi"].isoformat(),
            "mulai_upload": start_upload.isoformat(),
            "selesai_upload": datetime.now().isoformat()
        }
    }

    try:
        supabase.table("gambar_hama").insert(payload).execute()
        print("📤 Data hasil deteksi berhasil disimpan ke Supabase.")
        return True
    except Exception as e:
        print(f"⚠️ Gagal menyimpan hasil ke database: {e}")
        return False


# === LOOP UTAMA ===
def main_loop(model):
    print("\n--- Sistem Deteksi Multi-User Aktif ---")
    supabase = initialize_supabase()

    while True:
        try:
            command = get_pending_command(supabase)
            if command and command.get("action") == "capture":
                cmd_id = command["id"]
                user_id = command["user_id"]

                time_log = {
                    "ambil_perintah": datetime.now()
                }

                print(f"🚀 Perintah 'capture' diterima dari user {user_id}")

                frame = capture_image_from_webcam(WEBCAM_INDEX)
                if frame is None:
                    print("⚠️ Kamera gagal mengambil gambar. Perintah tidak dihapus.")
                    continue

                objs, annotated, (start_det, end_det) = process_detection(model, frame)
                time_log["mulai_deteksi"] = start_det
                time_log["selesai_deteksi"] = end_det

                success = upload_and_save_results(supabase, frame, annotated, objs, user_id, time_log)

                if success:
                    delete_command(supabase, cmd_id)
                else:
                    print("⚠️ Upload gagal, perintah tidak dihapus agar bisa diproses ulang nanti.")

            else:
                print(".", end="", flush=True)

            time.sleep(CHECK_INTERVAL)

        except requests.exceptions.ConnectionError:
            print("\n🌐 Koneksi ke Supabase terputus, mencoba ulang...")
            time.sleep(10)
            supabase = initialize_supabase()
        except KeyboardInterrupt:
            print("\n🛑 Dihentikan oleh pengguna.")
            break
        except Exception as e:
            print(f"\n⚠️ Error tidak terduga: {e}")
            time.sleep(5)


# === EKSEKUSI ===
if __name__ == "__main__":
    yolo_model = load_yolo_model()
    if yolo_model:
        main_loop(yolo_model)