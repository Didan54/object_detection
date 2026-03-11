import os
import time
import threading
from datetime import datetime
from supabase import create_client, Client
import httpx
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter

# --- KONFIGURASI UTAMA ---
SUPABASE_URL = "https://wxkoqtcvkwduzfejtxib.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind4a29xdGN2a3dkdXpmZWp0eGliIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MDIzNzcyNCwiZXhwIjoyMDc1ODEzNzI0fQ.ZAJBuiPV_FsNIQwiK40wZQ1vWLXzs-fMxFTEmYJUsqc"
BUCKET_NAME = "gambar-hasil-deteksi"
YOLO_MODEL_PATH = "best_final.pt"

CHECK_INTERVAL = 5

COLOR_MAP = {
    "daun_sehat": (0, 255, 0),
    "daun_kuning": (0, 255, 255),
    "kerusakan_hama": (0, 0, 255),
    "jamur_putih": (255, 0, 255),
    "bercak_daun": (255, 255, 0)
}

# === KONEKSI SUPABASE ===
def initialize_supabase():
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print("✅ Terhubung ke Supabase.")
        return supabase
    except Exception as e:
        print("⚠️ Supabase gagal terhubung:", e)
        return None

def delete_command(supabase, command_id):
    supabase.table("commands").delete().eq("id", command_id).execute()
    print(f"🗑️ Perintah ID {command_id} berhasil dihapus ✅")

def get_pending_command(supabase):
    try:
        resp = supabase.table("commands").select("*").order("created_at").limit(1).execute()
        return resp.data[0] if resp.data else None
    except Exception:
        return None
    
# === CAMERA ===
class CameraStream:
    def __init__(self, index=0):
        print(f"🎥 Membuka dan menginisialisasi kamera (Index: {index})...")
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        
        self.ret = False
        self.frame = None
        self.running = False

        if not self.cap.isOpened():
            print("❌ Kamera tidak ditemukan atau sedang digunakan.")
            return

        # --- PENGATURAN HARDWARE MANUAL ---
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 3)

        print("   Stabilkan kamera (warm-up)...")
        for _ in range(30):
            self.cap.read()
            time.sleep(0.02)

        self.ret, self.frame = self.cap.read()
        self.running = True

        # Memulai thread terpisah khusus untuk streaming kamera
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        print("🎥 Kamera berhasil berjalan (Standby) ✅")

    def update(self):
        # Terus perbarui frame dari kamera selama program jalan
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        # Ambil frame yang paling baru saat dipanggil
        if self.frame is not None:
            return self.ret, self.frame.copy()
        return self.ret, None

    def stop(self):
        # Menghentikan kamera saat program dimatikan
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        self.cap.release()
        print("🛑 Kamera dihentikan dengan aman.")

# === YOLO MODEL ===
def load_yolo_model():
    return YOLO(YOLO_MODEL_PATH)

def process_detection(model, image):
    start = datetime.now()
    results = model.predict(image, conf=0.5, verbose=False)
    end = datetime.now()

    objects = []
    annotated = image.copy()

    for box in results[0].boxes:
        cid = int(box.cls[0])
        cname = model.names.get(cid, "unknown")
        acc = float(box.conf[0])
        objects.append({"nama": cname, "akurasi": acc})

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = COLOR_MAP.get(cname, (255,255,255))
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
        cv2.putText(annotated, f"{cname} ({acc*100:.1f}%)",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    print(f"✅ Deteksi selesai {(end-start).total_seconds():.2f}s")
    return objects, annotated, start, end

# === UPLOAD DATA ===
def upload_and_save_results(supabase, img, anno, objects, user_id, time_log):
    ts = datetime.now()
    fname = ts.strftime("%Y%m%d_%H%M%S")

    _, b1 = cv2.imencode(".jpg", img)
    _, b2 = cv2.imencode(".jpg", anno)

    supabase.storage.from_(BUCKET_NAME).upload(f"original/{fname}.jpg", b1.tobytes())
    supabase.storage.from_(BUCKET_NAME).upload(f"hasil_deteksi/{fname}.jpg", b2.tobytes())

    url_orig = supabase.storage.from_(BUCKET_NAME).get_public_url(f"original/{fname}.jpg")
    url_anno = supabase.storage.from_(BUCKET_NAME).get_public_url(f"hasil_deteksi/{fname}.jpg")

    counts = Counter(o["nama"] for o in objects)

    payload = {
        "timestamp": ts.isoformat(),
        "image_url": url_orig,
        "url_hasil_deteksi": url_anno,
        "accuracy": [round(o["akurasi"]*100,2) for o in objects],
        "hama_deteksi": dict(counts),
        "status": "selesai",
        "user_id": user_id,
        "log_waktu": time_log
    }

    supabase.table("gambar_hama").insert(payload).execute()
    print("📤 Hasil deteksi disimpan ✅")

# === LOOP UTAMA ===
def main_loop():
    supabase = initialize_supabase()
    model = load_yolo_model()
    
    cam = CameraStream(index=1)
    
    if not cam.running:
        print("Sistem berhenti karena kamera gagal dimuat.")
        return

    while True:
        try:
            cmd = get_pending_command(supabase)
            if cmd and cmd.get("action") == "capture":
                uid = cmd["user_id"]
                cid = cmd["id"]
                print(f"\n🚀 Perintah dari user {uid}")

                time_log = {
                    "ambil_perintah": datetime.now().isoformat()
                }

                ret, frame = cam.read()
                if not ret or frame is None:
                    print("❌ Gagal menangkap gambar dari stream.")
                    continue
                print("📸 Gambar berhasil diambil dari stream ✅")

                objs, anno, t1, t2 = process_detection(model, frame)
                time_log["mulai_deteksi"] = t1.isoformat()
                time_log["selesai_deteksi"] = t2.isoformat()
                time_log["mulai_upload"] = datetime.now().isoformat()

                upload_and_save_results(supabase, frame, anno, objs, uid, time_log)

                time_log["selesai_upload"] = datetime.now().isoformat()

                delete_command(supabase, cid)

            else:
                print(".", end="", flush=True)

            time.sleep(CHECK_INTERVAL)

        except httpx.ConnectError:
            print("\n🌐 Koneksi internet putus! Reconnect...")
            supabase = initialize_supabase()
            time.sleep(5)

        except KeyboardInterrupt:
            print("\n🛑 Sistem dihentikan oleh pengguna ✅")
            break

        except Exception as e:
            print("\n⚠ ERROR:", e)
            time.sleep(3)
            
    cam.stop()

if __name__ == "__main__":
    main_loop()