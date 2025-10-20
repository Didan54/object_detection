import os
import time
from datetime import datetime
from supabase import create_client, Client
import requests
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter

# --- KONFIGURASI ---
SUPABASE_URL = "https://wxkoqtcvkwduzfejtxib.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind4a29xdGN2a3dkdXpmZWp0eGliIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MDIzNzcyNCwiZXhwIjoyMDc1ODEzNzI0fQ.ZAJBuiPV_FsNIQwiK40wZQ1vWLXzs-fMxFTEmYJUsqc"
BUCKET_NAME = "gambar-hasil-deteksi"
YOLO_MODEL_PATH = "best .pt"
OUTPUT_FOLDER = "hasil_deteksi"

# --- WARNA UNTUK MASING-MASING KELAS ---
COLOR_MAP = {
    "daun_sehat": (0, 255, 0),       # Hijau
    "daun_kuning": (0, 255, 255),    # Kuning
    "kerusakan_hama": (0, 0, 255),   # Merah
    "jamur_putih": (255, 0, 255),    # Ungu
    "bercak_daun": (255, 255, 0)     # Biru muda
}

# --- FUNGSI ---
def initialize_supabase():
    try:
        supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print("\n-> Koneksi baru ke Supabase berhasil dibuat.")
        return supabase_client
    except Exception as e:
        print(f"\n!!! Gagal terhubung ke Supabase: {e}")
        return None


def load_yolo_model():
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print(f"-> Model YOLO '{YOLO_MODEL_PATH}' berhasil dimuat.")
        print(f"-> Nama Kelas Model: {model.names}")
        return model
    except Exception as e:
        print(f"!!! Gagal memuat model YOLO: {e}")
        exit()


def process_detection(model, image_np):
    """Menjalankan deteksi dan memberi warna label sesuai kelas"""
    print("   -> Menjalankan deteksi...")
    results = model.predict(image_np, conf=0.5, verbose=False)

    detected_objects = []
    annotated_image = image_np.copy()

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        detected_objects.append({"nama": class_name, "akurasi": confidence})

        # Koordinat bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Warna kelas (default putih kalau tidak ada)
        color = COLOR_MAP.get(class_name, (255, 255, 255))

        # Gambar kotak
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)

        # Tulis label di atas kotak
        label = f"{class_name} ({confidence*100:.1f}%)"
        cv2.putText(
            annotated_image, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
        )

    return detected_objects, annotated_image


# --- LOOP UTAMA ---
def main_loop(model):
    print("\n--- Server sedang berjalan, memeriksa data baru setiap 5 detik ---")

    supabase = initialize_supabase()

    while True:
        job_id = None
        try:
            if supabase is None:
                print("\nKoneksi Supabase hilang, mencoba menghubungkan kembali...")
                time.sleep(10)
                supabase = initialize_supabase()
                if supabase is None:
                    continue

            response = (
                supabase.table('gambar_hama')
                .select('*')
                .eq('status', 'baru')
                .order('timestamp', desc=False)
                .limit(1)
                .execute()
            )

            if not response.data:
                print(".", end="", flush=True)
                time.sleep(5)
                continue

            job = response.data[0]
            job_id = job['id']
            image_url = job['image_url']

            print(f"\n[POLLING] Data baru ditemukan (ID: {job_id}).")
            supabase.table('gambar_hama').update({'status': 'memproses'}).eq('id', job_id).execute()

            img_response = requests.get(image_url, timeout=20)
            img_response.raise_for_status()
            image_np = cv2.imdecode(np.frombuffer(img_response.content, np.uint8), cv2.IMREAD_COLOR)
            if image_np is None:
                raise Exception("Gagal decode gambar.")

            hasil_deteksi_mentah, annotated_img = process_detection(model, image_np)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"{timestamp}_detected_{job_id}.jpg"
            result_path = os.path.join(OUTPUT_FOLDER, result_filename)
            cv2.imwrite(result_path, annotated_img)

            with open(result_path, 'rb') as f:
                supabase.storage.from_(BUCKET_NAME).upload(
                    f"hasil_deteksi/{result_filename}", f,
                    file_options={"content-type": "image/jpeg"}
                )

            if os.path.exists(result_path):
                os.remove(result_path)

            public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(
                f"hasil_deteksi/{result_filename}"
            )

            detection_counts = Counter(obj['nama'] for obj in hasil_deteksi_mentah)

            hama_deteksi = {nama: jumlah for nama, jumlah in detection_counts.items()}

            update_payload = {
                'status': 'selesai',
                'waktu_proses': datetime.now().isoformat(),
                'url_hasil_deteksi': public_url,
                'accuracy': [round(obj["akurasi"] * 100, 2) for obj in hasil_deteksi_mentah],
                'hama_deteksi': hama_deteksi
            }

            supabase.table('gambar_hama').update(update_payload).eq('id', job_id).execute()
            print(f"   -> Hasil monitoring ID {job_id} berhasil disimpan: {dict(detection_counts)}")

        except requests.exceptions.ConnectionError as e:
            print(f"\n!!! Terjadi error koneksi: {e}. Koneksi akan direset.")
            supabase = None
        except Exception as e:
            print(f"\n!!! Terjadi error saat polling: {e}")
            if job_id:
                try:
                    print("   -> Update status 'gagal'...")
                    temp_supabase = initialize_supabase()
                    if temp_supabase:
                        temp_supabase.table('gambar_hama').update({
                            'status': 'gagal',
                            'hama_deteksi': {'error': str(e)}
                        }).eq('id', job_id).execute()
                        print("   -> Status 'gagal' berhasil diupdate.")
                except Exception as update_error:
                    print(f"   -> Gagal update status error ke Supabase: {update_error}")

        time.sleep(5)


# --- EKSEKUSI ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    yolo_model = load_yolo_model()
    if yolo_model:
        main_loop(yolo_model)
