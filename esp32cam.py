import os
import time
from datetime import datetime
from supabase import create_client, Client
import requests
import cv2
import numpy as np
from ultralytics import YOLO

# --- KONFIGURASI ---
SUPABASE_URL = "https://bvndsoxfemunepctwaag.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ2bmRzb3hmZW11bmVwY3R3YWFnIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTI4NjAyOSwiZXhwIjoyMDcwODYyMDI5fQ.yowNa-4zLJJHaGIOe8ZHhayDoqxPqRxmMRcoc29AbuU"
BUCKET_NAME = "gambar-hasil-deteksi"
YOLO_MODEL_PATH = "best.pt"
OUTPUT_FOLDER = "hasil_deteksi"

# --- INISIALISASI ---
print("--- Server Deteksi Hama (Supabase - Mode Polling) ---")
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print("-> Koneksi ke Supabase berhasil.")
except Exception as e:
    print(f"!!! Gagal terhubung ke Supabase: {e}")
    exit()

try:
    model = YOLO(YOLO_MODEL_PATH)
    print(f"-> Model YOLO '{YOLO_MODEL_PATH}' berhasil dimuat.")
except Exception as e:
    print(f"!!! Gagal memuat model YOLO: {e}")
    exit()

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- FUNGSI DETEKSI ---
def process_detection(image_np):
    print("   -> Menjalankan deteksi...")
    results = model.predict(image_np, conf=0.5, verbose=False)  # verbose=False untuk hemat memori
    detection_summary = {"jumlah_deteksi": 0, "objek": []}
    annotated_image = results[0].plot()

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        detection_summary["objek"].append({"nama": class_name, "akurasi": confidence})

    detection_summary["jumlah_deteksi"] = len(detection_summary["objek"])
    return detection_summary, annotated_image

# --- FUNGSI UTAMA ---
def main_loop():
    print("\n--- Server sedang berjalan, memeriksa data baru setiap 5 detik ---")
    
    while True:
        job_id = None  # Deklarasi di awal untuk hindari error di except
        try:
            # 1. Cek data dengan status 'baru'
            response = supabase.table('gambar_hama').select('*').eq('status', 'baru').order('timestamp', desc=False).limit(1).execute()
            
            if not response.data:
                print(".", end="", flush=True)
                time.sleep(5)
                continue

            job = response.data[0]
            job_id = job['id']
            image_url = job['image_url']
            
            print(f"\n[POLLING] Data baru ditemukan (ID: {job_id}).")
            supabase.table('gambar_hama').update({'status': 'memproses'}).eq('id', job_id).execute()
            
            # 2. Download gambar
            img_response = requests.get(image_url, timeout=20)
            img_response.raise_for_status()
            image_np = cv2.imdecode(np.frombuffer(img_response.content, np.uint8), cv2.IMREAD_COLOR)
            if image_np is None:
                raise Exception("Gagal decode gambar.")

            # 3. Jalankan deteksi
            hasil_deteksi, annotated_img = process_detection(image_np)

            # 4. Simpan gambar hasil anotasi
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"{timestamp}_detected_{job_id}.jpg"
            result_path = os.path.join(OUTPUT_FOLDER, result_filename)
            cv2.imwrite(result_path, annotated_img)
            
            # 5. Upload ke Supabase Storage
            with open(result_path, 'rb') as f:
                supabase.storage.from_(BUCKET_NAME).upload(
                    f"hasil_deteksi/{result_filename}", 
                    f, 
                    file_options={"content-type": "image/jpeg"}
                )
            
            # 6. Hapus file lokal SEGERA setelah upload
            if os.path.exists(result_path):
                os.remove(result_path)
                print(f"   -> File lokal '{result_filename}' telah dihapus dari disk.")

            # 7. Dapatkan URL publik
            public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(f"hasil_deteksi/{result_filename}")
            
            # 8. Tentukan status akhir
            status_akhir = "Normal" if hasil_deteksi["jumlah_deteksi"] == 0 else "Abnormal"
            
            # 9. Ekstrak semua akurasi (dalam %)
            daftar_akurasi = [round(objek["akurasi"] * 100, 2) for objek in hasil_deteksi["objek"]]

            # 10. Update hasil ke database
            update_payload = {
                'status': status_akhir,
                'waktu_proses': datetime.now().isoformat(),
                'url_hasil_deteksi': public_url,
                'accuracy': daftar_akurasi,
                'hama_deteksi': hasil_deteksi
            }
            supabase.table('gambar_hama').update(update_payload).eq('id', job_id).execute()
            print(f"   -> Hasil deteksi (Status: {status_akhir}, Akurasi: {daftar_akurasi}) untuk ID {job_id} berhasil disimpan.")

        except Exception as e:
            print(f"\n!!! Terjadi error saat polling: {e}")
            if job_id:
                try:
                    supabase.table('gambar_hama').update({
                        'status': 'gagal', 
                        'hama_deteksi': {'error': str(e)}
                    }).eq('id', job_id).execute()
                except:
                    print("❌ Gagal update status error ke Supabase")

        time.sleep(5)

# --- JALANKAN PROGRAM ---
if __name__ == "__main__":
    main_loop()