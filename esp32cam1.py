# import os
# import json
# import time
# from datetime import datetime
# from supabase import create_client
# import requests
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from collections import Counter
# import websocket
# import threading

# # --- KONFIGURASI ---
# SUPABASE_URL = "https://wxkoqtcvkwduzfejtxib.supabase.co"
# SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind4a29xdGN2a3dkdXpmZWp0eGliIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MDIzNzcyNCwiZXhwIjoyMDc1ODEzNzI0fQ.ZAJBuiPV_FsNIQwiK40wZQ1vWLXzs-fMxFTEmYJUsqc"
# BUCKET_NAME = "gambar-hasil-deteksi"
# YOLO_MODEL_PATH = "best .pt"
# OUTPUT_FOLDER = "hasil_deteksi"

# # --- INISIALISASI ---
# supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
# model = YOLO(YOLO_MODEL_PATH)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # --- FUNGSI PENDUKUNG ---
# def process_image(job):
#     """Proses gambar saat ada insert baru."""
#     try:
#         job_id = job["id"]
#         image_url = job["image_url"]
#         print(f"\n📸 Gambar baru diterima (ID {job_id})")

#         # Update status ke 'memproses'
#         supabase.table("gambar_hama").update({"status": "memproses"}).eq("id", job_id).execute()

#         img_response = requests.get(image_url, timeout=20)
#         img_response.raise_for_status()
#         image_np = cv2.imdecode(np.frombuffer(img_response.content, np.uint8), cv2.IMREAD_COLOR)

#         results = model.predict(image_np, conf=0.5, verbose=False)
#         annotated_img = results[0].plot(font_size=15)

#         detected_objects = [
#             {"nama": model.names[int(b.cls[0])], "akurasi": float(b.conf[0])}
#             for b in results[0].boxes
#         ]
#         detection_counts = Counter(obj["nama"] for obj in detected_objects)

#         # Simpan hasil gambar
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"hasil_{timestamp}_{job_id}.jpg"
#         local_path = os.path.join(OUTPUT_FOLDER, filename)
#         cv2.imwrite(local_path, annotated_img)

#         # Upload ke storage
#         with open(local_path, "rb") as f:
#             supabase.storage.from_(BUCKET_NAME).upload(f"hasil_deteksi/{filename}", f, file_options={"content-type": "image/jpeg"})

#         public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(f"hasil_deteksi/{filename}")

#         # Update hasil
#         update_payload = {
#             "status": "selesai",
#             "url_hasil_deteksi": public_url,
#             "waktu_proses": datetime.now().isoformat(),
#             "accuracy": [round(o["akurasi"] * 100, 2) for o in detected_objects],
#             "hama_deteksi": dict(detection_counts)
#         }
#         supabase.table("gambar_hama").update(update_payload).eq("id", job_id).execute()
#         print(f"✅ Hasil deteksi disimpan untuk ID {job_id}: {detection_counts}")

#         os.remove(local_path)
#     except Exception as e:
#         print(f"❌ Error saat memproses gambar: {e}")
#         supabase.table("gambar_hama").update({"status": "gagal"}).eq("id", job["id"]).execute()

# # --- HANDLER REALTIME ---
# def on_message(ws, message):
#     data = json.loads(message)
#     if "event" in data and data["event"] == "INSERT":
#         payload = data.get("payload", {})
#         record = payload.get("record", {})
#         if record.get("status") == "baru":
#             process_image(record)

# def on_error(ws, error):
#     print("⚠️ WebSocket error:", error)

# def on_close(ws, close_status_code, close_msg):
#     print("🔌 WebSocket terputus:", close_msg)
#     # otomatis reconnect
#     time.sleep(5)
#     start_websocket()

# def on_open(ws):
#     print("✅ Realtime WebSocket terhubung!")
#     # join channel realtime tabel gambar_hama
#     join_msg = json.dumps({
#         "topic": "realtime:public:gambar_hama",
#         "event": "phx_join",
#         "payload": {},
#         "ref": "1"
#     })
#     ws.send(join_msg)
#     print("📡 Subscribed ke tabel 'gambar_hama'")

# # --- STARTING FUNCTION ---
# def start_websocket():
#     realtime_url = f"wss://wxkoqtcvkwduzfejtxib.supabase.co/realtime/v1/websocket?apikey={SUPABASE_SERVICE_KEY}&vsn=1.0.0"
#     ws = websocket.WebSocketApp(
#         realtime_url,
#         on_message=on_message,
#         on_error=on_error,
#         on_close=on_close,
#         on_open=on_open
#     )
#     threading.Thread(target=ws.run_forever, daemon=True).start()

# # --- MAIN ---
# if __name__ == "__main__":
#     print("🚀 Menjalankan deteksi realtime Supabase...")
#     start_websocket()

#     while True:
#         time.sleep(10)  # menjaga script tetap hidup
