# import os
# import cv2
# import glob
# from collections import Counter
# from datetime import datetime
# from ultralytics import YOLO

# # --- KONFIGURASI ---
# YOLO_MODEL_PATH = "best .pt"  # Ganti nama ini jika file model Anda berbeda
# INPUT_FOLDER = "input"
# OUTPUT_FOLDER = "output"

# # --- CARA PENGGUNAAN ---
# # 1. Simpan kode ini sebagai file Python (misal: test_model.py).
# # 2. Pastikan file model (best.pt) berada di folder yang sama dengan skrip ini.
# # 3. Buat folder bernama "input".
# # 4. Letakkan semua gambar yang ingin Anda uji di dalam folder "input".
# # 5. Jalankan skrip ini dari terminal: python test_model.py
# # 6. Hasil gambar akan tersimpan di folder "output".

# COLOR_PALETTE = [
#     (0, 0, 255),    # Merah
#     (0, 255, 255),  # Kuning
#     (0, 255, 0),    # Hijau
#     (255, 255, 0),  # Cyan
#     (255, 0, 255),  # Magenta
# ]

# # --- FUNGSI UTAMA ---

# def setup_folders():
#     """Membuat folder input dan output jika belum ada."""
#     os.makedirs(INPUT_FOLDER, exist_ok=True)
#     os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# def load_model():
#     """Memuat model YOLO dari path yang ditentukan."""
#     if not os.path.exists(YOLO_MODEL_PATH):
#         print(f"❌ File model '{YOLO_MODEL_PATH}' tidak ditemukan!")
#         print("   Pastikan file model (.pt) ada di folder yang sama dengan skrip ini.")
#         return None
#     try:
#         model = YOLO(YOLO_MODEL_PATH)
#         print(f"✅ Model YOLO '{YOLO_MODEL_PATH}' berhasil dimuat.")
#         print(f"   Model ini dapat mendeteksi kelas: {list(model.names.values())}")
#         return model
#     except Exception as e:
#         print(f"❌ Gagal memuat model: {e}")
#         return None

# def process_detection(model, image_path):
#     """Memproses satu gambar untuk deteksi objek."""
#     print(f"\n🔍 Memproses gambar: {os.path.basename(image_path)}")
    
#     image_np = cv2.imread(image_path)
#     if image_np is None:
#         print("   ❌ Gagal membaca gambar. Mungkin file rusak atau format tidak didukung.")
#         return

#     # Jalankan prediksi
#     results = model.predict(image_np, conf=0.5, verbose=False)
    
#     first_result = results[0]
#     annotated_image = first_result.orig_img.copy()
#     boxes = first_result.boxes

#     # --- PERUBAHAN 2: Loop ini dimodifikasi untuk menggunakan warna berbeda ---
#     for box in boxes:
#         # Dapatkan data deteksi
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         class_id = int(box.cls[0])
#         class_name = model.names[class_id]
#         confidence = float(box.conf[0])
        
#         # Pilih warna berdasarkan class_id
#         # Operator % digunakan sebagai pengaman jika jumlah kelas > jumlah warna
#         color = COLOR_PALETTE[class_id % len(COLOR_PALETTE)]
        
#         label = f"{class_name} {confidence:.2f}"
        
#         # Gambar kotak dengan warna yang sudah dipilih
#         cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        
#         # Gambar teks dengan latar belakang warna yang sama
#         font_scale = 0.5
#         font_thickness = 1
#         (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
#         cv2.rectangle(annotated_image, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
#         cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

#     # --------------------------------------------------------------------
    
#     # Hitung jumlah deteksi untuk ringkasan
#     detected_classes = [model.names[int(box.cls[0])] for box in boxes]
#     detection_counts = Counter(detected_classes)

#     # Simpan gambar hasil
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = os.path.basename(image_path)
#     output_filename = f"{timestamp}_detected_{filename}"
#     output_path = os.path.join(OUTPUT_FOLDER, output_filename)
#     cv2.imwrite(output_path, annotated_image)
#     print(f"   ✅ Hasil disimpan di: {output_path}")

#     # Tampilkan ringkasan yang rinci
#     print("   --- HASIL DETEKSI ---")
#     if not detection_counts:
#         print("   STATUS: Normal (Tidak ada objek terdeteksi)")
#     else:
#         print("   STATUS: Abnormal (Objek terdeteksi)")
#         print(f"   TOTAL DETEKSI: {len(detected_classes)}")
#         print("   RINCIAN:")
#         for class_name, count in detection_counts.items():
#             print(f"     - {class_name}: {count}")
#     # --------------------------

# def main():
#     """Fungsi utama untuk menjalankan seluruh proses."""
#     print("=" * 50)
#     print("=== PENGUJIAN MODEL DETEKSI PENYAKIT DAUN LADA ===")
#     print("=" * 50)

#     setup_folders()
#     model = load_model()

#     if model is None:
#         return

#     # PERUBAHAN: Menggunakan glob untuk mencari file gambar dengan lebih efisien
#     image_paths = glob.glob(os.path.join(INPUT_FOLDER, '*.jpg')) + \
#                   glob.glob(os.path.join(INPUT_FOLDER, '*.jpeg')) + \
#                   glob.glob(os.path.join(INPUT_FOLDER, '*.png'))

#     if not image_paths:
#         print(f"\n⚠️  Tidak ada gambar (.jpg, .jpeg, .png) di folder '{INPUT_FOLDER}'.")
#         print(f"    Silakan letakkan gambar di sana, lalu jalankan lagi.")
#     else:
#         print(f"\n✅ Ditemukan {len(image_paths)} gambar. Memulai deteksi...\n")
#         for img_path in image_paths:
#             process_detection(model, img_path)
#             print("-" * 50)
    
#     print("✅ Semua gambar telah diproses.")


# if __name__ == "__main__":
#     main()