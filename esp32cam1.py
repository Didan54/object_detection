import os
import cv2
import glob
from collections import Counter
from datetime import datetime
from ultralytics import YOLO

# --- KONFIGURASI (SESUAIKAN) ---
YOLO_MODEL_PATH = "best.pt" # Pastikan nama file model .pt Anda benar
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output_uji_coba" # Folder output berbeda agar tidak tercampur
CONFIDENCE_THRESHOLD = 0.5 # Minimal tingkat kepercayaan untuk menampilkan deteksi (misal 0.5 = 50%)

# --- WARNA UNTUK MASING-MASING KELAS (BGR Format) ---
# Pastikan urutan dan nama kelas ini sesuai dengan model Anda!
# Cek urutan ID dari file data.yaml Anda (ID 0, ID 1, dst.)
# Contoh (Sesuaikan dengan kelas Anda):
# ID 0: 'bercak_daun'
# ID 1: 'daun_kuning'
# ID 2: 'daun_sehat'
# ID 3: 'jamur_putih'
# ID 4: 'kerusakan_hama'
COLOR_MAP = {
    0: (255, 255, 0),    # Biru muda untuk ID 0 (bercak_daun)
    1: (0, 255, 255),    # Kuning untuk ID 1 (daun_kuning)
    2: (0, 255, 0),      # Hijau untuk ID 2 (daun_sehat)
    3: (255, 0, 255),    # Ungu untuk ID 3 (jamur_putih)
    4: (0, 0, 255)       # Merah untuk ID 4 (kerusakan_hama)
}
DEFAULT_COLOR = (255, 255, 255) # Putih jika ID kelas tidak ditemukan

# --- FUNGSI ---

def load_yolo_model():
    """Memuat model YOLO."""
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"❌ File model '{YOLO_MODEL_PATH}' tidak ditemukan!")
        print("   Pastikan file model (.pt) ada di folder yang sama.")
        return None
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print(f"✅ Model YOLO '{YOLO_MODEL_PATH}' berhasil dimuat.")
        print(f"   Nama Kelas Model: {model.names}")
        return model
    except Exception as e:
        print(f"❌ Gagal memuat model YOLO: {e}")
        return None

def test_single_image(model, image_path):
    """Memproses satu gambar untuk deteksi dan menyimpan hasilnya."""
    print(f"\n🔍 Menguji gambar: {os.path.basename(image_path)}")
    
    image_np = cv2.imread(image_path)
    if image_np is None:
        print("   ❌ Gagal membaca gambar.")
        return

    # Jalankan prediksi
    results = model.predict(image_np, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    # Salin gambar asli untuk dianotasi
    annotated_image = image_np.copy()
    detected_classes = []

    if results and results[0].boxes:
        boxes = results[0].boxes
        for box in boxes:
            # Dapatkan data deteksi
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = model.names.get(class_id, "Unknown")
            confidence = float(box.conf[0])
            
            detected_classes.append(class_name) # Kumpulkan nama kelas yang terdeteksi
            
            # Pilih warna berdasarkan class_id
            color = COLOR_MAP.get(class_id, DEFAULT_COLOR)
            
            label = f"{class_name} {confidence:.2f}"
            
            # Gambar kotak
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Gambar teks (ukuran font lebih kecil)
            font_scale = 0.5
            font_thickness = 1
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(annotated_image, (x1, y1 - h - 5), (x1 + w, y1), color, -1) # Latar teks
            cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA) # Teks hitam

    # Simpan gambar hasil anotasi
    os.makedirs(OUTPUT_FOLDER, exist_ok=True) # Buat folder output jika belum ada
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.basename(image_path)
    output_filename = f"{timestamp}_TEST_{filename}"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_path, annotated_image)
    print(f"   ✅ Hasil anotasi disimpan di: {output_path}")

    # Tampilkan ringkasan deteksi di konsol
    detection_counts = Counter(detected_classes)
    print("   --- RINGKASAN DETEKSI ---")
    if not detection_counts:
        print("   Tidak ada objek terdeteksi.")
    else:
        print(f"   TOTAL DETEKSI: {len(detected_classes)}")
        for class_name, count in detection_counts.items():
            print(f"     - {class_name}: {count}")

# --- EKSEKUSI UTAMA ---
if __name__ == "__main__":
    print("=" * 50)
    print("=== SKRIP UJI COBA MODEL DETEKSI ===")
    print("=" * 50)

    # Buat folder input jika belum ada
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"📁 Folder '{INPUT_FOLDER}' telah dibuat.")
        print("   Silakan letakkan gambar (.jpg, .jpeg, .png) di dalamnya.")
        exit()

    model = load_yolo_model()

    if model:
        # Cari semua file gambar di folder input
        image_paths = glob.glob(os.path.join(INPUT_FOLDER, '*.jpg')) + \
                      glob.glob(os.path.join(INPUT_FOLDER, '*.jpeg')) + \
                      glob.glob(os.path.join(INPUT_FOLDER, '*.png'))

        if not image_paths:
            print(f"\n⚠️  Tidak ada gambar (.jpg, .jpeg, .png) di folder '{INPUT_FOLDER}'.")
        else:
            print(f"\n✅ Ditemukan {len(image_paths)} gambar. Memulai pengujian...\n")
            for img_path in image_paths:
                test_single_image(model, img_path)
                print("-" * 50)
        
        print("\n🎉 Pengujian semua gambar selesai.")