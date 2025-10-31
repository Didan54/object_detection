import cv2

CAMERA_INDEX = 1  # ganti sesuai webcam kamu (1 = Logitech)

print(f"🔄 Mengembalikan pengaturan kamera index {CAMERA_INDEX} ke mode default...")

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("❌ Kamera gagal dibuka.")
    exit()

# Reset semua parameter penting ke mode otomatis / default
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)   # aktifkan auto exposure
cap.set(cv2.CAP_PROP_EXPOSURE, -4)          # biarkan sistem atur sendiri
cap.set(cv2.CAP_PROP_AUTO_WB, 1)            # aktifkan auto white balance
cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4000)  # nilai default umum
cap.set(cv2.CAP_PROP_SATURATION, 64)        # normalisasi warna
cap.set(cv2.CAP_PROP_CONTRAST, 32)          # kontras default
cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)       # kecerahan default
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)          # aktifkan autofocus
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # format normal warna

print("✅ Pengaturan kamera sudah dikembalikan ke mode warna normal.")
print("Tekan 'q' untuk keluar dari preview.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Tidak bisa membaca frame dari kamera.")
        break

    cv2.imshow("Kamera Mode Normal (Warna)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Kamera ditutup dan dikembalikan ke pengaturan semula.")
