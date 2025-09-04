# 1. Gunakan base image Python 3.10 yang ringan
FROM python:3.10-slim

# 2. Set direktori kerja di dalam container
WORKDIR /app

# 3. Install library sistem yang dibutuhkan OpenCV (INI KUNCINYA)
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# 4. Salin semua file proyek (termasuk requirements.txt, .py, .pt) ke dalam container
COPY . .

# 5. Install semua library Python dari requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 6. Perintah untuk menjalankan aplikasi saat container dimulai
CMD ["python", "esp32cam.py"]