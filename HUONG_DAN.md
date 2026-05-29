# BeeGuard Jetson Nano — Hướng dẫn cài đặt

## Tổng quan

Hệ thống chạy **headless** (không GUI). Cắm nguồn Jetson = tự chạy.
Điều khiển hoàn toàn qua app BeeGuard (Flutter) thông qua Firebase.

### Phần cứng cần có
| Thiết bị | Ghi chú |
|---|---|
| Jetson Nano A02 4GB + quạt tản nhiệt | **BẮT BUỘC** có quạt |
| Thẻ microSD 64GB+ (Class A2) | Khuyến nghị 128GB |
| Nguồn 5V/4A barrel jack | Không dùng Micro USB |
| Logitech C310 | USB camera |
| ESP32 + servo + pump + sensors | USB Serial |
| Cáp Ethernet hoặc USB WiFi | Để kết nối internet |

---

## Bước 1: Flash JetPack OS

> JetPack 4.6.x là bản cuối cùng hỗ trợ Jetson Nano A02.

1. Tải **JetPack 4.6.4 SD Card Image**:
   ```
   https://developer.nvidia.com/embedded/jetpack-sdk-464
   ```

2. Cài **balenaEtcher** trên PC Windows:
   ```
   https://etcher.balena.io/
   ```

3. Mở balenaEtcher → chọn file `.zip` vừa tải → chọn thẻ SD → Flash.

4. Cắm thẻ SD vào Jetson, kết nối: nguồn 5V/4A, màn hình HDMI, bàn phím USB, cáp mạng.

5. Bật nguồn → Jetson boot vào setup wizard:
   - Chọn ngôn ngữ, timezone
   - Tạo username và password (nhớ username, sẽ dùng nhiều)
   - Chọn **MAXN** power mode khi được hỏi
   - Đợi quá trình cài đặt hoàn tất → reboot

---

## Bước 2: Cấu hình cơ bản

Mở Terminal (Ctrl+Alt+T) và chạy lần lượt:

### 2.1 Cập nhật hệ thống
```bash
sudo apt update && sudo apt upgrade -y
```

### 2.2 Tạo swap 4GB (rất quan trọng với 4GB RAM)
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 2.3 Bật hiệu năng cao
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

### 2.4 Cài công cụ
```bash
sudo apt install -y python3-pip python3-venv git nano curl
```

### 2.5 Kiểm tra CUDA
```bash
nvcc --version
```
Kết quả phải hiện `Cuda compilation tools, release 10.2`.

---

## Bước 3: Clone code từ GitHub

```bash
cd ~
git clone https://github.com/napvip/beeguard_jetson.git beeguard
cd beeguard
```

> Nếu repo là private, cần tạo Personal Access Token trên GitHub:
> Settings → Developer settings → Personal access tokens → Generate new token.
> Khi clone, nhập token thay cho password.

---

## Bước 4: Copy file bí mật từ PC

Có 2 file không được push lên Git (chứa mật khẩu). Cần copy thủ công.

### Cách 1: Dùng USB flash drive

1. Trên PC Windows, copy 2 file sau vào USB:
   ```
   C:\Users\16toa\Downloads\tracking_v11\.env
   C:\Users\16toa\Downloads\tracking_v11\service_account.json
   ```

2. Cắm USB vào Jetson Nano.

3. Trên Jetson, mở Terminal:
   ```bash
   # Tìm USB (thường là /media/<username>/tên_usb)
   ls /media/$USER/

   # Copy vào dự án
   cp /media/$USER/<tên_usb>/.env ~/beeguard/
   cp /media/$USER/<tên_usb>/service_account.json ~/beeguard/
   ```

### Cách 2: Dùng SCP qua mạng LAN

Trên PC Windows (PowerShell), chạy:
```powershell
# Tìm IP Jetson (trên Jetson chạy: hostname -I)
scp "C:\Users\16toa\Downloads\tracking_v11\.env" user@<IP_JETSON>:~/beeguard/
scp "C:\Users\16toa\Downloads\tracking_v11\service_account.json" user@<IP_JETSON>:~/beeguard/
```
Thay `user` bằng username đã tạo ở Bước 1, thay `<IP_JETSON>` bằng IP thực tế.

### Kiểm tra
```bash
ls -la ~/beeguard/.env ~/beeguard/service_account.json
```
Phải thấy cả 2 file.

---

## Bước 5: Cài môi trường Python

### 5.1 Tạo virtual environment
```bash
cd ~/beeguard
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install --upgrade pip
```

### 5.2 Cài PyTorch cho Jetson Nano
```bash
# Cài đặt các thư viện toán học và OpenMPI (bắt buộc để load PyTorch)
sudo apt update && sudo apt install -y libopenmpi-dev libopenblas-dev liblapack-dev libblas-dev

# Tải PyTorch 1.10 (build sẵn cho JetPack 4.6)
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl \
     -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
```

### 5.3 Cài torchvision từ source
```bash
# Cài đặt Pillow 8.4.0 (bản cuối cùng tương thích Python 3.6 của Jetson) để tránh lỗi cú pháp Pillow mới
pip install Pillow==8.4.0

sudo apt install -y libjpeg-dev zlib1g-dev libpython3-dev \
     libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
python3 setup.py install
cd ~/beeguard
```

### 5.4 Kiểm tra PyTorch + CUDA
```bash
python3 -c "import torch; import torchvision; print('PyTorch:', torch.__version__, '| Torchvision:', torchvision.__version__, '| CUDA:', torch.cuda.is_available())"
```
Kết quả phải là: `PyTorch: 1.10.0 | Torchvision: 0.11.0... | CUDA: True`

### 5.5 Cài các thư viện còn lại
```bash
pip install -r requirements.txt
```

---

## Bước 6: Export model TensorRT

> Bước này mất 15-30 phút. Chỉ cần làm 1 lần.

```bash
cd ~/beeguard
source venv/bin/activate

# Export model sang TensorRT FP16 với inference size 320
yolo export model=model/best.pt format=engine device=0 half=True imgsz=320
```

Sau khi xong, kiểm tra:
```bash
ls -la model/
```
Phải thấy file `best.engine`.

---

## Bước 7: Test chạy thủ công

```bash
cd ~/beeguard
source venv/bin/activate
python3 main.py
```

Kết quả kỳ vọng:
```
[HH:MM:SS] === BeeGuard Jetson Nano ===
[HH:MM:SS] Loading model: best.engine
[HH:MM:SS] Model: Loaded best.engine on CUDA — 1 class(es)
[HH:MM:SS] Camera opened: 640x480
[HH:MM:SS] ESP32: connected to /dev/ttyUSB0
[HH:MM:SS] Tracking auto-enabled
[HH:MM:SS] Main loop started
[HH:MM:SS] FPS: 25.3 | Infer: 38ms | Objects: 0
```

Bấm `Ctrl+C` để dừng.

---

## Bước 8: Cài tự khởi động (systemd)

### 8.1 Tạo service file

```bash
sudo nano /etc/systemd/system/beeguard.service
```

Dán nội dung sau (thay `<USERNAME>` bằng username thực tế của bạn):

```ini
[Unit]
Description=BeeGuard Hornet Tracker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=<USERNAME>
WorkingDirectory=/home/<USERNAME>/beeguard
ExecStart=/home/<USERNAME>/beeguard/venv/bin/python3 main.py
Restart=always
RestartSec=5
Environment=DISPLAY=

[Install]
WantedBy=multi-user.target
```

Lưu: `Ctrl+O` → Enter → `Ctrl+X`

### 8.2 Kích hoạt
```bash
sudo systemctl daemon-reload
sudo systemctl enable beeguard.service
sudo systemctl start beeguard.service
```

### 8.3 Kiểm tra
```bash
# Xem log realtime
journalctl -u beeguard -f

# Xem trạng thái
sudo systemctl status beeguard.service
```

### 8.4 Thử reboot
```bash
sudo reboot
```
Sau khi boot lại, kiểm tra:
```bash
journalctl -u beeguard -f
```
Hệ thống phải tự chạy mà không cần đăng nhập.

---

## Lệnh hữu ích

```bash
# Dừng hệ thống
sudo systemctl stop beeguard.service

# Khởi động lại
sudo systemctl restart beeguard.service

# Tắt tự khởi động
sudo systemctl disable beeguard.service

# Xem log 50 dòng cuối
journalctl -u beeguard -n 50

# Xem nhiệt độ GPU
cat /sys/devices/virtual/thermal/thermal_zone1/temp
# Chia 1000 = °C (ví dụ 55000 = 55°C)

# Xem RAM usage
free -h

# Kiểm tra camera
ls /dev/video*

# Kiểm tra ESP32
ls /dev/ttyUSB*
```

---

## Cấu trúc dự án

```
beeguard/
├── main.py                 # Vòng lặp chính (headless)
├── detection_engine.py     # YOLO inference (TensorRT)
├── tracking_engine.py      # Pixel → Servo angle
├── servo_controller.py     # Serial protocol ESP32
├── firebase_alert.py       # SOS alert + sensor push
├── .env                    # Cấu hình Firebase (BÍ MẬT)
├── .env.example            # Mẫu .env
├── service_account.json    # Firebase credentials (BÍ MẬT)
├── requirements.txt        # Thư viện Python
├── model/
│   ├── best.pt             # Model gốc (push lên Git)
│   └── best.engine         # TensorRT (export trên Jetson, KHÔNG push Git)
└── .gitignore
```
