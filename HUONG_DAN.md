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
   D:\beeguard_jetson\.env
   D:\beeguard_jetson\service_account.json
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
scp "D:\beeguard_jetson\.env" user@<IP_JETSON>:~/beeguard/
scp "D:\beeguard_jetson\service_account.json" user@<IP_JETSON>:~/beeguard/
```
Thay `user` bằng username đã tạo ở Bước 1, thay `<IP_JETSON>` bằng IP thực tế.

### Kiểm tra
```bash
ls -la ~/beeguard/.env ~/beeguard/service_account.json
```
Phải thấy cả 2 file.

---

## Bước 5: Cài môi trường Python

> **Tại sao dùng TensorRT?**
> TensorRT là engine tăng tốc của NVIDIA, **đã cài sẵn** trong JetPack 4.6 và chạy **nhanh nhất** trên Jetson Nano (FP16).
> Ta không cài PyTorch / Ultralytics / ONNX Runtime (đều rườm rà hoặc không hợp Python 3.6).
> Chỉ cần thêm **pycuda** để Python điều khiển bộ nhớ GPU. Model `.pt` được export sang `.onnx` trên máy Windows, rồi **build thành `.engine`** ngay trên Jetson (Bước 6).

### 5.1 Tạo virtual environment
```bash
cd ~/beeguard
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install --upgrade pip
```

> `--system-site-packages` để venv dùng được **OpenCV, numpy và TensorRT bản hệ thống** đã có sẵn trên Jetson (không cần build lại, tiết kiệm cả tiếng).

### 5.2 Cài pycuda
```bash
# Cần CUDA trong PATH để pycuda compile được
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

sudo apt update && sudo apt install -y python3-dev build-essential

# pycuda 2020.1 tương thích Python 3.6 (mất ~5 phút để compile)
pip install pycuda==2020.1
```

### 5.3 Kiểm tra TensorRT + pycuda
```bash
python3 -c "import tensorrt as trt; print('TensorRT:', trt.__version__)"
python3 -c "import pycuda.autoinit, pycuda.driver as cuda; print('pycuda OK, GPU:', cuda.Device(0).name())"
```
Phải in ra phiên bản TensorRT (vd `8.0.x` / `8.2.x`) và tên GPU (`NVIDIA Tegra X1`).

### 5.4 Cài các thư viện còn lại
```bash
pip install -r requirements.txt
```

---

## Bước 6: Chuẩn bị model TensorRT (.engine)

> Quy trình: export `.pt → .onnx` trên **máy Windows** → chép `.onnx` sang Jetson → **build `.onnx → .engine`** trên **chính Jetson** bằng `trtexec`.
> File `.engine` **phụ thuộc thiết bị + phiên bản TensorRT**, nên **bắt buộc build trên Jetson**, không copy từ máy khác sang được.

### 6.1 Export `.pt` → `.onnx` trên máy Windows (PowerShell)
Trên máy Windows (đã cài `ultralytics`):
```powershell
yolo export model="D:\beeguard_jetson\model\best.pt" format=onnx imgsz=416 opset=11 simplify=False
```
→ tạo `D:\beeguard_jetson\model\best.onnx`.

> `imgsz=416` khớp kích thước lúc train. (Không bắt buộc khớp tay: khi load engine, code tự đọc kích thước từ engine.)

### 6.2 Chép `best.onnx` sang Jetson
```powershell
scp "D:\beeguard_jetson\model\best.onnx" viettoan@<IP_JETSON>:~/beeguard/model/
```

### 6.3 Build `.engine` trên Jetson (chạy trên Jetson)
```bash
cd ~/beeguard
/usr/src/tensorrt/bin/trtexec --onnx=model/best.onnx --saveEngine=model/best.engine --fp16 --workspace=2048
```
> Quá trình này mất **5–15 phút** (bình thường). Xong sẽ thấy `&&&& PASSED`.

### 6.4 Kiểm tra
```bash
ls -lh ~/beeguard/model/best.engine
```
Phải thấy file `best.engine` (vài MB).

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
[HH:MM:SS] Device ID: TRK-XXXXXXXX
[HH:MM:SS] Loading model: best.engine
[HH:MM:SS] Model: Loaded best.engine on TensorRT (FP16, 416px)
[HH:MM:SS] Camera opened: 640x480
[HH:MM:SS] ESP32: connected to /dev/ttyUSB0
[HH:MM:SS] Tracking auto-enabled
[HH:MM:SS] Main loop started
[HH:MM:SS] FPS: 25.0 | Infer: 35ms | Objects: 0
```

> - Dòng `Device ID:` là ID **tự sinh** cho máy này (lưu ở `device_id.txt`) — đây là ID sẽ hiện trên web để tạo QR.
> - FPS thực tế trên Jetson Nano ở 416px (TensorRT FP16) khoảng **12–18 FPS**.
> - Nếu báo `TensorRT ... chua cai` → pycuda chưa cài đúng (quay lại **Bước 5.2/5.3**). Nếu không thấy `best.engine` → chưa build (Bước 6.3).

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
├── main.py                 # Vòng lặp chính (headless) + tự sinh device_id
├── detection_engine.py     # YOLO inference (TensorRT / Ultralytics / ONNX / OpenCV)
├── tracking_engine.py      # Pixel → Servo angle
├── servo_controller.py     # Serial protocol ESP32
├── firebase_alert.py       # SOS alert + sensor push
├── .env                    # Cấu hình Firebase (BÍ MẬT)
├── .env.example            # Mẫu .env
├── service_account.json    # Firebase credentials (BÍ MẬT)
├── requirements.txt        # Thư viện Python
├── device_id.txt           # ID tự sinh cho máy này (KHÔNG push Git)
├── model/
│   ├── best.pt             # Model gốc (push lên Git)
│   ├── best.onnx           # Export từ PC (KHÔNG push Git) — dùng để build engine
│   └── best.engine         # Build bằng trtexec TRÊN Jetson (KHÔNG push Git)
└── .gitignore
```
