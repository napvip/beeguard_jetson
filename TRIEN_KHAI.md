# Hướng dẫn Test & Triển khai BeeGuard

Tài liệu này gom 2 việc: **(1) test nhanh trên máy Windows** trước, rồi **(2) triển khai/nâng cấp trên Jetson Nano**.

> Cài đặt Jetson lần đầu (flash JetPack, tạo swap, clone code, copy `.env`/`service_account.json`) xem `HUONG_DAN.md` Bước 1–4. Tài liệu này dành cho lúc **test** và **cập nhật code mới**.

---

## PHẦN 1 — Test nhanh trên máy Windows (khuyến nghị)

Dùng dự án **`tracking_v11`** (bản GUI Tkinter, có cửa sổ xem camera trực tiếp). Mục đích: kiểm tra **model nhận diện + logic tracking** bằng webcam trước khi mang lên Jetson.

> `tracking_v11` chạy model `.pt` thẳng bằng **ultralytics + PyTorch** (có GPU thì tự bật FP16). **Không** cần ONNX/TensorRT trên Windows.

### 1.1 Cài môi trường (chỉ làm 1 lần)
```powershell
cd D:\tracking_v11
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip

# PyTorch (chọn 1):
#  - Có GPU NVIDIA + CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#  - Chỉ CPU:
# pip install torch torchvision

pip install ultralytics          # cần để chạy .pt (nếu chưa có)
pip install -r requirements.txt
```

### 1.2 Chạy GUI
```powershell
cd D:\tracking_v11
venv\Scripts\activate
python hornet_tracker.py
```
- Cửa sổ hiện lên → chọn **camera** (webcam) → bấm bắt đầu. Sẽ thấy khung detect ong realtime.
- Chỉnh **Confidence** (0.3–0.5), **Inference Size** (320 nhanh / 960 xa hơn) ngay trên GUI.
- **ESP32 / servo**: cắm thì chọn COM port; không cắm vẫn chạy (chỉ không điều khiển servo).
- **Firebase**: cần `.env` + `service_account.json` trong `D:\tracking_v11`; không có thì vẫn detect được, chỉ không gửi cảnh báo.

### 1.3 Lưu ý quan trọng
- `tracking_v11` là bản **dev có GUI**; `beeguard_jetson` là bản **headless** (không GUI) đã tối ưu cho Jetson (TensorRT). Test ở đây để kiểm tra **model + tracking**, không phải đúng từng dòng code chạy trên Jetson.
- Model dùng chung: cùng `best.pt`. Test ổn ở đây → mang `best.pt` đó đi export ONNX rồi build engine cho Jetson (Phần 2).

---

## PHẦN 2 — Triển khai / nâng cấp trên Jetson Nano

Làm trên Jetson (qua màn hình hoặc RealVNC → mở **Terminal**).

### 2.1 Lấy code mới
```bash
cd ~/beeguard
git pull
```
> Nếu báo conflict (do lỡ sửa file tracked trên Jetson): `git stash` rồi `git pull`.
> File `.env` và `service_account.json` đã được gitignore nên `git pull` **không** đụng tới.

### 2.2 Gỡ thư viện cũ không cần (chỉ làm 1 lần khi chuyển sang TensorRT)
```bash
source venv/bin/activate
pip uninstall -y torch torchvision Pillow qrcode
```
> Chi tiết + cách lấy lại disk: xem `GO_CAI_DAT_CU.md`.

### 2.3 Cài pycuda (chỉ làm 1 lần)
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo apt update && sudo apt install -y python3-dev build-essential
pip install pycuda==2020.1
pip install -r requirements.txt
```
Kiểm tra:
```bash
python3 -c "import tensorrt as trt; print('TensorRT', trt.__version__)"
python3 -c "import pycuda.autoinit, pycuda.driver as cuda; print('pycuda OK, GPU:', cuda.Device(0).name())"
```

### 2.4 Tạo model TensorRT (.engine)
**Trên Windows** — export `.pt` → `.onnx`:
```powershell
yolo export model="D:\beeguard_jetson\model\best.pt" format=onnx imgsz=320 opset=11 simplify=False
```
Chép sang Jetson:
```powershell
scp "D:\beeguard_jetson\model\best.onnx" viettoan@192.168.1.22:~/beeguard/model/
```
**Trên Jetson** — build engine:
```bash
cd ~/beeguard
/usr/src/tensorrt/bin/trtexec --onnx=model/best.onnx --saveEngine=model/best.engine --fp16 --workspace=2048
ls -lh model/best.engine
```
> Build mất 5–15 phút, xong hiện `&&&& PASSED`.

### 2.5 Dọn device_id cũ (chuyển sang ID tự sinh)
```bash
cd ~/beeguard
sed -i 's/^DEVICE_ID=.*/DEVICE_ID=/' .env   # để trống → tự sinh ID duy nhất
rm -f device_id.txt                          # xoá ID cũ nếu có
```

### 2.6 Chạy thử
```bash
cd ~/beeguard
source venv/bin/activate
python3 main.py
```
Log đúng:
```
[..] === BeeGuard Jetson Nano ===
[..] Device ID: TRK-XXXXXXXX
[..] Loading model: best.engine
[..] Model: Loaded best.engine on TensorRT (FP16, 320px)
[..] FPS: 25.0 | Infer: 35ms | Objects: 0
```
- `Device ID` mới này sẽ **tự hiện trên web** → web tạo QR → quét bằng app để liên kết.
- `Ctrl+C` để dừng.

### 2.7 Bật tự khởi động (systemd)
Xem `HUONG_DAN.md` Bước 8. Sau khi đã cài service, **các lần cập nhật code sau** chỉ cần:
```bash
cd ~/beeguard && git pull
sudo systemctl restart beeguard.service
journalctl -u beeguard -f
```

---

## Sự cố thường gặp

| Lỗi | Nguyên nhân / cách xử |
|---|---|
| `TensorRT ... chua cai` | pycuda chưa cài → làm lại **2.3** |
| Không thấy `best.engine` | chưa build → **2.4** |
| `Failed to load TensorRT engine` | engine build từ máy/phiên bản TRT khác → build lại trên **chính Jetson** |
| Web vẫn trống | `main.py` chưa chạy qua bước load model (engine lỗi), hoặc Jetson chưa nối Internet |
| `git pull` conflict | `git stash && git pull` |
