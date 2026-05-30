# Gỡ các thứ không cần thiết (sau khi chuyển sang TensorRT)

> Dành cho bạn đã làm theo **guide cũ** tới bước `pip install -r requirements.txt`.
> Guide cũ cài `torch`, `torchvision`, `Pillow`, `qrcode` — **không cần** cho bản TensorRT mới.
> Gỡ chúng để tiết kiệm RAM + ~1.5 GB disk trên thẻ SD.

> ⚠️ **An toàn:** chỉ gỡ trong **venv** của dự án. KHÔNG gỡ `numpy`, `opencv`, `tensorrt` (là bản hệ thống của Jetson, dùng chung qua `--system-site-packages`).

---

## Bước 1: Vào venv của dự án
```bash
cd ~/beeguard
source venv/bin/activate
```
Dấu nhắc phải có `(venv)` ở đầu dòng.

## Bước 2: Gỡ các thư viện Python không dùng
```bash
pip uninstall -y torch torchvision Pillow qrcode
```

> Nếu `torchvision` báo *"not installed"* hoặc gỡ không sạch (do guide cũ build bằng `setup.py install`), xoá thủ công:
> ```bash
> rm -rf ~/beeguard/venv/lib/python3.6/site-packages/torchvision*
> ```

## Bước 3 (tuỳ chọn): Xoá file cài đặt thừa để lấy lại disk
```bash
# File wheel PyTorch đã tải về
rm -f ~/beeguard/torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Thư mục source torchvision đã clone (nếu còn)
rm -rf ~/beeguard/torchvision

# Thư mục QR đã sinh trước đây (nếu có)
rm -rf ~/beeguard/qr_codes
```

## Bước 4: Kiểm tra còn lại đúng các lib cần
```bash
pip list | grep -Ei "pyserial|dotenv|requests|google-auth|torch|pillow|qrcode|onnxruntime"
```
Kết quả **chỉ nên còn**: `pyserial`, `python-dotenv`, `requests`, `google-auth`.
Không nên còn: `torch`, `torchvision`, `Pillow`, `qrcode`, `onnxruntime`.

```bash
# Xác nhận numpy + opencv (bản hệ thống) vẫn còn — KHÔNG được mất
python3 -c "import cv2, numpy; print('OpenCV', cv2.__version__, '| numpy', numpy.__version__)"

# Xác nhận TensorRT hệ thống vẫn còn
python3 -c "import tensorrt; print('TensorRT', tensorrt.__version__)"
```

---

## Về các gói apt (không bắt buộc gỡ)

Guide cũ cài thêm vài thư viện hệ thống (`libopenmpi-dev`, `liblapack-dev`, `libavcodec-dev`...). **Không cần gỡ** — chúng vô hại và đôi khi là phụ thuộc của gói khác. Nếu muốn dọn rác apt nói chung:
```bash
sudo apt autoremove -y
```

---

## Sau khi gỡ xong

Tiếp tục theo `HUONG_DAN.md` (Bước 5–7):
1. Cài `pycuda` (Bước 5.2).
2. Copy `best.onnx` từ PC sang (Bước 6.2).
3. Build `best.engine` bằng `trtexec` trên Jetson (Bước 6.3).
4. Chạy `python3 main.py` (Bước 7).
