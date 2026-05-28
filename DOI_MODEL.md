# Hướng dẫn thay đổi Model AI

## Quy trình thay model mới

### 1. Copy model `.pt` mới sang Jetson

**Cách A — USB flash drive:**
1. Copy file model (ví dụ `yolov11n.pt`) vào USB.
2. Cắm USB vào Jetson.
3. Mở Terminal:
   ```bash
   cp /media/$USER/<tên_usb>/yolov11n.pt ~/beeguard/model/
   ```

**Cách B — SCP qua mạng LAN:**
Trên PC Windows (PowerShell):
```powershell
scp "D:\path\to\yolov11n.pt" user@<IP_JETSON>:~/beeguard/model/
```

### 2. Export sang TensorRT trên Jetson

```bash
cd ~/beeguard
source venv/bin/activate

# Dừng tracker trước khi export (cần toàn bộ GPU)
sudo systemctl stop beeguard.service

# Export (mất 15-30 phút)
yolo export model=model/yolov11n.pt format=engine device=0 half=True imgsz=320
```

Sau khi xong, kiểm tra:
```bash
ls -lh model/*.engine
```

### 3. Chọn model để chạy

Hệ thống tự động quét thư mục `model/` và ưu tiên theo thứ tự: `.engine` → `.pt` → `.onnx`.
Nếu có nhiều file `.engine`, nó sẽ lấy file đầu tiên tìm thấy.

**Cách đơn giản nhất:** đổi tên model mới thành `best.engine`:
```bash
cd ~/beeguard/model

# Backup model cũ
mv best.engine best_old.engine

# Đổi tên model mới
mv yolov11n.engine best.engine
```

### 4. Khởi động lại

```bash
sudo systemctl restart beeguard.service

# Xem log xác nhận model mới đã load
journalctl -u beeguard -f
```

---

## Lưu ý quan trọng

- File `.engine` **PHẢI** được export trên chính Jetson Nano. Không copy `.engine` từ PC sang.
- File `.pt` thì copy từ đâu cũng được.
- Nên dừng tracker (`systemctl stop`) trước khi export để GPU không bị tranh chấp.
- Mỗi lần đổi `imgsz` (320, 416, 640...) phải export lại `.engine` mới.

## So sánh kích thước inference

| imgsz | FPS (ước tính) | Độ chính xác | RAM |
|---|---|---|---|
| 320 | 25-30 | Tốt cho ong gần | ~2.0 GB |
| 416 | 15-20 | Cân bằng | ~2.5 GB |
| 640 | 8-12 | Tốt nhất, ong xa | ~3.5 GB |

Khuyến nghị: dùng **320** cho tốc độ, **416** nếu cần nhận diện ong ở xa hơn.
