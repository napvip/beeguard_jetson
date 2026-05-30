# Hướng dẫn thay đổi Model AI

> Trên Jetson Nano chạy bằng **TensorRT (.engine)**. Quy trình:
> export `.pt → .onnx` trên **máy Windows** → chép `.onnx` sang Jetson → **build `.engine`** bằng `trtexec` **trên chính Jetson**.
> Hỗ trợ cả **YOLOv5, YOLOv8, YOLOv11** (code tự nhận diện định dạng output).

## Quy trình thay model mới

### 1. Export `.pt` → `.onnx` trên máy Windows
```powershell
yolo export model="D:\beeguard_jetson\model\yolov11n.pt" format=onnx imgsz=320 opset=11 simplify=False
```
→ tạo `yolov11n.onnx`.

> `imgsz` quyết định kích thước inference. Khi load engine, code **tự đọc** kích thước từ engine nên không cần sửa tay `inference_size`.

### 2. Chép `.onnx` sang Jetson
```powershell
scp "D:\beeguard_jetson\model\yolov11n.onnx" user@<IP_JETSON>:~/beeguard/model/
```

### 3. Build `.engine` trên Jetson
```bash
cd ~/beeguard
# Dừng tracker trước khi build (cần GPU)
sudo systemctl stop beeguard.service

/usr/src/tensorrt/bin/trtexec --onnx=model/yolov11n.onnx --saveEngine=model/yolov11n.engine --fp16 --workspace=2048
```
Build mất ~5–15 phút.

### 4. Chọn model để chạy

Hệ thống tự quét `model/` và ưu tiên: `.engine` → `.onnx` → `.pt`.

**Cách đơn giản nhất:** đổi tên model mới thành `best.engine`:
```bash
cd ~/beeguard/model
mv best.engine best_old.engine     # backup model cũ
mv yolov11n.engine best.engine
```

### 5. Khởi động lại
```bash
sudo systemctl restart beeguard.service
journalctl -u beeguard -f          # phải thấy "Loaded best.engine on TensorRT (FP16, ...px)"
```

---

## Lưu ý quan trọng

- File `.engine` **PHẢI build trên chính Jetson** (đúng phiên bản TensorRT). **Không** copy `.engine` từ máy khác.
- File `.onnx`/`.pt` thì copy từ đâu cũng được.
- Đổi `imgsz` thì **export lại `.onnx` và build lại `.engine`** (code tự đọc imgsz từ engine).
- Model **khác số lớp (class)**: sửa `self.class_names` trong `detection_engine.py` cho khớp — parser dựa vào số lớp để nhận diện đúng YOLOv5 vs v8/v11.
- Nên dừng tracker (`systemctl stop`) trước khi build để GPU không bị tranh chấp.

## So sánh kích thước inference (Jetson Nano, TensorRT FP16)

| imgsz | FPS (ước tính) | Độ chính xác | RAM |
|---|---|---|---|
| 320 | 20-30 | Tốt cho ong gần | ~1.6 GB |
| 416 | 12-18 | Cân bằng | ~2.0 GB |
| 640 | 5-9  | Tốt nhất, ong xa | ~2.8 GB |

Khuyến nghị: dùng **320** cho tốc độ, **416** nếu cần nhận diện ong ở xa hơn.
