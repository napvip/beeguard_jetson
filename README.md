# BeeGuard Jetson Nano

Hệ thống bảo vệ đàn ong bằng AI chạy trên Jetson Nano A02 4GB.
Phiên bản headless (không GUI) — điều khiển qua app BeeGuard (Flutter).

## Tính năng
- Phát hiện ong bắp cày bằng YOLO (v5/v8/v11) + TensorRT FP16 (tối ưu cho Jetson Nano)
- Điều khiển servo + laser xua đuổi qua ESP32
- Cảnh báo SOS + ảnh detection qua Firebase → app BeeGuard
- Đọc cảm biến: nhiệt độ, độ ẩm, mực nước, khối lượng mật ong
- Tự khởi động khi cắm nguồn (systemd)
- Điều khiển từ xa qua app: bật/tắt camera, tracking, chỉnh thông số

## Xem hướng dẫn cài đặt
→ [HUONG_DAN.md](HUONG_DAN.md)
