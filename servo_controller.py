"""
Servo Controller Module
=======================
Handles serial communication with ESP32 for servo control.
Protocol: "PAN,TILT\n" at 115200 baud on COM port.

Hai kênh truyền TÁCH BIỆT để lệnh bơm KHÔNG bị luồng gửi góc ghi đè:
  - event_queue : lệnh RỜI RẠC (PUMP,FIRE / HOME / TARE) — FIFO, KHÔNG bao giờ bị xóa.
  - _latest_angle: chỉ giữ góc servo MỚI NHẤT (coalesce — ghi đè thoải mái).
Trước đây mọi lệnh đi chung 1 queue rồi bị "clear-before-put", nên góc (gửi mỗi
frame) nuốt mất PUMP,FIRE → bơm lúc bắn lúc không khi đang tracking.
"""

import serial
import serial.tools.list_ports
import threading
import time
import queue


class ServoController:
    """Manages serial connection and servo commands to ESP32."""

    def __init__(self, port="COM5", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.connected = False
        self.lock = threading.Lock()

        # Hai kênh truyền riêng biệt (xem docstring module).
        self.event_queue = queue.Queue()   # lệnh rời rạc — FIFO, KHÔNG xóa
        self._latest_angle = None          # góc mới nhất — latest-wins

        self.send_thread = None
        self.running = False
        self.last_response = ""

        # Servo state
        self.pan_angle = 90.0
        self.tilt_angle = 90.0
        # Pump state derived from ESP32 responses: "idle" | "firing" | "cooldown"
        self.pump_state = "idle"

        # Throttle bơm phía host: chặn gửi lại trong khoảng này. ESP32 mới là nơi
        # enforce cooldown thật. Dùng MỐC THỜI GIAN ĐƠN ĐIỆU (time.monotonic) thay cho
        # cổng pump_state — để 1 dòng OK:PUMP_READY bị sót KHÔNG làm kẹt bơm, và để
        # đồng hồ hệ thống nhảy (Jetson không có pin RTC) không khoá bơm hàng giờ.
        self._last_pump_fire_ts = 0.0      # mốc monotonic; 0.0 = cho phép bắn ngay
        self.pump_min_interval = 10.0      # giây (≈ 5s bắn + 5s nghỉ của ESP32)

        # Sensor data from ESP32
        self.sensor_data = {
            "temperature": None,
            "humidity": None,
            "water_distance_cm": None,
            "weight_kg": None,
        }
        self.on_sensor_data = None  # callback: fn(sensor_dict)

    @staticmethod
    def list_ports():
        """List available COM ports."""
        ports = serial.tools.list_ports.comports()
        return [p.device for p in ports]

    def connect(self, port=None):
        """Connect to ESP32 via serial (idempotent về luồng gửi)."""
        if port:
            self.port = port

        # Đảm bảo CHỈ có 1 luồng gửi: dừng luồng cũ (nếu còn) trước khi mở kết nối mới.
        if self.send_thread and self.send_thread.is_alive():
            self.running = False
            self.send_thread.join(timeout=1.0)
        self.send_thread = None

        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1,
                write_timeout=1
            )
            time.sleep(2)  # Wait for ESP32 reset

            # Phiên mới: xóa lệnh tồn đọng + reset throttle để KHÔNG có PUMP,FIRE cũ
            # bị bắn ngay lúc vừa kết nối lại.
            self._drain_event_queue()
            with self.lock:
                self._latest_angle = None
            self._last_pump_fire_ts = 0.0

            self.connected = True
            self.running = True

            # Start send thread (duy nhất)
            self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
            self.send_thread.start()

            # Read ready message
            self._read_responses()
            return True, "Connected"
        except Exception as e:
            self.connected = False
            self.running = False
            return False, str(e)

    def disconnect(self):
        """Disconnect from ESP32."""
        self.running = False
        self.connected = False

        # Dừng hẳn luồng gửi TRƯỚC khi đóng cổng để không thread nào còn chạm serial_conn.
        t = self.send_thread
        if t and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=1.0)
        self.send_thread = None

        # Xóa lệnh tồn đọng để lần connect sau không bắn lệnh cũ (vd PUMP,FIRE).
        self._drain_event_queue()
        with self.lock:
            self._latest_angle = None

        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
            except:
                pass
        self.serial_conn = None

    def _drain_event_queue(self):
        """Bỏ hết lệnh sự kiện đang chờ (dùng khi connect/disconnect)."""
        while True:
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break

    # ==================== Serial read ====================

    def _read_responses(self, max_lines=50):
        """Đọc HẾT các dòng ESP32 đang chờ trong buffer (không chỉ 1 dòng/lượt).

        Đọc 1 dòng/lượt như cũ khiến event bơm (OK:PUMP_*) bị đọc trễ và dồn ứ khi
        đang gửi góc liên tục → pump_state desync. Drain hết buffer giữ trạng thái kịp.
        Callback cảm biến được gọi NGOÀI lock (tránh tự khóa nếu callback gọi lại vào đây).
        """
        if not (self.serial_conn and self.serial_conn.is_open):
            return
        sensor_updates = []
        try:
            with self.lock:
                n = 0
                while self.serial_conn.in_waiting > 0 and n < max_lines:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    n += 1
                    if line:
                        snap = self._handle_line(line)
                        if snap is not None:
                            sensor_updates.append(snap)
        except Exception:
            pass

        # Gọi callback NGOÀI lock (lock không reentrant — callback có thể gọi lại vào đây).
        if self.on_sensor_data:
            for snap in sensor_updates:
                try:
                    self.on_sensor_data(snap)
                except Exception:
                    pass

    def _handle_line(self, line):
        """Phân loại 1 dòng phản hồi ESP32. Trả về dict cảm biến nếu là dòng SENSOR, else None."""
        self.last_response = line
        if line.startswith("SENSOR:"):
            return self._parse_sensor_line(line)
        elif line == "OK:PUMP_FIRE":
            self.pump_state = "firing"
        elif line == "OK:PUMP_STOP":
            self.pump_state = "cooldown"
        elif line == "OK:PUMP_READY":
            self.pump_state = "idle"
        elif line == "BUSY:PUMP":
            # Thông tin: ESP32 đang bận. Trạng thái thật do các dòng OK:PUMP_* quyết định.
            pass
        return None

    def _parse_sensor_line(self, line):
        """Parse 'SENSOR:temp,humidity,distance,weight_kg'. Trả về snapshot dict hoặc None."""
        try:
            parts = line[7:].split(",")  # Remove "SENSOR:" prefix
            if len(parts) < 3:
                return None
            temp = float(parts[0])
            hum = float(parts[1])
            dist = float(parts[2])
            weight = float(parts[3]) if len(parts) >= 4 else -999.0
            # Gửi nguyên giá trị kể cả khi DHT lỗi (125°C / 100%); HX711 trả -999 = chưa ready
            self.sensor_data["temperature"] = temp
            self.sensor_data["humidity"] = hum
            self.sensor_data["water_distance_cm"] = dist if dist >= 0 else None
            self.sensor_data["weight_kg"] = weight if weight > -990 else None
            return dict(self.sensor_data)
        except (ValueError, IndexError):
            return None

    # ==================== Serial write ====================

    def _write(self, cmd):
        """Ghi 1 lệnh xuống serial (giữ lock)."""
        with self.lock:
            if self.serial_conn and self.serial_conn.is_open:
                try:
                    self.serial_conn.write(cmd.encode('utf-8'))
                    self.serial_conn.flush()
                except Exception:
                    pass

    def _send_loop(self):
        """Luồng nền: gửi lệnh sự kiện (FIFO, ưu tiên) + góc mới nhất, rồi đọc serial.

        Mỗi vòng: (1) gửi HẾT lệnh sự kiện đang chờ → PUMP,FIRE không bao giờ bị bỏ;
        (2) gửi góc servo mới nhất nếu có (đã coalesce); (3) đọc hết phản hồi ESP32.
        get(timeout=0.02) vừa làm nhịp nghỉ khi rảnh, vừa phản hồi tức thì khi có event.
        """
        while self.running:
            # (1) Lệnh sự kiện — FIFO, không bao giờ bị xóa
            try:
                cmd = self.event_queue.get(timeout=0.02)
                self._write(cmd)
                while True:                      # drain các event còn lại (nếu có)
                    try:
                        self._write(self.event_queue.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                pass

            # (2) Góc servo mới nhất (latest-wins)
            ang = None
            with self.lock:
                if self._latest_angle is not None:
                    ang = self._latest_angle
                    self._latest_angle = None
            if ang is not None:
                self._write(ang)

            # (3) Luôn đọc dữ liệu về (sensor, OK/BUSY pump, ...)
            self._read_responses()

    # ==================== Public commands ====================

    def send_angles(self, pan, tilt):
        """Cập nhật góc servo mới nhất (coalesce — chỉ giá trị cuối được gửi đi)."""
        if not self.connected:
            return
        pan = max(0, min(180, pan))
        tilt = max(0, min(180, tilt))
        self.pan_angle = pan
        self.tilt_angle = tilt
        with self.lock:
            self._latest_angle = f"{pan:.1f},{tilt:.1f}\n"

    def send_home(self):
        """Gửi lệnh về home (qua hàng đợi sự kiện)."""
        if not self.connected:
            return
        self.pan_angle = 90.0
        self.tilt_angle = 90.0
        self.event_queue.put("HOME\n")

    def send_pump_fire(self):
        """Yêu cầu ESP32 bắn nước. Đưa vào hàng đợi sự kiện (KHÔNG bao giờ bị bỏ).

        Không gate theo pump_state (tránh kẹt nếu sót OK:PUMP_READY). Throttle bằng
        time.monotonic() phía host; ESP32 tự enforce cooldown và trả BUSY:PUMP vô hại
        nếu chưa tới lượt. Trả True nếu vừa THỰC SỰ gửi lệnh.
        """
        if not self.connected:
            return False
        now = time.monotonic()
        if now - self._last_pump_fire_ts < self.pump_min_interval:
            return False
        self._last_pump_fire_ts = now
        self.event_queue.put("PUMP,FIRE\n")
        return True

    def send_tare(self):
        """Tare load cell (qua hàng đợi sự kiện)."""
        if not self.connected:
            return
        self.event_queue.put("TARE\n")

    def ping(self):
        """Ping ESP32 to check connection (đồng bộ — chỉ dùng ngoài vòng lặp chính)."""
        if not self.connected:
            return False
        try:
            with self.lock:
                if self.serial_conn and self.serial_conn.is_open:
                    self.serial_conn.write(b"PING\n")
                    self.serial_conn.flush()
                    time.sleep(0.1)
                    resp = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    return resp == "PONG"
        except:
            return False
        return False
