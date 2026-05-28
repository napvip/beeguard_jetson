"""
Servo Controller Module
=======================
Handles serial communication with ESP32 for servo control.
Protocol: "PAN,TILT\n" at 115200 baud on COM port.
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
        self.cmd_queue = queue.Queue(maxsize=5)
        self.send_thread = None
        self.running = False
        self.last_response = ""

        # Servo state
        self.pan_angle = 90.0
        self.tilt_angle = 90.0
        # Pump state derived from ESP32 responses: "idle" | "firing" | "cooldown"
        self.pump_state = "idle"

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
        """Connect to ESP32 via serial."""
        if port:
            self.port = port
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1,
                write_timeout=1
            )
            time.sleep(2)  # Wait for ESP32 reset
            self.connected = True
            self.running = True

            # Start send thread
            self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
            self.send_thread.start()

            # Read ready message
            self._read_response()
            return True, "Connected"
        except Exception as e:
            self.connected = False
            return False, str(e)

    def disconnect(self):
        """Disconnect from ESP32."""
        self.running = False
        self.connected = False
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
            except:
                pass
        self.serial_conn = None

    def _read_response(self):
        """Read response from ESP32. Parses SENSOR: và PUMP events tự động."""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    self.last_response = line
                    if line.startswith("SENSOR:"):
                        self._parse_sensor_line(line)
                    elif line == "OK:PUMP_FIRE":
                        self.pump_state = "firing"
                    elif line == "OK:PUMP_STOP":
                        self.pump_state = "cooldown"
                    elif line == "OK:PUMP_READY":
                        self.pump_state = "idle"
                    return line
            except:
                pass
        return ""

    def _parse_sensor_line(self, line):
        """Parse 'SENSOR:temp,humidity,distance,weight_kg' from ESP32."""
        try:
            parts = line[7:].split(",")  # Remove "SENSOR:" prefix
            if len(parts) < 3:
                return
            temp = float(parts[0])
            hum = float(parts[1])
            dist = float(parts[2])
            weight = float(parts[3]) if len(parts) >= 4 else -999.0
            # Gửi nguyên giá trị kể cả khi DHT lỗi (125°C / 100%); HX711 trả -999 = chưa ready
            self.sensor_data["temperature"] = temp
            self.sensor_data["humidity"] = hum
            self.sensor_data["water_distance_cm"] = dist if dist >= 0 else None
            self.sensor_data["weight_kg"] = weight if weight > -990 else None
            if self.on_sensor_data:
                self.on_sensor_data(dict(self.sensor_data))
        except (ValueError, IndexError):
            pass

    def _send_loop(self):
        """Background thread to send queued commands and read incoming data."""
        while self.running:
            try:
                cmd = self.cmd_queue.get(timeout=0.05)
                with self.lock:
                    if self.serial_conn and self.serial_conn.is_open:
                        self.serial_conn.write(cmd.encode('utf-8'))
                        self.serial_conn.flush()
                        time.sleep(0.005)
            except queue.Empty:
                pass
            except Exception:
                pass

            # Always read incoming serial data (sensor lines, OK responses, etc.)
            with self.lock:
                self._read_response()

    def _send_cmd(self, cmd):
        """Queue a command to send."""
        if not self.connected:
            return
        try:
            # Clear old commands - only latest matters
            while not self.cmd_queue.empty():
                try:
                    self.cmd_queue.get_nowait()
                except queue.Empty:
                    break
            self.cmd_queue.put_nowait(cmd)
        except queue.Full:
            pass

    def send_angles(self, pan, tilt):
        """Send pan/tilt angles to ESP32."""
        pan = max(0, min(180, pan))
        tilt = max(0, min(180, tilt))
        self.pan_angle = pan
        self.tilt_angle = tilt
        self._send_cmd(f"{pan:.1f},{tilt:.1f}\n")

    def send_home(self):
        """Send home command."""
        self.pan_angle = 90.0
        self.tilt_angle = 90.0
        self._send_cmd("HOME\n")

    def send_pump_fire(self):
        """Yêu cầu ESP32 bắn nước 5s. ESP32 tự lo cooldown — gọi liên tục cũng vô hại.
        Trả về False nếu host đang biết là pump bận (giảm spam serial)."""
        if self.pump_state != "idle":
            return False
        self._send_cmd("PUMP,FIRE\n")
        return True

    def send_tare(self):
        """Tare load cell (zero point)."""
        self._send_cmd("TARE\n")

    def ping(self):
        """Ping ESP32 to check connection."""
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
