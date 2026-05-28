"""
BeeGuard Headless Tracker — Jetson Nano
========================================
Headless (no GUI) version of the hornet tracking system.
Designed for Jetson Nano: auto-start via systemd, control via Firebase app.

Camera: Logitech C310 USB (OpenCV V4L2)
ESP32: USB Serial (auto-detect /dev/ttyUSB*)
Model: TensorRT .engine (FP16, 320px)
"""

import os
import sys
import time
import signal
import threading
import cv2

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from servo_controller import ServoController
from detection_engine import DetectionEngine
from tracking_engine import TrackingEngine
from firebase_alert import FirebaseAlertSender


def log(msg):
    """Print timestamped log message."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


class HeadlessTracker:
    """Main headless tracker — no GUI, controlled via Firebase."""

    def __init__(self):
        # State
        self.camera_running = False
        self.tracking_active = False
        self.cap = None
        self.running = True

        # Modules
        self.servo = ServoController()
        self.detector = DetectionEngine()
        self.tracker = TrackingEngine()

        # Firebase
        self.device_id = os.environ.get("DEVICE_ID", "TRK-VUON001")
        DEVICE_NAME = os.environ.get("DEVICE_NAME", "Hornet Tracker Jetson")
        DB_URL = os.environ.get("DB_URL",
            "https://doan-hotronuoiong-default-rtdb.asia-southeast1.firebasedatabase.app")
        self.alert_sender = FirebaseAlertSender(self.device_id, DB_URL, DEVICE_NAME)

        self._last_heartbeat = 0.0
        self._last_state_push = 0.0

        # Sensor callback: ESP32 → Serial → Firebase
        self.servo.on_sensor_data = self._on_sensor_data

    # ======================== Setup ========================

    def setup(self):
        """Initialize all hardware. Returns True if ready to run."""
        log("=== BeeGuard Jetson Nano ===")

        # 0. Generate QR Code if not exists
        self._generate_qr_code()

        # 1. Load AI model
        model_path = self._find_model()
        if not model_path:
            log("ERROR: No model file found in model/ directory")
            return False

        log(f"Loading model: {os.path.basename(model_path)}")
        ok, msg = self.detector.load_model(model_path)
        if not ok:
            log(f"ERROR: {msg}")
            return False
        log(f"Model: {msg}")

        # 2. Open camera
        if not self._open_camera():
            log("ERROR: Cannot open camera")
            return False

        # 3. Connect ESP32 (auto-detect port)
        self._auto_connect_esp32()

        # 4. Firebase online
        self.alert_sender.update_status("online")
        self.alert_sender.start_command_listener(self._handle_remote_command)

        # 5. Auto-enable tracking
        self.camera_running = True
        self.tracking_active = True
        log("Tracking auto-enabled")

        self._push_state()
        return True

    def _find_model(self):
        """Find model file: prefer .engine, then .pt, then .onnx."""
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
        for ext in [".engine", ".pt", ".onnx"]:
            for f in os.listdir(model_dir) if os.path.isdir(model_dir) else []:
                if f.endswith(ext):
                    return os.path.join(model_dir, f)
        return None

    def _open_camera(self, index=0):
        """Open USB camera via V4L2."""
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log(f"Camera opened: {w}x{h}")
        return True

    def _auto_connect_esp32(self):
        """Auto-detect and connect to ESP32 serial port."""
        ports = ServoController.list_ports()
        if not ports:
            log("ESP32: no serial ports found — running without servo")
            return

        # Prefer /dev/ttyUSB* on Linux, COM* on Windows
        for port in ports:
            log(f"ESP32: trying {port}...")
            ok, msg = self.servo.connect(port)
            if ok:
                log(f"ESP32: connected to {port}")
                return
            log(f"ESP32: {port} failed — {msg}")

        log("ESP32: all ports failed — running without servo")

    # ======================== Main Loop ========================

    def run(self):
        """Main processing loop."""
        log("Main loop started")
        fps_count = 0
        fps_time = time.time()
        fps = 0.0

        while self.running:
            if not self.camera_running or self.cap is None:
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            self.tracker.set_frame_size(w, h)

            # Detection
            detections = []
            if self.detector.model_loaded:
                detections = self.detector.detect(frame)

            # SOS alert + pump when hornets detected
            if detections:
                avg_conf = sum(d[4] for d in detections) / len(detections)
                annotated = self.detector.draw_detections(frame.copy(), detections)
                _, jpeg_buf = cv2.imencode('.jpg', annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
                jpeg_bytes = bytes(jpeg_buf)
                threading.Thread(
                    target=self.alert_sender.send_hornet_alert,
                    args=(len(detections), avg_conf),
                    kwargs={"image_bytes": jpeg_bytes},
                    daemon=True,
                ).start()
                if self.servo.connected:
                    self.servo.send_pump_fire()

            # Tracking
            if self.tracking_active:
                selected = self.tracker.select_target(detections)
                tracking_det = None
                if selected is not None:
                    x1, y1, x2, y2, conf, cls, cx, cy = selected
                    tracking_det = (cx, cy, x2 - x1, y2 - y1)

                pan, tilt, active = self.tracker.update(tracking_det)
                if self.servo.connected:
                    self.servo.send_angles(pan, tilt)

            # FPS counter
            fps_count += 1
            now = time.time()
            if now - fps_time >= 5.0:
                fps = fps_count / (now - fps_time)
                fps_count = 0
                fps_time = now
                det_count = len(detections)
                infer = self.detector.avg_inference_ms
                log(f"FPS: {fps:.1f} | Infer: {infer:.0f}ms | Objects: {det_count}")

            # Heartbeat every 30s
            if now - self._last_heartbeat >= 30:
                threading.Thread(
                    target=self.alert_sender.update_status,
                    args=("online",),
                    daemon=True,
                ).start()
                self._last_heartbeat = now

            time.sleep(0.005)

    # ======================== Firebase Remote Control ========================

    def _current_state(self):
        """Snapshot current state for Firebase."""
        try:
            ports = ServoController.list_ports()
        except Exception:
            ports = []
        return {
            "camera_on": bool(self.camera_running),
            "tracking_on": bool(self.tracking_active),
            "confidence": round(self.detector.confidence, 3),
            "smooth_factor": round(self.tracker.smooth_factor, 3),
            "dead_zone_deg": round(self.tracker.dead_zone_deg, 3),
            "cal_pan_offset": round(self.tracker.cal_pan_offset, 2),
            "cal_tilt_offset": round(self.tracker.cal_tilt_offset, 2),
            "esp32_connected": bool(self.servo.connected),
            "esp32_port": self.servo.port or "",
            "com_ports": ports,
        }

    def _push_state(self):
        """Push state to Firebase (throttled 1s)."""
        now = time.time()
        if now - self._last_state_push < 1.0:
            return
        self._last_state_push = now
        snap = self._current_state()
        threading.Thread(
            target=self.alert_sender.push_state,
            args=(snap,),
            daemon=True,
        ).start()

    def _handle_remote_command(self, cmd_id, cmd):
        """Handle command from Firebase app. Returns (ok, error_msg)."""
        ctype = cmd.get("type", "")
        payload = cmd.get("payload", {}) or {}
        try:
            if ctype == "set_camera":
                want = bool(payload.get("on"))
                if want and not self.camera_running:
                    if self._open_camera():
                        self.camera_running = True
                        log("Camera ON (remote)")
                elif not want and self.camera_running:
                    self.camera_running = False
                    self.tracking_active = False
                    if self.cap:
                        self.cap.release()
                        self.cap = None
                    log("Camera OFF (remote)")

            elif ctype == "set_tracking":
                want = bool(payload.get("on"))
                if want and not self.tracking_active:
                    if not self.camera_running:
                        if self._open_camera():
                            self.camera_running = True
                    self.tracking_active = True
                    self.tracker.reset()
                    log("Tracking ON (remote)")
                elif not want and self.tracking_active:
                    self.tracking_active = False
                    self.tracker.reset()
                    log("Tracking OFF (remote)")

            elif ctype == "set_params":
                self._apply_params(payload)

            elif ctype == "set_esp32":
                want_on = bool(payload.get("on"))
                port = (payload.get("port") or "").strip()
                if want_on and not self.servo.connected:
                    target = port or (ServoController.list_ports() or [""])[0]
                    if target:
                        ok, msg = self.servo.connect(target)
                        log(f"ESP32 connect {target}: {msg}")
                elif not want_on and self.servo.connected:
                    self.servo.disconnect()
                    log("ESP32 disconnected (remote)")

            elif ctype == "refresh_ports":
                pass
            else:
                return False, f"unknown command type: {ctype}"

            time.sleep(0.3)
            self._push_state()
            log(f"Remote cmd: {ctype} {payload}")
            return True, ""
        except Exception as e:
            return False, str(e)

    def _apply_params(self, payload):
        """Apply parameter changes from Firebase."""
        if "confidence" in payload:
            v = max(0.1, min(1.0, float(payload["confidence"])))
            self.detector.set_confidence(v)
        if "smooth_factor" in payload:
            v = max(0.1, min(1.0, float(payload["smooth_factor"])))
            self.tracker.smooth_factor = v
        if "dead_zone_deg" in payload:
            v = max(0.0, min(3.0, float(payload["dead_zone_deg"])))
            self.tracker.dead_zone_deg = v
        if "cal_pan_offset" in payload:
            v = max(-15.0, min(15.0, float(payload["cal_pan_offset"])))
            self.tracker.cal_pan_offset = v
        if "cal_tilt_offset" in payload:
            v = max(-15.0, min(15.0, float(payload["cal_tilt_offset"])))
            self.tracker.cal_tilt_offset = v
        log(f"Params updated: {payload}")

    def _on_sensor_data(self, sensor_dict):
        """Callback: ESP32 sensor data → Firebase."""
        t = sensor_dict.get('temperature')
        h = sensor_dict.get('humidity')
        d = sensor_dict.get('water_distance_cm')
        w = sensor_dict.get('weight_kg')
        log(f"Sensor: temp={t}°C, hum={h}%, dist={d}cm, weight={w}kg")
        threading.Thread(
            target=self.alert_sender.push_sensor_data,
            args=(sensor_dict,),
            daemon=True,
        ).start()

    def _generate_qr_code(self):
        """Generate device ID QR code if it doesn't already exist."""
        try:
            import qrcode
            qr_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qr_codes")
            if not os.path.exists(qr_dir):
                os.makedirs(qr_dir)
            qr_path = os.path.join(qr_dir, f"{self.device_id}.png")
            if not os.path.exists(qr_path):
                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_H,
                    box_size=10,
                    border=4,
                )
                qr.add_data(self.device_id)
                qr.make(fit=True)
                img = qr.make_image(fill_color="black", back_color="white")
                img.save(qr_path)
                log(f"Generated QR code for device {self.device_id} at {qr_path}")
        except Exception as e:
            log(f"Failed to generate QR code: {e}")

    # ======================== Shutdown ========================

    def shutdown(self):
        """Clean shutdown."""
        log("Shutting down...")
        self.running = False
        self.camera_running = False
        self.tracking_active = False

        try:
            self.alert_sender.stop_command_listener()
        except Exception:
            pass

        self.alert_sender.update_status("offline")

        time.sleep(0.2)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass

        try:
            self.servo.disconnect()
        except Exception:
            pass

        log("Shutdown complete")


def main():
    app = HeadlessTracker()

    # Graceful shutdown on SIGTERM/SIGINT
    def _signal_handler(sig, frame):
        app.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    if not app.setup():
        log("Setup failed — exiting")
        sys.exit(1)

    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
