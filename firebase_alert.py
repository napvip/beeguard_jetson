"""Firebase Alert Module — gửi cảnh báo SOS ong bắp cày lên Realtime Database + FCM."""

import os
import time
import json
import hashlib
import threading
import requests
from typing import Union, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google.oauth2 import service_account
    import google.auth.transport.requests as google_requests
    HAS_GOOGLE_AUTH = True
except ImportError:
    HAS_GOOGLE_AUTH = False
    print("[Firebase] Canh bao: google-auth chua duoc cai. FCM push se bi tat.")
    print("[Firebase] -> Chay: pip install google-auth")

_SCOPES = [
    "https://www.googleapis.com/auth/firebase.messaging",
    "https://www.googleapis.com/auth/firebase.database",
    "https://www.googleapis.com/auth/userinfo.email",
]
_SERVICE_ACCOUNT_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "service_account.json")
_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "doan-hotronuoiong")

# Cloudinary config
_CLOUDINARY_CLOUD = os.environ.get("CLOUDINARY_CLOUD_NAME", "")
_CLOUDINARY_KEY   = os.environ.get("CLOUDINARY_API_KEY", "")
_CLOUDINARY_SECRET = os.environ.get("CLOUDINARY_API_SECRET", "")


def _load_credentials():
    """Nạp Google credentials từ service account file."""
    if not HAS_GOOGLE_AUTH:
        return None
    path = _SERVICE_ACCOUNT_PATH
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(path):
        print(f"[Firebase] Khong tim thay service_account.json: {path}")
        return None
    try:
        return service_account.Credentials.from_service_account_file(
            path, scopes=_SCOPES
        )
    except Exception as e:
        print(f"[Firebase] Loi nap credentials: {e}")
        return None


_credentials = _load_credentials()


def upload_detection_image(image_bytes: bytes) -> Optional[str]:
    """Upload ảnh detection (JPEG bytes) lên Cloudinary, trả về secure URL hoặc None."""
    try:
        ts = int(time.time())
        folder = "sos_detections"
        sign_str = f"folder={folder}&timestamp={ts}{_CLOUDINARY_SECRET}"
        signature = hashlib.sha1(sign_str.encode()).hexdigest()
        url = f"https://api.cloudinary.com/v1_1/{_CLOUDINARY_CLOUD}/image/upload"
        resp = requests.post(
            url,
            data={
                "api_key": _CLOUDINARY_KEY,
                "timestamp": str(ts),
                "signature": signature,
                "folder": folder,
            },
            files={"file": ("detection.jpg", image_bytes, "image/jpeg")},
            timeout=15,
        )
        if resp.status_code == 200:
            secure_url = resp.json().get("secure_url", "")
            print(f"[Cloudinary] Anh da upload: {secure_url}")
            return secure_url
        print(f"[Cloudinary] Upload that bai: HTTP {resp.status_code} — {resp.text[:120]}")
    except Exception as e:
        print(f"[Cloudinary] Loi upload anh: {e}")
    return None


def _get_access_token() -> Optional[str]:
    """Lấy OAuth2 access token, tự refresh khi hết hạn."""
    global _credentials
    if _credentials is None:
        return None
    try:
        req = google_requests.Request()
        _credentials.refresh(req)
        return _credentials.token
    except Exception as e:
        print(f"[Firebase] Loi refresh token: {e}")
        return None


def _auth_headers() -> dict:
    """Header Authorization cho RTDB REST API — dùng service account token (admin, bypass rules)."""
    token = _get_access_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


class FirebaseAlertSender:
    # Khớp chu kỳ pump ESP32 (5s bắn + 5s nghỉ) → mỗi vụ tấn công = 1 cảnh báo / 10s.
    ALERT_COOLDOWN = 10  # giây

    def __init__(self, device_id: str, database_url: str, device_name: str = ""):
        self.device_id = device_id
        self.device_name = device_name or device_id
        self.database_url = database_url.rstrip("/")
        self._last_alert_time = 0
        self._owner_uid = None
        self._hive_name = None

    def _fetch_device_info(self):
        """Lấy thông tin device từ DB, cache owner_uid và hive_name."""
        url = f"{self.database_url}/tracking_devices/{self.device_id}.json"
        try:
            resp = requests.get(url, headers=_auth_headers(), timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    self._owner_uid = data.get("owner_uid")
                    self._hive_name = data.get("hive_name", "")
                    print(f"[Firebase] Device info: owner_uid={self._owner_uid}, hive_name={self._hive_name}")
            else:
                print(f"[Firebase] Fetch device info that bai: HTTP {resp.status_code} — {resp.text[:120]}")
        except Exception as e:
            print(f"[Firebase] Loi khi lay thong tin device: {e}")

    def _fetch_fcm_token(self) -> Optional[str]:
        """Lấy FCM token của owner từ RTDB."""
        if not self._owner_uid:
            return None
        url = f"{self.database_url}/user_fcm_tokens/{self._owner_uid}.json"
        try:
            resp = requests.get(url, headers=_auth_headers(), timeout=5)
            if resp.status_code == 200:
                token = resp.json()
                return token if isinstance(token, str) else None
            else:
                print(f"[Firebase] Fetch FCM token that bai: HTTP {resp.status_code} — {resp.text[:120]}")
        except Exception as e:
            print(f"[Firebase] Loi lay FCM token: {e}")
        return None

    def _send_fcm_push(self, detection_count: int, confidence: float, image_url: str = "") -> bool:
        """Gửi FCM push notification qua FCM v1 API."""
        access_token = _get_access_token()
        if not access_token:
            return False

        fcm_token = self._fetch_fcm_token()
        if not fcm_token:
            print("[Firebase] Khong co FCM token — bo qua push notification")
            return False

        url = f"https://fcm.googleapis.com/v1/projects/{_PROJECT_ID}/messages:send"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        # Data-only message — không có field "notification" để Android không
        # tự hiện FCM system notification. Background handler của Flutter sẽ
        # tự hiện local notification → chỉ có 1 noti duy nhất.
        payload = {
            "message": {
                "token": fcm_token,
                "data": {
                    "type": "sos_alert",
                    "hive_name": self._hive_name or "",
                    "detection_count": str(detection_count),
                    "confidence": str(round(confidence, 4)),
                    "device_id": self.device_id,
                    "image_url": image_url,
                },
                "android": {
                    "priority": "high",
                },
            }
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                print(f"[Firebase] Da gui FCM push: {detection_count} con")
                return True
            else:
                print(f"[Firebase] FCM that bai: HTTP {resp.status_code} — {resp.text[:120]}")
                return False
        except Exception as e:
            print(f"[Firebase] Loi gui FCM: {e}")
            return False

    def send_hornet_alert(self, detection_count: int, confidence: float,
                          image_bytes: Optional[bytes] = None) -> bool:
        now = time.time()
        if now - self._last_alert_time < self.ALERT_COOLDOWN:
            return False
        # Đặt mốc thời gian NGAY LẬP TỨC để các thread song song không vượt qua cooldown
        self._last_alert_time = now

        if not self._owner_uid:
            self._fetch_device_info()

        if not self._owner_uid:
            print(f"[Firebase] Bo qua alert: device '{self.device_id}' chua duoc lien ket voi tai khoan nao.")
            print(f"[Firebase] -> Hay them/lien ket device ID nay tren web BeeGuard truoc.")
            return False

        # Upload ảnh detection lên Cloudinary (nếu có)
        image_url = ""
        if image_bytes:
            print("[Cloudinary] Dang upload anh detection...")
            image_url = upload_detection_image(image_bytes) or ""

        # Ghi vào RTDB
        payload = {
            "device_id": self.device_id,
            "owner_uid": self._owner_uid,
            "hive_name": self._hive_name or "",
            "alert_type": "hornet_attack",
            "detection_count": detection_count,
            "confidence": round(confidence, 4),
            "status": "active",
            "is_read": False,
            "created_at": int(time.time() * 1000),
            "image_url": image_url,
        }

        headers = _auth_headers()
        user_url = f"{self.database_url}/user_sos_alerts/{self._owner_uid}.json"
        admin_url = f"{self.database_url}/sos_alerts.json"
        try:
            resp = requests.post(user_url, json=payload, headers=headers, timeout=5)
            if resp.status_code not in (200, 201):
                print(f"[Firebase] Gui canh bao that bai: HTTP {resp.status_code}")
                return False

            requests.post(admin_url, json=payload, headers=headers, timeout=5)
            print(f"[Firebase] Da gui canh bao SOS: {detection_count} con, conf={confidence:.2f}, image={bool(image_url)}")
            self._send_fcm_push(detection_count, confidence, image_url)
            return True
        except Exception as e:
            print(f"[Firebase] Loi khi gui canh bao: {e}")
            return False

    def update_status(self, status: str = "online"):
        """Cập nhật trạng thái thiết bị. Tự tạo record đầy đủ nếu chưa tồn tại."""
        url = f"{self.database_url}/tracking_devices/{self.device_id}.json"
        now_ms = int(time.time() * 1000)
        headers = _auth_headers()

        try:
            check = requests.get(url, headers=headers, timeout=5)
            exists = check.status_code == 200 and check.json() is not None

            if exists:
                payload = {"status": status, "last_seen": now_ms}
                resp = requests.patch(url, json=payload, headers=headers, timeout=5)
            else:
                payload = {
                    "device_id": self.device_id,
                    "device_name": self.device_name,
                    "owner_uid": "",
                    "hive_name": "",
                    "status": status,
                    "created_at": now_ms,
                    "last_seen": now_ms,
                }
                resp = requests.put(url, json=payload, headers=headers, timeout=5)
                print(f"[Firebase] Da tao device moi: {self.device_id}")

            if resp.status_code == 200:
                print(f"[Firebase] Cap nhat trang thai: {status}")
            else:
                print(f"[Firebase] Cap nhat trang thai that bai: HTTP {resp.status_code} — {resp.text}")
        except Exception as e:
            print(f"[Firebase] Loi khi cap nhat trang thai: {e}")

    # ====================================================================
    # Remote Control — state publisher + command listener
    # ====================================================================
    def push_state(self, state_dict: dict) -> bool:
        """Đẩy trạng thái runtime của tracker lên RTDB để app đọc.

        state_dict ví dụ:
            {"camera_on": True, "tracking_on": False,
             "confidence": 0.5, "smooth_factor": 0.4, "dead_zone_deg": 0.8,
             "cal_pan_offset": 8.0, "cal_tilt_offset": -8.0}
        """
        url = f"{self.database_url}/tracking_devices/{self.device_id}/state.json"
        payload = dict(state_dict)
        payload["updated_at"] = int(time.time() * 1000)
        try:
            resp = requests.patch(url, json=payload, headers=_auth_headers(), timeout=5)
            if resp.status_code == 200:
                return True
            print(f"[Firebase] Push state that bai: HTTP {resp.status_code} — {resp.text[:120]}")
        except Exception as e:
            print(f"[Firebase] Loi push state: {e}")
        return False

    def fetch_pending_commands(self) -> dict:
        """Lấy các command chưa xử lý từ RTDB. Trả về dict {push_id: cmd}."""
        url = f"{self.database_url}/tracking_devices/{self.device_id}/commands.json"
        try:
            resp = requests.get(url, headers=_auth_headers(), timeout=5)
            if resp.status_code != 200:
                return {}
            data = resp.json() or {}
            return {
                cid: c for cid, c in data.items()
                if isinstance(c, dict) and c.get("status") == "pending"
            }
        except Exception as e:
            print(f"[Firebase] Loi fetch commands: {e}")
            return {}

    def ack_command(self, cmd_id: str, status: str = "done", error: str = "") -> bool:
        """Cập nhật trạng thái command sau khi xử lý."""
        url = f"{self.database_url}/tracking_devices/{self.device_id}/commands/{cmd_id}.json"
        payload = {
            "status": status,
            "processed_at": int(time.time() * 1000),
        }
        if error:
            payload["error"] = error
        try:
            resp = requests.patch(url, json=payload, headers=_auth_headers(), timeout=5)
            return resp.status_code == 200
        except Exception as e:
            print(f"[Firebase] Loi ack command: {e}")
            return False

    def start_command_listener(self, on_command, poll_interval: float = 2.0):
        """Bắt đầu thread polling commands. on_command(cmd_id, cmd_dict) là callback.

        Callback nên trả về (ok: bool, error_msg: str).
        """
        self._cmd_stop = threading.Event()

        def _loop():
            while not self._cmd_stop.is_set():
                try:
                    pending = self.fetch_pending_commands()
                    # Xử lý theo thứ tự created_at tăng dần
                    items = sorted(
                        pending.items(),
                        key=lambda kv: kv[1].get("created_at", 0),
                    )
                    for cmd_id, cmd in items:
                        try:
                            ok, err = on_command(cmd_id, cmd)
                            self.ack_command(
                                cmd_id,
                                status="done" if ok else "error",
                                error=err or "",
                            )
                        except Exception as e:
                            self.ack_command(cmd_id, status="error", error=str(e))
                except Exception as e:
                    print(f"[Firebase] Cmd loop error: {e}")
                self._cmd_stop.wait(poll_interval)

        t = threading.Thread(target=_loop, daemon=True, name="FirebaseCmdListener")
        t.start()
        self._cmd_thread = t
        return t

    def stop_command_listener(self):
        """Dừng listener (gọi khi shutdown)."""
        if hasattr(self, "_cmd_stop"):
            self._cmd_stop.set()

    def push_sensor_data(self, sensor_dict: dict) -> bool:
        """Đẩy dữ liệu cảm biến lên Firebase RTDB.

        Args:
            sensor_dict: {"temperature": float|None, "humidity": float|None,
                          "water_distance_cm": float|None, "weight_kg": float|None}
        Returns:
            True nếu gửi thành công.
        """
        url = f"{self.database_url}/tracking_devices/{self.device_id}/sensors.json"
        now_ms = int(time.time() * 1000)
        payload = {
            "temperature": sensor_dict.get("temperature"),
            "humidity": sensor_dict.get("humidity"),
            "water_distance_cm": sensor_dict.get("water_distance_cm"),
            "weight_kg": sensor_dict.get("weight_kg"),
            "updated_at": now_ms,
        }
        headers = _auth_headers()
        try:
            resp = requests.put(url, json=payload, headers=headers, timeout=5)
            if resp.status_code == 200:
                t = payload["temperature"]
                h = payload["humidity"]
                d = payload["water_distance_cm"]
                w = payload["weight_kg"]
                print(f"[Firebase] Sensor: temp={t}°C, hum={h}%, dist={d}cm, weight={w}kg")
                return True
            else:
                print(f"[Firebase] Push sensor that bai: HTTP {resp.status_code} — {resp.text[:120]}")
        except Exception as e:
            print(f"[Firebase] Loi push sensor: {e}")
        return False
