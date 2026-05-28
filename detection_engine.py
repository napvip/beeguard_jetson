"""
Detection Engine Module — YOLOv11s (Jetson Nano)
=================================================
Ultralytics YOLO model inference for hornet detection.
Optimized for Jetson Nano: TensorRT .engine, FP16, 320px default.

Supports: .pt, .onnx, .engine (TensorRT) model formats.
"""

import cv2
import numpy as np
import time
import os
import warnings

# Suppress non-critical warnings from ultralytics internals
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="ultralytics")


class DetectionEngine:
    """YOLO-based hornet detection engine (optimized for YOLOv11s)."""

    def __init__(self):
        self.model = None
        self.model_path = None
        self.confidence = 0.5
        self.iou_threshold = 0.45
        self.model_loaded = False
        self.class_names = ["ongbapcay"]

        # Inference settings (optimized for Jetson Nano 4GB)
        self.inference_size = 320
        self.augment = False
        self.max_det = 20
        self.half = True            # FP16 — Jetson Nano supports natively
        self.device = "cuda"        # Jetson Nano GPU (Maxwell)

        # Performance tracking
        self._inference_times = []
        self.avg_inference_ms = 0.0

    def load_model(self, model_path):
        """
        Load YOLO model using ultralytics.

        Supports YOLOv8, YOLOv9, YOLOv10, YOLOv11 (.pt, .onnx, .engine).
        No PosixPath fix needed — ultralytics handles paths natively.
        """
        if not os.path.exists(model_path):
            return False, f"Model not found: {model_path}"

        self.model_path = model_path

        try:
            from ultralytics import YOLO

            self.model = YOLO(model_path)

            # Get model info
            model_type = getattr(self.model, 'type', 'detect')
            task = getattr(self.model, 'task', 'detect')

            # Get class names from model
            if hasattr(self.model, 'names'):
                if isinstance(self.model.names, dict):
                    self.class_names = list(self.model.names.values())
                elif isinstance(self.model.names, list):
                    self.class_names = self.model.names

            # Pick device: CUDA if available, else CPU
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                    self.half = True
                    gpu = torch.cuda.get_device_name(0)
                    print(f"[DetectionEngine] CUDA: {gpu}, FP16 enabled")
                else:
                    self.device = "cpu"
                    self.half = False
                    print(f"[DetectionEngine] CUDA not available — running on CPU")
            except ImportError:
                self.device = "cpu"
                self.half = False

            # Warm up — first inference allocates GPU memory and is slow
            try:
                import numpy as np
                dummy = np.zeros((self.inference_size, self.inference_size, 3), dtype=np.uint8)
                self.model.predict(source=dummy, device=self.device, half=self.half,
                                   imgsz=self.inference_size, verbose=False)
            except Exception:
                pass

            self.model_loaded = True
            num_classes = len(self.class_names)
            model_name = os.path.basename(model_path)
            return True, f"Loaded {model_name} on {self.device.upper()} — {num_classes} class(es)"

        except ImportError:
            self.model_loaded = False
            return False, "ultralytics not installed. Run: pip install ultralytics>=8.3"
        except Exception as e:
            self.model_loaded = False
            return False, f"Failed to load model: {str(e)}"

    def set_confidence(self, conf):
        """Set detection confidence threshold."""
        self.confidence = max(0.1, min(1.0, conf))

    def set_iou_threshold(self, iou):
        """Set NMS IoU threshold."""
        self.iou_threshold = max(0.1, min(1.0, iou))

    def set_inference_size(self, size):
        """Set inference input size (pixels). Larger = better for small objects but slower."""
        self.inference_size = max(160, min(1280, int(size)))

    def set_augment(self, on):
        """Enable/disable test-time augmentation (slower but better recall)."""
        self.augment = bool(on)

    def detect(self, frame):
        """
        Run detection on a frame.

        Uses ultralytics predict() API directly.
        Inference size:
          - 320: fastest, for nearby/large objects
          - 640: default, best accuracy-speed balance
          - 960+: for very small/distant objects

        Returns list of (x1, y1, x2, y2, confidence, class_name, cx, cy).
        """
        if not self.model_loaded or frame is None:
            return []

        detections = []
        try:
            t_start = time.perf_counter()

            # Run inference — explicit device so Ultralytics doesn't fall back to CPU
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                imgsz=self.inference_size,
                augment=self.augment,
                max_det=self.max_det,
                half=self.half,
                device=self.device,
                verbose=False,
                stream=False,
            )

            t_end = time.perf_counter()

            # Track inference time
            inference_ms = (t_end - t_start) * 1000
            self._inference_times.append(inference_ms)
            if len(self._inference_times) > 30:
                self._inference_times.pop(0)
            self.avg_inference_ms = sum(self._inference_times) / len(self._inference_times)

            # Parse results
            for r in results:
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    # Vectorized extraction (faster than per-box loop)
                    xyxy = boxes.xyxy.cpu().numpy()       # (N, 4)
                    confs = boxes.conf.cpu().numpy()       # (N,)
                    cls_ids = boxes.cls.cpu().numpy().astype(int)  # (N,)

                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i]
                        conf = float(confs[i])
                        cls_id = int(cls_ids[i])
                        cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        detections.append((
                            int(x1), int(y1), int(x2), int(y2),
                            conf, cls_name, cx, cy
                        ))

        except Exception as e:
            print(f"Detection error: {e}")

        return detections

    def draw_detections(self, frame, detections):
        """Draw detection boxes and labels on frame."""
        for det in detections:
            x1, y1, x2, y2, conf, cls_name, cx, cy = det

            # Box color: bright green
            color = (0, 255, 100)
            thickness = 2

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Draw center point
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

            # Label
            label = f"{cls_name} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

        return frame

    def draw_crosshair(self, frame, target_cx=None, target_cy=None):
        """Draw tracking crosshair on frame center and target."""
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Center crosshair (thin, grey)
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y),
                 (150, 150, 150), 1)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20),
                 (150, 150, 150), 1)

        # Target crosshair (if tracking)
        if target_cx is not None and target_cy is not None:
            tcx, tcy = int(target_cx), int(target_cy)
            cv2.line(frame, (tcx - 15, tcy), (tcx + 15, tcy), (0, 0, 255), 2)
            cv2.line(frame, (tcx, tcy - 15), (tcx, tcy + 15), (0, 0, 255), 2)
            cv2.circle(frame, (tcx, tcy), 20, (0, 0, 255), 1)

        return frame

    def get_model_info(self):
        """Return model information dict."""
        if not self.model_loaded:
            return {"status": "not loaded"}

        info = {
            "status": "loaded",
            "path": self.model_path,
            "classes": self.class_names,
            "num_classes": len(self.class_names),
            "inference_size": self.inference_size,
            "half_precision": self.half,
            "avg_inference_ms": round(self.avg_inference_ms, 1),
        }
        return info
