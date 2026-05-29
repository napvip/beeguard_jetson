"""
Detection Engine Module — YOLOv11s (Jetson Nano & Windows)
===========================================================
Dual-Mode detection engine supporting:
1. Ultralytics YOLOv11s (.pt, .engine)
2. OpenCV DNN Fallback (.onnx) - Optimized for Jetson Nano (Python 3.6, GStreamer, CUDA)

No PyTorch or Ultralytics needed when running in ONNX mode!
"""

import cv2
import numpy as np
import time
import os
import warnings

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="ultralytics")

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


class DetectionEngine:
    """YOLO-based hornet detection engine with automatic OpenCV DNN ONNX fallback."""

    def __init__(self):
        self.model = None
        self.model_path = None
        self.confidence = 0.5
        self.iou_threshold = 0.45
        self.model_loaded = False
        self.class_names = ["ongbapcay"]
        self.use_onnx = False

        # Inference settings (optimized for Jetson Nano 4GB)
        self.inference_size = 320
        self.augment = False
        self.max_det = 20
        self.half = True            # FP16
        self.device = "cpu"         # Will be set dynamically

        # Performance tracking
        self._inference_times = []
        self.avg_inference_ms = 0.0

    def load_model(self, model_path):
        """
        Load model automatically choosing between Ultralytics YOLO and OpenCV DNN ONNX.
        """
        if not os.path.exists(model_path):
            return False, f"Model not found: {model_path}"

        self.model_path = model_path
        ext = os.path.splitext(model_path)[1].lower()

        # Check if we should use OpenCV DNN ONNX mode
        if ext == ".onnx" or not HAS_ULTRALYTICS:
            if ext != ".onnx" and not HAS_ULTRALYTICS:
                return False, "ultralytics is not installed. Please export your model to ONNX format (best.onnx) and load that instead."

            try:
                self.model = cv2.dnn.readNetFromONNX(model_path)
                try:
                    self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    self.device = "cuda"
                    print("[DetectionEngine] Loaded ONNX model using OpenCV DNN with CUDA backend")
                except Exception:
                    self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                    self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    self.device = "cpu"
                    print("[DetectionEngine] Loaded ONNX model using OpenCV DNN with CPU backend")

                self.model_loaded = True
                self.use_onnx = True
                self.half = False
                return True, f"Loaded {os.path.basename(model_path)} on OpenCV DNN ({self.device.upper()})"
            except Exception as e:
                self.model_loaded = False
                return False, f"Failed to load ONNX model via OpenCV DNN: {e}"

        # Otherwise, use standard Ultralytics YOLO
        try:
            self.model = YOLO(model_path)

            # Get class names from model if available
            if hasattr(self.model, 'names'):
                if isinstance(self.model.names, dict):
                    self.class_names = list(self.model.names.values())
                elif isinstance(self.model.names, list):
                    self.class_names = self.model.names

            # Pick device: CUDA if available
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

            # Warm up
            try:
                dummy = np.zeros((self.inference_size, self.inference_size, 3), dtype=np.uint8)
                self.model.predict(source=dummy, device=self.device, half=self.half,
                                   imgsz=self.inference_size, verbose=False)
            except Exception:
                pass

            self.model_loaded = True
            self.use_onnx = False
            num_classes = len(self.class_names)
            return True, f"Loaded {os.path.basename(model_path)} on {self.device.upper()} — {num_classes} class(es)"

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
        """Set inference input size (pixels)."""
        self.inference_size = max(160, min(1280, int(size)))

    def set_augment(self, on):
        """Enable/disable test-time augmentation."""
        self.augment = bool(on)

    def detect(self, frame):
        """
        Run detection on a frame.
        """
        if not self.model_loaded or frame is None:
            return []

        if self.use_onnx:
            return self._detect_onnx(frame)

        # Standard Ultralytics YOLO Inference
        detections = []
        try:
            t_start = time.perf_counter()

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
            inference_ms = (t_end - t_start) * 1000
            self._inference_times.append(inference_ms)
            if len(self._inference_times) > 30:
                self._inference_times.pop(0)
            self.avg_inference_ms = sum(self._inference_times) / len(self._inference_times)

            for r in results:
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy().astype(int)

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

    def _detect_onnx(self, frame):
        """
        Inference using pure OpenCV DNN with ONNX model.
        Extremely fast, zero dependencies (no torch/ultralytics required).
        """
        detections = []
        try:
            t_start = time.perf_counter()

            img_h, img_w = frame.shape[:2]
            x_factor = img_w / self.inference_size
            y_factor = img_h / self.inference_size

            # Create blob (YOLOv8/v11 expects RGB input, normalized by 1/255.0)
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (self.inference_size, self.inference_size), swapRB=True, crop=False)
            self.model.setInput(blob)
            outputs = self.model.forward()

            t_end = time.perf_counter()
            inference_ms = (t_end - t_start) * 1000
            self._inference_times.append(inference_ms)
            if len(self._inference_times) > 30:
                self._inference_times.pop(0)
            self.avg_inference_ms = sum(self._inference_times) / len(self._inference_times)

            # YOLOv8/v11 output is shape (1, 4 + num_classes, num_boxes)
            # For 1 class: (1, 5, 2100) or similar
            predictions = outputs[0]
            predictions = predictions.T  # Shape: (num_boxes, 5)

            boxes = []
            confidences = []

            for pred in predictions:
                conf = float(pred[4])
                if conf >= self.confidence:
                    xc, yc, w, h = pred[0], pred[1], pred[2], pred[3]

                    # Convert center coords to top-left coords and scale back to original resolution
                    x = int((xc - w/2) * x_factor)
                    y = int((yc - h/2) * y_factor)
                    w_scaled = int(w * x_factor)
                    h_scaled = int(h * y_factor)

                    boxes.append([x, y, w_scaled, h_scaled])
                    confidences.append(conf)

            # Non-Maximum Suppression (NMS)
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.iou_threshold)

            if len(indices) > 0:
                flat_indices = np.array(indices).flatten()
                for i in flat_indices:
                    x, y, w, h = boxes[i]
                    conf = confidences[i]
                    cls_name = self.class_names[0]
                    cx = x + w / 2
                    cy = y + h / 2
                    detections.append((
                        int(x), int(y), int(x + w), int(y + h),
                        conf, cls_name, cx, cy
                    ))

        except Exception as e:
            print(f"ONNX Detection error: {e}")

        return detections

    def draw_detections(self, frame, detections):
        """Draw detection boxes and labels on frame."""
        for det in detections:
            x1, y1, x2, y2, conf, cls_name, cx, cy = det
            color = (0, 255, 100)  # green
            thickness = 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

            label = f"{cls_name} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

        return frame

    def draw_crosshair(self, frame, target_cx=None, target_cy=None):
        """Draw tracking crosshair."""
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (150, 150, 150), 1)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (150, 150, 150), 1)

        if target_cx is not None and target_cy is not None:
            tcx, tcy = int(target_cx), int(target_cy)
            cv2.line(frame, (tcx - 15, tcy), (tcx + 15, tcy), (0, 0, 255), 2)
            cv2.line(frame, (tcx, tcy - 15), (tcx, tcy + 15), (0, 0, 255), 2)
            cv2.circle(frame, (tcx, tcy), 20, (0, 0, 255), 1)

        return frame

    def get_model_info(self):
        """Return model info."""
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
            "engine_type": "OpenCV_DNN_ONNX" if self.use_onnx else "Ultralytics_YOLO",
        }
        return info
