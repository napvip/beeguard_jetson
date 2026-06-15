"""
Detection Engine Module — YOLO v5 / v8 / v11 (Jetson Nano & Windows)
====================================================================
Multi-Mode detection engine supporting (in priority order):
1. TensorRT (.engine, FP16) — FASTEST, preferred on Jetson Nano (tensorrt + pycuda)
2. Ultralytics YOLO (.pt) — Full-featured, dev trên Windows (cần ultralytics)
3. ONNX Runtime (.onnx) — CUDA/CPU, fallback dùng trên Windows
4. OpenCV DNN (.onnx) — fallback cuối cùng (OpenCV mới; 4.1.1 của Jetson KHÔNG parse được)

Output được tự nhận diện cho cả YOLOv5 (có objectness) lẫn YOLOv8/v11 (không objectness).
Trên Jetson Nano chỉ cần TensorRT — không cần PyTorch / Ultralytics / ONNX Runtime.
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

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401  (khởi tạo CUDA context cho luồng chính)
    HAS_TENSORRT = True
except Exception:
    # Exception (không chỉ ImportError) vì pycuda.autoinit có thể lỗi khi không có GPU
    HAS_TENSORRT = False


class DetectionEngine:
    """YOLO-based hornet detection engine with automatic ONNX Runtime / OpenCV DNN fallback."""

    def __init__(self):
        self.model = None
        self.model_path = None
        self.confidence = 0.5
        self.iou_threshold = 0.45
        self.model_loaded = False
        self.class_names = ["ongbapcay"]
        self.engine_type = "none"  # "tensorrt", "ultralytics", "onnxruntime", "opencv_dnn"

        # TensorRT runtime objects (chỉ dùng khi engine_type == "tensorrt")
        self._trt_engine = None
        self._trt_context = None
        self._trt_stream = None
        self._trt_inputs = []
        self._trt_outputs = []
        self._trt_bindings = []

        # Inference settings (optimized for Jetson Nano 4GB)
        self.inference_size = 416
        self.augment = False
        self.max_det = 20
        self.half = True            # FP16
        self.device = "cpu"         # Will be set dynamically

        # Performance tracking
        self._inference_times = []
        self.avg_inference_ms = 0.0

    def load_model(self, model_path):
        """
        Load model automatically choosing between:
        1. TensorRT (.engine) — preferred on Jetson Nano (fastest)
        2. Ultralytics YOLO (.pt)
        3. ONNX Runtime (.onnx) → OpenCV DNN (.onnx)
        """
        if not os.path.exists(model_path):
            return False, "Model not found: {}".format(model_path)

        self.model_path = model_path
        ext = os.path.splitext(model_path)[1].lower()

        # ── TensorRT engine: fastest on Jetson ──
        if ext == ".engine":
            if HAS_TENSORRT:
                return self._load_tensorrt(model_path)
            if HAS_ULTRALYTICS:
                return self._load_ultralytics(model_path)
            return False, ("TensorRT (tensorrt + pycuda) chua cai. Cai pycuda va build engine "
                           "bang trtexec tren chinh Jetson, hoac dung .onnx/.pt.")

        # ── ONNX model: try ONNX Runtime first, then OpenCV DNN ──
        if ext == ".onnx":
            if HAS_ONNXRUNTIME:
                return self._load_onnxruntime(model_path)
            return self._load_opencv_dnn(model_path)

        # ── .pt: need Ultralytics ──
        if not HAS_ULTRALYTICS:
            return False, ("ultralytics is not installed. Export your model to ONNX (best.onnx) "
                           "or build a TensorRT .engine instead.")

        return self._load_ultralytics(model_path)

    def _load_tensorrt(self, model_path):
        """Load a TensorRT .engine and allocate I/O buffers via pycuda."""
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            with open(model_path, "rb") as f, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                return False, ("Khong deserialize duoc TensorRT engine. Engine phai duoc build "
                               "BANG trtexec TREN CHINH may nay (dung version TensorRT).")

            self._trt_engine = engine
            self._trt_context = engine.create_execution_context()
            self._trt_stream = cuda.Stream()
            self._trt_inputs = []
            self._trt_outputs = []
            self._trt_bindings = []

            for binding in engine:
                shape = engine.get_binding_shape(binding)
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                size = abs(int(trt.volume(shape)))
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self._trt_bindings.append(int(device_mem))
                buf = {"host": host_mem, "device": device_mem, "shape": tuple(shape), "dtype": dtype}
                if engine.binding_is_input(binding):
                    self._trt_inputs.append(buf)
                    if len(shape) == 4:           # (1, 3, H, W) → đồng bộ inference_size với engine
                        self.inference_size = int(shape[2])
                else:
                    self._trt_outputs.append(buf)

            self.model_loaded = True
            self.engine_type = "tensorrt"
            self.half = True
            self.device = "cuda"
            return True, "Loaded {} on TensorRT (FP16, {}px)".format(
                os.path.basename(model_path), self.inference_size)
        except Exception as e:
            self.model_loaded = False
            return False, "Failed to load TensorRT engine: {}".format(e)

    def _load_onnxruntime(self, model_path):
        """Load ONNX model using ONNX Runtime with CUDA or CPU."""
        try:
            providers = []
            # Try CUDA first
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self.device = "cuda"
            elif "TensorrtExecutionProvider" in available:
                providers = ["TensorrtExecutionProvider", "CPUExecutionProvider"]
                self.device = "tensorrt"
            else:
                providers = ["CPUExecutionProvider"]
                self.device = "cpu"

            self.model = ort.InferenceSession(model_path, providers=providers)
            self.model_loaded = True
            self.engine_type = "onnxruntime"
            self.half = False

            # Get actual provider used
            active_provider = self.model.get_providers()[0] if self.model.get_providers() else "Unknown"
            return True, "Loaded {} on ONNX Runtime ({})".format(os.path.basename(model_path), active_provider)
        except Exception as e:
            # If ONNX Runtime fails, try OpenCV DNN as fallback
            print("[DetectionEngine] ONNX Runtime failed: {}, trying OpenCV DNN...".format(e))
            return self._load_opencv_dnn(model_path)

    def _load_opencv_dnn(self, model_path):
        """Load ONNX model using OpenCV DNN (basic fallback)."""
        try:
            self.model = cv2.dnn.readNetFromONNX(model_path)
            try:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.device = "cuda"
            except Exception:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.device = "cpu"

            self.model_loaded = True
            self.engine_type = "opencv_dnn"
            self.half = False
            return True, "Loaded {} on OpenCV DNN ({})".format(os.path.basename(model_path), self.device.upper())
        except Exception as e:
            self.model_loaded = False
            return False, "Failed to load ONNX model: {}".format(e)

    def _load_ultralytics(self, model_path):
        """Load model using standard Ultralytics YOLO."""
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
                    print("[DetectionEngine] CUDA: {}, FP16 enabled".format(gpu))
                else:
                    self.device = "cpu"
                    self.half = False
                    print("[DetectionEngine] CUDA not available — running on CPU")
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
            self.engine_type = "ultralytics"
            num_classes = len(self.class_names)
            return True, "Loaded {} on {} — {} class(es)".format(os.path.basename(model_path), self.device.upper(), num_classes)

        except Exception as e:
            self.model_loaded = False
            return False, "Failed to load model: {}".format(e)

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

        if self.engine_type == "tensorrt":
            return self._detect_tensorrt(frame)
        elif self.engine_type == "onnxruntime":
            return self._detect_onnxruntime(frame)
        elif self.engine_type == "opencv_dnn":
            return self._detect_opencv_dnn(frame)
        else:
            return self._detect_ultralytics(frame)

    def _detect_ultralytics(self, frame):
        """Standard Ultralytics YOLO inference."""
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
            self._track_time(t_start, t_end)

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
                        cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else "class_{}".format(cls_id)
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        detections.append((
                            int(x1), int(y1), int(x2), int(y2),
                            conf, cls_name, cx, cy
                        ))

        except Exception as e:
            print("Detection error: {}".format(e))

        return detections

    def _detect_onnxruntime(self, frame):
        """Inference using ONNX Runtime (CUDA or CPU)."""
        detections = []
        try:
            t_start = time.perf_counter()

            img_h, img_w = frame.shape[:2]
            x_factor = img_w / self.inference_size
            y_factor = img_h / self.inference_size

            # Preprocess: resize, normalize, transpose to NCHW
            img = cv2.resize(frame, (self.inference_size, self.inference_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            img = np.expand_dims(img, axis=0)    # Add batch dimension: (1, 3, H, W)

            # Run inference
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: img})

            t_end = time.perf_counter()
            self._track_time(t_start, t_end)

            detections = self._parse_yolo_output(outputs[0], x_factor, y_factor)

        except Exception as e:
            print("ONNX Runtime Detection error: {}".format(e))

        return detections

    def _detect_opencv_dnn(self, frame):
        """Inference using pure OpenCV DNN with ONNX model (basic fallback)."""
        detections = []
        try:
            t_start = time.perf_counter()

            img_h, img_w = frame.shape[:2]
            x_factor = img_w / self.inference_size
            y_factor = img_h / self.inference_size

            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (self.inference_size, self.inference_size), swapRB=True, crop=False)
            self.model.setInput(blob)
            outputs = self.model.forward()

            t_end = time.perf_counter()
            self._track_time(t_start, t_end)

            detections = self._parse_yolo_output(outputs, x_factor, y_factor)

        except Exception as e:
            print("OpenCV DNN Detection error: {}".format(e))

        return detections

    def _detect_tensorrt(self, frame):
        """Inference using a TensorRT engine (FP16) via pycuda buffers."""
        detections = []
        try:
            t_start = time.perf_counter()

            img_h, img_w = frame.shape[:2]
            x_factor = img_w / self.inference_size
            y_factor = img_h / self.inference_size

            # Preprocess: resize, BGR->RGB, /255, HWC->CHW, contiguous
            img = cv2.resize(frame, (self.inference_size, self.inference_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.ascontiguousarray(img, dtype=self._trt_inputs[0]["dtype"])

            inp = self._trt_inputs[0]
            np.copyto(inp["host"], img.ravel())
            cuda.memcpy_htod_async(inp["device"], inp["host"], self._trt_stream)
            self._trt_context.execute_async_v2(
                bindings=self._trt_bindings, stream_handle=self._trt_stream.handle)
            for out in self._trt_outputs:
                cuda.memcpy_dtoh_async(out["host"], out["device"], self._trt_stream)
            self._trt_stream.synchronize()

            t_end = time.perf_counter()
            self._track_time(t_start, t_end)

            # Detection head = output có nhiều phần tử nhất (model 1 output thì lấy luôn nó)
            out = max(self._trt_outputs, key=lambda o: int(np.prod(o["shape"])))
            output = out["host"].reshape(out["shape"])
            detections = self._parse_yolo_output(output, x_factor, y_factor)

        except Exception as e:
            print("TensorRT Detection error: {}".format(e))

        return detections

    def _parse_yolo_output(self, output, x_factor, y_factor):
        """Parse raw YOLO output → list detections, tự nhận diện v5/v8/v11.

        - YOLOv8/v11: (1, 4+nc, N) → không objectness, conf = max(class_scores).
        - YOLOv5:     (1, N, 5+nc) → có objectness, conf = obj * max(class_scores).
        Tự xoay chiều (boxes ở chiều lớn hơn) và tự đoán có/không objectness theo số đặc trưng.
        """
        arr = np.asarray(output)
        # Bỏ batch dim → còn 2D
        while arr.ndim > 2:
            arr = arr[0]
        if arr.ndim != 2:
            return []
        # Xoay về (num_boxes, num_feat): num_boxes là chiều lớn hơn
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T

        num_feat = arr.shape[1]
        nc = len(self.class_names)
        if num_feat == 5 + nc:
            has_obj = True            # YOLOv5
        elif num_feat == 4 + nc:
            has_obj = False           # YOLOv8/v11
        else:
            has_obj = num_feat >= 6   # best-effort khi nc không khớp class_names

        # ===== Vector hóa bằng numpy (thay vòng for duyệt từng box) =====
        # Vòng for Python duyệt toàn bộ ~N box mỗi frame (vd ~2100 ở 320px) là nút
        # thắt CPU lớn nhất trên Jetson Nano và chạy NGOÀI đồng hồ đo inference.
        # numpy hoá → từ hàng nghìn vòng lặp Python còn vài phép toán mảng.
        arr = arr.astype(np.float32, copy=False)
        if has_obj:
            obj = arr[:, 4]
            cls_scores = arr[:, 5:]
        else:
            obj = None
            cls_scores = arr[:, 4:]

        if cls_scores.shape[1] == 0:
            return []

        class_ids_all = np.argmax(cls_scores, axis=1)
        top_scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids_all]
        confs_all = top_scores if obj is None else obj * top_scores

        keep = confs_all >= self.confidence
        if not np.any(keep):
            return []

        m = arr[keep]
        confs_k = confs_all[keep]
        cls_k = class_ids_all[keep]

        xc = m[:, 0]; yc = m[:, 1]; ww = m[:, 2]; hh = m[:, 3]
        xs = ((xc - ww / 2.0) * x_factor).astype(np.int32)
        ys = ((yc - hh / 2.0) * y_factor).astype(np.int32)
        ws = (ww * x_factor).astype(np.int32)
        hs = (hh * y_factor).astype(np.int32)

        boxes = np.stack([xs, ys, ws, hs], axis=1).tolist()
        confidences = confs_k.astype(np.float32).tolist()
        class_ids = cls_k.tolist()

        detections = []
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.iou_threshold)
        if len(indices) > 0:
            for i in np.array(indices).flatten():
                x, y, w, h = boxes[i]
                cls_id = class_ids[i]
                cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) \
                    else "class_{}".format(cls_id)
                cx = x + w / 2.0
                cy = y + h / 2.0
                detections.append((
                    int(x), int(y), int(x + w), int(y + h),
                    float(confidences[i]), cls_name, cx, cy
                ))
        return detections

    def _track_time(self, t_start, t_end):
        """Track inference time for FPS calculation."""
        inference_ms = (t_end - t_start) * 1000
        self._inference_times.append(inference_ms)
        if len(self._inference_times) > 30:
            self._inference_times.pop(0)
        self.avg_inference_ms = sum(self._inference_times) / len(self._inference_times)

    def draw_detections(self, frame, detections):
        """Draw detection boxes and labels on frame."""
        for det in detections:
            x1, y1, x2, y2, conf, cls_name, cx, cy = det
            color = (0, 255, 100)  # green
            thickness = 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

            label = "{} {:.2f}".format(cls_name, conf)
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
            "engine_type": self.engine_type,
        }
        return info
