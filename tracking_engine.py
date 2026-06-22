"""
Tracking Engine Module
======================
Converts detection pixel coordinates to servo angles using proper
geometry-based mapping for the 2-DOF laser turret.

Hardware geometry (all in cm):
  - Camera FOV on wall: 82cm wide × 33cm tall
  - Distance from camera/laser to wall: 70cm
  - Camera height from ground: 16cm
  - Laser height from ground: 16cm (top of arm at home)
  - Camera-to-laser horizontal offset: 11cm
  - Arm: base width 5cm, shoulder→elbow 5cm, elbow→laser tip 3cm

Coordinate system (looking AT the wall from behind):
  - X: horizontal, positive = right
  - Y: vertical, positive = up
  - Z: depth, towards wall

Servo HOME position (laser perpendicular to wall):
  - Pan  (shoulder, D26): HOME = 90°
    - < 90° = turn left, > 90° = turn right
  - Tilt (elbow, D25):   HOME = 90°
    - < 90° = tilt up,  > 90° = tilt down
"""

import math


class TrackingEngine:
    """
    Converts detection pixel coordinates to servo angles.

    Pipeline:
      1. Pixel (px, py) → Wall offset (cm) from camera optical center
      2. Wall offset → Laser target (cm) relative to laser base
      3. Laser target → Pan/Tilt servo angles (degrees)
      4. Fast exponential moving average (EMA) smoothing with dead-zone

    Servo HOME (laser perpendicular to wall):
      - Pan  = 90°  (shoulder servo, D26)
      - Tilt = 90°  (elbow servo, D25)
    """

    # ========== Hardware Constants (cm) ==========
    WALL_FOV_WIDTH  = 82.0   # Camera FOV width projected on wall
    WALL_FOV_HEIGHT = 33.0   # Camera FOV height projected on wall
    DIST_TO_WALL    = 70.0   # Distance from camera/laser assembly to wall

    CAM_HEIGHT      = 16.0   # Camera lens height from ground
    LASER_HEIGHT    = 16.0   # Laser tip height from ground (at home position)

    # Horizontal distance between camera lens and laser tip
    # Positive = camera is to the LEFT of laser when looking from behind
    CAM_LASER_H_OFFSET = 11.0

    # Vertical offset: laser tip is higher than camera
    # Camera and laser at same height → offset = 0
    CAM_LASER_V_OFFSET = 0.0

    # ========== Servo HOME angles (laser perpendicular to wall) ==========
    PAN_HOME  = 90.0    # Shoulder servo home angle
    TILT_HOME = 90.0    # Elbow servo home angle

    def __init__(self):
        # Current servo angles (start at home)
        self.pan_angle  = self.PAN_HOME
        self.tilt_angle = self.TILT_HOME

        # Smoothing (exponential moving average)
        self.smooth_factor = 1.0  # 1.0 = instant response (fastest tracking)
        self.smoothed_pan  = self.PAN_HOME
        self.smoothed_tilt = self.TILT_HOME

        # Dead zone in degrees — prevents micro-jitter
        self.dead_zone_deg = 0.3

        # Servo angle limits
        self.pan_min  = 30.0
        self.pan_max  = 160.0
        self.tilt_min = 60.0
        self.tilt_max = 170.0

        # Frame dimensions (updated each frame)
        self.frame_w = 640
        self.frame_h = 480

        # Target loss handling
        self.frames_without_target = 0
        self.max_loss_frames = 45  # ~1.5s at 30fps before drifting home

        # ========== Target Persistence (auto-handoff) ==========
        # Last known target center (pixel coords)
        self.last_target_cx = None
        self.last_target_cy = None
        # Max pixel distance to consider same target (prevents jumping)
        self.target_match_radius = 120  # pixels
        # Frames since tracked target was last seen (not same as frames_without_target)
        self.target_lost_frames = 0
        # How many frames to wait before declaring target truly gone
        self.target_lost_patience = 5  # ~0.17s at 30fps

        # Calibration fine-tune offsets (adjustable via app ±15°)
        self.cal_pan_offset  = 8.0   # degrees (calibrated from real testing)
        self.cal_tilt_offset = -8.0  # degrees (calibrated from real testing)

        # Bù cố định do thay servo mới (servo cũ HOME=125°, servo mới HOME=90°)
        # Giá trị này KHÔNG chỉnh từ app — chỉ sửa ở đây khi thay servo.
        self.tilt_servo_offset = -25.0

        # Direction multipliers (1 or -1) — adjust if servo moves backwards
        # Default: 1 (PID error = desired - current, applied directly)
        self.pan_direction  = 1
        self.tilt_direction = 1

    # ========== Configuration ==========

    def set_frame_size(self, w, h):
        """Update frame dimensions (called each frame)."""
        self.frame_w = w
        self.frame_h = h

    def set_smooth_factor(self, factor):
        """Set smoothing factor (0.1 = very smooth, 1.0 = instant)."""
        self.smooth_factor = max(0.1, min(1.0, factor))

    def reset(self):
        """Reset to home position (Pan=90°, Tilt=90°)."""
        self.pan_angle     = self.PAN_HOME
        self.tilt_angle    = self.TILT_HOME
        self.smoothed_pan  = self.PAN_HOME
        self.smoothed_tilt = self.TILT_HOME
        self.frames_without_target = 0
        self.last_target_cx = None
        self.last_target_cy = None
        self.target_lost_frames = 0

    # ========== Target Selection ==========

    def select_target(self, detections):
        """
        Smart target selection with persistence and auto-handoff.

        Strategy:
          1. If we have a tracked target, find the closest detection
             to its last known position (within match_radius).
          2. If tracked target is gone for a few frames, switch to
             the nearest detection to the current laser aim point.
          3. If no previous target, pick the largest detection.

        Args:
            detections: list of (x1, y1, x2, y2, conf, cls_name, cx, cy)

        Returns:
            Selected detection tuple, or None if no detections.
        """
        if not detections:
            self.target_lost_frames += 1
            return None

        # === Case 1: We have a tracked target — match by proximity ===
        if self.last_target_cx is not None:
            best_match = None
            best_dist = float('inf')

            for det in detections:
                cx, cy = det[6], det[7]
                dist = math.sqrt(
                    (cx - self.last_target_cx) ** 2 +
                    (cy - self.last_target_cy) ** 2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_match = det

            # If close enough, it's the same target
            if best_dist <= self.target_match_radius:
                self.target_lost_frames = 0
                self._update_target_position(best_match)
                return best_match

            # Target not found this frame
            self.target_lost_frames += 1

            # Wait a few frames before switching (brief occlusion protection)
            if self.target_lost_frames <= self.target_lost_patience:
                # Target temporarily lost — keep aiming at last position
                # but don't update target
                return None

            # === Case 2: Target truly gone — handoff to nearest remaining ===
            # Pick the detection closest to current laser aim point
            # (frame center offset by current servo angles)
            aim_cx = self.frame_w / 2
            aim_cy = self.frame_h / 2

            nearest = min(detections, key=lambda d: math.sqrt(
                (d[6] - aim_cx) ** 2 + (d[7] - aim_cy) ** 2
            ))
            self.target_lost_frames = 0
            self._update_target_position(nearest)
            return nearest

        # === Case 3: No previous target — pick the largest detection ===
        largest = max(detections, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
        self.target_lost_frames = 0
        self._update_target_position(largest)
        return largest

    def _update_target_position(self, detection):
        """Update last known target position from a detection."""
        self.last_target_cx = detection[6]  # cx
        self.last_target_cy = detection[7]  # cy

    # ========== Core Geometry ==========

    def pixel_to_wall_cm(self, px, py):
        """
        Convert pixel coordinates to wall offset (cm) relative to
        the camera's optical center.

        Returns (wx, wy):
          wx: horizontal offset on wall (positive = right of camera center)
          wy: vertical offset on wall (positive = below camera center)
        """
        # Normalize to [-0.5, +0.5]
        nx = (px / self.frame_w) - 0.5   # positive = right
        ny = (py / self.frame_h) - 0.5   # positive = down (image coords)

        # Convert to wall cm
        wx = nx * self.WALL_FOV_WIDTH    # cm, positive = right
        wy = ny * self.WALL_FOV_HEIGHT   # cm, positive = down

        return wx, wy

    def wall_to_laser_target(self, wx, wy):
        """
        Convert wall offset (from camera center) to target position
        relative to the laser base, accounting for camera-laser offset.

        Returns (tx, ty):
          tx: horizontal offset from laser (cm, positive = right)
          ty: vertical offset from laser (cm, positive = up)
        """
        # Camera sees from its own position, but laser needs to aim from
        # a different position (11cm away horizontally, same height).
        #
        # Camera at position (cam_x, cam_y) = (0, 16cm)
        # Laser at position (laser_x, laser_y) = (11, 16cm)
        # (camera is 11cm to the left of laser)
        #
        # Camera sees a target at wall offset (wx, wy) from its center.
        # Actual wall position: (cam_x + wx, cam_y - wy) = (wx, 16 - wy)
        # Laser needs to aim at: (wx - 11, 16 - wy - 16) = (wx - 11, -wy)
        #
        # tx = wx - CAM_LASER_H_OFFSET (target is to the left from laser's perspective)
        # ty = -CAM_LASER_V_OFFSET - wy (target is below laser, wy positive=down)

        tx = wx - self.CAM_LASER_H_OFFSET
        ty = -self.CAM_LASER_V_OFFSET - wy

        return tx, ty

    def target_to_desired_angles(self, tx, ty):
        """
        Convert laser target position (cm, relative to laser base) to
        desired servo angles using trigonometry.

        Home position (laser straight at wall):
          Pan  = 90° (shoulder), offset from center
          Tilt = 90° (elbow), offset from center

        When target is to the right (tx > 0), pan angle decreases (inverted for servo).
        When target is above (ty > 0), tilt angle increases (inverted for servo).
        """
        # Pan: horizontal angle offset from home (INVERTED for servo direction)
        pan_rad = math.atan2(tx, self.DIST_TO_WALL)
        desired_pan = self.PAN_HOME - math.degrees(pan_rad)

        # Tilt: vertical angle offset from home (INVERTED for servo direction)
        tilt_rad = math.atan2(ty, self.DIST_TO_WALL)
        desired_tilt = self.TILT_HOME + math.degrees(tilt_rad)

        return desired_pan, desired_tilt

    # ========== Main Update ==========

    def update(self, detection=None):
        """
        Update tracking with a detection.

        Uses direct exponential interpolation toward the computed target
        angle. This is more stable than PID for our use case since we
        compute exact desired angles from geometry (no feedback loop needed).

        Args:
            detection: (cx, cy, w, h) in pixels, or None if lost.

        Returns:
            (pan_angle, tilt_angle, is_tracking)
        """
        if detection is None:
            return self._handle_target_lost()

        self.frames_without_target = 0
        cx, cy, det_w, det_h = detection

        # === Step 1: Pixel → Wall offset (cm) ===
        wx, wy = self.pixel_to_wall_cm(cx, cy)

        # === Step 2: Wall offset → Laser target (cm) ===
        tx, ty = self.wall_to_laser_target(wx, wy)

        # === Step 3: Compute desired servo angles ===
        desired_pan, desired_tilt = self.target_to_desired_angles(tx, ty)

        # === Step 4: Apply calibration offsets ===
        desired_pan  += self.cal_pan_offset
        desired_tilt += self.tilt_servo_offset + self.cal_tilt_offset

        # Clamp desired to servo limits
        desired_pan  = max(self.pan_min, min(self.pan_max, desired_pan))
        desired_tilt = max(self.tilt_min, min(self.tilt_max, desired_tilt))

        # === Step 5: Dead zone — ignore tiny movements ===
        error_pan  = desired_pan  - self.pan_angle
        error_tilt = desired_tilt - self.tilt_angle

        if abs(error_pan) > self.dead_zone_deg:
            # Interpolate toward desired angle (smooth_factor controls speed)
            self.pan_angle += self.smooth_factor * error_pan
        if abs(error_tilt) > self.dead_zone_deg:
            self.tilt_angle += self.smooth_factor * error_tilt

        # Clamp to limits
        self.pan_angle  = max(self.pan_min, min(self.pan_max, self.pan_angle))
        self.tilt_angle = max(self.tilt_min, min(self.tilt_max, self.tilt_angle))

        # === Step 6: Output smoothing (EMA bypassed/instant for maximum speed) ===
        self.smoothed_pan  += 1.0 * (self.pan_angle  - self.smoothed_pan)
        self.smoothed_tilt += 1.0 * (self.tilt_angle - self.smoothed_tilt)

        return self.smoothed_pan, self.smoothed_tilt, True

    def _handle_target_lost(self):
        """Handle frames where no target is detected."""
        self.frames_without_target += 1

        if self.frames_without_target > self.max_loss_frames:
            # Slowly drift back to home position
            home_speed = 0.03  # degrees per frame toward home
            self.pan_angle  += (self.PAN_HOME  - self.pan_angle)  * home_speed
            self.tilt_angle += (self.TILT_HOME - self.tilt_angle) * home_speed
            self.smoothed_pan  = self.pan_angle
            self.smoothed_tilt = self.tilt_angle

        return self.smoothed_pan, self.smoothed_tilt, False

    def get_angles(self):
        """Return current smoothed angles."""
        return self.smoothed_pan, self.smoothed_tilt

    def get_debug_info(self):
        """Return debug information for GUI display."""
        return {
            "pan": self.smoothed_pan,
            "tilt": self.smoothed_tilt,
            "raw_pan": self.pan_angle,
            "raw_tilt": self.tilt_angle,
            "lost_frames": self.frames_without_target,
        }
