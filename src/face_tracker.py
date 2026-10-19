"""
Face tracking and landmark estimation module using MediaPipe Face Mesh.
Features:
- Sub-pixel landmark detection with iris refinement
- In-plane roll angle and approximate 3D head pose estimation (yaw, pitch)
- Exponential Moving Average (EMA) smoothing for jitter-free overlay tracking
- Automatic recovery when faces disappear and reappear
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2

try:
    import mediapipe as mp
except ImportError:
    mp = None

logger = logging.getLogger(__name__)


# Standard MediaPipe Face Mesh Key Landmark Indices
IDX_FOREHEAD = 10
IDX_NOSE_TIP = 1
IDX_NOSE_BRIDGE = 168
IDX_CHIN = 152
IDX_LEFT_TEMPLE = 234    # Person's right, viewer's left
IDX_RIGHT_TEMPLE = 454   # Person's left, viewer's right
IDX_LEFT_EYE_OUTER = 33
IDX_LEFT_EYE_INNER = 133
IDX_RIGHT_EYE_INNER = 362
IDX_RIGHT_EYE_OUTER = 263
IDX_MOUTH_TOP = 13
IDX_MOUTH_BOTTOM = 14
IDX_LEFT_EYEBROW = 70
IDX_RIGHT_EYEBROW = 300


@dataclass
class FaceData:
    """Structured container for real-time facial tracking data."""
    detected: bool = False
    landmarks_px: Optional[np.ndarray] = None  # (N, 2) pixel coords
    center: Tuple[int, int] = (0, 0)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    roll_deg: float = 0.0      # In-plane rotation (head tilt)
    yaw_deg: float = 0.0       # Left/Right head turn
    pitch_deg: float = 0.0     # Up/Down head tilt
    face_width: float = 0.0
    face_height: float = 0.0
    eye_distance: float = 0.0
    left_eye: Tuple[int, int] = (0, 0)
    right_eye: Tuple[int, int] = (0, 0)
    eyes_midpoint: Tuple[int, int] = (0, 0)
    nose_tip: Tuple[int, int] = (0, 0)
    forehead: Tuple[int, int] = (0, 0)
    chin: Tuple[int, int] = (0, 0)
    mouth_center: Tuple[int, int] = (0, 0)


class FaceTracker:
    """Tracks facial landmarks using MediaPipe Face Mesh with jitter reduction."""

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smoothing_alpha: float = 0.72,
        enable_smoothing: bool = True
    ) -> None:
        if mp is None:
            raise RuntimeError("MediaPipe is not installed. Please install mediapipe.")

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.smoothing_alpha = smoothing_alpha
        self.enable_smoothing = enable_smoothing

        # State for temporal smoothing
        self.smoothed_landmarks: Optional[np.ndarray] = None
        self.frames_since_last_detection: int = 999
        self.max_coast_frames: int = 4  # Reset smoothing if lost for >4 frames

    def process(self, frame_bgr: np.ndarray) -> FaceData:
        """
        Processes a BGR image frame and extracts stabilized facial metrics.
        Returns a FaceData instance.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return FaceData(detected=False)

        h, w = frame_bgr.shape[:2]

        # MediaPipe requires RGB format
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Performance optimization: set writeable to false before passing to MediaPipe
        frame_rgb.flags.writeable = False
        results = self.face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            self.frames_since_last_detection += 1
            if self.frames_since_last_detection > self.max_coast_frames:
                self.smoothed_landmarks = None
            return FaceData(detected=False)

        self.frames_since_last_detection = 0
        raw_face = results.multi_face_landmarks[0]

        # Convert normalized coordinates (0..1) to pixel coordinates (x, y)
        current_landmarks = np.array(
            [[lm.x * w, lm.y * h] for lm in raw_face.landmark],
            dtype=np.float32
        )

        # Apply Exponential Moving Average (EMA) smoothing
        if self.enable_smoothing:
            if self.smoothed_landmarks is None or len(self.smoothed_landmarks) != len(current_landmarks):
                self.smoothed_landmarks = current_landmarks.copy()
            else:
                self.smoothed_landmarks = (
                    self.smoothing_alpha * current_landmarks +
                    (1.0 - self.smoothing_alpha) * self.smoothed_landmarks
                )
            pts = self.smoothed_landmarks
        else:
            pts = current_landmarks

        # Extract primary landmark points
        def get_pt(idx: int) -> Tuple[int, int]:
            return int(round(pts[idx][0])), int(round(pts[idx][1]))

        forehead = get_pt(IDX_FOREHEAD)
        nose_tip = get_pt(IDX_NOSE_TIP)
        chin = get_pt(IDX_CHIN)
        left_temple = get_pt(IDX_LEFT_TEMPLE)
        right_temple = get_pt(IDX_RIGHT_TEMPLE)

        # Calculate eye centers from corner landmarks
        l_outer = pts[IDX_LEFT_EYE_OUTER]
        l_inner = pts[IDX_LEFT_EYE_INNER]
        left_eye_center = (
            int(round((l_outer[0] + l_inner[0]) / 2.0)),
            int(round((l_outer[1] + l_inner[1]) / 2.0))
        )

        r_inner = pts[IDX_RIGHT_EYE_INNER]
        r_outer = pts[IDX_RIGHT_EYE_OUTER]
        right_eye_center = (
            int(round((r_inner[0] + r_outer[0]) / 2.0)),
            int(round((r_inner[1] + r_outer[1]) / 2.0))
        )

        eyes_midpoint = (
            int(round((left_eye_center[0] + right_eye_center[0]) / 2.0)),
            int(round((left_eye_center[1] + right_eye_center[1]) / 2.0))
        )

        mouth_top = pts[IDX_MOUTH_TOP]
        mouth_bottom = pts[IDX_MOUTH_BOTTOM]
        mouth_center = (
            int(round((mouth_top[0] + mouth_bottom[0]) / 2.0)),
            int(round((mouth_top[1] + mouth_bottom[1]) / 2.0))
        )

        # Calculate distances & scales
        dx_eyes = right_eye_center[0] - left_eye_center[0]
        dy_eyes = right_eye_center[1] - left_eye_center[1]
        eye_distance = float(math.hypot(dx_eyes, dy_eyes))

        # Roll angle (degrees): tilt of the head in camera image plane
        roll_rad = math.atan2(dy_eyes, dx_eyes)
        roll_deg = math.degrees(roll_rad)

        # Face width (temple-to-temple) and height (forehead-to-chin)
        face_width = float(math.hypot(right_temple[0] - left_temple[0], right_temple[1] - left_temple[1]))
        face_height = float(math.hypot(chin[0] - forehead[0], chin[1] - forehead[1]))

        # Approximate yaw (left/right turn) using nose position relative to eye midpoint
        if eye_distance > 1e-3:
            # Vector along eye line
            eye_unit_x = dx_eyes / eye_distance
            eye_unit_y = dy_eyes / eye_distance
            # Vector from eyes midpoint to nose tip
            nose_vec_x = nose_tip[0] - eyes_midpoint[0]
            nose_vec_y = nose_tip[1] - eyes_midpoint[1]
            # Projection of nose onto eye line
            nose_proj = (nose_vec_x * eye_unit_x + nose_vec_y * eye_unit_y)
            # Normalized asymmetry (-1.0 to +1.0)
            yaw_ratio = nose_proj / (eye_distance * 0.5)
            yaw_deg = float(np.clip(yaw_ratio * 45.0, -60.0, 60.0))
        else:
            yaw_deg = 0.0

        # Approximate pitch (up/down tilt)
        if face_height > 1e-3:
            dist_forehead_nose = math.hypot(nose_tip[0] - forehead[0], nose_tip[1] - forehead[1])
            dist_nose_chin = math.hypot(chin[0] - nose_tip[0], chin[1] - nose_tip[1])
            pitch_ratio = (dist_forehead_nose - dist_nose_chin) / face_height
            pitch_deg = float(np.clip(pitch_ratio * 40.0, -50.0, 50.0))
        else:
            pitch_deg = 0.0

        # Bounding box
        min_x = int(max(0, np.min(pts[:, 0])))
        max_x = int(min(w, np.max(pts[:, 0])))
        min_y = int(max(0, np.min(pts[:, 1])))
        max_y = int(min(h, np.max(pts[:, 1])))
        bbox = (min_x, min_y, max(1, max_x - min_x), max(1, max_y - min_y))

        # Overall face center
        center = (
            int(round((min_x + max_x) / 2.0)),
            int(round((min_y + max_y) / 2.0))
        )

        return FaceData(
            detected=True,
            landmarks_px=pts,
            center=center,
            bbox=bbox,
            roll_deg=roll_deg,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            face_width=face_width,
            face_height=face_height,
            eye_distance=eye_distance,
            left_eye=left_eye_center,
            right_eye=right_eye_center,
            eyes_midpoint=eyes_midpoint,
            nose_tip=nose_tip,
            forehead=forehead,
            chin=chin,
            mouth_center=mouth_center
        )

    def draw_debug_landmarks(
        self,
        frame_bgr: np.ndarray,
        face_data: FaceData,
        draw_contours: bool = True
    ) -> np.ndarray:
        """Draws aesthetic tracking points, face outline, and orientation axis."""
        if not face_data.detected or face_data.landmarks_px is None:
            return frame_bgr

        overlay = frame_bgr.copy()

        # Draw subtle landmark points
        for pt in face_data.landmarks_px[::4]:  # Every 4th point for speed and clarity
            cv2.circle(overlay, (int(pt[0]), int(pt[1])), 1, (0, 220, 255), -1)

        # Draw key anchors
        anchors = [
            (face_data.left_eye, (255, 100, 0), "L Eye"),
            (face_data.right_eye, (0, 100, 255), "R Eye"),
            (face_data.nose_tip, (0, 255, 0), "Nose"),
            (face_data.forehead, (255, 0, 255), "Forehead"),
            (face_data.chin, (255, 255, 0), "Chin"),
            (face_data.mouth_center, (0, 255, 255), "Mouth")
        ]
        for (x, y), color, name in anchors:
            cv2.circle(overlay, (x, y), 4, color, -1)
            cv2.circle(overlay, (x, y), 6, (255, 255, 255), 1)

        # Eye connector line
        cv2.line(overlay, face_data.left_eye, face_data.right_eye, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw roll angle indicator
        cx, cy = face_data.eyes_midpoint
        length = int(face_data.eye_distance * 0.6)
        rad = math.radians(face_data.roll_deg)
        end_x = int(cx + length * math.cos(rad))
        end_y = int(cy + length * math.sin(rad))
        cv2.arrowedLine(overlay, (cx, cy), (end_x, end_y), (0, 0, 255), 2, tipLength=0.3)

        # Alpha blend overlay for smooth look
        return cv2.addWeighted(overlay, 0.8, frame_bgr, 0.2, 0)

    def release(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, 'face_mesh') and self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None

    def __del__(self) -> None:
        self.release()
