"""
Character transformation and facial overlay compositor.
Tracks 468 3D facial landmarks with MediaPipe Face Mesh:
- Continuously calculates eye centers, face center, interpupillary scale, and head roll angle
- Affine-warps and rotates transparent superhero masks and visors in real time
- Keeps overlays locked to the user's face during head movement and rotation
- Supports chest emblems anchored to upper torso
"""

import math
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

logger = logging.getLogger(__name__)

# Key Face Mesh Landmark Indices
IDX_FOREHEAD = 10
IDX_NOSE_TIP = 1
IDX_CHIN = 152
IDX_LEFT_TEMPLE = 234
IDX_RIGHT_TEMPLE = 454
IDX_LEFT_EYE_OUTER = 33
IDX_LEFT_EYE_INNER = 133
IDX_RIGHT_EYE_INNER = 362
IDX_RIGHT_EYE_OUTER = 263


class CharacterCompositor:
    """Renders face masks, eye lenses, visors, and chest emblems tracking facial landmarks."""

    def __init__(self, characters_dir: str = "assets/characters") -> None:
        self.characters_dir = Path(characters_dir)
        self.character_files: List[Path] = []
        self.cached_overlays: Dict[str, np.ndarray] = {}
        self.active_index: int = 0
        self.is_enabled: bool = False

        # MediaPipe Face Mesh for real-time sub-pixel facial tracking
        if mp is not None:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.55
            )
        else:
            self.face_mesh = None

        # Smoothing history for face metrics
        self.smoothed_center: Optional[Tuple[float, float]] = None
        self.smoothed_scale: Optional[float] = None
        self.smoothed_roll: Optional[float] = None
        self.smoothing_alpha: float = 0.82

        self.reload()

    def reload(self) -> None:
        """Discovers character assets in assets/characters/."""
        self.character_files.clear()
        self.cached_overlays.clear()

        if self.characters_dir.exists():
            for p in sorted(self.characters_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in [".png", ".webp"]:
                    self.character_files.append(p)

        if self.active_index >= len(self.character_files):
            self.active_index = 0

        logger.info(f"Loaded {len(self.character_files)} character overlays from {self.characters_dir}")

    def get_current_character_name(self) -> str:
        if self.character_files:
            return self.character_files[self.active_index].stem.replace("_", " ").title()
        return "None"

    def next_character(self) -> str:
        if self.character_files:
            self.active_index = (self.active_index + 1) % len(self.character_files)
            return self.get_current_character_name()
        return "None"

    def prev_character(self) -> str:
        if self.character_files:
            self.active_index = (self.active_index - 1) % len(self.character_files)
            return self.get_current_character_name()
        return "None"

    def toggle(self) -> bool:
        self.is_enabled = not self.is_enabled
        return self.is_enabled

    def composite(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts face landmarks and composites the active character overlay.
        The overlay continuously follows the face's position, scale, and head roll angle.
        """
        if not self.is_enabled or not self.character_files or self.face_mesh is None:
            return frame_bgr

        char_path = self.character_files[self.active_index]
        if char_path.name not in self.cached_overlays:
            overlay_bgra = cv2.imread(str(char_path), cv2.IMREAD_UNCHANGED)
            if overlay_bgra is None:
                return frame_bgr
            self.cached_overlays[char_path.name] = overlay_bgra

        overlay = self.cached_overlays[char_path.name]
        h, w = frame_bgr.shape[:2]

        # Extract Face Mesh landmarks
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            # Face not detected: return frame
            self.smoothed_center = None
            return frame_bgr

        face_lms = results.multi_face_landmarks[0].landmark
        pts = np.array([[lm.x * w, lm.y * h] for lm in face_lms], dtype=np.float32)

        # Eye centers
        l_outer, l_inner = pts[IDX_LEFT_EYE_OUTER], pts[IDX_LEFT_EYE_INNER]
        left_eye = ((l_outer[0] + l_inner[0]) / 2.0, (l_outer[1] + l_inner[1]) / 2.0)

        r_inner, r_outer = pts[IDX_RIGHT_EYE_INNER], pts[IDX_RIGHT_EYE_OUTER]
        right_eye = ((r_inner[0] + r_outer[0]) / 2.0, (r_inner[1] + r_outer[1]) / 2.0)

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        raw_eye_dist = max(10.0, float(math.hypot(dx, dy)))

        # Head roll angle in degrees (tilt)
        raw_roll_deg = math.degrees(math.atan2(dy, dx))

        # Face center
        eyes_mid = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
        nose = pts[IDX_NOSE_TIP]
        raw_center = (float(eyes_mid[0] * 0.4 + nose[0] * 0.6), float(eyes_mid[1] * 0.4 + nose[1] * 0.6))

        # Apply low-latency EMA smoothing
        if self.smoothed_center is not None:
            cx = self.smoothing_alpha * raw_center[0] + (1.0 - self.smoothing_alpha) * self.smoothed_center[0]
            cy = self.smoothing_alpha * raw_center[1] + (1.0 - self.smoothing_alpha) * self.smoothed_center[1]
            eye_dist = self.smoothing_alpha * raw_eye_dist + (1.0 - self.smoothing_alpha) * self.smoothed_scale
            roll_deg = self.smoothing_alpha * raw_roll_deg + (1.0 - self.smoothing_alpha) * self.smoothed_roll
        else:
            cx, cy = raw_center
            eye_dist = raw_eye_dist
            roll_deg = raw_roll_deg

        self.smoothed_center = (cx, cy)
        self.smoothed_scale = eye_dist
        self.smoothed_roll = roll_deg

        # Determine overlay anchor mode based on asset filename
        fname = char_path.name.lower()
        if "emblem" in fname or "chest" in fname:
            # Chest emblem: projected down from chin along face vertical axis
            rad = math.radians(roll_deg + 90.0)
            chin = pts[IDX_CHIN]
            emblem_cx = int(chin[0] + eye_dist * 1.5 * math.cos(rad))
            emblem_cy = int(chin[1] + eye_dist * 1.5 * math.sin(rad))
            target_w = int(eye_dist * 2.2)
            return self._render_rotated_overlay(frame_bgr, overlay, (emblem_cx, emblem_cy), target_w, -roll_deg)

        elif "mask" in fname:
            # Full Face Mask: scaled to face width and height
            target_w = int(eye_dist * 3.4)
            # Offset slightly upward towards forehead
            rad = math.radians(roll_deg - 90.0)
            mask_cx = int(cx + eye_dist * 0.15 * math.cos(rad))
            mask_cy = int(cy + eye_dist * 0.15 * math.sin(rad))
            return self._render_rotated_overlay(frame_bgr, overlay, (mask_cx, mask_cy), target_w, -roll_deg)

        else:
            # Eye lenses or Visor: anchored directly on eye midpoint
            target_w = int(eye_dist * 2.6)
            eye_cx = int(eyes_mid[0])
            eye_cy = int(eyes_mid[1])
            return self._render_rotated_overlay(frame_bgr, overlay, (eye_cx, eye_cy), target_w, -roll_deg)

    def _render_rotated_overlay(
        self,
        frame_bgr: np.ndarray,
        overlay_bgra: np.ndarray,
        center: Tuple[int, int],
        target_w: int,
        angle_deg: float
    ) -> np.ndarray:
        """Scales, rotates with expanded canvas, and alpha-blends overlay onto frame."""
        h, w = frame_bgr.shape[:2]
        img_h, img_w = overlay_bgra.shape[:2]

        if img_h == 0 or img_w == 0 or target_w <= 4:
            return frame_bgr

        target_w = max(10, min(target_w, w * 2))
        target_h = max(10, int(target_w * (img_h / float(img_w))))

        resized = cv2.resize(overlay_bgra, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Rotate with alpha channel & expanded bounds
        cx_sub, cy_sub = target_w / 2.0, target_h / 2.0
        M = cv2.getRotationMatrix2D((cx_sub, cy_sub), angle_deg, 1.0)
        cos_val = abs(M[0, 0])
        sin_val = abs(M[0, 1])
        new_w = int(math.ceil((target_h * sin_val) + (target_w * cos_val)))
        new_h = int(math.ceil((target_h * cos_val) + (target_w * sin_val)))
        M[0, 2] += (new_w / 2.0) - cx_sub
        M[1, 2] += (new_h / 2.0) - cy_sub

        rotated = cv2.warpAffine(
            resized, M, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        # Place onto canvas at center
        cx, cy = center
        x1 = cx - new_w // 2
        y1 = cy - new_h // 2
        x2 = x1 + new_w
        y2 = y1 + new_h

        fx1 = max(0, x1)
        fy1 = max(0, y1)
        fx2 = min(w, x2)
        fy2 = min(h, y2)

        if fx1 >= fx2 or fy1 >= fy2:
            return frame_bgr

        ox1 = fx1 - x1
        oy1 = fy1 - y1
        ox2 = ox1 + (fx2 - fx1)
        oy2 = oy1 + (fy2 - fy1)

        crop_ov = rotated[oy1:oy2, ox1:ox2]
        crop_base = frame_bgr[fy1:fy2, fx1:fx2]

        if crop_ov.shape[2] == 4:
            alpha = (crop_ov[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]
            rgb = crop_ov[:, :, :3].astype(np.float32)
            blended = alpha * rgb + (1.0 - alpha) * crop_base.astype(np.float32)
            frame_bgr[fy1:fy2, fx1:fx2] = np.clip(blended, 0, 255).astype(np.uint8)

        return frame_bgr

    def release(self) -> None:
        if self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None

    def __del__(self) -> None:
        self.release()
