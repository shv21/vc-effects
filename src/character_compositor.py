"""
Character overlay compositor module.
Anchors costume elements, superhero chest emblems, and high-tech visors
onto the live person using MediaPipe Pose landmarks as Layer 3 in the AR compositing pipeline.
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


class CharacterCompositor:
    """Anchors 2D character accessories and costume overlays onto the segmented person."""

    def __init__(self, characters_dir: str = "assets/characters") -> None:
        self.characters_dir = Path(characters_dir)
        self.character_files: List[Path] = []
        self.cached_overlays: Dict[str, np.ndarray] = {}
        self.active_index: int = 0
        self.is_enabled: bool = False

        # MediaPipe Pose for anchoring costume overlays
        if mp is not None:
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # 0 is fastest for real-time video
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.pose_detector = None

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
        Detects body anchors and overlays character costume elements onto the frame.
        """
        if not self.is_enabled or not self.character_files:
            return frame_bgr

        char_path = self.character_files[self.active_index]
        if char_path.name not in self.cached_overlays:
            overlay_bgra = cv2.imread(str(char_path), cv2.IMREAD_UNCHANGED)
            if overlay_bgra is None:
                return frame_bgr
            self.cached_overlays[char_path.name] = overlay_bgra

        overlay = self.cached_overlays[char_path.name]
        h, w = frame_bgr.shape[:2]

        # Extract anchor pose landmarks (shoulders, nose, eyes)
        anchor_center, target_width = self._find_anchor(frame_bgr, char_path.name)
        if anchor_center is None:
            return frame_bgr

        # Resize overlay maintaining aspect ratio
        ov_h, ov_w = overlay.shape[:2]
        target_w = max(20, int(target_width))
        target_h = max(20, int(target_w * (ov_h / float(ov_w))))
        resized_ov = cv2.resize(overlay, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Center placement with boundary safety
        cx, cy = anchor_center
        x1 = cx - target_w // 2
        y1 = cy - target_h // 2
        x2 = x1 + target_w
        y2 = y1 + target_h

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

        crop_ov = resized_ov[oy1:oy2, ox1:ox2]
        crop_base = frame_bgr[fy1:fy2, fx1:fx2]

        if crop_ov.shape[2] == 4:
            alpha = (crop_ov[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]
            rgb = crop_ov[:, :, :3].astype(np.float32)
            blended = alpha * rgb + (1.0 - alpha) * crop_base.astype(np.float32)
            frame_bgr[fy1:fy2, fx1:fx2] = np.clip(blended, 0, 255).astype(np.uint8)

        return frame_bgr

    def _find_anchor(self, frame_bgr: np.ndarray, filename: str) -> Tuple[Optional[Tuple[int, int]], float]:
        """Calculates anchor coordinates and scale based on landmark detection or heuristic fallback."""
        h, w = frame_bgr.shape[:2]
        is_visor = "visor" in filename.lower() or "eye" in filename.lower() or "mask" in filename.lower()

        if self.pose_detector is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self.pose_detector.process(rgb)
            rgb.flags.writeable = True

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                # Pose indices: Nose=0, Left Eye=2, Right Eye=5, Left Shoulder=11, Right Shoulder=12
                if is_visor:
                    nose = lms[0]
                    l_eye = lms[2]
                    r_eye = lms[5]
                    cx = int((l_eye.x + r_eye.x) / 2.0 * w)
                    cy = int((l_eye.y + r_eye.y) / 2.0 * h)
                    eye_dist = math.hypot((r_eye.x - l_eye.x) * w, (r_eye.y - l_eye.y) * h)
                    return (cx, cy), max(120.0, eye_dist * 3.4)
                else:
                    # Chest emblem: midpoint between shoulders
                    ls = lms[11]
                    rs = lms[12]
                    cx = int((ls.x + rs.x) / 2.0 * w)
                    # Slightly below shoulder line
                    cy = int(((ls.y + rs.y) / 2.0 + 0.08) * h)
                    shoulder_dist = math.hypot((rs.x - ls.x) * w, (rs.y - ls.y) * h)
                    return (cx, cy), max(100.0, shoulder_dist * 0.75)

        # Fallback heuristic: upper torso / chest area of screen
        cx = w // 2
        cy = int(h * (0.35 if is_visor else 0.60))
        target_w = w * (0.28 if is_visor else 0.32)
        return (cx, cy), target_w

    def release(self) -> None:
        if self.pose_detector is not None:
            self.pose_detector.close()
            self.pose_detector = None

    def __del__(self) -> None:
        self.release()
