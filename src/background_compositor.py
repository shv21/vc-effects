"""
Virtual background compositor and perspective portal module.
Supports:
1. Full background replacement: replaces room background using person segmentation mask
2. Hand-controlled Scene Portal: maps the selected scene into the hand-defined polygon
   using 4-point perspective transformation, creating an interactive portal into another world.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BackgroundCompositor:
    """Manages virtual background scenes, person compositing, and perspective portals."""

    def __init__(self, backgrounds_dir: str = "assets/backgrounds") -> None:
        self.backgrounds_dir = Path(backgrounds_dir)
        self.background_files: List[Path] = []
        self.cached_images: Dict[str, np.ndarray] = {}
        self.active_index: int = 0
        self.is_enabled: bool = False

        self.reload()

    def reload(self) -> None:
        """Discovers background scenes in assets/backgrounds/."""
        self.background_files.clear()
        self.cached_images.clear()

        if self.backgrounds_dir.exists():
            for p in sorted(self.backgrounds_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                    self.background_files.append(p)

        if self.active_index >= len(self.background_files):
            self.active_index = 0

        logger.info(f"Loaded {len(self.background_files)} backgrounds from {self.backgrounds_dir}")

    def get_current_background_name(self) -> str:
        if self.background_files:
            return self.background_files[self.active_index].stem.replace("_", " ").title()
        return "None"

    def next_background(self) -> str:
        if self.background_files:
            self.active_index = (self.active_index + 1) % len(self.background_files)
            return self.get_current_background_name()
        return "None"

    def prev_background(self) -> str:
        if self.background_files:
            self.active_index = (self.active_index - 1) % len(self.background_files)
            return self.get_current_background_name()
        return "None"

    def get_current_background_image(self, target_w: int, target_h: int) -> Optional[np.ndarray]:
        """Loads and center-crops the active background image to match target resolution."""
        if not self.background_files:
            return None

        bg_path = self.background_files[self.active_index]
        bg_key = f"{bg_path.name}_{target_w}x{target_h}"
        if bg_key not in self.cached_images:
            raw_bg = cv2.imread(str(bg_path))
            if raw_bg is None:
                return None

            bg_h, bg_w = raw_bg.shape[:2]
            scale = max(target_w / float(bg_w), target_h / float(bg_h))
            scaled_w = int(round(bg_w * scale))
            scaled_h = int(round(bg_h * scale))
            resized = cv2.resize(raw_bg, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

            crop_x = (scaled_w - target_w) // 2
            crop_y = (scaled_h - target_h) // 2
            cropped = resized[crop_y:crop_y + target_h, crop_x:crop_x + target_w]
            self.cached_images[bg_key] = cropped

        return self.cached_images[bg_key]

    def composite_full(
        self,
        frame_bgr: np.ndarray,
        person_mask: np.ndarray
    ) -> np.ndarray:
        """
        Full scene replacement: preserves person using person_mask and replaces room background.
        """
        h, w = frame_bgr.shape[:2]
        virtual_bg = self.get_current_background_image(w, h)
        if virtual_bg is None or person_mask is None:
            return frame_bgr

        if person_mask.shape[:2] != (h, w):
            person_mask = cv2.resize(person_mask, (w, h))

        p_mask_3d = person_mask[:, :, np.newaxis]
        composited = (
            p_mask_3d * frame_bgr.astype(np.float32) +
            (1.0 - p_mask_3d) * virtual_bg.astype(np.float32)
        )
        return np.clip(composited, 0, 255).astype(np.uint8)

    def render_portal_frame(
        self,
        base_frame_bgr: np.ndarray,
        polygon_pts: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Perspective-warps the virtual background image into the 4-corner hand polygon.
        Used for MODE B (Scene Portal) and MODE D.
        """
        h, w = base_frame_bgr.shape[:2]
        virtual_bg = self.get_current_background_image(w, h)
        if virtual_bg is None:
            return base_frame_bgr

        if polygon_pts is None or len(polygon_pts) != 4:
            return virtual_bg

        # Map virtual scene corners into the hand quadrilateral
        src_corners = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype=np.float32)

        dst_corners = polygon_pts.astype(np.float32)

        try:
            M = cv2.getPerspectiveTransform(src_corners, dst_corners)
            warped_portal = cv2.warpPerspective(
                virtual_bg,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            return warped_portal
        except Exception as e:
            logger.warning(f"Portal perspective warp error: {e}")
            return virtual_bg
