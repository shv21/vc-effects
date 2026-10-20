"""
Background compositor module for virtual scene replacement.
Replaces real room background with user-selected scenes from assets/backgrounds/,
compositing the live person cleanly with feathered alpha masks.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BackgroundCompositor:
    """Manages virtual background images and composites live person foreground."""

    def __init__(self, backgrounds_dir: str = "assets/backgrounds") -> None:
        self.backgrounds_dir = Path(backgrounds_dir)
        self.background_files: List[Path] = []
        self.cached_images: Dict[str, np.ndarray] = {}
        self.active_index: int = 0
        self.is_enabled: bool = False

        self.reload()

    def reload(self) -> None:
        """Discovers background images in assets/backgrounds/."""
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

    def toggle(self) -> bool:
        self.is_enabled = not self.is_enabled
        return self.is_enabled

    def composite(
        self,
        frame_bgr: np.ndarray,
        person_mask: np.ndarray
    ) -> np.ndarray:
        """
        Blends frame_bgr over the active virtual background using person_mask.
        If disabled or no backgrounds available, returns original frame_bgr.
        """
        if not self.is_enabled or not self.background_files or person_mask is None:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        bg_path = self.background_files[self.active_index]

        # Load or retrieve cached background
        bg_key = f"{bg_path.name}_{w}x{h}"
        if bg_key not in self.cached_images:
            raw_bg = cv2.imread(str(bg_path))
            if raw_bg is None:
                logger.error(f"Failed to read background {bg_path}")
                return frame_bgr

            # Aspect-ratio preserving center crop to match (w, h)
            bg_h, bg_w = raw_bg.shape[:2]
            scale = max(w / float(bg_w), h / float(bg_h))
            scaled_w = int(round(bg_w * scale))
            scaled_h = int(round(bg_h * scale))
            resized_bg = cv2.resize(raw_bg, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

            # Center crop
            crop_x = (scaled_w - w) // 2
            crop_y = (scaled_h - h) // 2
            cropped = resized_bg[crop_y:crop_y + h, crop_x:crop_x + w]
            self.cached_images[bg_key] = cropped

        virtual_bg = self.cached_images[bg_key]

        # Ensure person_mask matches dimensions
        if person_mask.shape[:2] != (h, w):
            person_mask = cv2.resize(person_mask, (w, h))

        # Alpha composite: Scene = PersonMask * LiveSubject + (1 - PersonMask) * VirtualBg
        p_mask_3d = person_mask[:, :, np.newaxis]
        composited = (
            p_mask_3d * frame_bgr.astype(np.float32) +
            (1.0 - p_mask_3d) * virtual_bg.astype(np.float32)
        )

        return np.clip(composited, 0, 255).astype(np.uint8)
