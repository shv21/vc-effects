"""
Mask renderer and compositing engine.
Applies visual transformations strictly inside the feathered polygon mask,
guaranteeing the rest of the camera frame remains 100% untouched.
Renders energetic glowing outlines around active hand-controlled regions.
"""

import time
import math
import logging
from typing import Optional, List, Tuple
import cv2
import numpy as np

from .region_controller import RegionState

logger = logging.getLogger(__name__)


class MaskRenderer:
    """Blends masked effects and draws dynamic HUD polygon boundaries."""

    @staticmethod
    def apply_masked_effect(
        base_frame_bgr: np.ndarray,
        effect_frame_bgr: np.ndarray,
        feathered_mask: np.ndarray,
        opacity: float = 0.95
    ) -> np.ndarray:
        """
        Blends effect_frame_bgr onto base_frame_bgr strictly where feathered_mask > 0.
        All other pixels remain 100% identical to base_frame_bgr.
        """
        if feathered_mask is None or base_frame_bgr is None or effect_frame_bgr is None:
            return base_frame_bgr

        # Ensure matching dimensions
        h, w = base_frame_bgr.shape[:2]
        if effect_frame_bgr.shape[:2] != (h, w):
            effect_frame_bgr = cv2.resize(effect_frame_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

        # Scale mask by opacity
        effective_mask = np.clip(feathered_mask * opacity, 0.0, 1.0)
        mask_3d = effective_mask[:, :, np.newaxis]

        # Alpha composite: Out = Mask * Effect + (1 - Mask) * Base
        blended = (
            mask_3d * effect_frame_bgr.astype(np.float32) +
            (1.0 - mask_3d) * base_frame_bgr.astype(np.float32)
        )

        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def draw_region_outline(
        frame_bgr: np.ndarray,
        polygon_pts: Optional[np.ndarray],
        state: str = RegionState.PREVIEW
    ) -> np.ndarray:
        """
        Draws an energetic, animated futuristic boundary around the hand-controlled polygon.
        """
        if polygon_pts is None or len(polygon_pts) < 3:
            return frame_bgr

        overlay = frame_bgr.copy()
        now = time.time()

        # Pulse brightness using sine wave
        pulse = 0.5 + 0.5 * math.sin(now * 6.0)

        if state == RegionState.LOCKED:
            # Pulsing Gold / Emerald Green for locked/active mode
            r = int(255 * pulse)
            g = int(200 + 55 * pulse)
            b = 0
            border_color = (b, g, r)
            glow_color = (0, 255, 128)
            corner_color = (0, 255, 255)
            line_thickness = 3
        else:
            # Electric Cyan for preview mode
            b = int(220 + 35 * pulse)
            g = int(180 + 75 * pulse)
            r = 0
            border_color = (b, g, r)
            glow_color = (255, 200, 0)
            corner_color = (255, 255, 255)
            line_thickness = 2

        # Outer subtle glow line
        cv2.polylines(overlay, [polygon_pts], isClosed=True, color=glow_color, thickness=line_thickness + 2, lineType=cv2.LINE_AA)
        # Sharp inner line
        cv2.polylines(overlay, [polygon_pts], isClosed=True, color=border_color, thickness=line_thickness, lineType=cv2.LINE_AA)

        # Corner anchor nodes
        for idx, pt in enumerate(polygon_pts):
            p = (int(pt[0]), int(pt[1]))
            # Outer ring
            cv2.circle(overlay, p, 7, glow_color, 1, lineType=cv2.LINE_AA)
            # Inner solid node
            cv2.circle(overlay, p, 4, corner_color, -1, lineType=cv2.LINE_AA)

        # Soft blend overlay
        return cv2.addWeighted(overlay, 0.85, frame_bgr, 0.15, 0)
