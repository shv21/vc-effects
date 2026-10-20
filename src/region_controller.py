"""
Two-hand dynamic region controller.
Creates, resizes, repositions, feathers, and smoothly fades the user-defined polygon region.
Ensures:
- Real-time continuous live tracking by default (no accidental locking)
- 4 control points calculated from Thumb tips and Index tips of both hands
- Clockwise convex quadrilateral sorting to prevent twisted/bowtie polygons
- Automatic orientation adaptation for rotated/tilted hands
- Grace period & smooth fadeout on tracking loss (never stays frozen indefinitely)
- Instant reacquisition when hands return
"""

import time
import math
import logging
from typing import Optional, List, Tuple
import cv2
import numpy as np

from .hand_tracker import DualHandData

logger = logging.getLogger(__name__)


class RegionState:
    INACTIVE = "INACTIVE"
    PREVIEW = "PREVIEW"
    LOCKED = "LOCKED"


class RegionController:
    """Manages the interactive quadrilateral region controlled continuously by both hands."""

    def __init__(
        self,
        feather_radius: int = 21,
        smoothing_alpha: float = 0.82,
        grace_period_sec: float = 0.40,
        min_size_px: int = 40
    ) -> None:
        self.feather_radius = feather_radius
        self.smoothing_alpha = smoothing_alpha
        self.grace_period_sec = grace_period_sec
        self.min_size_px = min_size_px

        self.polygon_pts: Optional[np.ndarray] = None  # (4, 2) float32
        self.is_frozen: bool = False
        self.frozen_pts: Optional[np.ndarray] = None
        self.inverted: bool = False

        # Fade & tracking loss state
        self.tracking_loss_time: Optional[float] = None
        self.current_opacity: float = 0.0  # Fades in/out smoothly

    @property
    def is_locked(self) -> bool:
        return self.is_frozen

    @property
    def state(self) -> str:
        if self.is_frozen:
            return RegionState.LOCKED
        elif self.current_opacity > 0.05:
            return RegionState.PREVIEW
        return RegionState.INACTIVE

    def update(self, dual: DualHandData, frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Updates the region polygon from fresh hand landmarks every frame.
        Follows both hands continuously without freezing.
        Returns (4, 2) int32 vertices or None.
        """
        now = time.time()
        h, w = frame_shape[:2]

        # Explicit user freeze takes precedence
        if self.is_frozen and self.frozen_pts is not None:
            self.current_opacity = 1.0
            return self.frozen_pts.astype(np.int32)

        # Both hands are visible: update continuously
        if dual.both_hands_visible:
            self.tracking_loss_time = None
            # Fast fade-in
            self.current_opacity = min(1.0, self.current_opacity + 0.20)

            lh = dual.left_hand
            rh = dual.right_hand

            # Calculate 4 control points from Left Hand (Index & Thumb) and Right Hand (Index & Thumb)
            lh_index = np.array([float(lh.index_tip[0]), float(lh.index_tip[1])], dtype=np.float32)
            lh_thumb = np.array([float(lh.thumb_tip[0]), float(lh.thumb_tip[1])], dtype=np.float32)
            rh_index = np.array([float(rh.index_tip[0]), float(rh.index_tip[1])], dtype=np.float32)
            rh_thumb = np.array([float(rh.thumb_tip[0]), float(rh.thumb_tip[1])], dtype=np.float32)

            # Vector between the two hand centers
            hand_dx = float(rh.palm_center[0] - lh.palm_center[0])
            hand_dy = float(rh.palm_center[1] - lh.palm_center[1])
            hand_dist = max(1.0, math.hypot(hand_dx, hand_dy))

            # Perpendicular normal unit vector (for height orientation)
            nx = -hand_dy / hand_dist
            ny = hand_dx / hand_dist

            # Left hand vertical span
            lh_finger_dist = math.hypot(lh_index[0] - lh_thumb[0], lh_index[1] - lh_thumb[1])
            lh_min_span = max(self.min_size_px, lh.hand_scale * 0.70)
            if lh_finger_dist < lh_min_span:
                # Pinching or close fingers: expand along perpendicular normal around pinch center
                lh_mid = (lh_index + lh_thumb) * 0.5
                lh_c0 = lh_mid - np.array([nx, ny], dtype=np.float32) * (lh_min_span * 0.5)
                lh_c3 = lh_mid + np.array([nx, ny], dtype=np.float32) * (lh_min_span * 0.5)
            else:
                lh_c0 = lh_index
                lh_c3 = lh_thumb

            # Right hand vertical span
            rh_finger_dist = math.hypot(rh_index[0] - rh_thumb[0], rh_index[1] - rh_thumb[1])
            rh_min_span = max(self.min_size_px, rh.hand_scale * 0.70)
            if rh_finger_dist < rh_min_span:
                rh_mid = (rh_index + rh_thumb) * 0.5
                rh_c1 = rh_mid - np.array([nx, ny], dtype=np.float32) * (rh_min_span * 0.5)
                rh_c2 = rh_mid + np.array([nx, ny], dtype=np.float32) * (rh_min_span * 0.5)
            else:
                rh_c1 = rh_index
                rh_c2 = rh_thumb

            raw_corners = [
                [float(lh_c0[0]), float(lh_c0[1])],
                [float(rh_c1[0]), float(rh_c1[1])],
                [float(rh_c2[0]), float(rh_c2[1])],
                [float(lh_c3[0]), float(lh_c3[1])]
            ]

            # Sort into convex clockwise quadrilateral starting from Top-Left
            sorted_quad = self._sort_convex_quadrilateral(raw_corners, h, w)

            # Low-latency temporal EMA smoothing (responsive follow without lagging)
            if self.polygon_pts is not None:
                self.polygon_pts = (
                    self.smoothing_alpha * sorted_quad +
                    (1.0 - self.smoothing_alpha) * self.polygon_pts
                )
            else:
                self.polygon_pts = sorted_quad.copy()

            return self.polygon_pts.astype(np.int32)

        else:
            # Tracking Loss: apply short grace period then fade out completely
            if self.polygon_pts is not None:
                if self.tracking_loss_time is None:
                    self.tracking_loss_time = now

                elapsed = now - self.tracking_loss_time
                if elapsed < self.grace_period_sec:
                    # Fade out smoothly over grace period
                    fade = max(0.0, 1.0 - (elapsed / self.grace_period_sec))
                    self.current_opacity = fade
                    return self.polygon_pts.astype(np.int32)
                else:
                    # Grace period expired: clear stale polygon so it never stays stuck
                    self.polygon_pts = None
                    self.current_opacity = 0.0
                    return None

            self.current_opacity = 0.0
            return None

    def _sort_convex_quadrilateral(self, pts: List[List[float]], h: int, w: int) -> np.ndarray:
        """
        Sorts 4 points into a clockwise convex quadrilateral starting at Top-Left.
        Prevents self-intersection, twisting, or bowtie distortion.
        """
        pts_arr = np.array(pts, dtype=np.float32)

        # Sort clockwise around centroid (computer graphics Y-down coordinate space)
        center = np.mean(pts_arr, axis=0)
        angles = np.arctan2(pts_arr[:, 1] - center[1], pts_arr[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        clockwise_pts = pts_arr[sorted_indices]

        # Rotate array so point closest to Top-Left (min x + y) is index 0
        sum_xy = clockwise_pts[:, 0] + clockwise_pts[:, 1]
        top_left_idx = np.argmin(sum_xy)
        ordered_pts = np.roll(clockwise_pts, -top_left_idx, axis=0)

        # Padding scale so region frames the control points comfortably
        padding_scale = 1.12
        expanded = center + (ordered_pts - center) * padding_scale

        # Clamp strictly inside camera frame boundaries
        expanded[:, 0] = np.clip(expanded[:, 0], 2, w - 3)
        expanded[:, 1] = np.clip(expanded[:, 1], 2, h - 3)

        return expanded

    def toggle_freeze(self) -> bool:
        """Explicit user toggle to freeze/unfreeze region."""
        self.is_frozen = not self.is_frozen
        if self.is_frozen and self.polygon_pts is not None:
            self.frozen_pts = self.polygon_pts.copy()
            logger.info("Region frozen in place by user command")
        else:
            self.frozen_pts = None
            logger.info("Region unfrozen: Tracking hands live")
        return self.is_frozen

    def toggle_lock(self) -> bool:
        return self.toggle_freeze()

    def unfreeze(self) -> None:
        self.is_frozen = False
        self.frozen_pts = None

    def toggle_invert(self) -> bool:
        self.inverted = not self.inverted
        return self.inverted

    def generate_mask(self, frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Generates feathered float32 mask (0.0 to 1.0) scaled by current fade opacity.
        Returns None if opacity is zero or no valid polygon exists.
        """
        if self.current_opacity <= 0.01:
            return None

        pts = self.get_current_polygon()
        if pts is None or len(pts) < 3:
            return None

        h, w = frame_shape[:2]
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(binary_mask, [pts], 255)

        if self.inverted:
            binary_mask = cv2.bitwise_not(binary_mask)

        # Feather boundaries with Gaussian blur
        radius = max(3, self.feather_radius)
        if radius % 2 == 0:
            radius += 1
        feathered = cv2.GaussianBlur(binary_mask, (radius, radius), 0)

        # Scale by fade opacity
        norm_mask = (feathered.astype(np.float32) / 255.0) * self.current_opacity
        return norm_mask

    def get_current_polygon(self) -> Optional[np.ndarray]:
        if self.is_frozen and self.frozen_pts is not None:
            return self.frozen_pts.astype(np.int32)
        elif self.polygon_pts is not None:
            return self.polygon_pts.astype(np.int32)
        return None

    def is_active(self) -> bool:
        return self.current_opacity > 0.05
