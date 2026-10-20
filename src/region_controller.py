"""
Two-hand dynamic region controller.
Creates, resizes, repositions, locks, and feathers the user-defined polygon region.
Handles single-hand dropouts, smooth EMA transitions, and generates feathered alpha masks.
"""

import math
import logging
from typing import Optional, List, Tuple, Dict
import cv2
import numpy as np

from .hand_tracker import DualHandData

logger = logging.getLogger(__name__)


class RegionState:
    INACTIVE = "INACTIVE"    # No hands detected
    PREVIEW = "PREVIEW"      # Dynamic region following hands
    LOCKED = "LOCKED"        # Region locked in space, effect actively burning inside


class RegionController:
    """Manages the interactive quadrilateral/polygon region controlled by both hands."""

    def __init__(
        self,
        feather_radius: int = 21,
        smoothing_alpha: float = 0.70
    ) -> None:
        self.feather_radius = feather_radius
        self.smoothing_alpha = smoothing_alpha

        self.state = RegionState.INACTIVE
        self.polygon_pts: Optional[np.ndarray] = None  # (4, 2) float points
        self.locked_polygon_pts: Optional[np.ndarray] = None
        self.inverted: bool = False

        # Scale modifier controlled by distance / fine pinch
        self.scale_factor: float = 1.0

    def update(self, dual_data: DualHandData, frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Updates the region polygon based on current hand positions.
        Returns the current polygon vertices as an (N, 2) int32 numpy array, or None.
        """
        h, w = frame_shape[:2]

        if self.state == RegionState.LOCKED and self.locked_polygon_pts is not None:
            # When locked, the region stays pinned in screen space
            return self.locked_polygon_pts.astype(np.int32)

        if not dual_data.both_hands_visible:
            # If only one hand is visible and we have a previous preview polygon, coast smoothly
            if (dual_data.left_hand or dual_data.right_hand) and self.polygon_pts is not None:
                # Keep existing polygon without collapsing
                return self.polygon_pts.astype(np.int32)
            else:
                self.state = RegionState.INACTIVE
                self.polygon_pts = None
                return None

        # Both hands are visible: compute dynamic 4-point region
        lh = dual_data.left_hand
        rh = dual_data.right_hand

        # Four interactive control corners:
        # P0: Top-Left  (Left Hand Index Tip)
        # P1: Top-Right (Right Hand Index Tip)
        # P2: Bottom-Right (Right Hand Thumb Tip or Wrist)
        # P3: Bottom-Left  (Left Hand Thumb Tip or Wrist)
        p0 = [float(lh.index_tip[0]), float(lh.index_tip[1])]
        p1 = [float(rh.index_tip[0]), float(rh.index_tip[1])]
        p2 = [float(rh.thumb_tip[0]), float(rh.thumb_tip[1])]
        p3 = [float(lh.thumb_tip[0]), float(lh.thumb_tip[1])]

        raw_pts = np.array([p0, p1, p2, p3], dtype=np.float32)

        # Ensure vertical separation so region does not flatten when fingers are horizontal
        dist_top = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        min_depth = max(50.0, dist_top * 0.45)

        dy_right = p2[1] - p1[1]
        if dy_right < min_depth:
            raw_pts[2][1] = p1[1] + min_depth

        dy_left = p3[1] - p0[1]
        if dy_left < min_depth:
            raw_pts[3][1] = p0[1] + min_depth

        # Centroid expansion based on scale factor
        centroid = np.mean(raw_pts, axis=0)
        # Expand slightly outwards to comfortably contain the area framed by hands
        padding_scale = 1.18 * self.scale_factor
        expanded_pts = centroid + (raw_pts - centroid) * padding_scale

        # Clip coordinates within camera boundaries
        expanded_pts[:, 0] = np.clip(expanded_pts[:, 0], 0, w - 1)
        expanded_pts[:, 1] = np.clip(expanded_pts[:, 1], 0, h - 1)

        # Smooth polygon transitions over time
        if self.polygon_pts is not None:
            self.polygon_pts = (
                self.smoothing_alpha * expanded_pts +
                (1.0 - self.smoothing_alpha) * self.polygon_pts
            )
        else:
            self.polygon_pts = expanded_pts.copy()

        self.state = RegionState.PREVIEW
        return self.polygon_pts.astype(np.int32)

    def toggle_lock(self) -> str:
        """Toggles between PREVIEW (movable) and LOCKED (fixed)."""
        if self.state == RegionState.LOCKED:
            self.state = RegionState.PREVIEW
            self.locked_polygon_pts = None
            logger.info("Region unlocked: Tracking hands")
            return "UNLOCKED"
        elif self.polygon_pts is not None:
            self.state = RegionState.LOCKED
            self.locked_polygon_pts = self.polygon_pts.copy()
            logger.info("Region locked in space")
            return "LOCKED"
        return "INACTIVE"

    def set_lock(self, locked: bool) -> None:
        if locked and self.polygon_pts is not None:
            self.state = RegionState.LOCKED
            self.locked_polygon_pts = self.polygon_pts.copy()
        else:
            self.state = RegionState.PREVIEW
            self.locked_polygon_pts = None

    def toggle_invert(self) -> bool:
        """Toggles inside/outside mask inversion."""
        self.inverted = not self.inverted
        return self.inverted

    def generate_mask(self, frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Generates a smooth feathered float mask (0.0 to 1.0) of size (H, W).
        Returns None if no active region exists.
        """
        pts = self.get_current_polygon()
        if pts is None or len(pts) < 3:
            return None

        h, w = frame_shape[:2]

        # 1. Binary mask
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(binary_mask, [pts], 255)

        # 2. Invert if inside/outside mode toggled
        if self.inverted:
            binary_mask = cv2.bitwise_not(binary_mask)

        # 3. Feather edges using Gaussian blur
        radius = max(3, self.feather_radius)
        if radius % 2 == 0:
            radius += 1

        feathered = cv2.GaussianBlur(binary_mask, (radius, radius), 0)

        # 4. Normalize to float32 [0.0, 1.0]
        return feathered.astype(np.float32) / 255.0

    def get_current_polygon(self) -> Optional[np.ndarray]:
        """Returns the active (4, 2) integer polygon vertices."""
        if self.state == RegionState.LOCKED and self.locked_polygon_pts is not None:
            return self.locked_polygon_pts.astype(np.int32)
        elif self.polygon_pts is not None:
            return self.polygon_pts.astype(np.int32)
        return None

    def is_active(self) -> bool:
        return self.state in [RegionState.PREVIEW, RegionState.LOCKED]
