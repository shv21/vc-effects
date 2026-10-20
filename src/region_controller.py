"""
Two-hand dynamic region controller and 4-point portal tracker.
Establishes 4 live control points from Thumb and Index fingertips of both hands.

Guarantees:
- Real-time continuous live tracking (never freezes at old coordinates)
- Configurable activation gesture rules ('dual_pinch', 'framing', 'both_hands')
- Follows hand movement continuously (expand, shrink, translate, change shape)
- Disappears cleanly when gesture is released or tracking is lost
- Topologically consistent corner ordering [TL, TR, BR, BL]
- Degenerate & self-intersecting polygon protection (hides portal safely rather than crashing)
- Low-latency adaptive smoothing (zero lag on fast moves, jitter-free when still)
"""

import time
import math
import logging
from typing import Optional, List, Tuple
import cv2
import numpy as np

from .hand_tracker import DualHandData, HandData

logger = logging.getLogger(__name__)


class RegionState:
    INACTIVE = "INACTIVE"
    PREVIEW = "PREVIEW"
    LOCKED = "LOCKED"


class ActivationGestureRule:
    DUAL_PINCH = "dual_pinch"  # Both hands pinching thumb and index (User prompt example)
    FRAMING = "framing"        # Both hands framing with thumb and index fingers extended
    BOTH_HANDS = "both_hands"  # Any two hands present in view

    ALL = [DUAL_PINCH, FRAMING, BOTH_HANDS]


class RegionController:
    """Manages the 4 live control points established from both hands to frame the scene portal."""

    def __init__(
        self,
        feather_radius: int = 0,
        smoothing_alpha: float = 0.85,
        grace_period_sec: float = 0.20,
        min_size_px: int = 30,
        activation_rule: str = ActivationGestureRule.DUAL_PINCH
    ) -> None:
        self.feather_radius = feather_radius
        self.base_smoothing_alpha = smoothing_alpha
        self.grace_period_sec = grace_period_sec
        self.min_size_px = min_size_px
        self.activation_rule = activation_rule

        self.polygon_pts: Optional[np.ndarray] = None  # (4, 2) float32 [TL, TR, BR, BL]
        self.is_frozen: bool = False
        self.frozen_pts: Optional[np.ndarray] = None
        self.inverted: bool = False

        # Fade & tracking loss state
        self.tracking_loss_time: Optional[float] = None
        self.current_opacity: float = 0.0

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

    def cycle_activation_rule(self) -> str:
        """Cycles through activation gesture rules."""
        idx = ActivationGestureRule.ALL.index(self.activation_rule)
        self.activation_rule = ActivationGestureRule.ALL[(idx + 1) % len(ActivationGestureRule.ALL)]
        return self.activation_rule

    def set_activation_rule(self, rule: str) -> None:
        if rule in ActivationGestureRule.ALL:
            self.activation_rule = rule

    def is_gesture_activated(self, dual: DualHandData) -> bool:
        """Evaluates whether current hand state satisfies the active gesture rule."""
        if not dual.both_hands_visible or dual.left_hand is None or dual.right_hand is None:
            return False

        lh = dual.left_hand
        rh = dual.right_hand

        if self.activation_rule == ActivationGestureRule.DUAL_PINCH:
            # Active when both hands are pinching thumb and index finger
            return bool(lh.is_pinching and rh.is_pinching)

        elif self.activation_rule == ActivationGestureRule.FRAMING:
            # Active when both hands are framing (index and thumb extended)
            lh_open = (lh.pinch_ratio > 0.35)
            rh_open = (rh.pinch_ratio > 0.35)
            return bool(lh_open and rh_open)

        elif self.activation_rule == ActivationGestureRule.BOTH_HANDS:
            # Active whenever both hands are tracked in view
            return True

        return False

    def update(self, dual: DualHandData, frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Updates the 4-point quadrilateral from fresh hand landmarks every frame.
        Follows hands continuously in real time.
        Returns (4, 2) int32 vertices [TL, TR, BR, BL] or None.
        """
        now = time.time()
        h, w = frame_shape[:2]

        # Explicit user freeze takes precedence if active
        if self.is_frozen and self.frozen_pts is not None:
            self.current_opacity = 1.0
            return self.frozen_pts.astype(np.int32)

        # Check if gesture conditions are met
        gesture_active = self.is_gesture_activated(dual)

        if gesture_active:
            self.tracking_loss_time = None
            # Fast fade-in
            self.current_opacity = min(1.0, self.current_opacity + 0.30)

            lh = dual.left_hand
            rh = dual.right_hand

            # Extract 4 live control points from Left Hand (Index & Thumb) and Right Hand (Index & Thumb)
            quad = self._extract_four_control_points(lh, rh, h, w)
            if quad is None:
                # Degenerate or invalid polygon: hide portal temporarily
                return None

            # Low-latency temporal EMA smoothing with velocity adaptation
            if self.polygon_pts is not None:
                movement = float(np.mean(np.linalg.norm(quad - self.polygon_pts, axis=1)))
                # Adaptive alpha: smooth when still, fast when moving rapidly
                adaptive_alpha = min(0.95, max(0.70, self.base_smoothing_alpha + movement * 0.02))
                self.polygon_pts = adaptive_alpha * quad + (1.0 - adaptive_alpha) * self.polygon_pts
            else:
                # First frame snap (zero latency)
                self.polygon_pts = quad.copy()

            return self.polygon_pts.astype(np.int32)

        else:
            # Tracking loss or gesture released: apply short smooth fadeout then clear
            if self.polygon_pts is not None:
                if self.tracking_loss_time is None:
                    self.tracking_loss_time = now

                elapsed = now - self.tracking_loss_time
                if elapsed < self.grace_period_sec:
                    # Smooth fadeout over grace period
                    fade = max(0.0, 1.0 - (elapsed / self.grace_period_sec))
                    self.current_opacity = fade
                    return self.polygon_pts.astype(np.int32)
                else:
                    # Grace period expired: clear stale polygon so it NEVER stays stuck
                    self.polygon_pts = None
                    self.current_opacity = 0.0
                    return None

            self.current_opacity = 0.0
            return None

    def _extract_four_control_points(
        self,
        lh: HandData,
        rh: HandData,
        h: int,
        w: int
    ) -> Optional[np.ndarray]:
        """
        Extracts and orders 4 live control points:
        [TL (Left Top), TR (Right Top), BR (Right Bottom), BL (Left Bottom)].

        Guarantees:
        - Natural fingertip mapping
        - Consistent corner orientation invariant to rotation/tilt
        - Hand scale expansion when pinching to prevent quad collapse
        - Protection against degenerate or self-intersecting quads
        """
        lh_index = np.array([float(lh.index_tip[0]), float(lh.index_tip[1])], dtype=np.float32)
        lh_thumb = np.array([float(lh.thumb_tip[0]), float(lh.thumb_tip[1])], dtype=np.float32)
        rh_index = np.array([float(rh.index_tip[0]), float(rh.index_tip[1])], dtype=np.float32)
        rh_thumb = np.array([float(rh.thumb_tip[0]), float(rh.thumb_tip[1])], dtype=np.float32)

        # Inter-hand axis vector (left hand to right hand)
        vx = float(rh.palm_center[0] - lh.palm_center[0])
        vy = float(rh.palm_center[1] - lh.palm_center[1])
        hand_dist = max(1.0, math.hypot(vx, vy))

        # Perpendicular normal pointing "upward" in computer graphics coordinates (Y-down)
        # Clockwise 90 deg rotation from (vx, vy) towards -Y is (vy, -vx)
        nx = vy / hand_dist
        ny = -vx / hand_dist

        # 1. Left Hand Corners (TL, BL)
        lh_finger_dist = math.hypot(lh_index[0] - lh_thumb[0], lh_index[1] - lh_thumb[1])
        lh_min_span = max(float(self.min_size_px), float(lh.hand_scale * 0.75))

        if lh_finger_dist < lh_min_span:
            # Pinching: expand along perpendicular normal around pinch center
            lh_mid = (lh_index + lh_thumb) * 0.5
            lh_tl = lh_mid + np.array([nx, ny], dtype=np.float32) * (lh_min_span * 0.5)
            lh_bl = lh_mid - np.array([nx, ny], dtype=np.float32) * (lh_min_span * 0.5)
        else:
            # Framing: project fingertips along normal to determine top vs bottom
            proj_idx = lh_index[0] * nx + lh_index[1] * ny
            proj_th = lh_thumb[0] * nx + lh_thumb[1] * ny
            if proj_idx >= proj_th:
                lh_tl = lh_index
                lh_bl = lh_thumb
            else:
                lh_tl = lh_thumb
                lh_bl = lh_index

        # 2. Right Hand Corners (TR, BR)
        rh_finger_dist = math.hypot(rh_index[0] - rh_thumb[0], rh_index[1] - rh_thumb[1])
        rh_min_span = max(float(self.min_size_px), float(rh.hand_scale * 0.75))

        if rh_finger_dist < rh_min_span:
            rh_mid = (rh_index + rh_thumb) * 0.5
            rh_tr = rh_mid + np.array([nx, ny], dtype=np.float32) * (rh_min_span * 0.5)
            rh_br = rh_mid - np.array([nx, ny], dtype=np.float32) * (rh_min_span * 0.5)
        else:
            proj_idx = rh_index[0] * nx + rh_index[1] * ny
            proj_th = rh_thumb[0] * nx + rh_thumb[1] * ny
            if proj_idx >= proj_th:
                rh_tr = rh_index
                rh_br = rh_thumb
            else:
                rh_tr = rh_thumb
                rh_br = rh_index

        # Assemble [TL, TR, BR, BL]
        ordered_pts = np.array([lh_tl, rh_tr, rh_br, lh_bl], dtype=np.float32)

        # Boundary clipping strictly inside camera frame
        ordered_pts[:, 0] = np.clip(ordered_pts[:, 0], 2.0, float(w - 3))
        ordered_pts[:, 1] = np.clip(ordered_pts[:, 1], 2.0, float(h - 3))

        # Protection against degenerate or self-intersecting quads
        edges = [
            ordered_pts[1] - ordered_pts[0],  # Top
            ordered_pts[2] - ordered_pts[1],  # Right
            ordered_pts[3] - ordered_pts[2],  # Bottom
            ordered_pts[0] - ordered_pts[3]   # Left
        ]
        edge_lens = [math.hypot(e[0], e[1]) for e in edges]
        if min(edge_lens) < 18.0:
            return None

        area = float(cv2.contourArea(ordered_pts))
        if area < 350.0:
            return None

        # Convexity & self-intersection check (cross products of consecutive edges)
        cross_products = []
        for i in range(4):
            e_curr = edges[i]
            e_next = edges[(i + 1) % 4]
            cp = float(e_curr[0] * e_next[1] - e_curr[1] * e_next[0])
            cross_products.append(cp)

        all_pos = all(cp > 1e-3 for cp in cross_products)
        all_neg = all(cp < -1e-3 for cp in cross_products)
        if not (all_pos or all_neg):
            # Degenerate or self-intersecting
            return None

        return ordered_pts

    def toggle_freeze(self) -> bool:
        """Explicit toggle for debugging / freeze-frame."""
        self.is_frozen = not self.is_frozen
        if self.is_frozen and self.polygon_pts is not None:
            self.frozen_pts = self.polygon_pts.copy()
            logger.info("Region frozen by user command")
        else:
            self.frozen_pts = None
            logger.info("Region unfrozen: Live hand tracking active")
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
        """Generates float32 mask scaled by current fade opacity."""
        if self.current_opacity <= 0.01:
            return None

        pts = self.get_current_polygon()
        if pts is None or len(pts) != 4:
            return None

        h, w = frame_shape[:2]
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(binary_mask, [pts], 255)

        if self.inverted:
            binary_mask = cv2.bitwise_not(binary_mask)

        if self.feather_radius > 0:
            radius = self.feather_radius if self.feather_radius % 2 == 1 else self.feather_radius + 1
            feathered = cv2.GaussianBlur(binary_mask, (radius, radius), 0)
            norm_mask = (feathered.astype(np.float32) / 255.0) * self.current_opacity
        else:
            norm_mask = (binary_mask.astype(np.float32) / 255.0) * self.current_opacity

        return norm_mask

    def get_current_polygon(self) -> Optional[np.ndarray]:
        if self.is_frozen and self.frozen_pts is not None:
            return self.frozen_pts.astype(np.int32)
        elif self.polygon_pts is not None:
            return self.polygon_pts.astype(np.int32)
        return None

    def is_active(self) -> bool:
        return self.current_opacity > 0.05
