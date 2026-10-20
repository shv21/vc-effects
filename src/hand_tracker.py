"""
Hand tracking module for two-handed gesture-controlled AR.
Tracks up to 2 hands using MediaPipe Hands, extracts 21 3D landmarks,
computes finger extension states, pinch distances, and applies temporal EMA smoothing.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

logger = logging.getLogger(__name__)

# Landmark Index Constants
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Hand Connections for Skeleton Rendering
HAND_CONNECTIONS = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (MIDDLE_MCP, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (RING_MCP, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    (WRIST, PINKY_MCP)
]


@dataclass
class HandData:
    """Individual hand tracking telemetry."""
    label: str  # 'Left' or 'Right'
    landmarks_px: np.ndarray  # (21, 2) pixel coordinates
    wrist: Tuple[int, int]
    thumb_tip: Tuple[int, int]
    thumb_ip: Tuple[int, int]
    thumb_mcp: Tuple[int, int]
    index_tip: Tuple[int, int]
    index_pip: Tuple[int, int]
    index_mcp: Tuple[int, int]
    middle_tip: Tuple[int, int]
    middle_mcp: Tuple[int, int]
    ring_tip: Tuple[int, int]
    pinky_tip: Tuple[int, int]
    palm_center: Tuple[int, int]
    pinch_distance: float
    is_pinching: bool
    fingers_extended: List[bool]  # [thumb, index, middle, ring, pinky]
    bbox: Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class DualHandData:
    """Dual-hand context container."""
    left_hand: Optional[HandData] = None
    right_hand: Optional[HandData] = None
    hands_count: int = 0
    both_hands_visible: bool = False
    inter_hand_distance: float = 0.0
    hands_midpoint: Tuple[int, int] = (0, 0)


class HandTracker:
    """Tracks both hands using MediaPipe Hands with jitter smoothing."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        smoothing_alpha: float = 0.72,
        pinch_threshold_px: float = 42.0
    ) -> None:
        if mp is None:
            raise RuntimeError("MediaPipe is required. Please install mediapipe.")

        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.smoothing_alpha = smoothing_alpha
        self.pinch_threshold_px = pinch_threshold_px

        # Temporal smoothing history keyed by hand label ('Left', 'Right')
        self.smoothed_history: Dict[str, np.ndarray] = {}

    def process(self, frame_bgr: np.ndarray) -> DualHandData:
        """
        Processes a BGR camera frame and returns structured DualHandData.
        Note: Assumes input frame is already horizontally mirrored if mirror preview is active.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return DualHandData()

        h, w = frame_bgr.shape[:2]

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands_detector.process(frame_rgb)
        frame_rgb.flags.writeable = True

        dual_data = DualHandData()

        if not results.multi_hand_landmarks or not results.multi_handedness:
            # Clear smoothing if both hands disappear
            self.smoothed_history.clear()
            return dual_data

        detected_labels = set()

        for idx, (hand_lms, handedness) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_handedness)
        ):
            # In flipped mirror view, MediaPipe's Left is visually on the left screen side
            raw_label = handedness.classification[0].label
            label = "Right" if raw_label == "Left" else "Left"

            detected_labels.add(label)

            # Convert normalized landmarks to pixel coordinates
            pts_px = np.array(
                [[lm.x * w, lm.y * h] for lm in hand_lms.landmark],
                dtype=np.float32
            )

            # Apply Exponential Moving Average (EMA) smoothing
            if label in self.smoothed_history:
                smoothed_pts = (
                    self.smoothing_alpha * pts_px +
                    (1.0 - self.smoothing_alpha) * self.smoothed_history[label]
                )
            else:
                smoothed_pts = pts_px.copy()

            self.smoothed_history[label] = smoothed_pts

            # Build HandData
            hand_obj = self._build_hand_data(smoothed_pts, label, h, w)

            if label == "Left":
                dual_data.left_hand = hand_obj
            else:
                dual_data.right_hand = hand_obj

        # Prune smoothing history for hands no longer detected
        for old_label in list(self.smoothed_history.keys()):
            if old_label not in detected_labels:
                del self.smoothed_history[old_label]

        # Dual hand metrics
        count = (1 if dual_data.left_hand else 0) + (1 if dual_data.right_hand else 0)
        dual_data.hands_count = count
        dual_data.both_hands_visible = (dual_data.left_hand is not None and dual_data.right_hand is not None)

        if dual_data.both_hands_visible:
            lh = dual_data.left_hand
            rh = dual_data.right_hand
            dual_data.inter_hand_distance = float(math.hypot(
                rh.palm_center[0] - lh.palm_center[0],
                rh.palm_center[1] - lh.palm_center[1]
            ))
            dual_data.hands_midpoint = (
                int((lh.palm_center[0] + rh.palm_center[0]) / 2),
                int((lh.palm_center[1] + rh.palm_center[1]) / 2)
            )

        return dual_data

    def _build_hand_data(self, pts: np.ndarray, label: str, frame_h: int, frame_w: int) -> HandData:
        """Extracts key anatomical points and finger states from 21 landmarks."""
        def p(idx: int) -> Tuple[int, int]:
            return int(round(pts[idx][0])), int(round(pts[idx][1]))

        wrist = p(WRIST)
        thumb_tip = p(THUMB_TIP)
        thumb_ip = p(THUMB_IP)
        thumb_mcp = p(THUMB_MCP)
        index_tip = p(INDEX_TIP)
        index_pip = p(INDEX_PIP)
        index_mcp = p(INDEX_MCP)
        middle_tip = p(MIDDLE_TIP)
        middle_mcp = p(MIDDLE_MCP)
        ring_tip = p(RING_TIP)
        pinky_tip = p(PINKY_TIP)

        # Palm center: weighted average of wrist and finger base joints
        palm_center = (
            int(round((wrist[0] + index_mcp[0] + middle_mcp[0]) / 3.0)),
            int(round((wrist[1] + index_mcp[1] + middle_mcp[1]) / 3.0))
        )

        # Pinch distance between thumb tip and index tip
        pinch_dist = float(math.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1]))
        is_pinching = (pinch_dist <= self.pinch_threshold_px)

        # Finger extended classification
        # For index, middle, ring, pinky: fingertip Y is above (smaller than) PIP joint Y in standard pose
        # More robust: distance from wrist to tip is greater than wrist to PIP
        def is_ext(tip_idx: int, pip_idx: int) -> bool:
            dist_tip = math.hypot(pts[tip_idx][0] - wrist[0], pts[tip_idx][1] - wrist[1])
            dist_pip = math.hypot(pts[pip_idx][0] - wrist[0], pts[pip_idx][1] - wrist[1])
            return dist_tip > (dist_pip * 1.1)

        # Thumb extension: thumb tip distance to pinky MCP vs thumb MCP to pinky MCP
        pinky_mcp_pt = pts[PINKY_MCP]
        dist_thumb_tip = math.hypot(pts[THUMB_TIP][0] - pinky_mcp_pt[0], pts[THUMB_TIP][1] - pinky_mcp_pt[1])
        dist_thumb_ip = math.hypot(pts[THUMB_IP][0] - pinky_mcp_pt[0], pts[THUMB_IP][1] - pinky_mcp_pt[1])
        thumb_ext = dist_thumb_tip > (dist_thumb_ip * 1.05)

        fingers_ext = [
            bool(thumb_ext),
            bool(is_ext(INDEX_TIP, INDEX_PIP)),
            bool(is_ext(MIDDLE_TIP, MIDDLE_PIP)),
            bool(is_ext(RING_TIP, RING_PIP)),
            bool(is_ext(PINKY_TIP, PINKY_PIP))
        ]

        # Bounding box
        min_x = int(max(0, np.min(pts[:, 0])))
        max_x = int(min(frame_w, np.max(pts[:, 0])))
        min_y = int(max(0, np.min(pts[:, 1])))
        max_y = int(min(frame_h, np.max(pts[:, 1])))
        bbox = (min_x, min_y, max(1, max_x - min_x), max(1, max_y - min_y))

        return HandData(
            label=label,
            landmarks_px=pts,
            wrist=wrist,
            thumb_tip=thumb_tip,
            thumb_ip=thumb_ip,
            thumb_mcp=thumb_mcp,
            index_tip=index_tip,
            index_pip=index_pip,
            index_mcp=index_mcp,
            middle_tip=middle_tip,
            middle_mcp=middle_mcp,
            ring_tip=ring_tip,
            pinky_tip=pinky_tip,
            palm_center=palm_center,
            pinch_distance=pinch_dist,
            is_pinching=is_pinching,
            fingers_extended=fingers_ext,
            bbox=bbox
        )

    def draw_debug_hands(self, frame_bgr: np.ndarray, dual_data: DualHandData) -> np.ndarray:
        """Renders subtle, glowing cybernetic hand skeletons onto frame."""
        overlay = frame_bgr.copy()

        for hand in [dual_data.left_hand, dual_data.right_hand]:
            if hand is None:
                continue

            # Electric cyan for Left hand, Gold/Amber for Right hand
            color_accent = (255, 220, 0) if hand.label == "Left" else (0, 215, 255)
            color_joint = (255, 255, 255)

            # Draw bones
            for start_idx, end_idx in HAND_CONNECTIONS:
                p1 = (int(hand.landmarks_px[start_idx][0]), int(hand.landmarks_px[start_idx][1]))
                p2 = (int(hand.landmarks_px[end_idx][0]), int(hand.landmarks_px[end_idx][1]))
                cv2.line(overlay, p1, p2, color_accent, 2, cv2.LINE_AA)

            # Draw joints
            for idx in range(21):
                pt = (int(hand.landmarks_px[idx][0]), int(hand.landmarks_px[idx][1]))
                # Tip joints are highlighted
                radius = 4 if idx in [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP] else 2
                cv2.circle(overlay, pt, radius, color_joint, -1)

            # Pinch indicator circle
            if hand.is_pinching:
                pinch_mid = (
                    int((hand.thumb_tip[0] + hand.index_tip[0]) / 2),
                    int((hand.thumb_tip[1] + hand.index_tip[1]) / 2)
                )
                cv2.circle(overlay, pinch_mid, 8, (0, 255, 128), 2, cv2.LINE_AA)

        return cv2.addWeighted(overlay, 0.75, frame_bgr, 0.25, 0)

    def release(self) -> None:
        if hasattr(self, 'hands_detector') and self.hands_detector is not None:
            self.hands_detector.close()
            self.hands_detector = None

    def __del__(self) -> None:
        self.release()
