"""
High-performance dual hand tracking module for gesture-controlled AR.
Features:
- Sub-pixel landmark detection with MediaPipe Hands
- Distance-invariant normalized pinch detection
- Velocity-adaptive EMA smoothing (zero lag during fast movement, jitter-free when stationary)
- Spatial continuity hand tracking to prevent left/right history cross-wiring
- Accurate camera-space coordinate mapping
"""

import math
import time
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

# Key MediaPipe Hand Landmark Indices
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
    label: str  # 'ScreenLeft' or 'ScreenRight'
    landmarks_px: np.ndarray  # (21, 2) float32
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
    pinch_ratio: float
    is_pinching: bool
    fingers_extended: List[bool]  # [thumb, index, middle, ring, pinky]
    bbox: Tuple[int, int, int, int]
    hand_scale: float


@dataclass
class DualHandData:
    """Dual-hand context container."""
    left_hand: Optional[HandData] = None
    right_hand: Optional[HandData] = None
    hands_count: int = 0
    both_hands_visible: bool = False
    inter_hand_distance: float = 0.0
    hands_midpoint: Tuple[int, int] = (0, 0)
    is_dual_pinching: bool = False


class HandTracker:
    """Tracks both hands using MediaPipe Hands with adaptive velocity smoothing."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.55,
        min_tracking_confidence: float = 0.50,
        base_smoothing_alpha: float = 0.78
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

        self.base_smoothing_alpha = base_smoothing_alpha
        # Spatial track continuity: track_id -> (last_seen_time, smoothed_landmarks)
        self.active_tracks: Dict[int, Tuple[float, np.ndarray]] = {}
        self.track_max_idle_sec: float = 0.35

    def process(self, frame_bgr: np.ndarray) -> DualHandData:
        """Processes BGR frame in camera coordinates and returns DualHandData."""
        if frame_bgr is None or frame_bgr.size == 0:
            return DualHandData()

        now = time.time()
        h, w = frame_bgr.shape[:2]

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands_detector.process(frame_rgb)
        frame_rgb.flags.writeable = True

        dual = DualHandData()

        # Clean up stale tracks
        stale_tracks = [tid for tid, (t, _) in self.active_tracks.items() if now - t > self.track_max_idle_sec]
        for tid in stale_tracks:
            del self.active_tracks[tid]

        if not results.multi_hand_landmarks:
            return dual

        raw_hands = []
        for lms in results.multi_hand_landmarks:
            # Map normalized coordinates directly to camera-frame pixel dimensions
            pts_px = np.array([[lm.x * w, lm.y * h] for lm in lms.landmark], dtype=np.float32)
            raw_hands.append(pts_px)

        # Associate detections to tracks using spatial distance
        smoothed_hands = []
        used_tracks = set()

        for pts in raw_hands:
            centroid = np.mean(pts, axis=0)
            best_tid = None
            best_dist = float("inf")

            for tid, (t_seen, prev_pts) in self.active_tracks.items():
                if tid in used_tracks:
                    continue
                prev_centroid = np.mean(prev_pts, axis=0)
                d = float(np.linalg.norm(centroid - prev_centroid))
                if d < best_dist and d < (w * 0.40):
                    best_dist = d
                    best_tid = tid

            if best_tid is not None:
                used_tracks.add(best_tid)
                _, prev_pts = self.active_tracks[best_tid]
                movement = float(np.mean(np.linalg.norm(pts - prev_pts, axis=1)))
                # Adaptive alpha: 0.70 when still (smooth), up to 0.96 when moving rapidly (zero lag)
                adaptive_alpha = min(0.96, max(0.70, self.base_smoothing_alpha + movement * 0.025))
                smoothed = adaptive_alpha * pts + (1.0 - adaptive_alpha) * prev_pts
                self.active_tracks[best_tid] = (now, smoothed)
            else:
                # New track
                new_tid = 0 if 0 not in self.active_tracks else (1 if 1 not in self.active_tracks else len(self.active_tracks))
                smoothed = pts.copy()
                self.active_tracks[new_tid] = (now, smoothed)
                used_tracks.add(new_tid)

            smoothed_hands.append(smoothed)

        if not smoothed_hands:
            return dual

        # Sort hands horizontally by centroid X for screen-space left/right consistency
        smoothed_hands.sort(key=lambda pts: float(np.mean(pts[:, 0])))

        hand_objects = []
        if len(smoothed_hands) == 1:
            pts = smoothed_hands[0]
            cx = float(np.mean(pts[:, 0]))
            label = "ScreenLeft" if cx < w * 0.5 else "ScreenRight"
            hand_obj = self._build_hand_data(pts, label, h, w)
            hand_objects.append(hand_obj)
            if label == "ScreenLeft":
                dual.left_hand = hand_obj
            else:
                dual.right_hand = hand_obj
        else:
            # First is ScreenLeft (smaller X), second is ScreenRight (larger X)
            lh_obj = self._build_hand_data(smoothed_hands[0], "ScreenLeft", h, w)
            rh_obj = self._build_hand_data(smoothed_hands[1], "ScreenRight", h, w)
            hand_objects.extend([lh_obj, rh_obj])
            dual.left_hand = lh_obj
            dual.right_hand = rh_obj

        dual.hands_count = len(hand_objects)
        dual.both_hands_visible = (dual.left_hand is not None and dual.right_hand is not None)

        if dual.both_hands_visible:
            lh = dual.left_hand
            rh = dual.right_hand
            dual.inter_hand_distance = float(math.hypot(
                rh.palm_center[0] - lh.palm_center[0],
                rh.palm_center[1] - lh.palm_center[1]
            ))
            dual.hands_midpoint = (
                int((lh.palm_center[0] + rh.palm_center[0]) / 2),
                int((lh.palm_center[1] + rh.palm_center[1]) / 2)
            )
            dual.is_dual_pinching = bool(lh.is_pinching and rh.is_pinching)

        return dual

    def _build_hand_data(self, pts: np.ndarray, label: str, frame_h: int, frame_w: int) -> HandData:
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

        palm_center = (
            int(round((wrist[0] + index_mcp[0] + middle_mcp[0]) / 3.0)),
            int(round((wrist[1] + index_mcp[1] + middle_mcp[1]) / 3.0))
        )

        # Anatomical scale reference: distance from wrist to index MCP
        hand_scale = max(20.0, float(math.hypot(index_mcp[0] - wrist[0], index_mcp[1] - wrist[1])))

        # Distance-invariant pinch ratio
        pinch_dist = float(math.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1]))
        pinch_ratio = pinch_dist / hand_scale
        is_pinching = (pinch_ratio <= 0.38)

        # Finger extension classification
        def is_ext(tip_i: int, pip_i: int) -> bool:
            dist_tip = math.hypot(pts[tip_i][0] - wrist[0], pts[tip_i][1] - wrist[1])
            dist_pip = math.hypot(pts[pip_i][0] - wrist[0], pts[pip_i][1] - wrist[1])
            return dist_tip > (dist_pip * 1.08)

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
            pinch_ratio=pinch_ratio,
            is_pinching=is_pinching,
            fingers_extended=fingers_ext,
            bbox=bbox,
            hand_scale=hand_scale
        )

    def draw_debug_hands(self, frame_bgr: np.ndarray, dual: DualHandData) -> np.ndarray:
        """Renders subtle hand skeletons in debug mode only."""
        overlay = frame_bgr.copy()
        for hand in [dual.left_hand, dual.right_hand]:
            if hand is None:
                continue

            color = (0, 220, 255) if "Left" in hand.label else (255, 180, 0)
            for s_idx, e_idx in HAND_CONNECTIONS:
                p1 = (int(hand.landmarks_px[s_idx][0]), int(hand.landmarks_px[s_idx][1]))
                p2 = (int(hand.landmarks_px[e_idx][0]), int(hand.landmarks_px[e_idx][1]))
                cv2.line(overlay, p1, p2, color, 2, cv2.LINE_AA)

            for idx in [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP, WRIST]:
                pt = (int(hand.landmarks_px[idx][0]), int(hand.landmarks_px[idx][1]))
                cv2.circle(overlay, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)

            if hand.is_pinching:
                mid = (
                    int((hand.thumb_tip[0] + hand.index_tip[0]) / 2),
                    int((hand.thumb_tip[1] + hand.index_tip[1]) / 2)
                )
                cv2.circle(overlay, mid, 10, (0, 255, 100), 2, cv2.LINE_AA)

        return cv2.addWeighted(overlay, 0.70, frame_bgr, 0.30, 0)

    def release(self) -> None:
        if hasattr(self, 'hands_detector') and self.hands_detector is not None:
            self.hands_detector.close()
            self.hands_detector = None

    def __del__(self) -> None:
        self.release()
