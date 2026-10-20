"""
Gesture recognition engine for hands-free AR interaction.
Detects:
- Dual Pinch / Single Pinch (Thumb + Index)
- Swipe Left / Swipe Right (dynamic hand velocity)
- Open Palm (neutral/preview)
- V-Sign / Peace (mode toggle)
- Fist (freeze/lock)
- Two-hand separation (scaling)
Includes temporal debouncing and action cooldown timers to prevent false triggers.
"""

import time
import math
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from collections import deque

from .hand_tracker import HandData, DualHandData

logger = logging.getLogger(__name__)


@dataclass
class GestureResult:
    """Telemetry report of currently recognized gestures and triggered actions."""
    primary_gesture: str = "NONE"
    left_gesture: str = "NONE"
    right_gesture: str = "NONE"
    action_triggered: Optional[str] = None
    confidence: float = 0.0
    description: str = "Waiting for hands..."


class GestureRecognizer:
    """Evaluates spatial landmarks and temporal trajectories to trigger actions."""

    def __init__(
        self,
        cooldown_sec: float = 0.55,
        swipe_velocity_min: float = 320.0,
        swipe_displacement_min: float = 70.0,
        hold_time_sec: float = 0.15
    ) -> None:
        self.cooldown_sec = cooldown_sec
        self.swipe_velocity_min = swipe_velocity_min
        self.swipe_displacement_min = swipe_displacement_min
        self.hold_time_sec = hold_time_sec

        # Velocity tracking: circular buffers of (timestamp, x, y)
        self.left_positions: deque = deque(maxlen=10)
        self.right_positions: deque = deque(maxlen=10)

        # State tracking for debouncing
        self.last_action_time: float = 0.0
        self.candidate_gesture: str = "NONE"
        self.candidate_first_seen: float = 0.0

    def process(self, dual_data: DualHandData) -> GestureResult:
        """
        Analyzes dual-hand landmarks and returns recognized gestures and actions.
        """
        now = time.time()
        result = GestureResult()

        # Update position history for swipe velocity
        if dual_data.left_hand:
            self.left_positions.append((now, dual_data.left_hand.palm_center[0], dual_data.left_hand.palm_center[1]))
        else:
            self.left_positions.clear()

        if dual_data.right_hand:
            self.right_positions.append((now, dual_data.right_hand.palm_center[0], dual_data.right_hand.palm_center[1]))
        else:
            self.right_positions.clear()

        # 1. Identify static gestures on individual hands
        left_g = self._classify_hand_gesture(dual_data.left_hand)
        right_g = self._classify_hand_gesture(dual_data.right_hand)
        result.left_gesture = left_g
        result.right_gesture = right_g

        # 2. Check for dynamic Swipes (Priority for hands-free switching)
        swipe_action = self._check_swipe(now)
        if swipe_action and (now - self.last_action_time >= self.cooldown_sec):
            self.last_action_time = now
            result.primary_gesture = "SWIPE_" + swipe_action.upper().split("_")[-1]
            result.action_triggered = swipe_action
            result.confidence = 0.95
            direction_label = "Next Effect ▶" if "next" in swipe_action else "◀ Previous Effect"
            result.description = f"Hand Swipe Detected: {direction_label}"
            return result

        # 3. Two-Hand Composite Gestures
        detected_primary = "NONE"
        action = None

        if dual_data.both_hands_visible:
            # Dual Pinch: Both hands pinching
            if dual_data.left_hand.is_pinching and dual_data.right_hand.is_pinching:
                detected_primary = "DUAL_PINCH"
                action = "toggle_lock_region"
                desc = "Dual Pinch: Lock / Unlock Region"

            # Both Open Palms
            elif left_g == "OPEN_PALM" and right_g == "OPEN_PALM":
                detected_primary = "DUAL_OPEN_PALM"
                action = "preview_mode"
                desc = "Open Palms: Dynamic Region Tracking"

            # V-Sign on either hand while other is visible
            elif left_g == "V_SIGN" or right_g == "V_SIGN":
                detected_primary = "V_SIGN"
                action = "cycle_scene_mode"
                desc = "Peace / V-Sign: Switching Scene Mode"

            # Frame / Boundary pose
            elif (dual_data.left_hand.fingers_extended[1] and dual_data.right_hand.fingers_extended[1]):
                detected_primary = "TWO_HAND_FRAME"
                desc = "Two Hands Framing Active Region"

            else:
                detected_primary = f"{left_g} + {right_g}"
                desc = "Tracking both hands"

        elif dual_data.left_hand or dual_data.right_hand:
            # Single Hand Active
            active_hand = dual_data.left_hand if dual_data.left_hand else dual_data.right_hand
            active_g = left_g if dual_data.left_hand else right_g

            if active_hand.is_pinching:
                detected_primary = "SINGLE_PINCH"
                desc = f"{active_hand.label} Hand Pinch"
            elif active_g == "V_SIGN":
                detected_primary = "V_SIGN"
                action = "cycle_scene_mode"
                desc = "Peace / V-Sign: Switching Scene Mode"
            elif active_g == "FIST":
                detected_primary = "FIST"
                action = "freeze_region"
                desc = "Fist: Freeze Region"
            elif active_g == "OPEN_PALM":
                detected_primary = "OPEN_PALM"
                desc = "Open Palm Neutral"
            else:
                detected_primary = active_g
                desc = f"{active_hand.label} Hand Active"
        else:
            detected_primary = "SEARCHING"
            desc = "Show hands to webcam to begin"

        # Debounce confirmation hold time
        if detected_primary == self.candidate_gesture:
            hold_duration = now - self.candidate_first_seen
            if hold_duration >= self.hold_time_sec:
                result.primary_gesture = detected_primary
                result.confidence = min(1.0, 0.7 + hold_duration * 0.5)
                result.description = desc

                # Fire action with cooldown
                if action and (now - self.last_action_time >= self.cooldown_sec):
                    self.last_action_time = now
                    result.action_triggered = action
        else:
            self.candidate_gesture = detected_primary
            self.candidate_first_seen = now
            result.primary_gesture = detected_primary
            result.confidence = 0.5
            result.description = desc

        return result

    def _classify_hand_gesture(self, hand: Optional[HandData]) -> str:
        """Classifies static posture of an individual hand."""
        if hand is None:
            return "NONE"

        f = hand.fingers_extended
        thumb, index, middle, ring, pinky = f[0], f[1], f[2], f[3], f[4]

        # Pinch: thumb + index close together
        if hand.is_pinching:
            return "PINCH"

        # Fist: all fingers curled
        if not index and not middle and not ring and not pinky:
            return "FIST"

        # Open Palm: all fingers extended
        if index and middle and ring and pinky:
            return "OPEN_PALM"

        # V-Sign / Peace: index + middle extended, ring + pinky curled
        if index and middle and not ring and not pinky:
            return "V_SIGN"

        # Pointing: only index extended
        if index and not middle and not ring and not pinky:
            return "POINT"

        # Thumbs Up
        if thumb and not index and not middle and not ring and not pinky:
            return "THUMBS_UP"

        return "TRACKING"

    def _check_swipe(self, now: float) -> Optional[str]:
        """Calculates hand velocities across recent trajectory to detect deliberate swipes."""
        for buffer in [self.right_positions, self.left_positions]:
            if len(buffer) < 4:
                continue

            t_old, x_old, _ = buffer[0]
            t_new, x_new, _ = buffer[-1]
            dt = t_new - t_old

            if 0.05 < dt < 0.35:
                dx = x_new - x_old
                velocity = dx / dt  # pixels per second

                if abs(velocity) >= self.swipe_velocity_min and abs(dx) >= self.swipe_displacement_min:
                    buffer.clear()  # Consume swipe
                    # dx > 0: moved towards right screen edge -> SWIPE RIGHT (next)
                    # dx < 0: moved towards left screen edge -> SWIPE LEFT (prev)
                    return "next_effect" if dx > 0 else "prev_effect"

        return None
