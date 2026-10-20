"""
Upgraded gesture recognizer for hands-free AR control.
Separates continuous spatial tracking from discrete action triggers.
Eliminates accidental locking loops and ensures live tracking is always default.
Actions:
- Swipe Left / Right: Change visual effect
- V-Sign / Peace (held 0.4s): Cycle Scene Mode (Mode A -> B -> C -> D)
- Closed Fist (held 0.6s): Deliberate Freeze/Lock toggle
- Open Palm: Instant release / unlock to live tracking
"""

import time
import math
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque

from .hand_tracker import HandData, DualHandData

logger = logging.getLogger(__name__)


@dataclass
class GestureResult:
    """Telemetry report of currently recognized gestures and triggered actions."""
    primary_gesture: str = "SEARCHING"
    action_triggered: Optional[str] = None
    confidence: float = 0.0
    description: str = "Bring both hands into view to control region"
    is_framing: bool = False
    is_dual_pinching: bool = False


class GestureRecognizer:
    """Evaluates spatial landmarks and temporal trajectories to trigger actions."""

    def __init__(
        self,
        cooldown_sec: float = 0.65,
        swipe_velocity_min: float = 300.0,
        swipe_displacement_min: float = 65.0,
        hold_time_sec: float = 0.35
    ) -> None:
        self.cooldown_sec = cooldown_sec
        self.swipe_velocity_min = swipe_velocity_min
        self.swipe_displacement_min = swipe_displacement_min
        self.hold_time_sec = hold_time_sec

        # Circular position buffers for velocity tracking
        self.left_history: deque = deque(maxlen=10)
        self.right_history: deque = deque(maxlen=10)

        # State tracking for debouncing
        self.last_action_time: float = 0.0
        self.candidate_action: Optional[str] = None
        self.candidate_first_seen: float = 0.0

    def process(self, dual: DualHandData) -> GestureResult:
        """Analyzes dual hand metrics and returns recognized gesture and triggered action."""
        now = time.time()
        result = GestureResult()

        # Update position history for swipe velocity
        if dual.left_hand:
            self.left_history.append((now, dual.left_hand.palm_center[0], dual.left_hand.palm_center[1]))
        else:
            self.left_history.clear()

        if dual.right_hand:
            self.right_history.append((now, dual.right_hand.palm_center[0], dual.right_hand.palm_center[1]))
        else:
            self.right_history.clear()

        if not dual.both_hands_visible:
            if dual.left_hand or dual.right_hand:
                result.primary_gesture = "SINGLE_HAND"
                result.description = "Show other hand to complete 4-corner region"
            else:
                result.primary_gesture = "SEARCHING"
                result.description = "Show both hands to camera to control effect"
            return result

        lh = dual.left_hand
        rh = dual.right_hand

        # Continuous status flags
        result.is_dual_pinching = (lh.is_pinching and rh.is_pinching)
        result.is_framing = True  # Both hands in view framing active region

        # 1. Dynamic Swipe Check (Priority for hands-free switching)
        swipe_act = self._check_swipe(now)
        if swipe_act and (now - self.last_action_time >= self.cooldown_sec):
            self.last_action_time = now
            result.primary_gesture = "SWIPE_RIGHT" if swipe_act == "next_effect" else "SWIPE_LEFT"
            result.action_triggered = swipe_act
            result.confidence = 0.95
            result.description = f"Hand Swipe: {'Next Effect ▶' if swipe_act == 'next_effect' else '◀ Previous Effect'}"
            return result

        # 2. Static Gesture Classifications
        lh_g = self._classify_hand_posture(lh)
        rh_g = self._classify_hand_posture(rh)

        # Check for deliberate mode switch (V-Sign held on either hand)
        if (lh_g == "V_SIGN" or rh_g == "V_SIGN") and not (lh.is_pinching or rh.is_pinching):
            self._evaluate_held_action(now, "cycle_scene_mode", "✌️ Peace Sign: Change Scene Mode", result, hold_required=0.35)
            return result

        # Check for deliberate freeze (Closed Fist held on either hand)
        if (lh_g == "FIST" or rh_g == "FIST") and not (lh.is_pinching or rh.is_pinching):
            self._evaluate_held_action(now, "toggle_freeze", "✊ Fist: Freeze/Unfreeze Region", result, hold_required=0.50)
            return result

        # Open Palms (Unlock / Reset to live follow)
        if lh_g == "OPEN_PALM" and rh_g == "OPEN_PALM":
            result.primary_gesture = "OPEN_PALM"
            result.description = "✋ Open Palms: Smooth Live Tracking"
            result.action_triggered = "ensure_unlocked"
            return result

        # Dual Pinch: Active fine-sizing
        if result.is_dual_pinching:
            result.primary_gesture = "DUAL_PINCH"
            result.description = "🤏 Dual Pinch: Active Precision Sizing"
            return result

        # Default continuous framing state
        result.primary_gesture = "TWO_HAND_FRAME"
        result.description = "Two Hands Framing Live Region"
        self.candidate_action = None
        return result

    def _evaluate_held_action(
        self,
        now: float,
        action: str,
        desc: str,
        result: GestureResult,
        hold_required: float = 0.35
    ) -> None:
        """Debounces deliberate gestures requiring a sustained posture."""
        if self.candidate_action == action:
            hold_time = now - self.candidate_first_seen
            if hold_time >= hold_required and (now - self.last_action_time >= self.cooldown_sec):
                self.last_action_time = now
                self.candidate_action = None
                result.action_triggered = action
                result.primary_gesture = action.upper()
                result.confidence = 1.0
                result.description = f"{desc} [TRIGGERED]"
            else:
                result.primary_gesture = action.upper()
                result.confidence = min(1.0, hold_time / hold_required)
                result.description = f"{desc} (Hold {int(hold_time / hold_required * 100)}%)"
        else:
            self.candidate_action = action
            self.candidate_first_seen = now
            result.primary_gesture = action.upper()
            result.description = desc

    def _classify_hand_posture(self, hand: HandData) -> str:
        """Classifies static hand posture."""
        f = hand.fingers_extended
        thumb, index, middle, ring, pinky = f[0], f[1], f[2], f[3], f[4]

        if hand.is_pinching:
            return "PINCH"
        if not index and not middle and not ring and not pinky:
            return "FIST"
        if index and middle and ring and pinky:
            return "OPEN_PALM"
        if index and middle and not ring and not pinky:
            return "V_SIGN"
        return "FRAME"

    def _check_swipe(self, now: float) -> Optional[str]:
        """Detects high-velocity horizontal hand swipes."""
        for buffer in [self.right_history, self.left_history]:
            if len(buffer) < 4:
                continue

            t_old, x_old, _ = buffer[0]
            t_new, x_new, _ = buffer[-1]
            dt = t_new - t_old

            if 0.06 < dt < 0.30:
                dx = x_new - x_old
                velocity = dx / dt
                if abs(velocity) >= self.swipe_velocity_min and abs(dx) >= self.swipe_displacement_min:
                    buffer.clear()
                    # dx > 0: hand moved towards right edge -> NEXT
                    # dx < 0: hand moved towards left edge -> PREVIOUS
                    return "next_effect" if dx > 0 else "prev_effect"

        return None
