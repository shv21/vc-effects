"""
Upgraded gesture recognizer for hands-free AR control.
Separates continuous spatial tracking from discrete action triggers.
Features:
- Edge-triggered gesture latching: held gestures trigger once and require release before re-triggering
- Prevents rapid mode overshooting while holding gestures
- Debounced swipe detection for hands-free effect switching
- Continuous live region control (never locks accidentally)
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
        cooldown_sec: float = 0.60,
        swipe_velocity_min: float = 280.0,
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

        # State tracking for debouncing & edge-triggered latching
        self.last_action_time: float = 0.0
        self.candidate_action: Optional[str] = None
        self.candidate_first_seen: float = 0.0
        self.latched_action: Optional[str] = None  # Prevents repeated triggering until gesture released

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
            self.latched_action = None
            self.candidate_action = None
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
        result.is_dual_pinching = bool(lh.is_pinching and rh.is_pinching)
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

        # 2. Static Posture Classifications
        lh_g = self._classify_hand_posture(lh)
        rh_g = self._classify_hand_posture(rh)

        # Mode switch: V-Sign held on either hand (Edge-triggered: triggers ONCE per gesture presentation)
        if (lh_g == "V_SIGN" or rh_g == "V_SIGN") and not (lh.is_pinching or rh.is_pinching):
            self._evaluate_held_action(now, "cycle_scene_mode", "✌️ Peace Sign: Change Mode", result, hold_required=self.hold_time_sec)
            return result
        else:
            # Gesture released: clear latch so V-sign can be used again
            if self.latched_action == "cycle_scene_mode":
                self.latched_action = None
                self.candidate_action = None

        # Dual Pinch: Active precision sizing
        if result.is_dual_pinching:
            result.primary_gesture = "DUAL_PINCH"
            result.description = "🤏 Dual Pinch: Active Region Control"
            self.candidate_action = None
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
        """Requires a gesture to be held continuously before triggering once (edge-triggered)."""
        # If already triggered for this gesture hold, wait until release
        if self.latched_action == action:
            result.primary_gesture = f"{action.upper()} (TRIGGERED)"
            result.description = f"Active: {desc} (Release to reset)"
            return

        if now - self.last_action_time < self.cooldown_sec:
            result.primary_gesture = f"{action.upper()}_COOLDOWN"
            return

        if self.candidate_action == action:
            elapsed = now - self.candidate_first_seen
            if elapsed >= hold_required:
                self.last_action_time = now
                self.latched_action = action  # Latch until hand changes posture
                self.candidate_action = None
                result.action_triggered = action
                result.primary_gesture = action.upper()
                result.description = f"Triggered: {desc}"
                logger.info(f"Gesture triggered: {action}")
            else:
                progress = int((elapsed / hold_required) * 100)
                result.primary_gesture = f"HOLDING ({progress}%)"
                result.description = f"Hold: {desc}"
        else:
            self.candidate_action = action
            self.candidate_first_seen = now
            result.primary_gesture = "STARTING_GESTURE"
            result.description = f"Hold: {desc}"

    def _classify_hand_posture(self, hand: HandData) -> str:
        """Classifies hand shape into posture categories."""
        ext = hand.fingers_extended
        # ext: [thumb, index, middle, ring, pinky]

        # V-Sign / Peace: Index + Middle extended, Ring + Pinky curled
        if ext[1] and ext[2] and not ext[3] and not ext[4]:
            return "V_SIGN"

        # Open Palm: 4 or 5 fingers extended
        if sum(ext) >= 4:
            return "OPEN_PALM"

        # Pinch
        if hand.is_pinching:
            return "PINCH"

        return "OTHER"

    def _check_swipe(self, now: float) -> Optional[str]:
        """Detects rapid horizontal hand velocity across recent frames."""
        for hist in [self.right_history, self.left_history]:
            if len(hist) < 4:
                continue

            dt = hist[-1][0] - hist[0][0]
            if dt <= 0.05 or dt > 0.45:
                continue

            dx = hist[-1][1] - hist[0][1]
            dy = hist[-1][2] - hist[0][2]

            # Mostly horizontal motion
            if abs(dx) < self.swipe_displacement_min or abs(dx) < abs(dy) * 1.5:
                continue

            velocity = abs(dx) / dt
            if velocity >= self.swipe_velocity_min:
                hist.clear()
                return "next_effect" if dx > 0 else "prev_effect"

        return None
