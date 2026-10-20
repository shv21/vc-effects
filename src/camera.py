"""
Camera capture module for Gesture-Controlled AR Experience.
Handles webcam initialization, DirectShow acceleration on Windows,
resolution configuration, and smoothed FPS calculation.
"""

import time
import logging
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """Manages webcam video capture, device selection, and FPS estimation."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30
    ) -> None:
        self.device_index = device_index
        self.desired_width = width
        self.desired_height = height
        self.desired_fps = fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False

        # FPS tracking
        self.prev_time = 0.0
        self.current_fps = 0.0
        self.fps_alpha = 0.9

    def start(self) -> bool:
        """
        Opens camera using DirectShow (preferred on Windows) or default backend.
        Returns True if successful, False otherwise.
        """
        if self.is_running and self.cap and self.cap.isOpened():
            return True

        logger.info(f"Opening camera index {self.device_index}...")

        # DirectShow is optimal on Windows for low latency
        try:
            self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            if not self.cap or not self.cap.isOpened():
                logger.warning("DirectShow failed. Falling back to default backend...")
                self.cap = cv2.VideoCapture(self.device_index)
        except Exception as e:
            logger.warning(f"Error opening camera with DirectShow: {e}")
            self.cap = cv2.VideoCapture(self.device_index)

        if not self.cap or not self.cap.isOpened():
            logger.error(f"Cannot open webcam index {self.device_index}.")
            self.is_running = False
            return False

        # Configure resolution & FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.desired_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.desired_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.desired_fps)

        self.is_running = True
        self.prev_time = time.time()
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera ready: {actual_w}x{actual_h} @ {self.desired_fps} FPS target")
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads next frame from webcam.
        Returns (success, frame_bgr).
        """
        if not self.is_running or self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            return False, None

        # Smooth FPS calculation
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now

        if dt > 0:
            instant_fps = 1.0 / dt
            self.current_fps = (self.fps_alpha * self.current_fps) + ((1.0 - self.fps_alpha) * instant_fps)

        return True, frame

    def get_fps(self) -> float:
        """Returns smoothed FPS estimate."""
        return self.current_fps

    def get_resolution(self) -> Tuple[int, int]:
        """Returns (width, height) of camera stream."""
        if self.cap and self.cap.isOpened():
            return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return self.desired_width, self.desired_height

    def is_opened(self) -> bool:
        return bool(self.is_running and self.cap and self.cap.isOpened())

    def release(self) -> None:
        self.is_running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
            self.cap = None

    @staticmethod
    def scan_available_cameras(max_tested: int = 5) -> list[int]:
        available = []
        for idx in range(max_tested):
            temp_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if temp_cap is not None and temp_cap.isOpened():
                available.append(idx)
                temp_cap.release()
            else:
                temp_cap = cv2.VideoCapture(idx)
                if temp_cap is not None and temp_cap.isOpened():
                    available.append(idx)
                    temp_cap.release()
        return available

    def __del__(self) -> None:
        self.release()
