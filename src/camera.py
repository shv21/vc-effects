"""
Camera capture module for Spider-Man AR Studio.
Handles webcam initialization, device selection, frame reading, and FPS calculation.
Optimized for Windows with DirectShow fallback.
"""

import time
import logging
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """Manages webcam capture, configuration, and FPS estimation."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
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
        self.fps_alpha = 0.9  # Smoothing factor for FPS

    def start(self) -> bool:
        """
        Open the camera using DirectShow (preferred on Windows) or default backend.
        Returns True if successful, False otherwise.
        """
        if self.is_running and self.cap and self.cap.isOpened():
            return True

        logger.info(f"Attempting to open camera index {self.device_index}...")

        # Try cv2.CAP_DSHOW first on Windows for fast startup and low latency
        try:
            self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            if not self.cap or not self.cap.isOpened():
                logger.warning("DirectShow failed or unsupported. Trying default backend...")
                self.cap = cv2.VideoCapture(self.device_index)
        except Exception as e:
            logger.warning(f"Error opening with DirectShow: {e}. Trying default...")
            self.cap = cv2.VideoCapture(self.device_index)

        if not self.cap or not self.cap.isOpened():
            logger.error(f"Failed to open camera with index {self.device_index}.")
            self.is_running = False
            return False

        # Configure camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.desired_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.desired_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.desired_fps)

        self.is_running = True
        self.prev_time = time.time()
        logger.info(
            f"Camera initialized: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
            f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {self.cap.get(cv2.CAP_PROP_FPS)} FPS"
        )
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads a frame from the webcam and updates FPS.
        Returns (success, frame_bgr).
        """
        if not self.is_running or self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            logger.warning("Empty frame received from webcam.")
            return False, None

        # Calculate FPS with exponential smoothing
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now

        if dt > 0:
            instant_fps = 1.0 / dt
            self.current_fps = (self.fps_alpha * self.current_fps) + ((1.0 - self.fps_alpha) * instant_fps)

        return True, frame

    def get_fps(self) -> float:
        """Returns the smoothed FPS estimate."""
        return self.current_fps

    def get_resolution(self) -> Tuple[int, int]:
        """Returns (width, height) of the active camera feed."""
        if self.cap and self.cap.isOpened():
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return 0, 0

    def is_opened(self) -> bool:
        """Checks if the camera stream is currently active."""
        return bool(self.is_running and self.cap and self.cap.isOpened())

    def release(self) -> None:
        """Cleanly releases the camera hardware."""
        self.is_running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
            self.cap = None
        logger.info("Camera released cleanly.")

    @staticmethod
    def scan_available_cameras(max_tested: int = 5) -> list[int]:
        """Scans indices 0..max_tested-1 to identify working webcams."""
        available = []
        for idx in range(max_tested):
            temp_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if temp_cap is not None and temp_cap.isOpened():
                available.append(idx)
                temp_cap.release()
            else:
                # Also test standard backend
                temp_cap = cv2.VideoCapture(idx)
                if temp_cap is not None and temp_cap.isOpened():
                    available.append(idx)
                    temp_cap.release()
        return available

    def __del__(self) -> None:
        self.release()
