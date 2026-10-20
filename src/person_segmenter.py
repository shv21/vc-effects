"""
Person silhouette segmentation module using MediaPipe Selfie Segmentation.
Separates the person foreground from the background with temporal smoothing
and boundary feathering to reduce halos around hair and shoulders.
"""

import logging
from typing import Optional, Tuple
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

logger = logging.getLogger(__name__)


class PersonSegmenter:
    """Extracts stabilized person foreground alpha mask in real time."""

    def __init__(
        self,
        model_selection: int = 0,
        threshold: float = 0.50,
        feather_radius: int = 9,
        smoothing_alpha: float = 0.75
    ) -> None:
        if mp is None:
            raise RuntimeError("MediaPipe is required for person segmentation.")

        self.mp_selfie = mp.solutions.selfie_segmentation
        # model_selection=0: general model (256x256), faster for real-time video
        self.segmenter = self.mp_selfie.SelfieSegmentation(model_selection=model_selection)

        self.threshold = threshold
        self.feather_radius = feather_radius
        self.smoothing_alpha = smoothing_alpha

        self.prev_mask: Optional[np.ndarray] = None

    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts feathered person foreground mask (0.0 = background, 1.0 = person).
        Returns a float32 array of shape (H, W).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return np.ones((480, 640), dtype=np.float32)

        h, w = frame_bgr.shape[:2]

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.segmenter.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if results.segmentation_mask is None:
            # If segmentation fails, return identity (full person)
            return np.ones((h, w), dtype=np.float32)

        raw_mask = results.segmentation_mask.astype(np.float32)

        # Apply temporal smoothing on probability field to kill high-frequency flicker
        if self.prev_mask is not None and self.prev_mask.shape == raw_mask.shape:
            smoothed = (
                self.smoothing_alpha * raw_mask +
                (1.0 - self.smoothing_alpha) * self.prev_mask
            )
        else:
            smoothed = raw_mask.copy()

        self.prev_mask = smoothed

        # Smooth boundary feathering via Gaussian blur
        k = max(3, self.feather_radius)
        if k % 2 == 0:
            k += 1

        blurred = cv2.GaussianBlur(smoothed, (k, k), 0)

        # Rescale & contrast enhancement
        contrasted = np.clip((blurred - (self.threshold - 0.2)) / 0.4, 0.0, 1.0)
        return contrasted.astype(np.float32)

    def release(self) -> None:
        if hasattr(self, 'segmenter') and self.segmenter is not None:
            self.segmenter.close()
            self.segmenter = None

    def __del__(self) -> None:
        self.release()
