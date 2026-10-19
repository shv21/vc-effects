"""
Photo capture and video recording module for Spider-Man AR Studio.
Features:
- Timestamped photo saving with high JPEG quality
- Asynchronous, queue-buffered video recording to prevent GUI frame drops
- Safe writer finalization to ensure non-corrupted MP4 files
"""

import os
import time
import queue
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PhotoSaver:
    """Handles snapshot saving with formatted timestamp filenames."""

    @staticmethod
    def save(
        frame_bgr: np.ndarray,
        output_dir: str = "output"
    ) -> Tuple[bool, str]:
        """
        Saves a BGR frame as a high-quality JPEG image.
        Returns (success, absolute_filepath_or_error_message).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return False, "Cannot save empty frame."

        try:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"photo_{timestamp}.jpg"
            full_path = out_path / filename

            # Save with high JPEG quality (95)
            success = cv2.imwrite(
                str(full_path),
                frame_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            )

            if success:
                logger.info(f"Photo saved successfully to {full_path}")
                return True, str(full_path.resolve())
            else:
                return False, f"OpenCV failed to write image to {full_path}"
        except Exception as e:
            logger.error(f"Error saving photo: {e}")
            return False, str(e)


class VideoRecorder:
    """
    Asynchronous video recorder with worker thread to guarantee zero camera lag
    during high-framerate capture.
    """

    def __init__(self, output_dir: str = "output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.is_recording = False
        self.writer: Optional[cv2.VideoWriter] = None
        self.current_filepath: Optional[Path] = None

        self.frame_queue: queue.Queue = queue.Queue(maxsize=150)
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        self.start_time: float = 0.0
        self.recorded_frames: int = 0

    def start(
        self,
        width: int,
        height: int,
        fps: float = 30.0,
        output_dir: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Initializes video writer and launches asynchronous writing thread.
        Returns (success, message_or_filepath).
        """
        if self.is_recording:
            return False, "Recording is already in progress."

        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_filepath = self.output_dir / f"video_{timestamp}.mp4"

        # Try 'mp4v' first, then 'XVID' fallback
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        effective_fps = max(10.0, min(60.0, fps))

        try:
            self.writer = cv2.VideoWriter(
                str(self.current_filepath),
                fourcc,
                effective_fps,
                (width, height)
            )

            if not self.writer.isOpened():
                logger.warning("mp4v codec failed. Falling back to XVID / AVI format...")
                self.current_filepath = self.output_dir / f"video_{timestamp}.avi"
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                self.writer = cv2.VideoWriter(
                    str(self.current_filepath),
                    fourcc,
                    effective_fps,
                    (width, height)
                )

            if not self.writer.isOpened():
                self.writer = None
                return False, "Could not open VideoWriter with available codecs."

        except Exception as e:
            return False, f"Failed to initialize video codec: {e}"

        # Initialize queue & thread
        self.stop_event.clear()
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        self.is_recording = True
        self.start_time = time.time()
        self.recorded_frames = 0

        self.worker_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.worker_thread.start()

        logger.info(f"Started recording video to {self.current_filepath} at {effective_fps:.1f} FPS")
        return True, str(self.current_filepath.resolve())

    def write_frame(self, frame_bgr: np.ndarray) -> None:
        """Pushes a frame to the async queue if recording is active."""
        if not self.is_recording or frame_bgr is None:
            return

        try:
            # Non-blocking put; if queue fills up, drop to prevent memory explosion
            self.frame_queue.put_nowait(frame_bgr.copy())
            self.recorded_frames += 1
        except queue.Full:
            logger.warning("Video writer queue full, dropped frame to keep preview responsive.")

    def _writer_loop(self) -> None:
        """Background thread that consumes frames from queue and writes to disk."""
        while not self.stop_event.is_set() or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                if self.writer and self.writer.isOpened():
                    self.writer.write(frame)
                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in video writer loop: {e}")

    def stop(self) -> Tuple[bool, str, int, float]:
        """
        Stops recording, flushes buffer, closes video file cleanly.
        Returns (success, filepath, total_frames, duration_seconds).
        """
        if not self.is_recording:
            return False, "Not recording.", 0, 0.0

        self.is_recording = False
        self.stop_event.set()

        duration = max(0.0, time.time() - self.start_time)
        filepath_str = str(self.current_filepath.resolve()) if self.current_filepath else ""

        # Wait for queue worker to flush
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3.0)
            self.worker_thread = None

        if self.writer:
            try:
                self.writer.release()
            except Exception as e:
                logger.error(f"Error closing video writer: {e}")
            self.writer = None

        logger.info(f"Video saved: {filepath_str} ({self.recorded_frames} frames, {duration:.1f}s)")
        return True, filepath_str, self.recorded_frames, duration

    def get_duration(self) -> float:
        """Returns elapsed recording time in seconds."""
        if self.is_recording:
            return time.time() - self.start_time
        return 0.0

    def __del__(self) -> None:
        if self.is_recording:
            self.stop()
