"""
Headless pipeline verification script for Spider-Man AR Studio.
Tests face tracker, overlay renderer, filter manager, photo saver, and recorder.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.filter_manager import FilterManager
from src.face_tracker import FaceTracker, FaceData
from src.overlay_renderer import OverlayRenderer
from src.recorder import PhotoSaver, VideoRecorder


def test_pipeline() -> None:
    print("=== Testing FilterManager ===")
    fm = FilterManager()
    print(f"Total filters loaded: {len(fm.filters)}")
    for item in fm.filters:
        print(f" - [{item.filter_type}] {item.display_name} (thumb: {Path(item.thumbnail_path).name if item.thumbnail_path else 'None'})")

    assert len(fm.filters) >= 10, f"Expected at least 10 filters, found {len(fm.filters)}"

    print("\n=== Testing Overlay Renderer & Clipping ===")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_frame[:] = [100, 100, 100]

    # Create dummy face data
    face = FaceData(
        detected=True,
        landmarks_px=np.array([[320, 240] for _ in range(468)], dtype=np.float32),
        center=(320, 240),
        bbox=(220, 140, 200, 220),
        roll_deg=15.0,
        yaw_deg=5.0,
        pitch_deg=-2.0,
        face_width=180.0,
        face_height=210.0,
        eye_distance=70.0,
        left_eye=(285, 220),
        right_eye=(355, 235),
        eyes_midpoint=(320, 227),
        nose_tip=(320, 250),
        forehead=(320, 150),
        chin=(320, 350),
        mouth_center=(320, 290)
    )

    # Test applying each filter on dummy frame
    for item in fm.filters:
        fm.select_filter_by_id(item.filter_id)
        out = fm.apply_current_filter(dummy_frame.copy(), face)
        assert out is not None and out.shape == dummy_frame.shape, f"Filter {item.display_name} output shape mismatch!"

    print("All filters applied successfully on synthetic frame!")

    print("\n=== Testing PhotoSaver ===")
    success, photo_path = PhotoSaver.save(dummy_frame, "output")
    assert success, f"PhotoSaver failed: {photo_path}"
    print(f"Saved photo test: {photo_path}")
    assert Path(photo_path).exists()

    print("\n=== Testing VideoRecorder ===")
    vr = VideoRecorder("output")
    started, rec_msg = vr.start(640, 480, 30.0)
    assert started, f"VideoRecorder failed to start: {rec_msg}"
    for _ in range(10):
        vr.write_frame(dummy_frame)
    stopped, vid_path, frames, dur = vr.stop()
    assert stopped, f"VideoRecorder failed to stop: {vid_path}"
    print(f"Saved video test: {vid_path} ({frames} frames, {dur:.2f}s)")
    assert Path(vid_path).exists()

    print("\n=== All pipeline verification tests PASSED! ===")


if __name__ == "__main__":
    test_pipeline()
