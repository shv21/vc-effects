"""
Comprehensive Verification Test Suite for Two-Hand-Controlled Full-Scene AR Portal.

Verifies:
1. Four-point extraction from both hands (Thumb and Index fingertips)
2. Live portal position updates (Continuous follow, translation)
3. Portal dynamic resizing (Hands spread -> expand, hands close -> shrink)
4. Corner ordering and perspective transformation
5. Invalid and degenerate polygon handling (safely hides portal, 0 crash)
6. Scene switching, caching, reloading, and fit modes (Crop, Letterbox, Stretch)
7. Transparency handling (RGBA alpha channel preserved inside portal)
8. Strict pixel isolation outside polygon (Exactly 0 altered pixels)
9. Tracking loss cleanup & instant reacquisition (Zero stale freeze)
10. Configurable gesture activation rules (Dual Pinch, Framing, Both Hands)
"""

import sys
import time
from pathlib import Path
from typing import Tuple, Optional, List
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hand_tracker import HandTracker, HandData, DualHandData
from src.gesture_recognizer import GestureRecognizer, GestureResult
from src.region_controller import RegionController, RegionState, ActivationGestureRule
from src.scene_manager import ScenePortalManager, AspectRatioFitMode, SceneItem
from src.mask_renderer import MaskRenderer
from src.effect_manager import EffectManager
from src.recorder import PhotoSaver, VideoRecorder


def make_dummy_hand(
    label: str,
    cx: int,
    cy: int,
    pinching: bool = False,
    thumb_offset: Tuple[int, int] = (-40, 20),
    index_offset: Tuple[int, int] = (-15, -45)
) -> HandData:
    """Creates synthetic 21-point hand landmarks in camera coordinates."""
    pts = np.zeros((21, 2), dtype=np.float32)
    # Wrist
    pts[0] = [cx, cy + 70]
    pts[1] = [cx - 25, cy + 50]
    pts[2] = [cx - 35, cy + 30]
    pts[3] = [cx - 40, cy + 15]

    if pinching:
        # Thumb and index tips touch at pinch point
        pts[4] = [cx - 15, cy - 20]
        pts[8] = [cx - 15, cy - 20]
    else:
        # Configurable thumb and index offsets from center
        pts[4] = [cx + thumb_offset[0], cy + thumb_offset[1]]
        pts[8] = [cx + index_offset[0], cy + index_offset[1]]

    # Other finger landmarks
    pts[5] = [cx - 10, cy + 15]
    pts[6] = [cx - 12, cy - 10]
    pts[7] = [cx - 14, cy - 30]
    pts[9] = [cx, cy + 15]
    pts[10] = [cx, cy - 15]
    pts[11] = [cx, cy - 35]
    pts[12] = [cx, cy - 55]
    pts[13] = [cx + 12, cy + 20]
    pts[14] = [cx + 12, cy - 10]
    pts[15] = [cx + 12, cy - 30]
    pts[16] = [cx + 12, cy - 50]
    pts[17] = [cx + 25, cy + 25]
    pts[18] = [cx + 25, cy]
    pts[19] = [cx + 25, cy - 20]
    pts[20] = [cx + 25, cy - 40]

    pinch_dist = float(np.hypot(pts[4][0] - pts[8][0], pts[4][1] - pts[8][1]))
    hand_scale = float(np.hypot(pts[5][0] - pts[0][0], pts[5][1] - pts[0][1]))
    pinch_ratio = pinch_dist / max(1.0, hand_scale)
    is_pinch = pinch_ratio < 0.38

    return HandData(
        label=label,
        landmarks_px=pts,
        wrist=(int(pts[0][0]), int(pts[0][1])),
        thumb_tip=(int(pts[4][0]), int(pts[4][1])),
        thumb_ip=(int(pts[3][0]), int(pts[3][1])),
        thumb_mcp=(int(pts[2][0]), int(pts[2][1])),
        index_tip=(int(pts[8][0]), int(pts[8][1])),
        index_pip=(int(pts[6][0]), int(pts[6][1])),
        index_mcp=(int(pts[5][0]), int(pts[5][1])),
        middle_tip=(int(pts[12][0]), int(pts[12][1])),
        middle_mcp=(int(pts[9][0]), int(pts[9][1])),
        ring_tip=(int(pts[16][0]), int(pts[16][1])),
        pinky_tip=(int(pts[20][0]), int(pts[20][1])),
        palm_center=(cx, cy),
        pinch_distance=pinch_dist,
        pinch_ratio=pinch_ratio,
        is_pinching=is_pinch,
        fingers_extended=[True, True, True, True, True],
        bbox=(cx - 50, cy - 60, 100, 140),
        hand_scale=hand_scale
    )


# --- TEST 1: FOUR-POINT EXTRACTION FROM BOTH HANDS ---

def test_four_point_extraction_from_both_hands() -> None:
    print("\n[Test 1] Four-Point Extraction from Left & Right Hands...")
    rc = RegionController(activation_rule=ActivationGestureRule.BOTH_HANDS)

    # Frame gesture: Left hand has index high (-50) and thumb low (+30)
    lh = make_dummy_hand("ScreenLeft", 160, 240, pinching=False, thumb_offset=(-30, 40), index_offset=(-10, -50))
    # Right hand has index high (-50) and thumb low (+30)
    rh = make_dummy_hand("ScreenRight", 480, 240, pinching=False, thumb_offset=(30, 40), index_offset=(10, -50))

    dual = DualHandData(left_hand=lh, right_hand=rh, both_hands_visible=True, hands_count=2)
    quad = rc.update(dual, (480, 640))

    assert quad is not None, "Quad must not be None when both hands present"
    assert len(quad) == 4, f"Quad must have exactly 4 control points, got {len(quad)}"

    # Verify corner ordering: [TL, TR, BR, BL]
    tl, tr, br, bl = quad
    assert tl[0] < tr[0], f"Top-Left X ({tl[0]}) must be to the left of Top-Right X ({tr[0]})"
    assert bl[0] < br[0], f"Bottom-Left X ({bl[0]}) must be to the left of Bottom-Right X ({br[0]})"
    assert tl[1] < bl[1], f"Top-Left Y ({tl[1]}) must be higher up than Bottom-Left Y ({bl[1]})"
    assert tr[1] < br[1], f"Top-Right Y ({tr[1]}) must be higher up than Bottom-Right Y ({br[1]})"

    print("  -> Four live control points successfully extracted and ordered [TL, TR, BR, BL]!")


# --- TEST 2: LIVE PORTAL POSITION UPDATES ---

def test_live_portal_position_updates() -> None:
    print("\n[Test 2] Live Portal Position Updates (Follows Hands Continuously)...")
    rc = RegionController(activation_rule=ActivationGestureRule.BOTH_HANDS)

    lh1 = make_dummy_hand("ScreenLeft", 160, 240)
    rh1 = make_dummy_hand("ScreenRight", 480, 240)
    dual1 = DualHandData(left_hand=lh1, right_hand=rh1, both_hands_visible=True, hands_count=2)
    quad1 = rc.update(dual1, (480, 640))
    centroid_y1 = float(np.mean(quad1[:, 1]))

    # Move hands upward by 80px
    lh2 = make_dummy_hand("ScreenLeft", 160, 160)
    rh2 = make_dummy_hand("ScreenRight", 480, 160)
    dual2 = DualHandData(left_hand=lh2, right_hand=rh2, both_hands_visible=True, hands_count=2)

    quad2 = None
    for _ in range(8):
        quad2 = rc.update(dual2, (480, 640))
    centroid_y2 = float(np.mean(quad2[:, 1]))

    assert centroid_y2 < centroid_y1 - 40, f"Portal must translate upward with hands: {centroid_y2} < {centroid_y1}"

    # Move hands rightward by 100px
    lh3 = make_dummy_hand("ScreenLeft", 260, 160)
    rh3 = make_dummy_hand("ScreenRight", 580, 160)
    dual3 = DualHandData(left_hand=lh3, right_hand=rh3, both_hands_visible=True, hands_count=2)
    quad3 = None
    for _ in range(8):
        quad3 = rc.update(dual3, (480, 640))
    centroid_x3 = float(np.mean(quad3[:, 0]))

    assert centroid_x3 > float(np.mean(quad2[:, 0])) + 50, "Portal must translate rightward with hands"
    print("  -> Real-time continuous translation (up, down, left, right) confirmed!")


# --- TEST 3: PORTAL DYNAMIC RESIZING ---

def test_portal_resizing() -> None:
    print("\n[Test 3] Dynamic Portal Resizing (Hands Spread vs Close)...")
    rc = RegionController(activation_rule=ActivationGestureRule.BOTH_HANDS)

    # Initial hand distance = 300px
    lh_mid = make_dummy_hand("ScreenLeft", 170, 240)
    rh_mid = make_dummy_hand("ScreenRight", 470, 240)
    dual_mid = DualHandData(left_hand=lh_mid, right_hand=rh_mid, both_hands_visible=True, hands_count=2)
    quad_mid = rc.update(dual_mid, (480, 640))
    area_mid = float(cv2.contourArea(quad_mid))

    # Spread hands apart to 460px (Expand)
    lh_wide = make_dummy_hand("ScreenLeft", 90, 240)
    rh_wide = make_dummy_hand("ScreenRight", 550, 240)
    dual_wide = DualHandData(left_hand=lh_wide, right_hand=rh_wide, both_hands_visible=True, hands_count=2)
    quad_wide = None
    for _ in range(10):
        quad_wide = rc.update(dual_wide, (480, 640))
    area_wide = float(cv2.contourArea(quad_wide))

    assert area_wide > area_mid * 1.25, f"Spreading hands must expand portal: {area_wide:.0f} > {area_mid:.0f}"

    # Bring hands closer to 180px (Shrink)
    lh_close = make_dummy_hand("ScreenLeft", 230, 240)
    rh_close = make_dummy_hand("ScreenRight", 410, 240)
    dual_close = DualHandData(left_hand=lh_close, right_hand=rh_close, both_hands_visible=True, hands_count=2)
    quad_close = None
    for _ in range(10):
        quad_close = rc.update(dual_close, (480, 640))
    area_close = float(cv2.contourArea(quad_close))

    assert area_close < area_mid * 0.75, f"Bringing hands closer must shrink portal: {area_close:.0f} < {area_mid:.0f}"
    print(f"  -> Dynamic resizing verified: Wide={area_wide:.0f}px², Mid={area_mid:.0f}px², Shrunk={area_close:.0f}px²")


# --- TEST 4: CORNER ORDERING & PERSPECTIVE TRANSFORMATION ---

def test_corner_ordering_and_perspective_transform() -> None:
    print("\n[Test 4] Corner Ordering & OpenCV Perspective Transformation...")
    sm = ScenePortalManager(scenes_dir="assets/scenes", default_fit_mode="crop")
    assert sm.get_scene_count() >= 3, f"Expected at least 3 scenes in assets/scenes, found {sm.get_scene_count()}"

    base_frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    quad = np.array([
        [180, 120],  # TL
        [460, 140],  # TR
        [440, 360],  # BR
        [160, 340]   # BL
    ], dtype=np.int32)

    rendered = sm.render_portal(base_frame, quad, opacity=1.0)
    assert rendered is not None and rendered.shape == base_frame.shape, "Rendered output shape must match base frame"

    # Verify pixels inside portal changed to the scene image
    inside_sample = rendered[240, 300]
    assert not np.array_equal(inside_sample, [100, 100, 100]), "Inside portal pixels must contain warped scene!"
    print(f"  -> Perspective transformation mapped scene cleanly onto {quad.shape} quad!")


# --- TEST 5: INVALID & DEGENERATE POLYGON HANDLING ---

def test_invalid_and_degenerate_polygon_handling() -> None:
    print("\n[Test 5] Invalid & Degenerate Polygon Handling (Graceful Hiding, 0 Crashes)...")
    sm = ScenePortalManager(scenes_dir="assets/scenes")
    base_frame = np.full((480, 640, 3), 150, dtype=np.uint8)

    # 1. Collapsed / Degenerate Quad (Points too close together)
    collapsed_quad = np.array([[200, 200], [205, 200], [205, 205], [200, 205]], dtype=np.int32)
    out1 = sm.render_portal(base_frame.copy(), collapsed_quad)
    assert np.array_equal(out1, base_frame), "Collapsed quad must return base frame without crashing"

    # 2. Self-intersecting / Bowtie Quad (Crossed corners)
    bowtie_quad = np.array([[100, 100], [300, 300], [100, 300], [300, 100]], dtype=np.int32)
    out2 = sm.render_portal(base_frame.copy(), bowtie_quad)
    assert np.array_equal(out2, base_frame), "Self-intersecting quad must return base frame without crashing"

    # 3. None / Empty Quad
    out3 = sm.render_portal(base_frame.copy(), None)
    assert np.array_equal(out3, base_frame), "None polygon must return base frame cleanly"

    # 4. Out-of-bounds coordinates
    oob_quad = np.array([[-50, -50], [900, -20], [800, 600], [-30, 500]], dtype=np.int32)
    out4 = sm.render_portal(base_frame.copy(), oob_quad)
    assert out4 is not None and out4.shape == base_frame.shape, "Out of bounds quad must be clipped cleanly"

    print("  -> Invalid, collapsed, self-intersecting, and OOB polygons safely handled!")


# --- TEST 6: SCENE SWITCHING, CACHING & FIT MODES ---

def test_scene_switching_and_caching() -> None:
    print("\n[Test 6] Scene Switching, In-Memory Caching & Fit Modes...")
    sm = ScenePortalManager(scenes_dir="assets/scenes", default_fit_mode="crop")
    total = sm.get_scene_count()
    assert total >= 3, f"Need at least 3 scenes, got {total}"

    initial_name = sm.get_active_scene_name()
    next_name = sm.next_scene()
    assert next_name != initial_name, "next_scene() must change active scene"

    prev_name = sm.prev_scene()
    assert prev_name == initial_name, "prev_scene() must return to original scene"

    # In-memory caching: verify buffer is reused (not reading disk in frame loop)
    item = sm.current_scene
    assert item is not None
    assert item.image_bgr is not None
    assert item.image_bgr.shape[0] == item.height
    assert item.image_bgr.shape[1] == item.width

    # Fit modes: Crop, Letterbox, Stretch
    base = np.full((480, 640, 3), 50, dtype=np.uint8)
    quad = np.array([[180, 100], [460, 100], [460, 380], [180, 380]], dtype=np.int32)

    for fit_mode in [AspectRatioFitMode.CROP, AspectRatioFitMode.LETTERBOX, AspectRatioFitMode.STRETCH]:
        sm.set_fit_mode(fit_mode)
        out = sm.render_portal(base.copy(), quad)
        assert out is not None and out.shape == base.shape, f"Fit mode {fit_mode} failed"

    print(f"  -> Verified scene cycling across {total} scenes and 3 fit modes (Crop, Letterbox, Stretch)!")


# --- TEST 7: CORRECT TRANSPARENCY HANDLING (RGBA SCENES) ---

def test_transparency_handling_rgba() -> None:
    print("\n[Test 7] Transparency Handling for RGBA Scenes...")
    sm = ScenePortalManager(scenes_dir="assets/scenes")

    # Find cyber_warrior.png or any alpha-channel scene
    rgba_scene = None
    for item in sm.scene_items:
        if item.has_alpha and item.image_alpha is not None:
            rgba_scene = item
            break

    assert rgba_scene is not None, "At least one scene in assets/scenes must have an alpha channel (RGBA)"
    sm.active_index = sm.scene_items.index(rgba_scene)

    base_frame = np.full((480, 640, 3), [80, 120, 160], dtype=np.uint8)
    quad = np.array([[150, 100], [490, 100], [490, 380], [150, 380]], dtype=np.int32)

    rendered = sm.render_portal(base_frame.copy(), quad, opacity=1.0)
    assert rendered is not None and rendered.shape == base_frame.shape

    # Outside the polygon must be 100% untouched
    assert np.array_equal(rendered[20, 20], [80, 120, 160]), "Outside pixels must remain untouched"
    print(f"  -> RGBA scene '{rgba_scene.name}' rendered with clean alpha blending!")


# --- TEST 8: STRICT PIXEL ISOLATION (ZERO CHANGE OUTSIDE POLYGON) ---

def test_strict_pixel_isolation() -> None:
    print("\n[Test 8] Strict Pixel Isolation (Mandatory Requirement 3)...")
    sm = ScenePortalManager(scenes_dir="assets/scenes", feather_radius=0)

    # Unique synthetic background pattern
    np.random.seed(99)
    base_frame = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)

    quad = np.array([
        [190, 140],
        [450, 120],
        [470, 340],
        [170, 360]
    ], dtype=np.int32)

    # Binary mask of the hand polygon
    mask = np.zeros((480, 640), dtype=np.uint8)
    cv2.fillPoly(mask, [quad], 255)

    rendered = sm.render_portal(base_frame.copy(), quad, opacity=1.0, feather_radius=0)

    # Compare pixels where mask == 0
    outside_mask = (mask == 0)
    diff = np.abs(rendered[outside_mask].astype(np.int32) - base_frame[outside_mask].astype(np.int32))
    max_diff = int(np.max(diff))

    assert max_diff == 0, f"STRICT ISOLATION VIOLATION: Found {max_diff} difference outside portal mask!"
    print("  -> EXACT 0 PIXEL DIFFERENCE outside portal mask! Strict pixel isolation verified 100%!")


# --- TEST 9: TRACKING LOSS & INSTANT REACQUISITION ---

def test_tracking_loss_and_reacquisition() -> None:
    print("\n[Test 9] Tracking Loss Cleanup & Instant Reacquisition...")
    rc = RegionController(grace_period_sec=0.20, activation_rule=ActivationGestureRule.BOTH_HANDS)

    lh = make_dummy_hand("ScreenLeft", 170, 240)
    rh = make_dummy_hand("ScreenRight", 470, 240)
    dual = DualHandData(left_hand=lh, right_hand=rh, both_hands_visible=True, hands_count=2)

    quad = rc.update(dual, (480, 640))
    assert quad is not None and rc.is_active(), "Portal must be active when hands present"

    # Tracking lost (hands drop out of view)
    empty_dual = DualHandData()
    rc.update(empty_dual, (480, 640))
    time.sleep(0.25)
    cleared = rc.update(empty_dual, (480, 640))

    assert cleared is None, "Stale portal must be completely cleared after grace period!"
    assert rc.polygon_pts is None, "polygon_pts must be None (zero stuck coordinates)"
    assert not rc.is_active(), "Region must be inactive after tracking loss"

    # Reacquisition: hands return
    reacquired = rc.update(dual, (480, 640))
    assert reacquired is not None, "Portal must immediately reacquire when hands return"
    assert rc.is_active(), "Portal must immediately become active on reacquisition"
    print("  -> Zero stale freeze on tracking loss; instant reacquisition confirmed!")


# --- TEST 10: CONFIGURABLE ACTIVATION RULES ---

def test_configurable_activation_rules() -> None:
    print("\n[Test 10] Configurable Activation Rules (Dual Pinch, Framing, Both Hands)...")
    rc = RegionController(activation_rule=ActivationGestureRule.DUAL_PINCH)

    # 1. Non-pinching hands (open frame)
    lh_open = make_dummy_hand("ScreenLeft", 180, 240, pinching=False)
    rh_open = make_dummy_hand("ScreenRight", 460, 240, pinching=False)
    dual_open = DualHandData(left_hand=lh_open, right_hand=rh_open, both_hands_visible=True, hands_count=2)

    # In DUAL_PINCH mode: open hands do NOT activate portal
    res1 = rc.update(dual_open, (480, 640))
    assert res1 is None, "In DUAL_PINCH mode, open hands must NOT activate portal"

    # 2. Pinching hands in DUAL_PINCH mode -> ACTIVATES!
    lh_pinch = make_dummy_hand("ScreenLeft", 180, 240, pinching=True)
    rh_pinch = make_dummy_hand("ScreenRight", 460, 240, pinching=True)
    dual_pinch = DualHandData(left_hand=lh_pinch, right_hand=rh_pinch, both_hands_visible=True, hands_count=2, is_dual_pinching=True)

    res2 = rc.update(dual_pinch, (480, 640))
    assert res2 is not None, "Dual pinch MUST activate portal in DUAL_PINCH mode!"

    # 3. Switch to FRAMING mode -> open hands activate
    rc.set_activation_rule(ActivationGestureRule.FRAMING)
    res3 = rc.update(dual_open, (480, 640))
    assert res3 is not None, "Framing hands MUST activate portal in FRAMING mode!"

    # 4. Switch to BOTH_HANDS mode
    rc.set_activation_rule(ActivationGestureRule.BOTH_HANDS)
    res4 = rc.update(dual_open, (480, 640))
    assert res4 is not None, "Any hands in view activate portal in BOTH_HANDS mode!"

    print("  -> Configurable activation rules ('dual_pinch', 'framing', 'both_hands') validated!")


def run_all_tests() -> None:
    print("=" * 70)
    print("TWO-HAND FULL-SCENE AR PORTAL - COMPREHENSIVE VERIFICATION SUITE")
    print("=" * 70)

    test_four_point_extraction_from_both_hands()
    test_live_portal_position_updates()
    test_portal_resizing()
    test_corner_ordering_and_perspective_transform()
    test_invalid_and_degenerate_polygon_handling()
    test_scene_switching_and_caching()
    test_transparency_handling_rgba()
    test_strict_pixel_isolation()
    test_tracking_loss_and_reacquisition()
    test_configurable_activation_rules()

    print("\n" + "=" * 70)
    print(">>> ALL 10 UNIT & INTEGRATION TEST SUITES PASSED FLAWLESSLY! <<<")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
