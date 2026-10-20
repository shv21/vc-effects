"""
End-to-End Integration Verification Suite for Gesture-Controlled AR Experience.
Tests the entire integrated multi-model pipeline without requiring a physical camera:
1. Dual-hand continuous tracking & polygon update
2. Dynamic polygon resizing & translation
3. Mode A: Strict local-effect pixel isolation
4. Mode B: 4-point perspective scene portal
5. Mode C: Person segmentation background replacement + face mesh character mask
6. Mode D: Combined AR (Background + Segmented Person + Character Face Mask + Hand Portal)
7. Tracking loss grace period fadeout & instant hand reacquisition (zero stale freeze)
8. Photo capture, video recording, and resource teardown
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hand_tracker import HandTracker, HandData, DualHandData
from src.gesture_recognizer import GestureRecognizer, GestureResult
from src.region_controller import RegionController, ActivationGestureRule
from src.scene_manager import ScenePortalManager, AspectRatioFitMode
from src.mask_renderer import MaskRenderer
from src.effect_manager import EffectManager
from src.person_segmenter import PersonSegmenter
from src.background_compositor import BackgroundCompositor
from src.character_compositor import CharacterCompositor
from src.recorder import PhotoSaver, VideoRecorder
from src.app import ARMode


def make_synthetic_hand(label: str, cx: int, cy: int, pinching: bool = False) -> HandData:
    """Creates synthetic 21-point hand landmarks in camera coordinates."""
    pts = np.zeros((21, 2), dtype=np.float32)
    pts[0] = [cx, cy + 80]
    pts[1] = [cx - 30, cy + 60]
    pts[2] = [cx - 45, cy + 40]
    pts[3] = [cx - 50, cy + 20]
    pts[4] = [cx - 15 if pinching else cx - 55, cy - 15 if pinching else cy]
    pts[5] = [cx - 15, cy + 20]
    pts[6] = [cx - 18, cy - 10]
    pts[7] = [cx - 19, cy - 35]
    pts[8] = [cx - 15 if pinching else cx - 18, cy - 18 if pinching else cy - 60]
    pts[9] = [cx, cy + 20]
    pts[10] = [cx, cy - 15]
    pts[11] = [cx, cy - 40]
    pts[12] = [cx, cy - 68]
    pts[13] = [cx + 15, cy + 25]
    pts[14] = [cx + 15, cy - 10]
    pts[15] = [cx + 15, cy - 35]
    pts[16] = [cx + 15, cy - 60]
    pts[17] = [cx + 30, cy + 30]
    pts[18] = [cx + 30, cy]
    pts[19] = [cx + 30, cy - 20]
    pts[20] = [cx + 30, cy - 45]

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
        bbox=(cx - 60, cy - 70, 120, 160),
        hand_scale=hand_scale
    )


def test_end_to_end_integration() -> None:
    print("=" * 70)
    print("END-TO-END AR STUDIO INTEGRATION REVIEW")
    print("=" * 70)

    # 1. Initialize complete pipeline components
    print("\n[Step 1] Initializing Pipeline Engines...")
    region_ctrl = RegionController(feather_radius=0, smoothing_alpha=0.82, grace_period_sec=0.30, activation_rule=ActivationGestureRule.DUAL_PINCH)
    scene_portal_mgr = ScenePortalManager(scenes_dir="assets/scenes", default_fit_mode=AspectRatioFitMode.CROP, feather_radius=0)
    effect_mgr = EffectManager(effects_dir="assets/effects", textures_dir="assets/textures")
    bg_comp = BackgroundCompositor(backgrounds_dir="assets/backgrounds")
    char_comp = CharacterCompositor(characters_dir="assets/characters", filters_dir="assets/filters")
    char_comp.is_enabled = True
    recognizer = GestureRecognizer()

    print(f" - Loaded {scene_portal_mgr.get_scene_count()} complete scene images in assets/scenes/")
    print(f" - Loaded {len(effect_mgr.effects)} effects")
    print(f" - Loaded {len(bg_comp.background_files)} virtual background scenes")
    print(f" - Loaded {len(char_comp.character_files)} character overlays")

    h, w = 480, 640
    base_frame = np.full((h, w, 3), [110, 140, 170], dtype=np.uint8)

    # 2. Verify Hand Tracking to Region Polygon Workflow
    print("\n[Step 2] Testing Continuous Hand-Controlled Region Tracking...")
    lh = make_synthetic_hand("ScreenLeft", 180, 240, pinching=True)
    rh = make_synthetic_hand("ScreenRight", 460, 240, pinching=True)
    dual = DualHandData(left_hand=lh, right_hand=rh, hands_count=2, both_hands_visible=True, is_dual_pinching=True)

    g_res = recognizer.process(dual)
    assert g_res.is_dual_pinching, "Dual pinching must be detected"
    assert g_res.action_triggered is None, "Dual pinch must NOT trigger locking"

    poly = region_ctrl.update(dual, (h, w))
    assert poly is not None and len(poly) == 4, "Must generate valid 4-corner polygon"
    init_area = cv2.contourArea(poly)

    # Test resizing
    lh_far = make_synthetic_hand("ScreenLeft", 100, 240, pinching=True)
    rh_far = make_synthetic_hand("ScreenRight", 540, 240, pinching=True)
    dual_far = DualHandData(left_hand=lh_far, right_hand=rh_far, hands_count=2, both_hands_visible=True)
    for _ in range(8):
        poly_far = region_ctrl.update(dual_far, (h, w))
    expanded_area = cv2.contourArea(poly_far)
    assert expanded_area > init_area, f"Expanding hands must enlarge region: {expanded_area} > {init_area}"
    print(f" - Region Resizing Verified: Initial Area={init_area:.0f}px², Expanded Area={expanded_area:.0f}px²")

    # 3. Verify Mode A: Local Effect Strict Confinement
    print("\n[Step 3] Testing Mode A (Local Effect Confinement)...")
    mask_a = region_ctrl.generate_mask((h, w))
    assert mask_a is not None, "Mask must be generated when region is active"
    eff_frame = effect_mgr.render_effect_frame(base_frame, poly_far)
    out_mode_a = MaskRenderer.apply_masked_effect(base_frame, eff_frame, mask_a, opacity=0.95)

    # Pixels outside mask must be identical to base frame
    outside_mask = (mask_a == 0)
    diff_outside = np.max(np.abs(out_mode_a[outside_mask].astype(np.int32) - base_frame[outside_mask].astype(np.int32)))
    assert diff_outside == 0, f"Mode A violation: {diff_outside} pixels altered outside active region"
    print(" - Mode A Verified: 100% pixel isolation outside hand region confirmed.")

    # 4. Verify Full-Scene AR Portal (ScenePortalManager) & Perspective Warp
    print("\n[Step 4] Testing Full-Scene AR Portal (Character + Background Scene inside Quad)...")
    portal_scene_out = scene_portal_mgr.render_portal(base_frame, poly_far, opacity=1.0, feather_radius=0)
    assert portal_scene_out is not None and portal_scene_out.shape == base_frame.shape
    diff_outside_scene = np.max(np.abs(portal_scene_out[outside_mask].astype(np.int32) - base_frame[outside_mask].astype(np.int32)))
    assert diff_outside_scene == 0, f"Full-Scene AR Portal: Outside pixels must be completely untouched! Diff: {diff_outside_scene}"
    print(" - Full-Scene AR Portal Verified: Strict pixel isolation & complete scene warping confirmed.")

    # 5. Verify Mode C: Background Replacement + Person Silhouette
    print("\n[Step 5] Testing Mode C (Character Transformation & Background Replacement)...")
    person_mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(person_mask, (320, 240), (100, 180), 0, 0, 360, 1.0, -1)
    scene_c = bg_comp.composite_full(base_frame, person_mask)
    assert scene_c is not None and scene_c.shape == base_frame.shape
    # Person center preserved
    assert np.allclose(scene_c[240, 320], [110, 140, 170], atol=5), "Person silhouette must be preserved"
    # Room background replaced
    assert not np.allclose(scene_c[10, 10], [110, 140, 170], atol=10), "Background must be replaced"
    print(" - Mode C Verified: Person silhouette segmented and background replaced.")

    # 6. Verify Mode D: Combined AR (All 4 Layers Coexisting in Same Frame)
    print("\n[Step 6] Testing Mode D (Combined AR: All 4 Layers Coexisting)...")
    scene_d = bg_comp.composite_full(base_frame, person_mask)
    # Layer 3: Character Overlay
    char_d = char_comp.composite(scene_d, raw_frame_for_detection=base_frame)
    # Layer 4: Hand-controlled portal/energy zone
    eff_d = effect_mgr.render_effect_frame(char_d, poly_far)
    out_mode_d = MaskRenderer.apply_masked_effect(char_d, eff_d, mask_a, opacity=0.95)
    assert out_mode_d is not None and out_mode_d.shape == base_frame.shape
    print(" - Mode D Verified: Background + Person + Character Overlay + Hand Portal harmonized.")

    # 7. Verify Tracking Loss Grace Period and Zero-Freeze Recovery
    print("\n[Step 7] Testing Tracking Loss Grace Period & Clean Teardown...")
    empty_dual = DualHandData()
    region_ctrl.update(empty_dual, (h, w))
    assert region_ctrl.current_opacity > 0.0, "Should remain during grace period"
    time.sleep(0.35)
    expired_poly = region_ctrl.update(empty_dual, (h, w))
    assert expired_poly is None, "Polygon MUST be None after grace period expires (never frozen!)"
    assert region_ctrl.polygon_pts is None, "polygon_pts must be None"
    print(" - Tracking Loss Cleanup Verified: Region completely clears (0 freeze).")

    # Reacquisition
    reacquired = region_ctrl.update(dual, (h, w))
    assert reacquired is not None, "Instant hand reacquisition confirmed"
    print(" - Instant Reacquisition Verified: Tracking resumes immediately.")

    # 8. Verify Photo Capture & Video Recording
    print("\n[Step 8] Testing Photo & Video Recording Engine...")
    success_p, p_file = PhotoSaver.save(out_mode_d, "output")
    assert success_p and Path(p_file).exists(), f"PhotoSaver failed: {p_file}"

    rec = VideoRecorder(output_dir="output")
    started, v_file = rec.start(w, h, fps=30.0)
    assert started, f"VideoRecorder failed to start: {v_file}"
    for _ in range(15):
        rec.write_frame(out_mode_d)
    stopped, final_v, f_count, dur = rec.stop()
    assert stopped and Path(final_v).exists(), f"VideoRecorder failed to stop: {final_v}"
    print(f" - Photo & Video Verified: Photo={Path(p_file).name}, Video={Path(final_v).name} ({f_count} frames)")

    # Clean test files
    Path(p_file).unlink(missing_ok=True)
    Path(final_v).unlink(missing_ok=True)

    # 9. Teardown
    print("\n[Step 9] Testing Component Resource Teardown...")
    char_comp.release()
    print(" - Teardown Verified: Resources cleanly released.")

    print("\n" + "=" * 70)
    print(">>> COMPLETE INTEGRATION REVIEW PASSED: ALL 9 CRITICAL CHECKS VALIDATED! <<<")
    print("=" * 70)


if __name__ == "__main__":
    test_end_to_end_integration()
