"""
Overlay renderer module for transparent PNG image compositing on face landmarks.
Handles scaling, orientation alignment, alpha blending, boundary clipping,
and opacity control without edge artifacts.
"""

import math
import logging
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np

from .face_tracker import FaceData

logger = logging.getLogger(__name__)


class OverlayRenderer:
    """Renders transparent PNG overlays aligned with facial landmarks."""

    @staticmethod
    def rotate_image_with_alpha(
        image_bgra: np.ndarray,
        angle_deg: float
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Rotates a 4-channel BGRA image by angle_deg without clipping corners.
        Returns (rotated_image, (new_center_x, new_center_y)).
        """
        h, w = image_bgra.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        # Compute rotation matrix
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

        # Compute new bounding dimensions so corners aren't cut off
        cos_val = abs(M[0, 0])
        sin_val = abs(M[0, 1])
        new_w = int(math.ceil((h * sin_val) + (w * cos_val)))
        new_h = int(math.ceil((h * cos_val) + (w * sin_val)))

        # Adjust the translation component of the matrix
        M[0, 2] += (new_w / 2.0) - cx
        M[1, 2] += (new_h / 2.0) - cy

        # Warp affine with transparent border (0, 0, 0, 0)
        rotated = cv2.warpAffine(
            image_bgra,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        return rotated, (new_w // 2, new_h // 2)

    @staticmethod
    def render_overlay(
        frame_bgr: np.ndarray,
        overlay_bgra: np.ndarray,
        face_data: FaceData,
        config: Dict[str, Any]
    ) -> np.ndarray:
        """
        Composites a transparent PNG overlay onto frame_bgr based on face_data and config.

        Config fields:
          - anchor: 'eyes', 'full_face', 'forehead', 'nose', 'chin', 'mouth'
          - scale_x: float scale multiplier
          - scale_y: float scale multiplier (optional, defaults to scale_x)
          - offset_x: float offset multiplier (relative to face size / eye distance)
          - offset_y: float offset multiplier (relative to face size / eye distance)
          - rotation_offset: additional degrees to rotate
          - opacity: float (0.0 to 1.0)
          - mirror: bool
          - coverage_type: 'eyes', 'full_face', 'forehead', etc.
        """
        if not face_data.detected or overlay_bgra is None or overlay_bgra.size == 0:
            return frame_bgr

        frame_h, frame_w = frame_bgr.shape[:2]
        img_h, img_w = overlay_bgra.shape[:2]

        if img_h == 0 or img_w == 0:
            return frame_bgr

        # Ensure overlay has 4 channels (BGRA)
        if overlay_bgra.shape[2] == 3:
            alpha = np.full((img_h, img_w, 1), 255, dtype=np.uint8)
            overlay_bgra = np.concatenate([overlay_bgra, alpha], axis=2)

        # Parse config with sensible fallbacks
        anchor_name = config.get("anchor", "eyes").lower()
        scale_x = float(config.get("scale_x", 1.0))
        scale_y = float(config.get("scale_y", scale_x))
        offset_x_ratio = float(config.get("offset_x", 0.0))
        offset_y_ratio = float(config.get("offset_y", 0.0))
        rotation_offset = float(config.get("rotation_offset", 0.0))
        opacity = float(np.clip(config.get("opacity", 1.0), 0.0, 1.0))
        mirror = bool(config.get("mirror", False))
        coverage_type = config.get("coverage_type", anchor_name).lower()

        # Determine reference anchor position and base reference dimension
        if anchor_name == "eyes":
            anchor_pt = face_data.eyes_midpoint
            ref_dim = face_data.eye_distance if face_data.eye_distance > 5 else 60.0
            base_target_w = ref_dim * 2.3 * scale_x
        elif anchor_name == "full_face":
            anchor_pt = face_data.center
            ref_dim = face_data.face_width if face_data.face_width > 10 else 150.0
            base_target_w = ref_dim * 1.35 * scale_x
        elif anchor_name == "forehead":
            anchor_pt = face_data.forehead
            ref_dim = face_data.face_width if face_data.face_width > 10 else 150.0
            base_target_w = ref_dim * 1.15 * scale_x
        elif anchor_name == "nose":
            anchor_pt = face_data.nose_tip
            ref_dim = face_data.eye_distance if face_data.eye_distance > 5 else 60.0
            base_target_w = ref_dim * 1.2 * scale_x
        elif anchor_name == "chin":
            anchor_pt = face_data.chin
            ref_dim = face_data.eye_distance if face_data.eye_distance > 5 else 60.0
            base_target_w = ref_dim * 1.5 * scale_x
        elif anchor_name == "mouth":
            anchor_pt = face_data.mouth_center
            ref_dim = face_data.eye_distance if face_data.eye_distance > 5 else 60.0
            base_target_w = ref_dim * 1.4 * scale_x
        else:
            anchor_pt = face_data.center
            ref_dim = face_data.face_width if face_data.face_width > 10 else 150.0
            base_target_w = ref_dim * scale_x

        # Calculate target dimensions while preserving aspect ratio scaled by scale_y/scale_x
        target_w = int(max(4, round(base_target_w)))
        aspect_ratio = img_h / float(img_w)
        target_h = int(max(4, round(target_w * aspect_ratio * (scale_y / scale_x))))

        # Resize overlay image
        interp = cv2.INTER_AREA if (target_w < img_w and target_h < img_h) else cv2.INTER_LINEAR
        resized_overlay = cv2.resize(overlay_bgra, (target_w, target_h), interpolation=interp)

        # Mirror horizontally if requested
        if mirror:
            resized_overlay = cv2.flip(resized_overlay, 1)

        # Calculate rotation: OpenCV rotation is counter-clockwise for positive angles
        # face_data.roll_deg is clockwise positive from image coordinates, so we invert
        total_angle = -face_data.roll_deg + rotation_offset

        # Rotate overlay with alpha channel
        rotated_overlay, (rot_cx, rot_cy) = OverlayRenderer.rotate_image_with_alpha(
            resized_overlay, total_angle
        )
        rot_h, rot_w = rotated_overlay.shape[:2]

        # Calculate rotated offsets in head coordinate space
        # offset_x_ratio and offset_y_ratio are scaled relative to ref_dim
        dx_local = offset_x_ratio * ref_dim
        dy_local = offset_y_ratio * ref_dim

        # Rotate local offset into screen space using face roll angle
        rad = math.radians(face_data.roll_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        dx_screen = dx_local * cos_a - dy_local * sin_a
        dy_screen = dx_local * sin_a + dy_local * cos_a

        # Screen coordinates of overlay center
        center_x = int(round(anchor_pt[0] + dx_screen))
        center_y = int(round(anchor_pt[1] + dy_screen))

        # Target box on canvas
        top_left_x = center_x - rot_cx
        top_left_y = center_y - rot_cy
        bottom_right_x = top_left_x + rot_w
        bottom_right_y = top_left_y + rot_h

        # Compute intersection with frame boundary (safe clipping)
        frame_x1 = max(0, top_left_x)
        frame_y1 = max(0, top_left_y)
        frame_x2 = min(frame_w, bottom_right_x)
        frame_y2 = min(frame_h, bottom_right_y)

        # Completely out of screen bounds
        if frame_x1 >= frame_x2 or frame_y1 >= frame_y2:
            return frame_bgr

        # Compute corresponding source overlay sub-rectangle
        overlay_x1 = frame_x1 - top_left_x
        overlay_y1 = frame_y1 - top_left_y
        overlay_x2 = overlay_x1 + (frame_x2 - frame_x1)
        overlay_y2 = overlay_y1 + (frame_y2 - frame_y1)

        # Extract regions
        target_roi = frame_bgr[frame_y1:frame_y2, frame_x1:frame_x2]
        overlay_crop = rotated_overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

        # Alpha blending
        overlay_bgr = overlay_crop[:, :, :3]
        overlay_alpha = (overlay_crop[:, :, 3].astype(np.float32) / 255.0) * opacity

        # Expand alpha to 3 channels for vector multiplication
        alpha_3d = overlay_alpha[:, :, np.newaxis]

        # Blending formula: Blended = alpha * Overlay + (1 - alpha) * Frame
        blended_roi = (
            alpha_3d * overlay_bgr.astype(np.float32) +
            (1.0 - alpha_3d) * target_roi.astype(np.float32)
        )

        frame_bgr[frame_y1:frame_y2, frame_x1:frame_x2] = np.clip(blended_roi, 0, 255).astype(np.uint8)
        return frame_bgr
