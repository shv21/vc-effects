"""
Full-Scene AR Portal Engine & Scene Asset Manager.
Manages user-selected scene images containing BOTH character and background,
warping the entire complete scene into a hand-controlled 4-point quadrilateral portal.

Guarantees:
- Strict pixel isolation outside the hand polygon
- Perspective warping with OpenCV
- Configurable aspect-ratio fit modes ('crop', 'letterbox', 'stretch')
- Transparent PNG (RGBA) support
- In-memory caching (zero disk reads inside the frame loop)
- Degenerate & self-intersecting polygon protection (never crashes)
"""

import os
import math
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class AspectRatioFitMode:
    CROP = "crop"            # Preserve aspect ratio by cropping margins (fills portal, character visible)
    LETTERBOX = "letterbox"  # Preserve aspect ratio by letterboxing (full scene visible)
    STRETCH = "stretch"      # Direct quad warp stretching to match corners exactly

    ALL = [CROP, LETTERBOX, STRETCH]


@dataclass
class SceneItem:
    """Metadata and cached pixel buffer for a loaded scene."""
    name: str
    file_path: Path
    image_bgr: np.ndarray             # (H, W, 3) uint8
    image_alpha: Optional[np.ndarray] # (H, W) uint8 or None
    width: int
    height: int
    aspect_ratio: float
    has_alpha: bool


class ScenePortalManager:
    """Manages scene discovery, image caching, perspective warping, and strict masked compositing."""

    def __init__(
        self,
        scenes_dir: str = "assets/scenes",
        default_fit_mode: str = AspectRatioFitMode.CROP,
        feather_radius: int = 0
    ) -> None:
        self.scenes_dir = Path(scenes_dir)
        self.fit_mode = default_fit_mode if default_fit_mode in AspectRatioFitMode.ALL else AspectRatioFitMode.CROP
        self.feather_radius = max(0, feather_radius)

        self.scene_items: List[SceneItem] = []
        self.active_index: int = 0
        self.last_load_error: Optional[str] = None

        self.reload()

    def reload(self) -> int:
        """Discovers and caches all valid scene images from assets/scenes/."""
        self.scene_items.clear()
        self.last_load_error = None

        if not self.scenes_dir.exists():
            try:
                self.scenes_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.last_load_error = f"Cannot create scenes directory: {e}"
                logger.error(self.last_load_error)
                return 0

        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        discovered_paths = sorted([
            p for p in self.scenes_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        ])

        for path in discovered_paths:
            item = self._load_scene_from_disk(path)
            if item is not None:
                self.scene_items.append(item)

        if self.active_index >= len(self.scene_items):
            self.active_index = 0

        logger.info(f"Loaded {len(self.scene_items)} complete scenes from {self.scenes_dir}")
        return len(self.scene_items)

    def _load_scene_from_disk(self, path: Path) -> Optional[SceneItem]:
        """Loads and caches an image with full bit depth, handling RGB and RGBA."""
        try:
            # Load with IMREAD_UNCHANGED to preserve alpha channel if PNG
            raw_img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw_img is None or raw_img.size == 0:
                logger.warning(f"Failed to read image file: {path}")
                return None

            h, w = raw_img.shape[:2]
            if h < 10 or w < 10:
                logger.warning(f"Image too small ({w}x{h}): {path}")
                return None

            has_alpha = False
            alpha_channel = None

            if len(raw_img.shape) == 3 and raw_img.shape[2] == 4:
                # BGRA image
                bgr = raw_img[:, :, :3]
                alpha_channel = raw_img[:, :, 3]
                has_alpha = True
            elif len(raw_img.shape) == 3 and raw_img.shape[2] == 3:
                bgr = raw_img
            elif len(raw_img.shape) == 2:
                # Grayscale to BGR
                bgr = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
            else:
                logger.warning(f"Unsupported image shape {raw_img.shape}: {path}")
                return None

            name = path.stem.replace("_", " ").title()
            ar = float(w) / float(max(1, h))

            return SceneItem(
                name=name,
                file_path=path,
                image_bgr=bgr,
                image_alpha=alpha_channel,
                width=w,
                height=h,
                aspect_ratio=ar,
                has_alpha=has_alpha
            )
        except Exception as e:
            logger.error(f"Error loading scene {path}: {e}")
            return None

    def load_custom_scene(self, file_path: str) -> bool:
        """Loads a user-specified image file from anywhere on disk and sets it active."""
        p = Path(file_path).resolve()
        if not p.exists() or not p.is_file():
            self.last_load_error = f"File not found: {p}"
            return False

        item = self._load_scene_from_disk(p)
        if item is None:
            self.last_load_error = f"Could not decode scene image: {p.name}"
            return False

        # Check if already in list
        for idx, existing in enumerate(self.scene_items):
            if existing.file_path == p:
                self.active_index = idx
                return True

        self.scene_items.append(item)
        self.active_index = len(self.scene_items) - 1
        logger.info(f"Custom scene activated: {item.name}")
        return True

    @property
    def current_scene(self) -> Optional[SceneItem]:
        if 0 <= self.active_index < len(self.scene_items):
            return self.scene_items[self.active_index]
        return None

    def get_active_scene_name(self) -> str:
        sc = self.current_scene
        return sc.name if sc else "No Scene Loaded"

    def get_scene_count(self) -> int:
        return len(self.scene_items)

    def next_scene(self) -> str:
        if self.scene_items:
            self.active_index = (self.active_index + 1) % len(self.scene_items)
            return self.get_active_scene_name()
        return "No Scenes"

    def prev_scene(self) -> str:
        if self.scene_items:
            self.active_index = (self.active_index - 1) % len(self.scene_items)
            return self.get_active_scene_name()
        return "No Scenes"

    def set_fit_mode(self, mode: str) -> None:
        if mode in AspectRatioFitMode.ALL:
            self.fit_mode = mode

    def cycle_fit_mode(self) -> str:
        idx = AspectRatioFitMode.ALL.index(self.fit_mode)
        self.fit_mode = AspectRatioFitMode.ALL[(idx + 1) % len(AspectRatioFitMode.ALL)]
        return self.fit_mode.upper()

    # --- PERSPECTIVE WARPING & STRICT MASKED COMPOSITING ---

    @staticmethod
    def is_valid_quadrilateral(
        pts: np.ndarray,
        frame_shape: Tuple[int, int],
        min_edge_px: float = 16.0,
        min_area_px: float = 350.0
    ) -> bool:
        """
        Validates quadrilateral geometry:
        - Exactly 4 vertices
        - Minimum edge lengths
        - Non-degenerate area
        - Strict convexity without self-intersection (bowtie distortion)
        """
        if pts is None or len(pts) != 4:
            return False

        pts_f = pts.astype(np.float32)

        # 1. Edge length checks
        edges = [
            pts_f[1] - pts_f[0],  # Top: TL -> TR
            pts_f[2] - pts_f[1],  # Right: TR -> BR
            pts_f[3] - pts_f[2],  # Bottom: BR -> BL
            pts_f[0] - pts_f[3]   # Left: BL -> TL
        ]
        edge_lens = [float(np.hypot(e[0], e[1])) for e in edges]
        if min(edge_lens) < min_edge_px:
            return False

        # 2. Minimum Area Check
        area = float(cv2.contourArea(pts_f))
        if area < min_area_px:
            return False

        # 3. Strict Convexity & Non-Self-Intersection
        # Cross products of consecutive edges must all have the same sign
        cross_products = []
        for i in range(4):
            e_curr = edges[i]
            e_next = edges[(i + 1) % 4]
            cp = float(e_curr[0] * e_next[1] - e_curr[1] * e_next[0])
            cross_products.append(cp)

        # All cross products must be strictly positive (clockwise) or negative (counter-clockwise)
        all_pos = all(cp > 1e-3 for cp in cross_products)
        all_neg = all(cp < -1e-3 for cp in cross_products)
        if not (all_pos or all_neg):
            # Polygon is self-intersecting (bowtie) or concave!
            return False

        return True

    def calculate_source_quadrilateral(
        self,
        scene: SceneItem,
        dst_quad: np.ndarray
    ) -> np.ndarray:
        """
        Calculates source quadrilateral corners according to configured fit mode:
        - CROP: Crop scene margins to preserve aspect ratio (fills portal cleanly)
        - LETTERBOX: Pad scene margins with letterboxing (entire image visible)
        - STRETCH: Direct 1-to-1 stretch mapping
        Returns (4, 2) float32 array: [TL, TR, BR, BL]
        """
        sw = float(scene.width)
        sh = float(scene.height)
        src_ar = scene.aspect_ratio

        if self.fit_mode == AspectRatioFitMode.STRETCH:
            return np.array([
                [0.0, 0.0],
                [sw - 1.0, 0.0],
                [sw - 1.0, sh - 1.0],
                [0.0, sh - 1.0]
            ], dtype=np.float32)

        # Calculate average width and height of destination quadrilateral
        dst_w = 0.5 * (
            np.hypot(dst_quad[1, 0] - dst_quad[0, 0], dst_quad[1, 1] - dst_quad[0, 1]) +
            np.hypot(dst_quad[2, 0] - dst_quad[3, 0], dst_quad[2, 1] - dst_quad[3, 1])
        )
        dst_h = 0.5 * (
            np.hypot(dst_quad[3, 0] - dst_quad[0, 0], dst_quad[3, 1] - dst_quad[0, 1]) +
            np.hypot(dst_quad[2, 0] - dst_quad[1, 0], dst_quad[2, 1] - dst_quad[1, 1])
        )
        dst_ar = max(0.01, float(dst_w) / max(1.0, float(dst_h)))

        if self.fit_mode == AspectRatioFitMode.CROP:
            # Crop source to match dst_ar
            if src_ar > dst_ar:
                # Source is wider: crop horizontally (keep vertical center)
                crop_w = sh * dst_ar
                crop_x = (sw - crop_w) * 0.5
                return np.array([
                    [crop_x, 0.0],
                    [crop_x + crop_w - 1.0, 0.0],
                    [crop_x + crop_w - 1.0, sh - 1.0],
                    [crop_x, sh - 1.0]
                ], dtype=np.float32)
            else:
                # Source is taller: crop vertically (keep horizontal center)
                crop_h = sw / dst_ar
                crop_y = (sh - crop_h) * 0.5
                return np.array([
                    [0.0, crop_y],
                    [sw - 1.0, crop_y],
                    [sw - 1.0, crop_y + crop_h - 1.0],
                    [0.0, crop_y + crop_h - 1.0]
                ], dtype=np.float32)

        elif self.fit_mode == AspectRatioFitMode.LETTERBOX:
            # Letterbox: expand source canvas bounds with margins
            if src_ar > dst_ar:
                # Target is taller: add vertical margins
                effective_h = sw / dst_ar
                pad_y = (effective_h - sh) * 0.5
                return np.array([
                    [0.0, -pad_y],
                    [sw - 1.0, -pad_y],
                    [sw - 1.0, sh - 1.0 + pad_y],
                    [0.0, sh - 1.0 + pad_y]
                ], dtype=np.float32)
            else:
                # Target is wider: add horizontal margins
                effective_w = sh * dst_ar
                pad_x = (effective_w - sw) * 0.5
                return np.array([
                    [-pad_x, 0.0],
                    [sw - 1.0 + pad_x, 0.0],
                    [sw - 1.0 + pad_x, sh - 1.0],
                    [-pad_x, sh - 1.0]
                ], dtype=np.float32)

        # Fallback
        return np.array([
            [0.0, 0.0],
            [sw - 1.0, 0.0],
            [sw - 1.0, sh - 1.0],
            [0.0, sh - 1.0]
        ], dtype=np.float32)

    def render_portal(
        self,
        base_frame_bgr: np.ndarray,
        polygon_pts: Optional[np.ndarray],
        opacity: float = 1.0,
        feather_radius: Optional[int] = None
    ) -> np.ndarray:
        """
        Renders the complete scene image strictly inside the 4-corner hand polygon.

        Guarantees:
        - Strict pixel isolation outside polygon when feather_radius == 0
        - Correct perspective mapping
        - Boundary clipping
        - Graceful hiding if quadrilateral is degenerate or self-intersecting
        """
        if base_frame_bgr is None or base_frame_bgr.size == 0:
            return base_frame_bgr

        if polygon_pts is None:
            return base_frame_bgr

        scene = self.current_scene
        if scene is None:
            return base_frame_bgr

        h, w = base_frame_bgr.shape[:2]
        effective_feather = self.feather_radius if feather_radius is None else max(0, feather_radius)

        # 1. Image-boundary clipping
        dst_quad = polygon_pts.astype(np.float32).copy()
        dst_quad[:, 0] = np.clip(dst_quad[:, 0], 0, w - 1)
        dst_quad[:, 1] = np.clip(dst_quad[:, 1], 0, h - 1)

        # 2. Validate quadrilateral geometry
        if not self.is_valid_quadrilateral(dst_quad, (h, w)):
            # Invalid or degenerate polygon: hide portal cleanly rather than crashing
            return base_frame_bgr

        # 3. Calculate source corners according to fit mode
        src_quad = self.calculate_source_quadrilateral(scene, dst_quad)

        # 4. Compute OpenCV Perspective Transformation Matrix
        try:
            M = cv2.getPerspectiveTransform(src_quad, dst_quad)
            # Warp scene image directly to frame dimensions
            warped_scene = cv2.warpPerspective(
                scene.image_bgr,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

            # Warp alpha channel if present
            warped_alpha = None
            if scene.has_alpha and scene.image_alpha is not None:
                warped_alpha = cv2.warpPerspective(
                    scene.image_alpha,
                    M,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
        except Exception as e:
            logger.warning(f"Perspective transform error: {e}")
            return base_frame_bgr

        # 5. Polygon Mask Creation
        poly_int = dst_quad.astype(np.int32)
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(binary_mask, [poly_int], 255)

        # 6. Optional edge feathering
        if effective_feather > 0:
            k = effective_feather if effective_feather % 2 == 1 else effective_feather + 1
            feathered = cv2.GaussianBlur(binary_mask, (k, k), 0)
            norm_mask = (feathered.astype(np.float32) / 255.0)
        else:
            # Strictly binary: 0.0 or 1.0 (exact pixel isolation)
            norm_mask = (binary_mask.astype(np.float32) / 255.0)

        # 7. Alpha blending calculation
        effective_mask = norm_mask * max(0.0, min(1.0, opacity))

        if warped_alpha is not None:
            alpha_norm = warped_alpha.astype(np.float32) / 255.0
            effective_mask = effective_mask * alpha_norm

        # 8. Composite: Out = Mask * WarpedScene + (1 - Mask) * BaseCameraFrame
        # Note: Outside polygon where effective_mask == 0.0, pixels are 100% identical to base_frame_bgr
        mask_3d = effective_mask[:, :, np.newaxis]
        composited = (
            mask_3d * warped_scene.astype(np.float32) +
            (1.0 - mask_3d) * base_frame_bgr.astype(np.float32)
        )
        return np.clip(composited, 0, 255).astype(np.uint8)
