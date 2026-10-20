"""
Effect manager module for hand-controlled AR transformations.
Provides both high-performance procedural effects and image/texture overlays
warped into the hand-defined polygon with perspective transformation.
"""

import os
import time
import math
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EffectItem:
    """Represents an active effect option."""
    effect_id: str
    display_name: str
    effect_type: str  # 'procedural' or 'image'
    category: str
    description: str = ""
    image_bgra: Optional[np.ndarray] = None
    generator_func: Optional[Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]] = None


class EffectManager:
    """Discovers, caches, generates, and perspective-warps effects for the hand region."""

    def __init__(
        self,
        effects_dir: str = "assets/effects",
        textures_dir: str = "assets/textures"
    ) -> None:
        self.effects_dir = Path(effects_dir)
        self.textures_dir = Path(textures_dir)

        self.effects: List[EffectItem] = []
        self.active_index: int = 0

        self.reload()

    def reload(self) -> None:
        """Discovers procedural generators and disk image assets."""
        self.effects.clear()

        # 1. Register Procedural Effects
        self.effects.extend([
            EffectItem(
                effect_id="proc_magic_portal",
                display_name="Magic Energy Portal",
                effect_type="procedural",
                category="Energy",
                description="Swirling golden plasma portal with animated runic rings",
                generator_func=self._gen_magic_portal
            ),
            EffectItem(
                effect_id="proc_spider_web",
                display_name="Spider-Web Zone",
                effect_type="procedural",
                category="Superhero",
                description="Intricate spider-web lattice geometry warped inside hand bounds",
                generator_func=self._gen_spider_web
            ),
            EffectItem(
                effect_id="proc_cyber_matrix",
                display_name="Cyber Matrix Grid",
                effect_type="procedural",
                category="Sci-Fi",
                description="Digital neon HUD telemetry grid and animated scanlines",
                generator_func=self._gen_cyber_matrix
            ),
            EffectItem(
                effect_id="proc_cosmic_nebula",
                display_name="Cosmic Nebula",
                effect_type="procedural",
                category="Space",
                description="Swirling deep-space starlight and violet nebula cloud",
                generator_func=self._gen_cosmic_nebula
            ),
            EffectItem(
                effect_id="proc_thermal_vision",
                display_name="Thermal Heat Signature",
                effect_type="procedural",
                category="Vision",
                description="Infrared heat signature spectrum inside hand boundary",
                generator_func=self._gen_thermal_vision
            ),
            EffectItem(
                effect_id="proc_flame_aura",
                display_name="Fire & Plasma Aura",
                effect_type="procedural",
                category="Energy",
                description="Dynamic turbulent flame and plasma vortex",
                generator_func=self._gen_flame_aura
            ),
            EffectItem(
                effect_id="proc_comic_halftone",
                display_name="Comic Pop-Art Halftone",
                effect_type="procedural",
                category="Style",
                description="Pop-art comic printing dot screen and ink edges",
                generator_func=self._gen_comic_halftone
            )
        ])

        # 2. Register Image Assets from assets/effects/ and assets/textures/
        for folder, cat in [(self.effects_dir, "Custom Effects"), (self.textures_dir, "Textures")]:
            if folder.exists():
                for p in sorted(folder.iterdir()):
                    if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            dname = p.stem.replace("_", " ").title()
                            self.effects.append(
                                EffectItem(
                                    effect_id=f"img_{p.name}",
                                    display_name=dname,
                                    effect_type="image",
                                    category=cat,
                                    description=f"Warped texture: {p.name}",
                                    image_bgra=img
                                )
                            )

        if self.active_index >= len(self.effects):
            self.active_index = 0

        logger.info(f"Loaded {len(self.effects)} effects into catalog.")

    def get_active_effect(self) -> EffectItem:
        if 0 <= self.active_index < len(self.effects):
            return self.effects[self.active_index]
        return self.effects[0]

    def next_effect(self) -> EffectItem:
        """Switches to the next effect (triggered by right swipe)."""
        if self.effects:
            self.active_index = (self.active_index + 1) % len(self.effects)
            return self.get_active_effect()
        return None

    def prev_effect(self) -> EffectItem:
        """Switches to the previous effect (triggered by left swipe)."""
        if self.effects:
            self.active_index = (self.active_index - 1) % len(self.effects)
            return self.get_active_effect()
        return None

    def render_effect_frame(
        self,
        base_frame_bgr: np.ndarray,
        polygon_pts: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Renders the active effect transformed to match the quadrilateral polygon.
        Returns a full-sized BGR image containing the effect to be masked.
        """
        active = self.get_active_effect()

        if active.effect_type == "procedural" and active.generator_func:
            return active.generator_func(base_frame_bgr, polygon_pts)

        elif active.effect_type == "image" and active.image_bgra is not None:
            return self._warp_image_to_polygon(base_frame_bgr, active.image_bgra, polygon_pts)

        return base_frame_bgr

    def _warp_image_to_polygon(
        self,
        base_frame_bgr: np.ndarray,
        image_bgra: np.ndarray,
        polygon_pts: Optional[np.ndarray]
    ) -> np.ndarray:
        """Performs 4-point perspective warp mapping the image corners to the hand quadrilateral."""
        h, w = base_frame_bgr.shape[:2]

        if polygon_pts is None or len(polygon_pts) != 4:
            # Fallback to direct resize if polygon is not a 4-point quadrilateral
            resized = cv2.resize(image_bgra, (w, h))
            if resized.shape[2] == 4:
                return cv2.cvtColor(resized, cv2.COLOR_BGRA2BGR)
            return resized

        img_h, img_w = image_bgra.shape[:2]
        src_corners = np.array([
            [0, 0],
            [img_w - 1, 0],
            [img_w - 1, img_h - 1],
            [0, img_h - 1]
        ], dtype=np.float32)

        dst_corners = polygon_pts.astype(np.float32)

        # Compute perspective transform matrix
        try:
            M = cv2.getPerspectiveTransform(src_corners, dst_corners)
            warped = cv2.warpPerspective(
                image_bgra,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )

            # If image has alpha channel, blend with base frame
            if warped.shape[2] == 4:
                alpha = (warped[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]
                rgb = warped[:, :, :3]
                blended = alpha * rgb.astype(np.float32) + (1.0 - alpha) * base_frame_bgr.astype(np.float32)
                return np.clip(blended, 0, 255).astype(np.uint8)
            return warped
        except Exception as e:
            logger.warning(f"Perspective warp failed: {e}")
            return base_frame_bgr

    # --- PROCEDURAL SHADER-LIKE EFFECT GENERATORS ---

    def _gen_magic_portal(self, base_frame: np.ndarray, polygon_pts: Optional[np.ndarray]) -> np.ndarray:
        """Animated swirling golden plasma vortex with runic rings."""
        h, w = base_frame.shape[:2]
        now = time.time()
        cx, cy = (w // 2, h // 2) if polygon_pts is None else (int(np.mean(polygon_pts[:, 0])), int(np.mean(polygon_pts[:, 1])))

        effect = np.zeros((h, w, 3), dtype=np.uint8)

        # Concentric glowing rings
        for r_step in range(30, 260, 25):
            radius = int(r_step + 10 * math.sin(now * 4.0 + r_step * 0.1))
            col = (
                int(20 + 30 * math.sin(now + r_step)),
                int(160 + 90 * math.cos(now * 2.0 + r_step)),
                int(255)
            )
            cv2.circle(effect, (cx, cy), max(2, radius), col, 3, cv2.LINE_AA)

        # Radial rotating energy spokes
        num_spokes = 16
        rot_offset = now * 1.5
        for i in range(num_spokes):
            ang = rot_offset + i * (2.0 * math.pi / num_spokes)
            ex = int(cx + 240 * math.cos(ang))
            ey = int(cy + 240 * math.sin(ang))
            cv2.line(effect, (cx, cy), (ex, ey), (50, 200, 255), 2, cv2.LINE_AA)

        # Central glowing plasma core
        cv2.circle(effect, (cx, cy), 35, (220, 255, 255), -1, cv2.LINE_AA)
        return cv2.addWeighted(base_frame, 0.25, effect, 0.75, 0)

    def _gen_spider_web(self, base_frame: np.ndarray, polygon_pts: Optional[np.ndarray]) -> np.ndarray:
        """Geometric spider-web lattice anchored to hand bounds."""
        h, w = base_frame.shape[:2]
        cx, cy = (w // 2, h // 2) if polygon_pts is None else (int(np.mean(polygon_pts[:, 0])), int(np.mean(polygon_pts[:, 1])))

        overlay = base_frame.copy()
        max_r = 240 if polygon_pts is None else int(np.max(np.linalg.norm(polygon_pts - [cx, cy], axis=1)))
        max_r = max(50, min(max_r, 450))

        # Radial spokes
        num_spokes = 14
        angles = [i * (2.0 * math.pi / num_spokes) for i in range(num_spokes)]
        for a in angles:
            ex = int(cx + max_r * math.cos(a))
            ey = int(cy + max_r * math.sin(a))
            cv2.line(overlay, (cx, cy), (ex, ey), (255, 255, 255), 2, cv2.LINE_AA)

        # Concentric sagging web rings
        for r in range(40, max_r, 45):
            pts = [(int(cx + r * math.cos(a)), int(cy + r * math.sin(a))) for a in angles]
            for i in range(num_spokes):
                p1 = pts[i]
                p2 = pts[(i + 1) % num_spokes]
                mid_a = (angles[i] + angles[(i + 1) % num_spokes]) / 2.0
                sag_x = int(cx + r * 0.92 * math.cos(mid_a))
                sag_y = int(cy + r * 0.92 * math.sin(mid_a))
                cv2.line(overlay, p1, (sag_x, sag_y), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(overlay, (sag_x, sag_y), p2, (255, 255, 255), 2, cv2.LINE_AA)

        # Tint background reddish-blue
        tint = np.zeros_like(base_frame)
        tint[:] = [120, 20, 200]
        return cv2.addWeighted(overlay, 0.70, tint, 0.30, 0)

    def _gen_cyber_matrix(self, base_frame: np.ndarray, polygon_pts: Optional[np.ndarray]) -> np.ndarray:
        """Digital neon HUD telemetry grid and green/cyan terminal scanlines."""
        h, w = base_frame.shape[:2]
        now = time.time()
        grid = np.zeros_like(base_frame)

        # Horizontal scanlines
        scan_offset = int((now * 80) % 20)
        for y in range(scan_offset, h, 20):
            cv2.line(grid, (0, y), (w, y), (0, 160, 220), 1)

        # Vertical matrix lines
        for x in range(0, w, 28):
            cv2.line(grid, (x, 0), (x, h), (0, 100, 180), 1)

        # Glowing target reticle in center
        cx, cy = (w // 2, h // 2) if polygon_pts is None else (int(np.mean(polygon_pts[:, 0])), int(np.mean(polygon_pts[:, 1])))
        cv2.circle(grid, (cx, cy), 60, (0, 255, 240), 2, cv2.LINE_AA)
        cv2.circle(grid, (cx, cy), 12, (0, 255, 240), -1, cv2.LINE_AA)

        # Telemetry text
        cv2.putText(grid, "SYS_SCAN // ACTIVE", (cx - 70, cy - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 240), 1, cv2.LINE_AA)

        return cv2.addWeighted(base_frame, 0.35, grid, 0.65, 0)

    def _gen_cosmic_nebula(self, base_frame: np.ndarray, polygon_pts: Optional[np.ndarray]) -> np.ndarray:
        """Swirling deep-space starlight and violet nebula cloud."""
        h, w = base_frame.shape[:2]
        nebula = np.zeros((h, w, 3), dtype=np.uint8)
        now = time.time()

        cx, cy = (w // 2, h // 2) if polygon_pts is None else (int(np.mean(polygon_pts[:, 0])), int(np.mean(polygon_pts[:, 1])))

        # Multi-layer cosmic spiral
        for arm in range(3):
            base_a = arm * (2.0 * math.pi / 3.0) + now * 0.8
            for step in range(20, 220, 8):
                a = base_a + step * 0.04
                sx = int(cx + step * 1.2 * math.cos(a))
                sy = int(cy + step * 1.2 * math.sin(a))
                if 0 <= sx < w and 0 <= sy < h:
                    col = (255, 80, 180) if arm % 2 == 0 else (180, 255, 0)
                    cv2.circle(nebula, (sx, sy), int(step / 30) + 2, col, -1)

        return cv2.addWeighted(base_frame, 0.35, nebula, 0.65, 0)

    def _gen_thermal_vision(self, base_frame: np.ndarray, polygon_pts: Optional[np.ndarray]) -> np.ndarray:
        """Infrared heat signature spectrum inside hand boundary."""
        gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        return cv2.addWeighted(base_frame, 0.20, thermal, 0.80, 0)

    def _gen_flame_aura(self, base_frame: np.ndarray, polygon_pts: Optional[np.ndarray]) -> np.ndarray:
        """Dynamic turbulent flame and plasma vortex."""
        gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
        fire = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
        return cv2.addWeighted(base_frame, 0.25, fire, 0.75, 0)

    def _gen_comic_halftone(self, base_frame: np.ndarray, polygon_pts: Optional[np.ndarray]) -> np.ndarray:
        """Pop-art comic printing dot screen and ink edges."""
        gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Color quantization via bilateral filter
        color = base_frame
        for _ in range(2):
            color = cv2.bilateralFilter(color, 7, 75, 75)

        return cv2.bitwise_and(color, edges_bgr)
