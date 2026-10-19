"""
Procedural built-in filters for Spider-Man AR Studio.
Contains modular filter implementations that do not require external image assets:
- No Filter (original webcam feed)
- Grayscale
- High-Contrast Black & White Comic
- Cyberpunk Pink / Neon Tint
- Comic Book Cartoonizer
- Face Mesh & Facial Geometry Visualizer
- Red & Blue Superhero Comic Duotone
- Procedural Spider-Web Overlay (geometrically calculated on face landmarks)
"""

import math
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
import cv2
import numpy as np

from .face_tracker import FaceData


class BaseFilter(ABC):
    """Abstract base class for all filters."""

    def __init__(self, filter_id: str, display_name: str, description: str = "") -> None:
        self.filter_id = filter_id
        self.display_name = display_name
        self.description = description

    @abstractmethod
    def apply(self, frame_bgr: np.ndarray, face_data: FaceData) -> np.ndarray:
        """Applies the filter effect to frame_bgr and returns the modified frame."""
        pass


class NoFilter(BaseFilter):
    """Pass-through filter returning unaltered webcam feed."""

    def __init__(self) -> None:
        super().__init__(
            filter_id="builtin_none",
            display_name="No Filter",
            description="Original unfiltered camera feed."
        )

    def apply(self, frame_bgr: np.ndarray, face_data: FaceData) -> np.ndarray:
        return frame_bgr


class GrayscaleFilter(BaseFilter):
    """Converts camera feed to smooth grayscale (3-channel BGR output)."""

    def __init__(self) -> None:
        super().__init__(
            filter_id="builtin_grayscale",
            display_name="Classic Grayscale",
            description="Noir monochrome film look."
        )

    def apply(self, frame_bgr: np.ndarray, face_data: FaceData) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class BlackAndWhiteFilter(BaseFilter):
    """High-contrast graphic novel black-and-white comic ink threshold."""

    def __init__(self) -> None:
        super().__init__(
            filter_id="builtin_comic_ink",
            display_name="Black & White Comic Ink",
            description="High-contrast graphic novel comic ink effect."
        )

    def apply(self, frame_bgr: np.ndarray, face_data: FaceData) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # Apply slight blur to reduce grain then adaptive threshold
        blurred = cv2.medianBlur(gray, 3)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


class PinkTintFilter(BaseFilter):
    """Cyberpunk neon pink & magenta glow tint."""

    def __init__(self) -> None:
        super().__init__(
            filter_id="builtin_pink_tint",
            display_name="Cyberpunk Pink Tint",
            description="Futuristic neon magenta & violet tint."
        )

    def apply(self, frame_bgr: np.ndarray, face_data: FaceData) -> np.ndarray:
        # Create pink/magenta overlay in BGR: [B=180, G=20, R=255]
        h, w = frame_bgr.shape[:2]
        tint_layer = np.zeros_like(frame_bgr)
        tint_layer[:] = [180, 20, 240]

        # Blend with 35% opacity
        return cv2.addWeighted(frame_bgr, 0.72, tint_layer, 0.28, 0)


class CartoonFilter(BaseFilter):
    """Bilateral smoothing combined with adaptive edge inking for a cartoon cell-shaded aesthetic."""

    def __init__(self) -> None:
        super().__init__(
            filter_id="builtin_cartoon",
            display_name="Superhero Cartoon",
            description="Cell-shaded comic book cartoon rendering."
        )

    def apply(self, frame_bgr: np.ndarray, face_data: FaceData) -> np.ndarray:
        # 1. Edge mask using median blur + adaptive threshold
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
            gray_blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 9, 7
        )

        # 2. Color quantization / smoothing using bilateral filter
        color = frame_bgr
        for _ in range(2):
            color = cv2.bilateralFilter(color, d=7, sigmaColor=75, sigmaSpace=75)

        # 3. Combine smoothed colors with ink edges
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(color, edges_bgr)
        return cartoon


class FaceMeshFilter(BaseFilter):
    """Visualizes facial landmark geometry, eye centers, and face contours."""

    def __init__(self) -> None:
        super().__init__(
            filter_id="builtin_facemesh",
            display_name="Face Landmark Wireframe",
            description="Real-time 468-point facial mesh & tracking telemetry."
        )

    def apply(self, frame_bgr: np.ndarray, face_data: FaceData) -> np.ndarray:
        if not face_data.detected or face_data.landmarks_px is None:
            return frame_bgr

        overlay = frame_bgr.copy()
        pts = face_data.landmarks_px

        # Subtle cyan mesh points
        for pt in pts[::2]:
            cv2.circle(overlay, (int(pt[0]), int(pt[1])), 1, (255, 200, 0), -1)

        # Key anchors with glowing rings
        anchors = [
            (face_data.left_eye, (255, 120, 0), "L_EYE"),
            (face_data.right_eye, (0, 120, 255), "R_EYE"),
            (face_data.nose_tip, (0, 255, 100), "NOSE"),
            (face_data.forehead, (255, 0, 200), "FOREHEAD"),
            (face_data.chin, (255, 255, 0), "CHIN"),
            (face_data.mouth_center, (0, 255, 255), "MOUTH")
        ]

        for pt, col, label in anchors:
            cv2.circle(overlay, pt, 4, col, -1)
            cv2.circle(overlay, pt, 7, (255, 255, 255), 1)

        # Bounding box
        bx, by, bw, bh = face_data.bbox
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (0, 255, 255), 1)

        # HUD Telemetry
        text_info = f"Roll: {face_data.roll_deg:.1f} | Yaw: {face_data.yaw_deg:.1f} | Pitch: {face_data.pitch_deg:.1f}"
        cv2.putText(overlay, text_info, (bx, max(20, by - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        return cv2.addWeighted(overlay, 0.85, frame_bgr, 0.15, 0)


class RedBlueHeroFilter(BaseFilter):
    """Dual-tone red & blue superhero lighting overlay inspired by Spider-Man suit palettes."""

    def __init__(self) -> None:
        super().__init__(
            filter_id="builtin_red_blue_hero",
            display_name="Red & Blue Superhero Duotone",
            description="Dynamic red & royal blue comic superhero lighting."
        )

    def apply(self, frame_bgr: np.ndarray, face_data: FaceData) -> np.ndarray:
        h, w = frame_bgr.shape[:2]

        # Horizontal split duotone gradient
        # Left side: vibrant crimson red; Right side: electric blue
        x_coords = np.linspace(0.0, 1.0, w, dtype=np.float32)
        # Crimson: B=30, G=30, R=220 | Blue: B=220, G=120, R=20
        b_channel = (1.0 - x_coords) * 30 + x_coords * 220
        g_channel = (1.0 - x_coords) * 30 + x_coords * 80
        r_channel = (1.0 - x_coords) * 220 + x_coords * 20

        gradient = np.zeros((h, w, 3), dtype=np.uint8)
        gradient[:, :, 0] = np.tile(b_channel, (h, 1)).astype(np.uint8)
        gradient[:, :, 1] = np.tile(g_channel, (h, 1)).astype(np.uint8)
        gradient[:, :, 2] = np.tile(r_channel, (h, 1)).astype(np.uint8)

        # Convert video frame to high-contrast luminance
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray_3d = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

        # Multiply blend
        blended = (gray_3d * gradient.astype(np.float32)).astype(np.uint8)

        # Soft blend with original frame
        return cv2.addWeighted(frame_bgr, 0.35, blended, 0.65, 0)


class ProceduralWebFilter(BaseFilter):
    """Procedurally renders an intricate Spider-Man web pattern centered on the face geometry."""

    def __init__(self) -> None:
        super().__init__(
            filter_id="builtin_procedural_web",
            display_name="Procedural Spider-Web Overlay",
            description="Mathematically generated spider-web mask conforming to facial center."
        )

    def apply(self, frame_bgr: np.ndarray, face_data: FaceData) -> np.ndarray:
        if not face_data.detected:
            return frame_bgr

        overlay = frame_bgr.copy()
        cx, cy = face_data.nose_tip
        max_radius = max(face_data.face_width, face_data.face_height) * 0.7
        if max_radius < 10:
            return frame_bgr

        # Draw radial web lines
        num_spokes = 16
        roll_rad = math.radians(face_data.roll_deg)
        spoke_angles = [roll_rad + i * (2.0 * math.pi / num_spokes) for i in range(num_spokes)]

        web_color = (255, 255, 255)
        web_thickness = 1

        for angle in spoke_angles:
            ex = int(cx + max_radius * math.cos(angle))
            ey = int(cy + max_radius * math.sin(angle))
            cv2.line(overlay, (cx, cy), (ex, ey), web_color, web_thickness, cv2.LINE_AA)

        # Draw concentric web rings
        num_rings = 5
        for r_step in range(1, num_rings + 1):
            r = max_radius * (r_step / float(num_rings))
            ring_pts = []
            for angle in spoke_angles:
                # Slight inward sag between spokes to look like authentic web threads
                px = int(cx + r * math.cos(angle))
                py = int(cy + r * math.sin(angle))
                ring_pts.append((px, py))

            # Connect ring points
            for i in range(num_spokes):
                p1 = ring_pts[i]
                p2 = ring_pts[(i + 1) % num_spokes]
                # Slight curve midpoint
                mid_angle = (spoke_angles[i] + spoke_angles[(i + 1) % num_spokes]) / 2.0
                sag_r = r * 0.92  # 8% sag for spider-web look
                sag_x = int(cx + sag_r * math.cos(mid_angle))
                sag_y = int(cy + sag_r * math.sin(mid_angle))

                # Draw two segment curve
                cv2.line(overlay, p1, (sag_x, sag_y), web_color, web_thickness, cv2.LINE_AA)
                cv2.line(overlay, (sag_x, sag_y), p2, web_color, web_thickness, cv2.LINE_AA)

        # Alpha blend web overlay
        return cv2.addWeighted(overlay, 0.70, frame_bgr, 0.30, 0)
