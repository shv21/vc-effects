"""
Filter manager for Spider-Man AR Studio.
Handles automatic discovery of PNG overlays in assets/filters/,
reads configuration from config/filters.json with fallback defaults,
caches image data, and auto-generates thumbnails for GUI galleries.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any
import cv2
import numpy as np
from PIL import Image, ImageDraw

from .built_in_filters import (
    BaseFilter,
    NoFilter,
    GrayscaleFilter,
    BlackAndWhiteFilter,
    PinkTintFilter,
    CartoonFilter,
    FaceMeshFilter,
    RedBlueHeroFilter,
    ProceduralWebFilter
)
from .overlay_renderer import OverlayRenderer
from .face_tracker import FaceData

logger = logging.getLogger(__name__)


@dataclass
class FilterItem:
    """Represents an active or selectable filter in the studio."""
    filter_id: str
    display_name: str
    filter_type: str  # 'builtin' or 'overlay'
    description: str = ""
    builtin_instance: Optional[BaseFilter] = None
    image_bgra: Optional[np.ndarray] = None
    config: Dict[str, Any] = field(default_factory=dict)
    thumbnail_path: Optional[str] = None
    category: str = "General"


class FilterManager:
    """Scans, organizes, and applies filters dynamically."""

    def __init__(
        self,
        filters_dir: str = "assets/filters",
        thumbnails_dir: str = "assets/thumbnails",
        config_path: str = "config/filters.json"
    ) -> None:
        self.filters_dir = Path(filters_dir)
        self.thumbnails_dir = Path(thumbnails_dir)
        self.config_path = Path(config_path)

        # Ensure directories exist
        self.filters_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)

        self.filters: List[FilterItem] = []
        self.active_index: int = 0
        self.config_cache: Dict[str, Any] = {}

        # Initial load
        self.reload()

    def reload(self) -> None:
        """Reloads configurations, scans filter folder, and rebuilds catalog."""
        logger.info("Reloading filter manager catalog...")
        self.load_config()

        new_filters: List[FilterItem] = []

        # 1. Register Procedural Built-In Filters
        built_ins: List[BaseFilter] = [
            NoFilter(),
            GrayscaleFilter(),
            BlackAndWhiteFilter(),
            PinkTintFilter(),
            CartoonFilter(),
            RedBlueHeroFilter(),
            ProceduralWebFilter(),
            FaceMeshFilter(),
        ]

        for b in built_ins:
            thumb_path = self._generate_builtin_thumbnail(b.filter_id, b.display_name)
            new_filters.append(
                FilterItem(
                    filter_id=b.filter_id,
                    display_name=b.display_name,
                    filter_type="builtin",
                    description=b.description,
                    builtin_instance=b,
                    thumbnail_path=thumb_path,
                    category="Built-In Effects"
                )
            )

        # 2. Discover PNG & image overlays in assets/filters/
        supported_exts = {".png", ".webp"}
        if self.filters_dir.exists():
            for file_path in sorted(self.filters_dir.iterdir()):
                if file_path.is_file() and file_path.suffix.lower() in supported_exts:
                    filter_item = self._load_image_filter(file_path)
                    if filter_item:
                        new_filters.append(filter_item)

        self.filters = new_filters

        # Preserve active index if valid
        if self.active_index >= len(self.filters):
            self.active_index = 0

        logger.info(f"Filter catalog loaded: {len(self.filters)} filters ready.")

    def load_config(self) -> None:
        """Loads filter settings from config/filters.json."""
        if not self.config_path.exists():
            logger.warning(f"Config file {self.config_path} not found. Creating default config.")
            self._create_default_config()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.config_cache = data.get("filters", {})
        except Exception as e:
            logger.error(f"Failed to read {self.config_path}: {e}. Using empty config.")
            self.config_cache = {}

    def _create_default_config(self) -> None:
        """Writes default configurations if config/filters.json doesn't exist."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        default_data = {
            "filters": {
                "spiderman_eyes.png": {
                    "display_name": "Spider-Man Classic Eyes",
                    "anchor": "eyes",
                    "coverage_type": "eyes",
                    "scale_x": 2.45,
                    "scale_y": 2.45,
                    "offset_x": 0.0,
                    "offset_y": -0.05,
                    "rotation_offset": 0.0,
                    "opacity": 1.0,
                    "mirror": False,
                    "category": "Spider-Man / Hero"
                },
                "spiderman_mask.png": {
                    "display_name": "Full Spider-Man Mask",
                    "anchor": "full_face",
                    "coverage_type": "full_face",
                    "scale_x": 1.35,
                    "scale_y": 1.30,
                    "offset_x": 0.0,
                    "offset_y": -0.04,
                    "rotation_offset": 0.0,
                    "opacity": 0.98,
                    "mirror": False,
                    "category": "Spider-Man / Hero"
                },
                "spiderman_web_overlay.png": {
                    "display_name": "Spider-Man Web Mask",
                    "anchor": "full_face",
                    "coverage_type": "full_face",
                    "scale_x": 1.32,
                    "scale_y": 1.28,
                    "offset_x": 0.0,
                    "offset_y": -0.02,
                    "rotation_offset": 0.0,
                    "opacity": 0.88,
                    "mirror": False,
                    "category": "Spider-Man / Hero"
                },
                "iron_hero_mask.png": {
                    "display_name": "Iron Hero Faceplate",
                    "anchor": "full_face",
                    "coverage_type": "full_face",
                    "scale_x": 1.30,
                    "scale_y": 1.32,
                    "offset_x": 0.0,
                    "offset_y": -0.03,
                    "rotation_offset": 0.0,
                    "opacity": 1.0,
                    "mirror": False,
                    "category": "Spider-Man / Hero"
                },
                "sunglasses.png": {
                    "display_name": "Hero Aviator Sunglasses",
                    "anchor": "eyes",
                    "coverage_type": "eyes",
                    "scale_x": 2.30,
                    "scale_y": 2.30,
                    "offset_x": 0.0,
                    "offset_y": -0.02,
                    "rotation_offset": 0.0,
                    "opacity": 0.95,
                    "mirror": False,
                    "category": "Accessories"
                },
                "cat_ears.png": {
                    "display_name": "Anime Cat Ears",
                    "anchor": "forehead",
                    "coverage_type": "forehead",
                    "scale_x": 1.55,
                    "scale_y": 1.45,
                    "offset_x": 0.0,
                    "offset_y": -0.42,
                    "rotation_offset": 0.0,
                    "opacity": 1.0,
                    "mirror": False,
                    "category": "Accessories"
                }
            }
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(default_data, f, indent=4)

    def _load_image_filter(self, file_path: Path) -> Optional[FilterItem]:
        """Loads and caches an image filter with sensible default configurations."""
        try:
            # Read image with alpha channel
            image_bgra = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if image_bgra is None:
                logger.error(f"Cannot load image: {file_path}")
                return None

            filename = file_path.name
            user_config = self.config_cache.get(filename, {})

            # Synthesize sensible defaults based on file name heuristics if unconfigured
            display_name = user_config.get(
                "display_name",
                filename.replace("_", " ").replace("-", " ").rsplit(".", 1)[0].title()
            )

            # Heuristics for default anchor
            name_lower = filename.lower()
            if "eye" in name_lower or "glass" in name_lower:
                default_anchor = "eyes"
                default_scale = 2.3
            elif "ear" in name_lower or "hat" in name_lower or "horn" in name_lower:
                default_anchor = "forehead"
                default_scale = 1.4
            elif "mouth" in name_lower or "mustache" in name_lower or "beard" in name_lower:
                default_anchor = "mouth"
                default_scale = 1.3
            elif "nose" in name_lower:
                default_anchor = "nose"
                default_scale = 1.2
            else:
                default_anchor = "full_face"
                default_scale = 1.3

            config = {
                "anchor": user_config.get("anchor", default_anchor),
                "coverage_type": user_config.get("coverage_type", default_anchor),
                "scale_x": user_config.get("scale_x", default_scale),
                "scale_y": user_config.get("scale_y", default_scale),
                "offset_x": user_config.get("offset_x", 0.0),
                "offset_y": user_config.get("offset_y", 0.0),
                "rotation_offset": user_config.get("rotation_offset", 0.0),
                "opacity": user_config.get("opacity", 1.0),
                "mirror": user_config.get("mirror", False),
            }

            category = user_config.get("category", "Custom Filters")

            # Generate or cache thumbnail
            thumb_path = self._generate_image_thumbnail(file_path, image_bgra)

            return FilterItem(
                filter_id=f"overlay_{filename}",
                display_name=display_name,
                filter_type="overlay",
                description=f"Overlay image: {filename}",
                image_bgra=image_bgra,
                config=config,
                thumbnail_path=thumb_path,
                category=category
            )
        except Exception as e:
            logger.error(f"Error loading filter {file_path}: {e}")
            return None

    def _generate_image_thumbnail(self, file_path: Path, image_bgra: np.ndarray, size: int = 72) -> str:
        """Generates a high-quality square thumbnail for the UI gallery."""
        thumb_file = self.thumbnails_dir / f"thumb_{file_path.stem}.png"

        # Check cache freshness
        if thumb_file.exists() and thumb_file.stat().st_mtime >= file_path.stat().st_mtime:
            return str(thumb_file)

        try:
            h, w = image_bgra.shape[:2]
            scale = min((size - 12) / float(w), (size - 12) / float(h))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))

            resized = cv2.resize(image_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Create circular dark badge background
            canvas = Image.new("RGBA", (size, size), (25, 27, 36, 255))
            draw = ImageDraw.Draw(canvas)
            draw.ellipse((2, 2, size - 3, size - 3), fill=(38, 41, 56, 255), outline=(70, 75, 100, 255), width=2)

            # Paste resized image centered
            if resized.shape[2] == 4:
                # Convert BGRA to RGBA for PIL
                rgba = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGBA)
                pil_overlay = Image.fromarray(rgba)
                offset_x = (size - new_w) // 2
                offset_y = (size - new_h) // 2
                canvas.alpha_composite(pil_overlay, (offset_x, offset_y))
            else:
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pil_overlay = Image.fromarray(rgb)
                offset_x = (size - new_w) // 2
                offset_y = (size - new_h) // 2
                canvas.paste(pil_overlay, (offset_x, offset_y))

            canvas.save(str(thumb_file), "PNG")
            return str(thumb_file)
        except Exception as e:
            logger.warning(f"Could not generate thumbnail for {file_path}: {e}")
            return ""

    def _generate_builtin_thumbnail(self, filter_id: str, display_name: str, size: int = 72) -> str:
        """Generates a stylish graphic thumbnail for procedural filters."""
        thumb_file = self.thumbnails_dir / f"thumb_{filter_id}.png"
        if thumb_file.exists():
            return str(thumb_file)

        try:
            canvas = Image.new("RGBA", (size, size), (25, 27, 36, 255))
            draw = ImageDraw.Draw(canvas)

            # Aesthetic background circle with distinct accent colors per filter
            accent_map = {
                "builtin_none": ((60, 65, 85), (100, 105, 125), "OFF"),
                "builtin_grayscale": ((70, 70, 70), (180, 180, 180), "B/W"),
                "builtin_comic_ink": ((20, 20, 20), (240, 240, 240), "INK"),
                "builtin_pink_tint": ((180, 20, 140), (255, 100, 220), "PINK"),
                "builtin_cartoon": ((200, 120, 20), (255, 220, 40), "TOON"),
                "builtin_red_blue_hero": ((200, 30, 40), (40, 100, 240), "HERO"),
                "builtin_procedural_web": ((40, 45, 60), (220, 220, 230), "WEB"),
                "builtin_facemesh": ((0, 140, 180), (0, 255, 240), "MESH"),
            }
            bg_col, ring_col, badge_text = accent_map.get(filter_id, ((50, 50, 60), (120, 120, 140), "FX"))

            draw.ellipse((2, 2, size - 3, size - 3), fill=(*bg_col, 255), outline=(*ring_col, 255), width=2)
            # Center text badge
            draw.text((size // 2, size // 2), badge_text, fill=(255, 255, 255, 255), anchor="mm")

            canvas.save(str(thumb_file), "PNG")
            return str(thumb_file)
        except Exception as e:
            logger.warning(f"Error creating thumbnail for {filter_id}: {e}")
            return ""

    def get_current_filter(self) -> Optional[FilterItem]:
        """Returns the currently active filter item."""
        if 0 <= self.active_index < len(self.filters):
            return self.filters[self.active_index]
        return None

    def select_filter(self, index: int) -> bool:
        """Selects a filter by its catalog index."""
        if 0 <= index < len(self.filters):
            self.active_index = index
            return True
        return False

    def select_filter_by_id(self, filter_id: str) -> bool:
        """Selects a filter by ID."""
        for idx, item in enumerate(self.filters):
            if item.filter_id == filter_id:
                self.active_index = idx
                return True
        return False

    def next_filter(self) -> FilterItem:
        """Switches to the next filter cyclically."""
        if self.filters:
            self.active_index = (self.active_index + 1) % len(self.filters)
            return self.filters[self.active_index]
        return None

    def prev_filter(self) -> FilterItem:
        """Switches to the previous filter cyclically."""
        if self.filters:
            self.active_index = (self.active_index - 1) % len(self.filters)
            return self.filters[self.active_index]
        return None

    def select_no_filter(self) -> None:
        """Quickly resets to 'No Filter'."""
        self.select_filter_by_id("builtin_none")

    def apply_current_filter(
        self,
        frame_bgr: np.ndarray,
        face_data: FaceData,
        opacity_override: Optional[float] = None
    ) -> np.ndarray:
        """Applies the currently active filter effect to frame_bgr."""
        current = self.get_current_filter()
        if not current:
            return frame_bgr

        if current.filter_type == "builtin" and current.builtin_instance:
            return current.builtin_instance.apply(frame_bgr, face_data)

        elif current.filter_type == "overlay" and current.image_bgra is not None:
            # Copy config and apply user opacity override if specified
            cfg = dict(current.config)
            if opacity_override is not None:
                cfg["opacity"] = opacity_override
            return OverlayRenderer.render_overlay(frame_bgr, current.image_bgra, face_data, cfg)

        return frame_bgr
