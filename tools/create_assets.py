"""
Asset generator tool for Spider-Man AR Studio.
Procedurally draws high-resolution transparent PNG filters for immediate testing and use.
Run once or anytime to regenerate original superhero assets.
"""

import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_spiderman_eyes(output_path: Path, width: int = 800, height: int = 400) -> None:
    """Generates classic comic Spider-Man white lenses with bold black borders."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = width // 2, height // 2

    # Left eye points (viewer's left)
    # Spider-Man lens: curved top, sweeping sharp outer edge, angled bottom
    left_outer_pts = [
        (cx - 30, cy + 20),    # inner bottom
        (cx - 70, cy - 80),    # inner top
        (cx - 240, cy - 120),  # top peak
        (cx - 360, cy - 70),   # outer top tip
        (cx - 340, cy + 40),   # outer bottom
        (cx - 180, cy + 110),  # bottom curve
        (cx - 50, cy + 50),    # inner bottom curve
    ]

    left_inner_pts = [
        (cx - 45, cy + 15),
        (cx - 80, cy - 65),
        (cx - 235, cy - 100),
        (cx - 335, cy - 60),
        (cx - 315, cy + 30),
        (cx - 180, cy + 88),
        (cx - 62, cy + 42),
    ]

    # Draw Left Eye outer black border
    draw.polygon(left_outer_pts, fill=(15, 15, 18, 255))
    # Draw Left Eye white lens
    draw.polygon(left_inner_pts, fill=(245, 248, 255, 255))

    # Add subtle comic shading gradient inside white lens
    for r in range(1, 8):
        scaled_inner = [
            (int(px * 0.96 + (cx - 200) * 0.04), int(py * 0.96 + cy * 0.04))
            for px, py in left_inner_pts
        ]
        draw.polygon(scaled_inner, fill=(255, 255, 255, 255))

    # Right eye points (mirror of left eye)
    right_outer_pts = [(cx + (cx - px), py) for px, py in left_outer_pts]
    right_inner_pts = [(cx + (cx - px), py) for px, py in left_inner_pts]

    draw.polygon(right_outer_pts, fill=(15, 15, 18, 255))
    draw.polygon(right_inner_pts, fill=(245, 248, 255, 255))

    img.save(str(output_path), "PNG")
    print(f"Created: {output_path}")


def create_spiderman_mask(output_path: Path, width: int = 800, height: int = 950) -> None:
    """Generates a full Spider-Man mask with web pattern and iconic eye lenses."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = width // 2, height // 2

    # Mask silhouette contour points
    mask_pts = [
        (cx, cy - 430),         # top forehead center
        (cx - 180, cy - 410),   # top left
        (cx - 310, cy - 310),   # temple left
        (cx - 360, cy - 120),   # cheek left upper
        (cx - 350, cy + 100),   # cheek left lower
        (cx - 270, cy + 280),   # jaw left
        (cx - 140, cy + 410),   # chin left
        (cx, cy + 450),         # chin tip
        (cx + 140, cy + 410),   # chin right
        (cx + 270, cy + 280),   # jaw right
        (cx + 350, cy + 100),   # cheek right lower
        (cx + 360, cy - 120),   # cheek right upper
        (cx + 310, cy - 310),   # temple right
        (cx + 180, cy - 410),   # top right
    ]

    # Draw vibrant crimson red base mask
    draw.polygon(mask_pts, fill=(215, 25, 35, 255), outline=(120, 10, 20, 255))

    # Mask shading contour (darker red around edges for 3D depth)
    inner_mask_pts = [(int(cx + (px - cx) * 0.94), int(cy + (py - cy) * 0.94)) for px, py in mask_pts]
    draw.polygon(inner_mask_pts, fill=(230, 32, 45, 255))

    # Radial Web lines originating from nose bridge (cx, cy - 10)
    web_center_x, web_center_y = cx, cy - 15
    num_spokes = 20
    spoke_angles = [i * (2.0 * math.pi / num_spokes) for i in range(num_spokes)]

    # Draw black web lines
    for angle in spoke_angles:
        ex = int(web_center_x + 550 * math.cos(angle))
        ey = int(web_center_y + 550 * math.sin(angle))
        draw.line([(web_center_x, web_center_y), (ex, ey)], fill=(20, 20, 25, 220), width=3)

    # Concentric web rings
    for ring_r in [60, 120, 190, 270, 360, 460]:
        ring_pts = []
        for angle in spoke_angles:
            px = int(web_center_x + ring_r * math.cos(angle))
            py = int(web_center_y + ring_r * math.sin(angle))
            ring_pts.append((px, py))

        for i in range(num_spokes):
            p1 = ring_pts[i]
            p2 = ring_pts[(i + 1) % num_spokes]
            mid_angle = (spoke_angles[i] + spoke_angles[(i + 1) % num_spokes]) / 2.0
            sag_r = ring_r * 0.92
            sag_x = int(web_center_x + sag_r * math.cos(mid_angle))
            sag_y = int(web_center_y + sag_r * math.sin(mid_angle))
            draw.line([p1, (sag_x, sag_y)], fill=(20, 20, 25, 220), width=3)
            draw.line([(sag_x, sag_y), p2], fill=(20, 20, 25, 220), width=3)

    # Mask outline border
    draw.polygon(mask_pts, outline=(30, 10, 15, 255), width=5)

    # Overlay eye lenses on mask
    # Left eye
    eye_offset_y = cy - 45
    l_outer = [
        (cx - 25, eye_offset_y + 20),
        (cx - 60, eye_offset_y - 75),
        (cx - 190, eye_offset_y - 110),
        (cx - 300, eye_offset_y - 60),
        (cx - 280, eye_offset_y + 40),
        (cx - 150, eye_offset_y + 105),
        (cx - 40, eye_offset_y + 45),
    ]
    l_inner = [
        (cx - 38, eye_offset_y + 16),
        (cx - 70, eye_offset_y - 62),
        (cx - 185, eye_offset_y - 92),
        (cx - 275, eye_offset_y - 50),
        (cx - 258, eye_offset_y + 32),
        (cx - 150, eye_offset_y + 85),
        (cx - 52, eye_offset_y + 38),
    ]

    draw.polygon(l_outer, fill=(10, 10, 12, 255))
    draw.polygon(l_inner, fill=(255, 255, 255, 255))

    # Right eye mirror
    r_outer = [(cx + (cx - px), py) for px, py in l_outer]
    r_inner = [(cx + (cx - px), py) for px, py in l_inner]

    draw.polygon(r_outer, fill=(10, 10, 12, 255))
    draw.polygon(r_inner, fill=(255, 255, 255, 255))

    img.save(str(output_path), "PNG")
    print(f"Created: {output_path}")


def create_spiderman_web_overlay(output_path: Path, width: int = 800, height: int = 900) -> None:
    """Generates transparent comic web lines overlay that aligns on the user's real face."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = width // 2, height // 2 - 30
    num_spokes = 18
    spoke_angles = [i * (2.0 * math.pi / num_spokes) for i in range(num_spokes)]

    # Draw black shadow lines then bright white web lines
    for angle in spoke_angles:
        ex = int(cx + 450 * math.cos(angle))
        ey = int(cy + 450 * math.sin(angle))
        draw.line([(cx, cy), (ex, ey)], fill=(10, 10, 15, 240), width=4)
        draw.line([(cx, cy), (ex, ey)], fill=(240, 245, 255, 220), width=2)

    for ring_r in [70, 140, 220, 310, 410]:
        ring_pts = []
        for angle in spoke_angles:
            px = int(cx + ring_r * math.cos(angle))
            py = int(cy + ring_r * math.sin(angle))
            ring_pts.append((px, py))

        for i in range(num_spokes):
            p1 = ring_pts[i]
            p2 = ring_pts[(i + 1) % num_spokes]
            mid_angle = (spoke_angles[i] + spoke_angles[(i + 1) % num_spokes]) / 2.0
            sag_r = ring_r * 0.90
            sag_x = int(cx + sag_r * math.cos(mid_angle))
            sag_y = int(cy + sag_r * math.sin(mid_angle))

            draw.line([p1, (sag_x, sag_y)], fill=(10, 10, 15, 240), width=4)
            draw.line([(sag_x, sag_y), p2], fill=(10, 10, 15, 240), width=4)
            draw.line([p1, (sag_x, sag_y)], fill=(240, 245, 255, 220), width=2)
            draw.line([(sag_x, sag_y), p2], fill=(240, 245, 255, 220), width=2)

    # Stylized spider symbol in center
    draw.ellipse((cx - 15, cy - 25, cx + 15, cy + 5), fill=(20, 20, 25, 255))
    draw.ellipse((cx - 12, cy + 2, cx + 12, cy + 30), fill=(20, 20, 25, 255))

    img.save(str(output_path), "PNG")
    print(f"Created: {output_path}")


def create_iron_hero_mask(output_path: Path, width: int = 800, height: int = 950) -> None:
    """Generates an Iron Hero high-tech armored faceplate with gold center & glowing cyan eyes."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = width // 2, height // 2

    # Red outer helmet shell
    outer_helmet = [
        (cx, cy - 420),
        (cx - 200, cy - 400),
        (cx - 320, cy - 280),
        (cx - 350, cy - 100),
        (cx - 340, cy + 120),
        (cx - 260, cy + 300),
        (cx - 130, cy + 410),
        (cx, cy + 440),
        (cx + 130, cy + 410),
        (cx + 260, cy + 300),
        (cx + 340, cy + 120),
        (cx + 350, cy - 100),
        (cx + 320, cy - 280),
        (cx + 200, cy - 400),
    ]
    draw.polygon(outer_helmet, fill=(185, 20, 25, 255), outline=(100, 10, 15, 255), width=4)

    # Gold face plate
    gold_plate = [
        (cx, cy - 260),
        (cx - 170, cy - 250),
        (cx - 230, cy - 160),
        (cx - 250, cy + 20),
        (cx - 200, cy + 180),
        (cx - 130, cy + 290),
        (cx, cy + 330),
        (cx + 130, cy + 290),
        (cx + 200, cy + 180),
        (cx + 250, cy + 20),
        (cx + 230, cy - 160),
        (cx + 170, cy - 250),
    ]
    draw.polygon(gold_plate, fill=(235, 185, 45, 255), outline=(160, 115, 20, 255), width=4)

    # Faceplate geometric groove lines
    draw.line([(cx - 170, cy - 250), (cx - 70, cy - 120)], fill=(140, 100, 15, 255), width=3)
    draw.line([(cx + 170, cy - 250), (cx + 70, cy - 120)], fill=(140, 100, 15, 255), width=3)
    draw.line([(cx - 70, cy - 120), (cx + 70, cy - 120)], fill=(140, 100, 15, 255), width=3)

    # Glowing Cyan Eye Slits
    eye_y = cy - 40
    # Left eye slit
    l_eye = [
        (cx - 60, eye_y - 12),
        (cx - 180, eye_y - 8),
        (cx - 170, eye_y + 12),
        (cx - 65, eye_y + 12),
    ]
    draw.polygon(l_eye, fill=(20, 240, 255, 255), outline=(0, 120, 150, 255), width=2)

    # Right eye slit
    r_eye = [(cx + (cx - px), py) for px, py in l_eye]
    draw.polygon(r_eye, fill=(20, 240, 255, 255), outline=(0, 120, 150, 255), width=2)

    img.save(str(output_path), "PNG")
    print(f"Created: {output_path}")


def create_sunglasses(output_path: Path, width: int = 800, height: int = 360) -> None:
    """Generates classic dark aviator sunglasses with reflective lens sheen."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = width // 2, height // 2

    # Left frame
    left_lens = [
        (cx - 30, cy - 40),
        (cx - 160, cy - 50),
        (cx - 320, cy - 40),
        (cx - 350, cy + 30),
        (cx - 280, cy + 120),
        (cx - 140, cy + 130),
        (cx - 50, cy + 70),
    ]
    # Draw frame border
    draw.polygon(left_lens, fill=(25, 25, 30, 240), outline=(210, 175, 50, 255), width=7)

    # Right frame mirror
    right_lens = [(cx + (cx - px), py) for px, py in left_lens]
    draw.polygon(right_lens, fill=(25, 25, 30, 240), outline=(210, 175, 50, 255), width=7)

    # Bridge bar across nose
    draw.line([(cx - 35, cy - 35), (cx + 35, cy - 35)], fill=(210, 175, 50, 255), width=6)
    draw.line([(cx - 30, cy - 15), (cx + 30, cy - 15)], fill=(210, 175, 50, 255), width=5)

    # Specular white gloss line on lenses
    draw.line([(cx - 280, cy - 20), (cx - 200, cy + 70)], fill=(255, 255, 255, 120), width=8)
    draw.line([(cx + 120, cy - 20), (cx + 200, cy + 70)], fill=(255, 255, 255, 120), width=8)

    img.save(str(output_path), "PNG")
    print(f"Created: {output_path}")


def create_cat_ears(output_path: Path, width: int = 800, height: int = 450) -> None:
    """Generates cute feline / anime cat ears with pink inner tufts."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx = width // 2

    # Left Ear
    l_outer = [
        (cx - 70, 360),
        (cx - 220, 60),   # Ear tip
        (cx - 350, 330),
    ]
    l_inner = [
        (cx - 100, 330),
        (cx - 215, 110),
        (cx - 300, 310),
    ]
    draw.polygon(l_outer, fill=(35, 35, 45, 255), outline=(20, 20, 25, 255), width=5)
    draw.polygon(l_inner, fill=(255, 140, 175, 255))

    # Right Ear
    r_outer = [(cx + (cx - px), py) for px, py in l_outer]
    r_inner = [(cx + (cx - px), py) for px, py in l_inner]
    draw.polygon(r_outer, fill=(35, 35, 45, 255), outline=(20, 20, 25, 255), width=5)
    draw.polygon(r_inner, fill=(255, 140, 175, 255))

    # Headband connecting ears
    draw.arc((cx - 280, 280, cx + 280, 480), start=200, end=340, fill=(30, 30, 35, 255), width=7)

    img.save(str(output_path), "PNG")
    print(f"Created: {output_path}")


def generate_all_assets() -> None:
    """Generates all standard filter assets into assets/filters/."""
    filters_dir = Path("assets/filters")
    filters_dir.mkdir(parents=True, exist_ok=True)

    create_spiderman_eyes(filters_dir / "spiderman_eyes.png")
    create_spiderman_mask(filters_dir / "spiderman_mask.png")
    create_spiderman_web_overlay(filters_dir / "spiderman_web_overlay.png")
    create_iron_hero_mask(filters_dir / "iron_hero_mask.png")
    create_sunglasses(filters_dir / "sunglasses.png")
    create_cat_ears(filters_dir / "cat_ears.png")
    print("All assets successfully generated!")


if __name__ == "__main__":
    generate_all_assets()
