"""
Generator for rich sample AR Scene images.
Generates complete scenes containing BOTH a character and a background:
1. anime_hoodie_building.jpg - Anime character in black hoodie in modern architectural building
2. superhero_city.jpg        - Superhero standing on rooftop overlooking vibrant city
3. fantasy_portal.jpg         - Character in magical fantasy realm with glowing runes
4. cyber_warrior.png         - Character with alpha transparency layer for RGBA testing
"""

import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def generate_anime_hoodie_building(path: Path, width: int = 1280, height: int = 720) -> None:
    """Anime character in black hoodie inside a sleek modern building."""
    img = Image.new("RGB", (width, height), (25, 28, 38))
    draw = ImageDraw.Draw(img)

    # 1. Modern Building Interior Background
    # Polished floor gradient
    for y in range(height // 2, height):
        ratio = (y - height // 2) / float(height // 2)
        r = int(20 + ratio * 35)
        g = int(22 + ratio * 38)
        b = int(32 + ratio * 48)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Ceiling and architectural panels
    for y in range(0, height // 3):
        ratio = 1.0 - (y / float(height // 3))
        r = int(18 + ratio * 20)
        g = int(20 + ratio * 25)
        b = int(30 + ratio * 35)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Sleek recessed ceiling LED strip lights
    for x in range(100, width, 250):
        draw.line([(x, 10), (x + 80, height // 3)], fill=(220, 240, 255), width=4)
        draw.line([(x + 2, 10), (x + 82, height // 3)], fill=(0, 200, 255), width=2)

    # Floor-to-ceiling glass windows with city view
    win_left, win_top, win_right, win_bot = 120, 80, width - 120, height - 120
    # Window glass gradient with distant dusk city glow
    for y in range(win_top, win_bot):
        ratio = (y - win_top) / float(win_bot - win_top)
        r = int(40 + ratio * 60)
        g = int(35 + ratio * 25)
        b = int(80 + ratio * 45)
        draw.line([(win_left, y), (win_right, y)], fill=(r, g, b))

    # Distant skyscraper silhouettes through glass
    np.random.seed(101)
    bx = win_left + 20
    while bx < win_right - 40:
        bw = np.random.randint(45, 90)
        bh = np.random.randint(150, 300)
        btop = win_bot - bh
        draw.rectangle([bx, btop, bx + bw, win_bot], fill=(22, 18, 38))
        # Window specks
        for wy in range(btop + 15, win_bot - 15, 20):
            for wx in range(bx + 8, bx + bw - 8, 14):
                if np.random.rand() > 0.4:
                    draw.rectangle([wx, wy, wx + 4, wy + 8], fill=(255, 220, 140))
        bx += bw + 15

    # Glass mullions (window frames)
    for fx in range(win_left, win_right, 220):
        draw.line([(fx, win_top), (fx, win_bot)], fill=(40, 48, 65), width=6)
        draw.line([(fx, win_top), (fx, win_bot)], fill=(80, 95, 125), width=2)
    draw.line([(win_left, win_top), (win_right, win_top)], fill=(45, 52, 70), width=8)
    draw.line([(win_left, win_bot), (win_right, win_bot)], fill=(45, 52, 70), width=8)

    # Modern steel interior pillar
    draw.rectangle([width - 240, 0, width - 140, height], fill=(30, 34, 46))
    draw.line([(width - 240, 0), (width - 240, height)], fill=(70, 80, 105), width=3)
    draw.line([(width - 140, 0), (width - 140, height)], fill=(70, 80, 105), width=3)

    # 2. Anime Character Wearing Black Hoodie (Standing in center)
    cx, cy = width // 2 - 20, height // 2 + 30

    # Character shadow on polished floor
    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    s_draw = ImageDraw.Draw(shadow)
    s_draw.ellipse([cx - 100, height - 140, cx + 100, height - 80], fill=(5, 6, 10, 160))
    img = Image.alpha_composite(img.convert("RGBA"), shadow).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Legs / Dark Pants
    draw.polygon([(cx - 45, cy + 130), (cx - 15, cy + 130), (cx - 20, height - 100), (cx - 55, height - 100)], fill=(15, 16, 22))
    draw.polygon([(cx + 15, cy + 130), (cx + 45, cy + 130), (cx + 55, height - 100), (cx + 20, height - 100)], fill=(18, 19, 26))
    # Shoes
    draw.ellipse([cx - 65, height - 110, cx - 15, height - 85], fill=(35, 40, 50))
    draw.ellipse([cx + 15, height - 110, cx + 65, height - 85], fill=(35, 40, 50))

    # Black Hoodie Body
    draw.polygon([
        (cx - 75, cy - 20),
        (cx + 75, cy - 20),
        (cx + 90, cy + 70),
        (cx + 60, cy + 145),
        (cx - 60, cy + 145),
        (cx - 90, cy + 70)
    ], fill=(16, 17, 22))
    # Hoodie zipper & pocket details
    draw.line([(cx, cy - 10), (cx, cy + 140)], fill=(45, 50, 62), width=3)
    draw.polygon([(cx - 40, cy + 85), (cx + 40, cy + 85), (cx + 35, cy + 125), (cx - 35, cy + 125)], outline=(35, 38, 48), fill=(20, 22, 28))

    # Hoodie Arms (hands in pockets)
    draw.polygon([(cx - 75, cy - 20), (cx - 105, cy + 45), (cx - 70, cy + 110), (cx - 45, cy + 85)], fill=(14, 15, 20))
    draw.polygon([(cx + 75, cy - 20), (cx + 105, cy + 45), (cx + 70, cy + 110), (cx + 45, cy + 85)], fill=(18, 20, 25))

    # Oversized Black Hood
    draw.ellipse([cx - 70, cy - 150, cx + 70, cy - 10], fill=(14, 15, 19))
    draw.ellipse([cx - 55, cy - 135, cx + 55, cy - 25], fill=(8, 9, 12))  # Inner hood shadow

    # Anime Face inside hood
    draw.ellipse([cx - 36, cy - 115, cx + 36, cy - 45], fill=(245, 218, 195))  # Skin tone
    # Anime Hair locks framing face
    hair_color = (25, 25, 30)
    draw.polygon([(cx - 35, cy - 110), (cx - 10, cy - 75), (cx - 20, cy - 115)], fill=hair_color)
    draw.polygon([(cx - 15, cy - 120), (cx + 10, cy - 70), (cx + 5, cy - 120)], fill=hair_color)
    draw.polygon([(cx + 10, cy - 115), (cx + 32, cy - 80), (cx + 35, cy - 110)], fill=hair_color)

    # Stylized Glowing Eyes
    eye_col = (0, 225, 255)
    draw.ellipse([cx - 24, cy - 82, cx - 12, cy - 74], fill=eye_col)
    draw.ellipse([cx + 12, cy - 82, cx + 24, cy - 74], fill=eye_col)
    draw.ellipse([cx - 20, cy - 80, cx - 16, cy - 76], fill=(255, 255, 255))
    draw.ellipse([cx + 16, cy - 80, cx + 20, cy - 76], fill=(255, 255, 255))

    # Sleek cyan accent lines on hoodie
    draw.line([(cx - 40, cy - 15), (cx - 40, cy + 60)], fill=(0, 210, 255), width=2)
    draw.line([(cx + 40, cy - 15), (cx + 40, cy + 60)], fill=(0, 210, 255), width=2)

    # Title label at top corner
    draw.text((30, 25), "ANIME PROTOCOL // SECTOR 7", fill=(0, 210, 255))

    img.save(str(path), "JPEG", quality=95)
    print(f"Generated: {path}")


def generate_superhero_city(path: Path, width: int = 1280, height: int = 720) -> None:
    """Complete scene: superhero in iconic red/blue suit standing on rooftop overlooking city."""
    img = Image.new("RGB", (width, height), (12, 10, 28))
    draw = ImageDraw.Draw(img)

    # Twilight skyline gradient
    for y in range(height):
        ratio = y / float(height)
        r = int(18 + ratio * 85)
        g = int(12 + ratio * 35)
        b = int(45 + ratio * 65)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Glowing moon
    draw.ellipse([width - 320, 70, width - 180, 210], fill=(255, 245, 220))
    # Soft moon glow
    moon_glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    mg_draw = ImageDraw.Draw(moon_glow)
    mg_draw.ellipse([width - 360, 30, width - 140, 250], fill=(255, 240, 200, 50))
    img = Image.alpha_composite(img.convert("RGBA"), moon_glow).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Skyline skyscraper silhouettes
    np.random.seed(42)
    x = 0
    while x < width:
        bw = np.random.randint(60, 130)
        bh = np.random.randint(240, 480)
        draw.rectangle([x, height - bh, x + bw, height], fill=(16, 12, 32))
        for wy in range(height - bh + 25, height - 120, 22):
            for wx in range(x + 10, x + bw - 10, 18):
                if np.random.rand() > 0.45:
                    draw.rectangle([wx, wy, wx + 6, wy + 10], fill=(255, 215, 120))
        x += bw + 10

    # Rooftop foreground ledge
    draw.polygon([(0, height - 140), (width, height - 140), (width, height), (0, height)], fill=(25, 28, 38))
    draw.line([(0, height - 140), (width, height - 140)], fill=(75, 85, 110), width=6)

    # Hero figure in center
    cx, cy = width // 2, height - 260
    # Red & Blue superhero body
    # Legs
    draw.line([(cx - 30, cy + 70), (cx - 45, cy + 140)], fill=(20, 60, 160), width=24)
    draw.line([(cx + 30, cy + 70), (cx + 45, cy + 140)], fill=(20, 60, 160), width=24)
    # Red Boots
    draw.line([(cx - 45, cy + 120), (cx - 50, cy + 145)], fill=(220, 30, 40), width=26)
    draw.line([(cx + 45, cy + 120), (cx + 50, cy + 145)], fill=(220, 30, 40), width=26)
    # Torso (Red & Blue)
    draw.polygon([(cx - 55, cy - 40), (cx + 55, cy - 40), (cx + 35, cy + 75), (cx - 35, cy + 75)], fill=(220, 30, 45))
    draw.polygon([(cx - 55, cy - 10), (cx - 35, cy + 75), (cx - 20, cy + 75), (cx - 35, cy - 10)], fill=(25, 65, 175))
    draw.polygon([(cx + 55, cy - 10), (cx + 35, cy + 75), (cx + 20, cy + 75), (cx + 35, cy - 10)], fill=(25, 65, 175))
    # Spider emblem on chest
    draw.ellipse([cx - 8, cy - 5, cx + 8, cy + 15], fill=(15, 15, 20))
    # Spider mask head
    draw.ellipse([cx - 35, cy - 110, cx + 35, cy - 40], fill=(225, 30, 45))
    # White reflective superhero eyes
    draw.polygon([(cx - 26, cy - 82), (cx - 6, cy - 72), (cx - 10, cy - 65), (cx - 28, cy - 70)], fill=(255, 255, 255))
    draw.polygon([(cx + 26, cy - 82), (cx + 6, cy - 72), (cx + 10, cy - 65), (cx + 28, cy - 70)], fill=(255, 255, 255))
    draw.polygon([(cx - 28, cy - 84), (cx - 4, cy - 70), (cx - 8, cy - 63), (cx - 30, cy - 68)], outline=(10, 10, 15), width=2)
    draw.polygon([(cx + 28, cy - 84), (cx + 4, cy - 70), (cx + 8, cy - 63), (cx + 30, cy - 68)], outline=(10, 10, 15), width=2)
    # Arms in heroic stance
    draw.line([(cx - 50, cy - 35), (cx - 80, cy + 20)], fill=(220, 30, 45), width=20)
    draw.line([(cx + 50, cy - 35), (cx + 80, cy + 20)], fill=(220, 30, 45), width=20)

    img.save(str(path), "JPEG", quality=95)
    print(f"Generated: {path}")


def generate_fantasy_portal(path: Path, width: int = 1280, height: int = 720) -> None:
    """Complete fantasy scene with character in magical stone shrine with glowing runes."""
    img = Image.new("RGB", (width, height), (15, 10, 30))
    draw = ImageDraw.Draw(img)

    # Cosmic mystical sky
    for y in range(height):
        ratio = y / float(height)
        r = int(25 + math.sin(ratio * 3.14) * 45)
        g = int(10 + ratio * 30)
        b = int(50 + ratio * 85)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Giant glowing celestial portal ring in background
    ring = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    r_draw = ImageDraw.Draw(ring)
    cx, cy = width // 2, height // 2 - 20
    for radius in range(260, 290, 4):
        alpha = int(180 - abs(radius - 275) * 8)
        r_draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=(180, 50, 255, alpha), width=3)
    r_draw.ellipse([cx - 240, cy - 240, cx + 240, cy + 240], fill=(60, 10, 110, 110))
    img = Image.alpha_composite(img.convert("RGBA"), ring).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Ancient Stone Pillars on sides
    for px in [140, width - 220]:
        draw.polygon([(px, 80), (px + 80, 80), (px + 90, height), (px - 10, height)], fill=(45, 38, 55))
        # Glowing arcane runes on pillars
        for ry in range(160, height - 120, 60):
            draw.line([(px + 20, ry), (px + 60, ry + 20)], fill=(0, 235, 255), width=3)
            draw.ellipse([px + 35, ry + 15, px + 45, ry + 25], fill=(255, 100, 255))

    # Ground stone dias
    draw.polygon([(180, height - 150), (width - 180, height - 150), (width, height), (0, height)], fill=(32, 28, 42))
    draw.line([(180, height - 150), (width - 180, height - 150)], fill=(140, 60, 220), width=4)

    # Fantasy Mage / Warrior Character standing in center
    ch_x, ch_y = cx, height - 240
    # Cloak / Robes
    draw.polygon([
        (ch_x - 30, ch_y - 80),
        (ch_x + 30, ch_y - 80),
        (ch_x + 75, ch_y + 110),
        (ch_x - 75, ch_y + 110)
    ], fill=(40, 15, 65))
    # Inner Tunic
    draw.polygon([(ch_x - 25, ch_y - 80), (ch_x + 25, ch_y - 80), (ch_x + 30, ch_y + 80), (ch_x - 30, ch_y + 80)], fill=(200, 180, 140))
    # Golden Armor Trim
    draw.line([(ch_x - 30, ch_y - 80), (ch_x + 30, ch_y - 80)], fill=(255, 215, 0), width=5)
    draw.line([(ch_x, ch_y - 80), (ch_x, ch_y + 80)], fill=(255, 215, 0), width=3)
    # Head & Hood
    draw.ellipse([ch_x - 30, ch_y - 135, ch_x + 30, ch_y - 75], fill=(35, 12, 58))
    draw.ellipse([ch_x - 18, ch_y - 118, ch_x + 18, ch_y - 88], fill=(240, 210, 185))
    # Glowing eyes
    draw.ellipse([ch_x - 12, ch_y - 105, ch_x - 5, ch_y - 99], fill=(120, 240, 255))
    draw.ellipse([ch_x + 5, ch_y - 105, ch_x + 12, ch_y - 99], fill=(120, 240, 255))
    # Staff with glowing crystal in right hand
    draw.line([(ch_x + 65, ch_y - 150), (ch_x + 65, ch_y + 110)], fill=(90, 70, 50), width=7)
    draw.ellipse([ch_x + 50, ch_y - 175, ch_x + 80, ch_y - 145], fill=(0, 240, 255))

    img.save(str(path), "JPEG", quality=95)
    print(f"Generated: {path}")


def generate_cyber_character_png(path: Path, width: int = 1280, height: int = 720) -> None:
    """Complete RGBA transparent scene image for testing alpha transparency preservation."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Semi-transparent high-tech grid arena backdrop
    grid = Image.new("RGBA", (width, height), (10, 15, 30, 200))
    g_draw = ImageDraw.Draw(grid)
    for x in range(0, width, 50):
        g_draw.line([(x, 0), (x, height)], fill=(0, 200, 255, 40), width=1)
    for y in range(0, height, 50):
        g_draw.line([(0, y), (width, y)], fill=(0, 200, 255, 40), width=1)
    # Circular tech rings
    cx, cy = width // 2, height // 2
    for r in [180, 240, 300]:
        g_draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 230, 255, 80), width=2)
    img = Image.alpha_composite(img, grid)
    draw = ImageDraw.Draw(img)

    # Cybernetic Character (Full solid Alpha)
    # Legs
    draw.polygon([(cx - 35, cy + 90), (cx - 15, cy + 90), (cx - 25, cy + 220), (cx - 50, cy + 220)], fill=(20, 25, 38, 255))
    draw.polygon([(cx + 15, cy + 90), (cx + 35, cy + 90), (cx + 50, cy + 220), (cx + 25, cy + 220)], fill=(20, 25, 38, 255))
    # Exosuit Torso
    draw.polygon([(cx - 60, cy - 50), (cx + 60, cy - 50), (cx + 40, cy + 100), (cx - 40, cy + 100)], fill=(35, 42, 58, 255))
    # Glowing energy core on chest
    draw.ellipse([cx - 20, cy - 10, cx + 20, cy + 30], fill=(0, 240, 255, 255))
    draw.ellipse([cx - 10, cy, cx + 10, cy + 20], fill=(255, 255, 255, 255))
    # Helmet / Visor
    draw.ellipse([cx - 40, cy - 140, cx + 40, cy - 55], fill=(30, 36, 50, 255))
    draw.polygon([(cx - 32, cy - 110), (cx + 32, cy - 110), (cx + 25, cy - 85), (cx - 25, cy - 85)], fill=(255, 160, 0, 255))

    img.save(str(path), "PNG")
    print(f"Generated: {path}")


def main() -> None:
    scenes_dir = Path("assets/scenes")
    scenes_dir.mkdir(parents=True, exist_ok=True)

    generate_anime_hoodie_building(scenes_dir / "anime_hoodie_building.jpg")
    generate_superhero_city(scenes_dir / "superhero_city.jpg")
    generate_fantasy_portal(scenes_dir / "fantasy_portal.jpg")
    generate_cyber_character_png(scenes_dir / "cyber_warrior.png")
    print("\n[SUCCESS] Generated rich scene assets in assets/scenes/!")


if __name__ == "__main__":
    main()
