"""
Asset generator for Gesture-Controlled AR Experience.
Procedurally draws high-resolution assets for:
- assets/effects/ (energy portals, magic rings, web overlays, shields)
- assets/backgrounds/ (cyberpunk city, superhero rooftop, quantum lab, alien skyline)
- assets/characters/ (spider suit emblem, cyber visor, energy core)
- assets/textures/ (web lattice, halftone dots, hex grid)
"""

import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def ensure_dirs() -> dict[str, Path]:
    dirs = {
        "effects": Path("assets/effects"),
        "backgrounds": Path("assets/backgrounds"),
        "characters": Path("assets/characters"),
        "textures": Path("assets/textures")
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


# --- BACKGROUNDS ---

def draw_cyberpunk_city(path: Path, width: int = 1280, height: int = 720) -> None:
    """Neon cyberpunk skyline at night with glowing skyscraper windows and cyan/magenta fog."""
    img = Image.new("RGB", (width, height), (10, 8, 22))
    draw = ImageDraw.Draw(img)

    # Sky gradient (dark navy to violet)
    for y in range(height):
        ratio = y / float(height)
        r = int(12 + ratio * 45)
        g = int(8 + ratio * 15)
        b = int(24 + ratio * 70)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Distant glowing fog
    fog = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    fog_draw = ImageDraw.Draw(fog)
    fog_draw.ellipse((-200, height // 2, width + 200, height + 300), fill=(180, 20, 140, 60))
    fog_draw.ellipse((width // 4, height // 3, width * 3 // 4, height + 200), fill=(0, 200, 255, 45))
    img = Image.alpha_composite(img.convert("RGBA"), fog).convert("RGB")
    draw = ImageDraw.Draw(img)

    # City Skyscrapers silhouette layers
    # Layer 1: Background silhouette buildings
    np.random.seed(42)
    x = 0
    while x < width:
        bw = np.random.randint(40, 90)
        bh = np.random.randint(280, 480)
        top = height - bh
        draw.rectangle([x, top, x + bw, height], fill=(18, 14, 35))
        # Small window dots
        for wy in range(top + 20, height - 40, 18):
            for wx in range(x + 8, x + bw - 8, 14):
                if np.random.rand() > 0.4:
                    col = (0, 220, 255) if np.random.rand() > 0.5 else (255, 60, 180)
                    draw.point((wx, wy), fill=col)
        x += bw + np.random.randint(2, 10)

    # Layer 2: Foreground sharp skyscrapers
    x = 0
    np.random.seed(101)
    while x < width:
        bw = np.random.randint(60, 130)
        bh = np.random.randint(180, 380)
        top = height - bh
        draw.rectangle([x, top, x + bw, height], fill=(12, 10, 20))
        draw.line([(x, top), (x + bw, top)], fill=(0, 240, 255), width=2)
        # Windows
        for wy in range(top + 15, height - 20, 16):
            for wx in range(x + 10, x + bw - 10, 12):
                if np.random.rand() > 0.3:
                    col = (255, 230, 80) if np.random.rand() > 0.6 else (0, 255, 200)
                    draw.point((wx, wy), fill=col)
        x += bw + np.random.randint(5, 20)

    # Neon ground glow
    draw.rectangle([0, height - 30, width, height], fill=(5, 5, 12))
    draw.line([(0, height - 30), (width, height - 30)], fill=(255, 0, 128), width=3)

    img.save(str(path), "JPEG", quality=92)
    print(f"Generated: {path}")


def draw_superhero_rooftop(path: Path, width: int = 1280, height: int = 720) -> None:
    """Dramatic twilight superhero rooftop overlooking sprawling city skyline with glowing moon."""
    img = Image.new("RGB", (width, height), (15, 20, 35))
    draw = ImageDraw.Draw(img)

    # Sunset twilight gradient
    for y in range(height):
        ratio = y / float(height)
        r = int(25 + (1.0 - ratio) * 200)
        g = int(30 + (1.0 - ratio) * 80)
        b = int(60 + (1.0 - ratio) * 40)
        draw.line([(0, y), (width, y)], fill=(min(255, r), min(255, g), min(255, b)))

    # Large stylized full moon
    moon_x, moon_y = width * 3 // 4, height // 4
    draw.ellipse((moon_x - 70, moon_y - 70, moon_x + 70, moon_y + 70), fill=(255, 250, 220))

    # Distant skyline silhouette
    np.random.seed(77)
    x = 0
    while x < width:
        bw = np.random.randint(50, 110)
        bh = np.random.randint(220, 420)
        top = height - bh - 80
        draw.rectangle([x, top, x + bw, height - 80], fill=(20, 25, 45))
        # Spire or antenna
        if np.random.rand() > 0.6:
            draw.line([(x + bw // 2, top), (x + bw // 2, top - 35)], fill=(255, 100, 50), width=2)
        x += bw + 8

    # Foreground rooftop ledge (where hero stands)
    ledge_top = height - 120
    draw.rectangle([0, ledge_top, width, height], fill=(12, 14, 20))
    # Concrete ledge parapet
    draw.rectangle([0, ledge_top, width, ledge_top + 24], fill=(25, 28, 38))
    draw.line([(0, ledge_top), (width, ledge_top)], fill=(80, 90, 115), width=2)
    # Brick / vent details
    draw.rectangle([width // 8, ledge_top - 50, width // 8 + 90, ledge_top], fill=(16, 18, 26))
    draw.rectangle([width * 3 // 4, ledge_top - 70, width * 3 // 4 + 110, ledge_top], fill=(16, 18, 26))

    img.save(str(path), "JPEG", quality=92)
    print(f"Generated: {path}")


def draw_quantum_lab(path: Path, width: int = 1280, height: int = 720) -> None:
    """High-tech stark quantum laboratory chamber with glowing circular reactor rings."""
    img = Image.new("RGB", (width, height), (8, 12, 18))
    draw = ImageDraw.Draw(img)

    cx, cy = width // 2, height // 2 - 20

    # Perspective grid floor
    for x in range(0, width, 60):
        draw.line([(x, height), (cx + (x - cx) // 4, cy + 100)], fill=(0, 120, 180), width=1)
    for y in range(cy + 100, height, 30):
        draw.line([(0, y), (width, y)], fill=(0, 100, 160), width=1)

    # Concentric quantum reactor rings
    for r in [260, 220, 180, 140, 90, 40]:
        col = (0, 220, 255) if r % 80 == 0 else (0, 120, 180)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=col, width=3)

    # Radial energy conduits
    for angle_deg in range(0, 360, 30):
        rad = math.radians(angle_deg)
        x1 = int(cx + 90 * math.cos(rad))
        y1 = int(cy + 90 * math.sin(rad))
        x2 = int(cx + 280 * math.cos(rad))
        y2 = int(cy + 280 * math.sin(rad))
        draw.line([(x1, y1), (x2, y2)], fill=(0, 240, 255), width=2)

    # Central glowing quantum core
    draw.ellipse((cx - 30, cy - 30, cx + 30, cy + 30), fill=(220, 250, 255))

    img.save(str(path), "JPEG", quality=92)
    print(f"Generated: {path}")


def draw_alien_skyline(path: Path, width: int = 1280, height: int = 720) -> None:
    """Cosmic celestial landscape with binary moons, deep-space purple nebula, and crystalline spires."""
    img = Image.new("RGB", (width, height), (14, 6, 26))
    draw = ImageDraw.Draw(img)

    # Cosmic gradient
    for y in range(height):
        ratio = y / float(height)
        r = int(40 + ratio * 80)
        g = int(10 + ratio * 20)
        b = int(70 + ratio * 90)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Binary Moons
    draw.ellipse((220, 120, 360, 260), fill=(255, 120, 180))  # Large pink moon
    draw.ellipse((380, 90, 450, 160), fill=(100, 240, 255))   # Small cyan moon

    # Star particles
    np.random.seed(99)
    for _ in range(120):
        sx = np.random.randint(0, width)
        sy = np.random.randint(0, height // 2)
        draw.point((sx, sy), fill=(255, 255, 255))

    # Alien mountain range
    pts = [(0, height - 120)]
    for x in range(0, width + 50, 60):
        my = height - 200 - int(np.random.rand() * 120)
        pts.append((x, my))
    pts.extend([(width, height), (0, height)])
    draw.polygon(pts, fill=(28, 12, 45))

    # Crystalline spires
    for sx in [180, 320, 650, 880, 1050]:
        sh = np.random.randint(240, 360)
        spire_pts = [
            (sx, height - sh),
            (sx - 20, height),
            (sx + 20, height)
        ]
        draw.polygon(spire_pts, fill=(60, 20, 90), outline=(180, 80, 255), width=2)

    img.save(str(path), "JPEG", quality=92)
    print(f"Generated: {path}")


# --- EFFECTS (TRANSPARENT PNGs) ---

def draw_magic_portal(path: Path, size: int = 800) -> None:
    """Swirling mystic runes energy portal with glowing circular perimeter and transparent center."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2
    r_outer = size // 2 - 30

    # Multi-layered concentric rune rings
    draw.ellipse((cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer), outline=(255, 140, 0, 240), width=6)
    draw.ellipse((cx - r_outer + 25, cy - r_outer + 25, cx + r_outer - 25, cy + r_outer - 25), outline=(255, 220, 40, 220), width=3)
    draw.ellipse((cx - r_outer + 60, cy - r_outer + 60, cx + r_outer - 60, cy + r_outer - 60), outline=(255, 90, 0, 200), width=4)

    # Runic spokes
    num_spokes = 24
    for i in range(num_spokes):
        angle = i * (2.0 * math.pi / num_spokes)
        x1 = cx + int((r_outer - 60) * math.cos(angle))
        y1 = cy + int((r_outer - 60) * math.sin(angle))
        x2 = cx + int(r_outer * math.cos(angle))
        y2 = cy + int(r_outer * math.sin(angle))
        draw.line([(x1, y1), (x2, y2)], fill=(255, 180, 20, 230), width=3)

        # Runic node circles
        nx = cx + int((r_outer - 25) * math.cos(angle))
        ny = cy + int((r_outer - 25) * math.sin(angle))
        draw.ellipse((nx - 6, ny - 6, nx + 6, ny + 6), fill=(255, 240, 100, 255))

    # Geometric inner star hexagram
    hex_pts = []
    for k in range(6):
        a = k * (math.pi / 3.0)
        px = cx + int((r_outer - 70) * math.cos(a))
        py = cy + int((r_outer - 70) * math.sin(a))
        hex_pts.append((px, py))
    for idx in range(6):
        draw.line([hex_pts[idx], hex_pts[(idx + 2) % 6]], fill=(255, 150, 30, 190), width=3)

    img.save(str(path), "PNG")
    print(f"Generated: {path}")


def draw_cosmic_vortex(path: Path, size: int = 800) -> None:
    """Cosmic swirling vortex of celestial energy with violet & cyan spiral arms."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2

    # Draw spiral arms
    num_arms = 4
    for arm in range(num_arms):
        base_angle = arm * (2.0 * math.pi / num_arms)
        for t in np.linspace(0.1, 4.0 * math.pi, 300):
            r = t * 28.0
            if r > size // 2 - 20:
                break
            angle = base_angle + t
            x = int(cx + r * math.cos(angle))
            y = int(cy + r * math.sin(angle))
            dot_size = max(2, int(r / 35.0))
            col = (180, 50, 255, min(240, int(t * 20))) if arm % 2 == 0 else (0, 220, 255, min(240, int(t * 20)))
            draw.ellipse((x - dot_size, y - dot_size, x + dot_size, y + dot_size), fill=col)

    # Pulsing core
    draw.ellipse((cx - 45, cy - 45, cx + 45, cy + 45), fill=(255, 255, 255, 220))

    img.save(str(path), "PNG")
    print(f"Generated: {path}")


def draw_energy_shield(path: Path, size: int = 800) -> None:
    """Hexagonal holographic forcefield shield with electric cyan grid."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2
    r_shield = size // 2 - 40

    # Outer shield perimeter
    draw.ellipse((cx - r_shield, cy - r_shield, cx + r_shield, cy + r_shield), outline=(0, 240, 255, 240), width=6)
    draw.ellipse((cx - r_shield + 15, cy - r_shield + 15, cx + r_shield - 15, cy + r_shield - 15), outline=(0, 180, 220, 160), width=2)

    # Hexagonal tessellation within shield radius
    hex_r = 45.0
    dx = hex_r * 1.5
    dy = hex_r * math.sqrt(3.0)

    for col in range(-8, 9):
        for row in range(-8, 9):
            hx = cx + col * dx
            hy = cy + row * dy + (col % 2) * (dy / 2.0)
            dist = math.hypot(hx - cx, hy - cy)
            if dist < r_shield - 30:
                hex_points = []
                for i in range(6):
                    ha = i * (math.pi / 3.0)
                    px = hx + (hex_r - 4) * math.cos(ha)
                    py = hy + (hex_r - 4) * math.sin(ha)
                    hex_points.append((px, py))
                alpha = int(max(40, 180 - dist * 0.4))
                draw.polygon(hex_points, outline=(0, 240, 255, alpha), width=2)

    img.save(str(path), "PNG")
    print(f"Generated: {path}")


# --- CHARACTERS / COSTUMES (TRANSPARENT PNGs) ---

def draw_spider_emblem(path: Path, size: int = 600) -> None:
    """Iconic Spider-Man chest spider emblem with aggressive angular legs."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2

    # Spider body (top head, central thorax, lower abdomen)
    # Head
    draw.ellipse((cx - 16, cy - 90, cx + 16, cy - 60), fill=(230, 25, 35, 255))
    # Thorax
    draw.ellipse((cx - 24, cy - 65, cx + 24, cy - 10), fill=(230, 25, 35, 255))
    # Abdomen diamond
    draw.polygon([(cx, cy + 90), (cx - 20, cy - 15), (cx + 20, cy - 15)], fill=(230, 25, 35, 255))

    # Upper 4 legs (sweeping upward then outward)
    leg_coords_upper = [
        # Leg 1 (inner high)
        [(cx - 18, cy - 50), (cx - 70, cy - 120), (cx - 130, cy - 190), (cx - 150, cy - 220)],
        # Leg 2 (outer high)
        [(cx - 18, cy - 35), (cx - 90, cy - 80), (cx - 170, cy - 120), (cx - 220, cy - 130)],
    ]

    for leg in leg_coords_upper:
        # Left leg
        for seg in range(len(leg) - 1):
            draw.line([leg[seg], leg[seg + 1]], fill=(230, 25, 35, 255), width=8 - seg)
        # Right leg (mirrored)
        m_leg = [(cx + (cx - px), py) for px, py in leg]
        for seg in range(len(m_leg) - 1):
            draw.line([m_leg[seg], m_leg[seg + 1]], fill=(230, 25, 35, 255), width=8 - seg)

    # Lower 4 legs (sweeping downward then inward)
    leg_coords_lower = [
        # Leg 3 (mid low)
        [(cx - 18, cy - 10), (cx - 95, cy + 30), (cx - 180, cy + 90), (cx - 200, cy + 160)],
        # Leg 4 (inner low)
        [(cx - 14, cy + 10), (cx - 70, cy + 70), (cx - 120, cy + 150), (cx - 130, cy + 220)],
    ]

    for leg in leg_coords_lower:
        for seg in range(len(leg) - 1):
            draw.line([leg[seg], leg[seg + 1]], fill=(230, 25, 35, 255), width=8 - seg)
        m_leg = [(cx + (cx - px), py) for px, py in leg]
        for seg in range(len(m_leg) - 1):
            draw.line([m_leg[seg], m_leg[seg + 1]], fill=(230, 25, 35, 255), width=8 - seg)

    img.save(str(path), "PNG")
    print(f"Generated: {path}")


def draw_cyber_visor(path: Path, width: int = 700, height: int = 350) -> None:
    """Cybernetic HUD eye visor with glowing target telemetry and scanlines."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = width // 2, height // 2

    # Visor frame
    frame_pts = [
        (cx - 280, cy - 40),
        (cx + 280, cy - 40),
        (cx + 240, cy + 50),
        (cx + 50, cy + 50),
        (cx, cy + 20),
        (cx - 50, cy + 50),
        (cx - 240, cy + 50),
    ]
    draw.polygon(frame_pts, fill=(0, 200, 255, 110), outline=(0, 240, 255, 255), width=4)

    # HUD reticle target rings
    draw.ellipse((cx - 160 - 30, cy - 30, cx - 160 + 30, cy + 30), outline=(255, 220, 0, 200), width=2)
    draw.ellipse((cx + 160 - 30, cy - 30, cx + 160 + 30, cy + 30), outline=(255, 220, 0, 200), width=2)

    # Telemetry brackets
    draw.line([(cx - 220, cy - 20), (cx - 220, cy + 30)], fill=(0, 255, 200, 220), width=3)
    draw.line([(cx + 220, cy - 20), (cx + 220, cy + 30)], fill=(0, 255, 200, 220), width=3)

    img.save(str(path), "PNG")
    print(f"Generated: {path}")


# --- TEXTURES ---

def draw_web_lattice(path: Path, size: int = 800) -> None:
    """Geometric spider-web lattice texture."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2
    num_spokes = 16
    spoke_angles = [i * (2.0 * math.pi / num_spokes) for i in range(num_spokes)]

    for angle in spoke_angles:
        ex = int(cx + (size // 2 - 20) * math.cos(angle))
        ey = int(cy + (size // 2 - 20) * math.sin(angle))
        draw.line([(cx, cy), (ex, ey)], fill=(240, 245, 255, 210), width=2)

    for ring_r in range(50, size // 2 - 10, 55):
        ring_pts = []
        for angle in spoke_angles:
            px = int(cx + ring_r * math.cos(angle))
            py = int(cy + ring_r * math.sin(angle))
            ring_pts.append((px, py))

        for i in range(num_spokes):
            p1 = ring_pts[i]
            p2 = ring_pts[(i + 1) % num_spokes]
            mid_angle = (spoke_angles[i] + spoke_angles[(i + 1) % num_spokes]) / 2.0
            sag_r = ring_r * 0.91
            sag_x = int(cx + sag_r * math.cos(mid_angle))
            sag_y = int(cy + sag_r * math.sin(mid_angle))
            draw.line([p1, (sag_x, sag_y)], fill=(240, 245, 255, 210), width=2)
            draw.line([(sag_x, sag_y), p2], fill=(240, 245, 255, 210), width=2)

    img.save(str(path), "PNG")
    print(f"Generated: {path}")


def draw_halftone_dots(path: Path, size: int = 800) -> None:
    """Pop-art comic printing dot matrix pattern."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    dot_spacing = 30
    for y in range(dot_spacing // 2, size, dot_spacing):
        for x in range(dot_spacing // 2, size, dot_spacing):
            # Radial distance gradient for dynamic dot size
            dist = math.hypot(x - size // 2, y - size // 2)
            r = max(2, int(11 - dist / (size * 0.5) * 8))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 40, 100, 220))

    img.save(str(path), "PNG")
    print(f"Generated: {path}")


def generate_all() -> None:
    dirs = ensure_dirs()

    # Backgrounds
    draw_cyberpunk_city(dirs["backgrounds"] / "cyberpunk_city.jpg")
    draw_superhero_rooftop(dirs["backgrounds"] / "superhero_rooftop.jpg")
    draw_quantum_lab(dirs["backgrounds"] / "quantum_lab.jpg")
    draw_alien_skyline(dirs["backgrounds"] / "alien_skyline.jpg")

    # Effects
    draw_magic_portal(dirs["effects"] / "magic_portal.png")
    draw_cosmic_vortex(dirs["effects"] / "cosmic_vortex.png")
    draw_energy_shield(dirs["effects"] / "energy_shield.png")

    # Characters
    draw_spider_emblem(dirs["characters"] / "spider_emblem.png")
    draw_cyber_visor(dirs["characters"] / "cyber_visor.png")

    # Textures
    draw_web_lattice(dirs["textures"] / "web_lattice.png")
    draw_halftone_dots(dirs["textures"] / "halftone_dots.png")

    print("\n[SUCCESS] All high-resolution AR assets generated successfully!")


if __name__ == "__main__":
    generate_all()
