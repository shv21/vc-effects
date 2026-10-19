# 🕷️ Spider-Man AR Studio: Custom Real-Time Face Filter Engine

A high-performance, real-time augmented reality desktop studio built from scratch in Python using **OpenCV**, **MediaPipe Face Mesh**, and **CustomTkinter**.

The application tracks facial landmarks with sub-pixel precision and orientation smoothing, applying dynamic superhero-inspired effects (Spider-Man eye lenses, full superhero masks, web overlays, iron armor plates) and custom transparent PNG filters with automatic discovery and zero-code hot reloading.

---

## 🌟 Key Features

- **Real-Time Facial Landmark Tracking**: 468-point 3D landmark mesh detection with Iris tracking powered by Google MediaPipe.
- **Orientation & Pose Estimation**: Computes in-plane head tilt (roll angle $\theta$) and 3D head yaw/pitch for perspective-aligned overlays.
- **Jitter-Free Exponential Moving Average (EMA) Smoothing**: Stabilizes overlay positions during rapid head movements and handles face recovery seamlessly.
- **Automatic PNG Filter Discovery**: Drop any transparent PNG into `assets/filters/` and click **Refresh** (or press `F5`)—no Python code changes required.
- **Config-Driven Calibration (`config/filters.json`)**: Fine-tune anchors (`eyes`, `full_face`, `forehead`, `nose`, `mouth`), scale multipliers, offsets, rotation adjustments, and opacity.
- **Safe Alpha Compositing & Boundary Clipping**: Clean mathematical alpha blending with seamless edge clipping when overlays partially hang off-screen.
- **Spider-Man & Superhero Asset Suite**: Includes high-resolution comic-styled Spider-Man eye lenses, full mask, face web lattice, iron hero faceplate, sunglasses, and cat ears.
- **Built-in Procedural Effects**: Noir Grayscale, High-Contrast Comic Ink, Cyberpunk Pink Glow, Cell-Shaded Cartoon, Red & Blue Superhero Duotone, Procedural Geometric Web, and Wireframe Mesh.
- **HD Snapshot & Video Recording**: High-quality JPEG photo capture with visual shutter feedback, and queue-buffered asynchronous MP4 video recording for lag-free performance.
- **Sleek Modern Dark GUI**: Built with CustomTkinter featuring live video canvas, thumbnail cards, opacity sliders, tracking telemetry badges, and full keyboard shortcuts.

---

## 📁 Project Architecture

```
spidermaneffect/
├── main.py                     # CLI launcher & desktop application entry point
├── requirements.txt            # Pinned, mutually compatible Python dependencies
├── README.md                   # Complete documentation, setup & configuration guide
├── .gitignore                  # Git exclusions for environments, outputs & caches
├── config/
│   └── filters.json            # Configuration catalog for anchor points, scale & offsets
├── src/
│   ├── __init__.py             # Package initializer
│   ├── camera.py               # DirectShow/OpenCV webcam capture & real-time FPS engine
│   ├── face_tracker.py         # MediaPipe Face Mesh tracking, pose estimation & EMA smoothing
│   ├── overlay_renderer.py     # Alpha-channel affine warping, boundary clipping & blending
│   ├── built_in_filters.py     # Procedural filters (NoFilter, Cartoon, Web, Mesh, etc.)
│   ├── filter_manager.py       # Auto-discovery, thumbnail generator & filter catalog
│   ├── recorder.py             # Asynchronous video writer & timestamped photo saver
│   └── app.py                  # CustomTkinter desktop interface and event controller
├── assets/
│   ├── filters/                # User PNG overlays (drop new images here!)
│   │   ├── spiderman_eyes.png
│   │   ├── spiderman_mask.png
│   │   ├── spiderman_web_overlay.png
│   │   ├── iron_hero_mask.png
│   │   ├── sunglasses.png
│   │   └── cat_ears.png
│   └── thumbnails/             # Auto-generated 72x72 gallery preview badges
├── output/                     # Saved photos (*.jpg) and recorded videos (*.mp4)
└── tools/
    ├── create_assets.py        # Vector asset generator for high-res transparent PNGs
    └── test_pipeline.py        # Automated headless test suite for all modules
```

---

## 🛠️ System Requirements & Prerequisites

- **Operating System**: Windows 10 or Windows 11 (64-bit)
- **Webcam**: Standard integrated webcam or USB camera (720p / 1080p recommended)
- **Python**: **Python 3.10** (recommended for full MediaPipe and Tkinter compatibility)

---

## 🚀 Installation & Quick Start (Windows PowerShell)

Open **PowerShell** in the project directory:

### 1. Create a Python 3.10 Virtual Environment

If using `uv` (recommended for instant setup):
```powershell
uv venv --python 3.10 venv
.\venv\Scripts\activate
```

Or using standard Python 3.10:
```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
```

> *Tip: If PowerShell displays an execution policy warning, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`.*

### 2. Install Pinned Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Generate or Verify Filter Assets

Generate the high-resolution superhero assets:
```powershell
python tools/create_assets.py
```

### 4. Run Pipeline Verification Tests (Optional)

Verify that the tracker, overlay renderer, photo saver, and recorder pass all checks:
```powershell
python tools/test_pipeline.py
```

### 5. Launch the Application

```powershell
python main.py
```

---

## ⌨️ Desktop Studio Controls & Keyboard Shortcuts

| Shortcut | Action | Description |
| :--- | :--- | :--- |
| `◀` / `▶` Left / Right Arrow | **Previous / Next Filter** | Cycles through all built-in and PNG filters |
| `Spacebar` | **Capture Photo** | Saves high-res snapshot with visual shutter flash |
| `R` / `r` | **Toggle Recording** | Starts / stops recording video to `output/` |
| `0` | **Reset to No Filter** | Returns to original webcam feed instantly |
| `F5` | **Refresh Filters** | Scans `assets/filters/` for newly added PNGs |
| `ESC` | **Exit Studio** | Cleanly releases camera, flushes video writer, exits |

---

## 🎨 How to Add Your Own Custom PNG Filters

You can add any transparent PNG filter (e.g. superhero masks, anime accessories, hats, glasses, logos) **without writing or editing a single line of Python code**:

### Step 1: Save Your Image
Export or download your effect as a **transparent PNG** (RGBA format) and place it in the `assets/filters/` folder:
```
assets/filters/my_custom_mask.png
```

### Step 2: Auto-Discovery
Launch the app or press **🔄 Refresh** (or press `F5`).
- The application automatically discovers your image.
- A 72x72 gallery thumbnail is automatically generated in `assets/thumbnails/`.
- Heuristic defaults automatically position your filter based on its filename (e.g., if the filename contains `glasses` or `eyes`, it automatically anchors to the eyes!).

### Step 3: Custom Configuration (Optional)
To fine-tune scale, positioning, or opacity, open `config/filters.json` and add an entry matching your image filename:

```json
"my_custom_mask.png": {
    "display_name": "My Custom Hero Mask",
    "anchor": "full_face",
    "coverage_type": "full_face",
    "scale_x": 1.35,
    "scale_y": 1.30,
    "offset_x": 0.0,
    "offset_y": -0.04,
    "rotation_offset": 0.0,
    "opacity": 1.0,
    "mirror": false,
    "category": "Custom Hero"
}
```

### Configuration Parameters Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `display_name` | string | Filename | The user-friendly label displayed in the GUI gallery. |
| `anchor` | string | `"full_face"` | Tracking anchor point: `"eyes"`, `"full_face"`, `"forehead"`, `"nose"`, `"chin"`, `"mouth"`. |
| `scale_x` | float | `1.0` | Width scale multiplier relative to reference facial feature. |
| `scale_y` | float | `scale_x` | Height scale multiplier (preserves aspect ratio by default). |
| `offset_x` | float | `0.0` | Horizontal offset in face coordinate space (shifts with head tilt). |
| `offset_y` | float | `0.0` | Vertical offset in face coordinate space (negative = upward). |
| `rotation_offset` | float | `0.0` | Fine-tune angle adjustment in degrees. |
| `opacity` | float | `1.0` | Overlay transparency multiplier (`0.0` to `1.0`). |
| `mirror` | bool | `false` | Flips image horizontally when `true`. |
| `category` | string | `"Custom"` | Grouping category badge for the UI gallery. |

---

## 📷 Command-Line Options

You can launch `main.py` with custom flags:

```powershell
# Discover connected webcam indices (e.g., [0], [1])
python main.py --scan-cameras

# Launch studio using camera index 1
python main.py --camera 1

# Save photos and recordings to a custom directory
python main.py --output "D:\AR_Recordings"

# List all active built-in and discovered filters
python main.py --list-filters

# Enable verbose debug logging
python main.py --verbose
```

---

## 🔧 Troubleshooting Guide

### 1. Camera Error: "Could not open camera with index 0"
- **Windows Privacy Settings**: Open **Windows Settings > Privacy & Security > Camera** and verify that **Camera access** and **Let desktop apps access your camera** are toggled **ON**.
- **Another App is Using the Camera**: Close applications like Zoom, Microsoft Teams, Discord, Skype, or the Windows Camera app.
- **Multiple Webcams**: Run `python main.py --scan-cameras` to check which device indices are active, then launch with `python main.py --camera <index>`.

### 2. Missing `_tkinter` / `ModuleNotFoundError: No module named 'tkinter'`
- Standard Python on Windows includes Tkinter by default. If using an embeddable or minimal Python runtime that omitted Tcl/Tk, install full CPython:
  ```powershell
  uv python install 3.10
  uv venv --python 3.10 venv
  .\venv\Scripts\activate
  pip install -r requirements.txt
  ```

### 3. MediaPipe / NumPy Version Conflicts
- MediaPipe 0.10.x requires `numpy < 2.0.0`. `requirements.txt` is strictly pinned to `numpy>=1.24.0,<2.0.0` and `mediapipe>=0.10.9,<=0.10.14` to prevent any binary incompatibility.

### 4. Head Turning & 2D vs. 3D Tracking Limitations
- The application uses 2D affine perspective projection derived from 3D facial landmarks. When tilting your head side-to-side (roll), the overlay rotates accurately.
- When turning your head past ~45° profile view (extreme yaw), 2D overlays cannot account for self-occlusion (the back of the head). For full profile views, specialized 3D mesh rendering / OpenGL UV shaders would be required.

---

## ⚖️ License & Intellectual Property Notice

This application's source code and procedural asset generation tools are original open-source software.

**Notice on Third-Party Intellectual Property**:
- "Spider-Man", "Iron Man", and related superhero character names, insignia, and costume designs are trademarks and copyrights of **Marvel Characters, Inc. / The Walt Disney Company**.
- The sample assets provided in this repository are original geometric artwork created strictly for educational, demonstration, and personal prototyping purposes.
- If you distribute or commercialize filters or applications using third-party character likenesses, logos, or artwork, you are solely responsible for obtaining the appropriate licenses or using original, unencumbered artwork.
