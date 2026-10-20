# ⚡ Gesture-Controlled AR Studio: Hand-Defined Effects & Scene Transformation

A real-time, hands-free Augmented Reality desktop studio built from scratch in Python using **OpenCV**, **MediaPipe (Hands, Selfie Segmentation, Pose)**, **NumPy**, and **CustomTkinter**.

Instead of conventional face-filter button clicking, this application uses **both hands** to dynamically control where visual effects appear, how large the affected region is, which effect is active, and how the background and character appearance change.

---

## 🌟 Key Features

- **Immersive Full-Screen AR Experience**:
  - Edge-to-edge camera canvas with zero desktop dashboard clutter.
  - Supports true full-screen mode (`F11`) and minimal, non-intrusive HUD notifications.
  - Optional debug overlay (`D`) toggles landmarks and diagnostics on demand.
- **Continuous Two-Hand Dynamic Region Control**:
  - Uses index fingertips and thumbs of both hands to define an interactive 4-corner polygon (convex quadrilateral).
  - Continuous real-time movement and resizing (hands closer = shrink, hands farther = enlarge).
  - Convex hull sorting prevents twisted, self-intersecting quads even when hands tilt or cross.
  - Exponential Moving Average (EMA) smoothing eliminates landmark jitter without lag.
  - Single-hand/dual-hand tracking loss triggers a clean 0.4s grace period with smooth fadeout — **never leaves stale or frozen regions on screen**.
  - Region remains live by default; explicit freeze via sustained fist gesture or `F` key.
- **Strict Mask Confinement & Feathering**:
  - Hand effects are applied **strictly inside** the hand-controlled polygon mask.
  - Outside pixels remain **100% untouched** (verified by pixel-difference testing).
- **Face Mesh-Anchored Character Overlays**:
  - MediaPipe Face Mesh extracts eye centers, face center, width, and in-plane head roll rotation angle.
  - Masks and eye lens overlays rotate and scale dynamically with head movement.
- **4 Precise Interaction Modes**:
  - **MODE A — Local Effect**: Both hands control a polygon region; selected effect renders inside with feathered boundaries.
  - **MODE B — Scene Portal**: Virtual background is warped directly into the hand-defined polygon with 4-point perspective transform (`cv2.warpPerspective`).
  - **MODE C — Character Transformation**: Virtual background replaces the room (person segmented via Selfie Segmentation), and face/character overlay is anchored to face landmarks.
  - **MODE D — Combined AR**: Full virtual background replacement + segmented person + character overlay + two hands controlling an active transformation portal/effect.
- **Hands-Free Gesture Recognition**:
  - 👋 **Swipe Left / Right**: Instant switch to previous or next visual effect.
  - ✌️ **Peace / V-Sign**: Cycle through Modes A, B, C, and D.
  - 🤏 **Dual Pinch**: Deliberate activation / fine-tuning gesture (keeps tracking live).
  - ✊ **Fist (Hold)**: Deliberate region freeze/lock toggle.
  - ✋ **Open Palms**: Re-engage live dynamic tracking.

---

## 📁 Project Architecture

```
spidermaneffect/
├── main.py                     # CLI launcher & desktop application entry point
├── app.py                      # Root convenience forwarding entry point
├── run.bat                     # Windows double-click launcher
├── requirements.txt            # Pinned, mutually compatible Python dependencies
├── README.md                   # Complete documentation, setup & gesture guide
├── .gitignore                  # Git exclusions for environments, outputs & caches
├── config/
│   ├── gestures.json           # Gesture recognition thresholds, hold times & cooldowns
│   └── effects.json            # Effect configuration catalog, feathering & opacity
├── src/
│   ├── __init__.py             # Package initializer (v2.1.0)
│   ├── camera.py               # DirectShow/OpenCV capture & real-time FPS engine
│   ├── hand_tracker.py         # Dual-hand tracking, screen-space horizontal assignment & EMA smoothing
│   ├── gesture_recognizer.py   # Two-hand gesture recognition, swipe detection & debouncing
│   ├── region_controller.py    # Convex quad sorting, scale expansion & tracking-loss grace fadeout
│   ├── mask_renderer.py        # Strict alpha blending, feathered masking & glowing HUD outline
│   ├── effect_manager.py       # Procedural effects & 4-point perspective warping
│   ├── person_segmenter.py     # MediaPipe Selfie Segmentation with temporal halo reduction
│   ├── background_compositor.py # Virtual background scene replacement & 4-point perspective scene portals
│   ├── character_compositor.py # MediaPipe Face Mesh-anchored character overlays with head roll rotation
│   ├── recorder.py             # Asynchronous MP4 video recorder & timestamped photo saver
│   └── app.py                  # Immersive full-screen AR studio application
├── assets/
│   ├── effects/                # Transparent PNG effect overlays (magic_portal, energy_shield, etc.)
│   ├── backgrounds/            # Virtual background scenes (cyberpunk_city, superhero_rooftop, etc.)
│   ├── characters/             # Character costume overlays (spiderman_mask, spiderman_eyes, etc.)
│   └── textures/               # Background fill textures (web_lattice, halftone_dots, etc.)
├── output/                     # Saved photos (*.jpg) and recorded videos (*.mp4)
└── tools/
    ├── create_assets.py        # Procedural asset generator for high-res images across all 4 categories
    └── test_pipeline.py        # Automated 6-suite verification test script
```

---

## 🛠️ System Requirements & Prerequisites

- **Operating System**: Windows 10 or Windows 11 (64-bit)
- **Webcam**: Standard integrated webcam or USB camera (640x480 default recommended)
- **Python**: **Python 3.10** (recommended for binary compatibility with MediaPipe and CustomTkinter)

---

## 🚀 Installation & Quick Start (Windows PowerShell)

### 1. Open PowerShell in Project Folder

```powershell
cd c:\Users\shivr\OneDrive\Desktop\spidermaneffect
```

### 2. Activate Virtual Environment or Install Dependencies

The project is already pre-configured with a Python 3.10 virtual environment (`.\venv`).
If setting up fresh on another machine:

```powershell
uv venv --python 3.10 venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Generate High-Resolution Assets

Generate high-resolution procedural backgrounds, effects, characters, and textures:

```powershell
python tools/create_assets.py
```

### 4. Run Automated Pipeline Verification Tests

Run the headless verification suite:

```powershell
python tools/test_pipeline.py
```

All 6 test suites will run and verify:
- Hand tracking metric extraction & dual-pinch stability
- Continuous region resizing & tracking-loss grace cleanup (no freeze)
- Strict pixel isolation (0 pixels altered outside mask)
- Perspective warping
- Background replacement, 4-point scene portal & Face Mesh character compositing
- Photo & video recording

### 5. Launch the Application

```powershell
python main.py
```
*(Or simply double-click `run.bat` or run `py main.py` — automatic venv delegation is built-in!)*

---

## 🖐️ Hands-Free Gesture Interaction Guide

| Gesture | Action | Visual Feedback |
| :--- | :--- | :--- |
| **Both Hands in View** | **Defines Active Region** | Animated cyan quadrilateral frames the area between hands |
| **Move Hands Apart / Together** | **Enlarge / Shrink Region** | Region dynamically scales outward/inward |
| **👋 Swipe Hand Right** | **Next Effect ▶** | Cycles to next visual effect (portal, web, matrix, cosmic, etc.) |
| **👋 Swipe Hand Left** | **◀ Previous Effect** | Cycles to previous visual effect |
| **✌️ Peace / V-Sign** | **Cycle Scene Mode** | Cycles through Mode A → Mode B → Mode C → Mode D |
| **✊ Fist (Hold)** | **Toggle Freeze / Lock** | Freezes region coordinates in place (unfreezes on open hands) |
| **✋ Open Palms** | **Dynamic Follow** | Resumes live tracking |

---

## ⌨️ Accessibility & Debugging Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `F11` | Toggle Full-Screen Mode |
| `D` / `d` | Toggle Debug Mode (Landmarks, Hand Skeletons, Face Mesh & Diagnostics) |
| `Spacebar` | Capture Photo Snapshot (with white shutter flash) |
| `R` / `r` | Start / Stop Video Recording |
| `M` / `m` | Cycle Scene Mode (Mode A → B → C → D) |
| `B` / `b` | Cycle Virtual Background Scene |
| `C` / `c` | Cycle Character Costume Overlay |
| `F` / `f` | Toggle Region Freeze / Lock |
| `I` / `i` | Invert Mask (Inside / Outside Effect) |
| `◀` / `▶` Left / Right Arrow | Cycle to Previous / Next Effect |
| `F5` | Reload all assets from disk |
| `ESC` | Exit Studio cleanly |

---

## 🎨 Adding Your Own Images & Assets

You can add your own custom images without editing any Python code:

### 1. Add Effect Overlays (`assets/effects/`)
Drop any transparent PNG (e.g. `energy_ring.png`, `magic_rune.png`).
- When selected, the image is automatically warped to match the 4-corner perspective of your hands using `cv2.warpPerspective`!

### 2. Add Virtual Backgrounds (`assets/backgrounds/`)
Drop any JPG or PNG (e.g. `mountains.jpg`, `mars_base.png`).
- The application automatically resizes and center-crops the image to fit your webcam resolution, placing you in the scene!

### 3. Add Character Overlays (`assets/characters/`)
Drop any transparent PNG (e.g. `hero_armor.png`, `tactical_mask.png`).
- The application anchors the asset relative to detected upper-body shoulders or eyes.

### 4. Hot Reload
Press `F5` in the studio or click **Reload Assets** to discover newly added files instantly.

---

## 🔬 Architecture & Compositing Pipeline

The application processes frames through a decoupled 4-stage pipeline:

```
[Webcam Frame (640x480)]
         │
         ├──> [HandTracker] ──> [GestureRecognizer] ──> Hands-Free Actions (Swipe, Pinch, V-Sign)
         │           │
         │           └──> [RegionController] ──> Feathered Polygon Mask & Glowing HUD
         │
         └──> [PersonSegmenter] ──> Foreground Silhouette Alpha Mask
                     │
                     ▼
         ┌────────────────────────────────────────────────────────┐
         │              LAYERED AR COMPOSITING                     │
         │                                                        │
         │  Layer 1: Virtual Background Image                     │
         │     ▲                                                  │
         │     │ (Composited using Person Silhouette Mask)        │
         │  Layer 2: Live Segmented Person                        │
         │     ▲                                                  │
         │     │ (Anchored via MediaPipe Face Mesh + Head Roll)                    │
         │  Layer 3: Character Costume / Visor Overlay            │
         │     ▲                                                  │
         │     │ (Restricted strictly to Hand Polygon Mask)       │
         │  Layer 4: Hand-Controlled Visual Effect / Portal       │
         └────────────────────────────────────────────────────────┘
                     │
                     ▼
         [Minimal Immersive Viewport (Canvas / Photo / Video)]
```

### 2D Layered Compositing vs. 3D Retargeting
- **2D Layered Compositing (Implemented)**: High-performance, photorealistic alpha-blended compositing of real-time segmentation, pose-anchored overlays, and perspective-warped polygon effects at 30-60 FPS on standard CPUs.
- **Full 3D Character Motion Retargeting**: Requires complex rigged 3D meshes, OpenGL/Vulkan rendering pipelines, and full-body depth sensors. The current system prioritizes ultra-responsive 2D AR compositing with zero GPU requirements.

---

## 🔧 Troubleshooting Guide

### 1. Webcam Not Opening
- Open **Windows Settings > Privacy & Security > Camera** and verify that camera access is allowed for desktop apps.
- Close other apps using the webcam (Zoom, Teams, Discord, etc.).
- Run `python main.py --scan-cameras` to see connected camera indices, then launch with `python main.py --camera <index>`.

### 2. Hand Tracking Sensitivity
- Ensure adequate lighting on your hands.
- Avoid extreme backlight (e.g. sitting directly in front of a bright window).
- Both hands should be reasonably open and facing toward the camera for initial region detection.

### 3. Virtual Background Halos
- The segmenter applies a Gaussian feathering kernel (`feather_radius=9`) and temporal EMA smoothing (`smoothing_alpha=0.75`).
- Ensure contrast between your clothing and the physical room background for cleanest segmentation edges.

---

## ⚖️ License & Intellectual Property Notice

This application is original software developed for educational and personal augmented reality prototyping. Third-party character names, emblems, and insignia (e.g. Spider-Man, Iron Man) are trademarks of their respective owners. Sample vector assets provided in `assets/` are original geometric demonstration artwork.
