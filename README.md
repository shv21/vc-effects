# ⚡ Gesture-Controlled AR Studio: Hand-Defined Effects & Scene Transformation

A real-time, hands-free Augmented Reality desktop studio built in Python using **OpenCV**, **MediaPipe (Hands, Selfie Segmentation, Face Mesh)**, **NumPy**, and **CustomTkinter**.

Instead of conventional face-filter button clicking, this application uses **both hands** to dynamically control where visual effects appear, how large the affected region is, which effect is active, and how the background and character appearance change.

---

## 🌟 Key Features

- **Immersive Full-Screen AR Experience**:
  - Edge-to-edge camera canvas with zero desktop dashboard clutter.
  - Supports true full-screen mode (`F11` or `F`) and minimal, non-intrusive HUD notifications.
  - Skeletons, polygon outlines, and telemetry are **hidden by default** in clean presentation mode and can be toggled on demand with Debug Mode (`D`).
- **Continuous Two-Hand Dynamic Region Control**:
  - Uses the thumb tips and index fingertips of both hands to define an interactive 4-corner polygon (convex quadrilateral).
  - Continuous real-time movement and resizing (hands closer = shrink, hands farther = enlarge).
  - Clockwise convex sorting prevents twisted, self-intersecting quads even when hands tilt or cross.
  - Orientation-adaptive span maintains stable geometry under any hand tilt.
  - Velocity-adaptive Exponential Moving Average (EMA) smoothing eliminates landmark jitter without lag.
  - Tracking loss triggers a clean 0.4s grace period with smooth fadeout — **never leaves stale or frozen regions on screen**.
  - **Live tracking by default**: Region follows hands continuously unless the user deliberately presses `L` to lock.
- **Strict Mask Confinement & Feathering**:
  - Hand effects are applied **strictly inside** the hand-controlled polygon mask.
  - Outside pixels remain **100% untouched** (verified by pixel-difference testing).
- **Face Mesh-Anchored Character Overlays**:
  - MediaPipe Face Mesh extracts eye centers, face center, width, and in-plane head roll rotation angle.
  - Masks and eye lens overlays rotate and scale dynamically with head movement.
  - Supports full Spider-Man masks, classic eye lenses, web overlays, iron hero faceplates, visors, and chest emblems.
- **4 Precise Interaction Modes**:
  - **MODE A — Local Effect**: Both hands control a polygon region; selected visual effect renders inside with feathered boundaries. Outside is 100% original webcam feed.
  - **MODE B — Scene Portal**: Virtual background is warped directly into the hand-defined polygon with 4-point perspective transform (`cv2.warpPerspective`). Outside is 100% original webcam feed.
  - **MODE C — Character Transformation**: Virtual background replaces the room (person segmented via Selfie Segmentation), and character overlay is anchored to facial landmarks.
  - **MODE D — Combined AR**: Full virtual background replacement + segmented person + character overlay + two hands controlling an active transformation portal/effect.
- **Hands-Free Gesture Recognition**:
  - 👋 **Swipe Left / Right**: Instant switch to previous or next visual effect.
  - ✌️ **Peace / V-Sign**: Cycle through Modes A, B, C, and D.
  - 🤏 **Dual Pinch**: Deliberate activation / fine-tuning gesture (keeps tracking live).
  - ✋ **Open Palms**: Re-engage live dynamic tracking.

---

## 📁 Project Architecture

```
spidermaneffect/
├── main.py                     # CLI launcher & desktop application entry point
├── app.py                      # Root convenience forwarding entry point (auto venv delegation)
├── run.bat                     # Windows double-click launcher
├── requirements.txt            # Pinned, mutually compatible Python dependencies
├── README.md                   # Complete documentation, setup & gesture guide
├── .gitignore                  # Git exclusions for environments, outputs & caches
├── config/
│   ├── gestures.json           # Gesture recognition thresholds, hold times & cooldowns
│   └── effects.json            # Effect configuration catalog, feathering & opacity
├── src/
│   ├── __init__.py             # Package initializer
│   ├── camera.py               # DirectShow/OpenCV capture & real-time FPS engine
│   ├── hand_tracker.py         # Dual-hand tracking, spatial continuity & adaptive EMA smoothing
│   ├── gesture_recognizer.py   # Two-hand gesture recognition, swipe detection & debouncing
│   ├── region_controller.py    # Convex quad sorting, scale expansion & tracking-loss grace fadeout
│   ├── mask_renderer.py        # Strict alpha blending, feathered masking & glowing HUD outline
│   ├── effect_manager.py       # Procedural effects & 4-point perspective warping
│   ├── person_segmenter.py     # MediaPipe Selfie Segmentation with temporal halo reduction
│   ├── background_compositor.py # Virtual background scene replacement & 4-point perspective portals
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
    └── test_pipeline.py        # Automated 7-suite verification test script
```

---

## 🚀 Installation & Quick Start (Windows PowerShell)

### 1. Launch the Application

You can launch using any of these commands:

```powershell
# Using the root app.py (automatically detects and uses the venv)
py app.py

# Or using main.py
py main.py

# Or run directly via venv python
.\venv\Scripts\python.exe main.py

# Or double-click run.bat in File Explorer
.\run.bat
```

### 2. Run Automated Verification Tests

Verify all 7 test suites (tracking, resizing, isolation, warping, segmentation, overlays, recording):

```powershell
.\venv\Scripts\python.exe tools/test_pipeline.py
```

---

## 🖐️ Hands-Free Gesture Interaction Guide

| Gesture | Action | Visual Feedback |
| :--- | :--- | :--- |
| **Both Hands in View** | **Defines Active Region** | Animated quadrilateral frames the area between hands |
| **Move Hands Apart / Together** | **Enlarge / Shrink Region** | Region dynamically scales outward/inward |
| **Move Hands Up / Down / Across** | **Follows Hand Position** | Region moves smoothly with hands in real time |
| **👋 Swipe Hand Right** | **Next Effect ▶** | Cycles to next visual effect |
| **👋 Swipe Hand Left** | **◀ Previous Effect** | Cycles to previous visual effect |
| **✌️ Peace / V-Sign** | **Cycle Scene Mode** | Cycles through Mode A → Mode B → Mode C → Mode D |
| **✋ Open Palms** | **Dynamic Follow** | Resumes live tracking |

---

## ⌨️ Accessibility & Controls Keyboard Shortcuts

| Shortcut | Action | Description |
| :--- | :--- | :--- |
| `F11` / `F` | **Toggle Fullscreen** | Expands canvas edge-to-edge |
| `D` | **Toggle Debug Mode** | Shows/hides hand skeletons, polygon outlines, and FPS diagnostics |
| `1`, `2`, `3`, `4` | **Select Mode** | Selects Mode A, B, C, or D directly |
| `M` | **Cycle Mode** | Cycles Mode A → B → C → D |
| `◀` / `▶` Left / Right | **Previous / Next Effect** | Cycles through active visual effects / textures |
| `N` / `P` | **Next / Prev Effect** | Alternative effect navigation |
| `B` | **Next Scene Background** | Cycles through virtual environments (rooftop, city, lab, etc.) |
| `C` | **Next Character Mask** | Cycles through face masks (Spider-Man, Eyes, Iron Hero, Visor, etc.) |
| `L` | **Toggle Lock / Freeze** | Explicitly freezes region in place or resumes live tracking |
| `Spacebar` | **Capture Snapshot** | Saves high-res photo to `output/` with shutter flash |
| `R` | **Record Video** | Starts / stops recording video to `output/` |
| `F5` | **Reload Assets** | Re-scans `assets/` folders for newly added images |
| `ESC` | **Exit** | Exits fullscreen or closes application cleanly |

---

## 🐛 Root Causes of Previous Bugs & Fixes Implemented

1. **Accidental Locking Bug (Frozen Region)**:
   - *Cause*: In `gesture_recognizer.py`, any relaxed hand with curled fingers was classified as a `FIST`, which triggered `toggle_freeze`. Once frozen, `region_controller.py` returned static coordinates on every frame, ignoring new hand movements. Furthermore, `config/gestures.json` previously mapped `dual_pinch` to `toggle_lock_region`.
   - *Fix*: Removed automatic freeze triggers from relaxed hand movement. Mapped `dual_pinch` to live continuous control. Made region locking an **explicit user command** (`L` key), ensuring tracking remains live by default.
2. **Hand Track Cross-Wiring**:
   - *Cause*: Sorting hands purely by horizontal position caused identity flips when hands crossed or when one hand moved across the screen center, blending smoothed history across different hands.
   - *Fix*: Implemented spatial track continuity in `hand_tracker.py` with Euclidean distance association, maintaining consistent tracking history for each hand.
3. **Display Scaling Typo**:
   - *Cause*: In `app.py`, `scale = min(cw / fw, ch / ch)` had a typo (`ch / ch = 1.0`), preventing the video canvas from scaling up to fill the window in fullscreen mode.
   - *Fix*: Corrected to `scale = min(cw / fw, ch / fh)` with aspect-ratio preserving window fill.
4. **Interface Dashboard Clutter**:
   - *Cause*: Heavy header bars, outlines, and diagnostic skeletons were displayed constantly.
   - *Fix*: Replaced with a pure, immersive AR presentation. Polygon outlines, skeletons, and telemetry are hidden by default and only visible when Debug Mode (`D`) is active.
5. **Person Transformation (Mode C & Mode D)**:
   - *Cause*: Face overlays and backgrounds were previously disconnected from hand interactions.
   - *Fix*: Integrated 4-layer compositing: virtual background, segmented person silhouette, Face Mesh-anchored character masks with roll angle rotation, and hand-controlled transformation portals.
