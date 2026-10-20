# 🌀 Two-Hand-Controlled Full-Scene AR Portal

A high-performance, real-time Augmented Reality application built in Python using **OpenCV**, **MediaPipe (Hands, Selfie Segmentation, Face Mesh)**, **NumPy**, and **CustomTkinter**.

---

## 🌟 What is the Full-Scene AR Portal?

This is **NOT a face filter** and **NOT full-screen background replacement**.

The user selects a complete scene image that contains **BOTH a character and a background** (for example, an anime character in a black hoodie inside a modern building, a superhero on a city rooftop, or a fantasy warrior in a glowing shrine).

When the user presents both hands to the webcam:
- The **ENTIRE selected scene image** (character + background) appears perspective-warped **strictly inside the region enclosed by the hands**.
- Outside the hand-defined region, the original live webcam video remains **100% UNCHANGED** (exact pixel isolation).
- Moving your hands moves the entire scene portal in real time.
- Spreading your hands apart enlarges the portal.
- Bringing your hands closer shrinks it.
- Moving individual fingertips alters the portal's shape.
- When the activation gesture is released or tracking is lost, the portal disappears smoothly without freezing at old coordinates.

---

## 🖐️ Hand Control & 4-Point Geometry

1. **MediaPipe Hand Landmark Tracking**:
   - Detects both hands in every frame in mirrored camera coordinates.
   - Extracts four live control points from the **Thumb tips** and **Index fingertips** of the Left and Right hands.
2. **Topological Corner Ordering `[TL, TR, BR, BL]`**:
   - Computes the inter-hand axis and normal vector to guarantee Top-Left, Top-Right, Bottom-Right, and Bottom-Left assignments remain invariant under rotation or tilt.
   - Prevents self-intersecting or twisted (bowtie) polygons.
3. **Pinch & Framing Geometry**:
   - When pinching, an anatomical span expands around the pinch point to prevent the quad from collapsing into a 2-point line, allowing users to "hold" the portal between two pinches.
   - When framing (fingers open), the 4 fingertips directly define the 4 corners of the quad.
4. **Configurable Activation Gesture Rules**:
   - **`dual_pinch`** *(Default)*: Both hands pinching thumb and index finger activates and maintains the portal. Releasing pinch closes the portal.
   - **`framing`**: Both hands framing with thumb and index fingers extended activates the portal.
   - **`both_hands`**: Any two hands in view activate the portal.
   - Cycle gesture rules on the fly with the **`[G]`** key.
5. **Low-Latency Adaptive Filtering**:
   - Velocity-adaptive EMA smoothing eliminates landmark jitter while stationary and delivers zero lag during rapid hand movements.
   - Instant reacquisition with zero drift when hands return to view.

---

## 🖼️ Complete Scene Images & Loading (`assets/scenes/`)

The application discovers all `.jpg`, `.jpeg`, `.png`, and `.webp` images in `assets/scenes/`:

| Scene Image | Description | Format |
| :--- | :--- | :--- |
| `anime_hoodie_building.jpg` | Anime character wearing a black hoodie in a modern building with floor-to-ceiling glass | JPEG (1280x720) |
| `superhero_city.jpg` | Superhero standing on a rooftop overlooking a vibrant dusk city skyline | JPEG (1280x720) |
| `fantasy_portal.jpg` | Mage / Warrior character in an ancient stone shrine with glowing runes | JPEG (1280x720) |
| `solo_leveling.jpg` | Full character standing in dramatic fantasy environment | JPEG (736x1308) |
| `cyber_warrior.png` | Cybernetic character with semi-transparent tech arena (Alpha Channel) | PNG RGBA (1280x720) |

### Image Fit Modes (Toggle with `[A]`):
- **`CROP`** *(Default)*: Preserves source aspect ratio by center-cropping margins. Fills the portal without black bars while keeping the character visible.
- **`LETTERBOX`**: Preserves aspect ratio by padding margins with black borders. The entire image is visible inside the portal.
- **`STRETCH`**: Directly maps the 4 image corners to the 4 hand control points.

### Loading Custom Images:
- **Drop files directly**: Add any `.jpg`, `.jpeg`, `.png`, or `.webp` into `assets/scenes/` and press **`[F5]`** to reload.
- **Open File Dialog**: Press **`[O]`** anytime to open a standard file browser and select any image on your computer.

---

## 🔒 Strict Pixel Isolation (Requirement 3)

1. The live camera frame is the base image.
2. The complete scene is warped into a separate image using `cv2.warpPerspective`.
3. A single-channel mask is created strictly from the hand-controlled quadrilateral polygon.
4. The warped scene is composited onto the camera frame using this mask.
5. Outside the polygon mask (with edge feathering disabled), **the camera frame is 100% mathematically unchanged** (max difference = 0).

---

## 🖥️ Minimal Interface (Requirement 7)

- **Edge-to-Edge Clean Presentation**:
  - Fullscreen compatible (`F11` or `F`).
  - No large telemetry dashboards or header panels.
  - Skeletons, landmark dots, polygon outlines, and FPS counters are **hidden by default**.
- **Debug Mode (`[D]`)**:
  - Toggles hand skeletons, corner nodes (`TL`, `TR`, `BR`, `BL`), polygon boundary outlines, and live FPS diagnostics on demand.
- **Subtle On-Screen Guidance**:
  - Non-intrusive instruction pill at bottom appears only when hands are not in view.
  - Floating toast notifications appear briefly on scene or mode switches and automatically fade out.

---

## ⌨️ Keyboard Shortcuts

| Key | Action | Description |
| :--- | :--- | :--- |
| **`S`** / **`▶`** | **Next Scene** | Cycles to next complete scene in `assets/scenes/` |
| **`◀`** | **Previous Scene** | Cycles to previous scene in `assets/scenes/` |
| **`A`** | **Cycle Fit Mode** | Cycles through `Crop` (fill) → `Letterbox` → `Stretch` |
| **`G`** | **Cycle Gesture Rule** | Cycles through `Dual Pinch` → `Framing` → `Both Hands` |
| **`O`** | **Open Custom Scene** | Opens file dialog to load any local image from disk |
| **`F5`** | **Reload Scenes** | Re-scans `assets/scenes/` for newly added files |
| **`D`** | **Toggle Debug Mode** | Shows/hides hand landmarks, corner labels & FPS |
| **`F11`** / **`F`** | **Toggle Fullscreen** | Fills screen edge-to-edge |
| **`Space`** | **Snapshot Photo** | Saves high-res photo to `output/` with shutter flash |
| **`R`** | **Record Video** | Starts / stops video recording to `output/` |
| **`L`** | **Toggle Lock / Freeze** | Explicitly locks portal in place or resumes live tracking |
| **`1`** – **`4`** | **Select AR Mode** | Jump to Mode 1 (Portal), 2 (Effect), 3 (Character), 4 (Combined) |
| **`M`** | **Cycle AR Mode** | Cycles through available interaction modes |
| **`Esc`** | **Exit** | Exits fullscreen or closes application |

---

## 🚀 Running the Application

### 1. Launch with Default Settings (Full-Scene Portal Mode)
```powershell
# Using the automatic venv launcher
py main.py

# Or run directly via venv python
.\venv\Scripts\python.exe main.py
```

### 2. Launch Options & CLI Flags
```powershell
# Select a specific webcam index
python main.py --camera 1

# Launch with a specific activation gesture rule
python main.py --activation pinch      # Dual pinch to activate (default)
python main.py --activation framing    # Frame with spread fingers to activate
python main.py --activation both       # Any 2 hands activate immediately

# List discovered scenes in assets/scenes/
python main.py --list-scenes

# Discover connected webcam indices
python main.py --scan-cameras
```

---

## 🧪 Automated Testing

### Run Comprehensive Test Suite:
```powershell
.\venv\Scripts\python.exe tools/test_pipeline.py
```
This runs 10 automated test suites verifying:
1. Four-point extraction from thumb & index fingertips of both hands.
2. Live portal position updates & continuous translation follow.
3. Portal dynamic resizing (hands apart = enlarge, hands close = shrink).
4. Corner ordering and OpenCV perspective transformation.
5. Degenerate & self-intersecting polygon handling (safely hides portal, 0 crashes).
6. Scene switching, in-memory caching, reloading, and fit modes (`Crop`, `Letterbox`, `Stretch`).
7. RGBA alpha transparency preservation inside the portal.
8. Strict pixel isolation (guaranteed 0 altered pixels outside the polygon).
9. Tracking loss cleanup (grace period fadeout, 0 stale freeze) & instant reacquisition.
10. Configurable gesture activation rules (`dual_pinch`, `framing`, `both_hands`).

### Run End-to-End Integration Suite:
```powershell
.\venv\Scripts\python.exe tools/test_integration.py
```

---

## 📹 Manual Webcam Testing Steps

If running with a physical webcam:
1. Launch the app: `.\venv\Scripts\python.exe main.py`
2. Bring both hands in front of the camera.
3. **Pinch**: Touch thumb and index finger on your left hand, and thumb and index finger on your right hand.
4. Observe the complete scene image (e.g. anime character in building) appear warped between your hands.
5. Move your hands wider apart: the portal enlarges.
6. Bring your hands closer: the portal shrinks.
7. Move your hands up, down, left, right: the portal follows your hands smoothly.
8. Release the pinch: the portal cleanly deactivates/fades out. Notice the camera pixels outside the portal were completely unchanged.
9. Press **`S`** to cycle to the next scene (e.g. superhero city, fantasy shrine).
10. Press **`A`** to switch fit modes (`Crop` -> `Letterbox` -> `Stretch`).
11. Press **`D`** to inspect the 4 corner landmarks (`TL`, `TR`, `BR`, `BL`) and FPS telemetry.
12. Press **`Esc`** to exit.
