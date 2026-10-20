"""
Gesture-Controlled AR Experience - Application Launcher
Two-Hand-Controlled Full-Scene AR Portal

Usage:
    python main.py                     # Launch Two-Hand Scene Portal AR Studio
    python main.py --camera 0          # Select webcam index
    python main.py --activation pinch  # Activation rule ('pinch', 'framing', 'both')
    python main.py --scan-cameras      # Discover connected webcams
    python main.py --list-scenes       # List available complete scene images
    python main.py --list-effects      # List available hand-controlled effects
"""

import sys
import os
import subprocess
import warnings
from pathlib import Path

# Suppress noisy deprecation warnings from third-party protobuf
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# Automatic venv delegation:
ROOT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = ROOT_DIR / "venv" / "Scripts" / "python.exe"

if VENV_PYTHON.exists():
    try:
        current_exe = Path(sys.executable).resolve()
        target_exe = VENV_PYTHON.resolve()
        if current_exe != target_exe:
            try:
                exit_code = subprocess.call([str(target_exe)] + sys.argv)
                sys.exit(exit_code)
            except KeyboardInterrupt:
                sys.exit(0)
    except Exception:
        pass

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import logging

try:
    import cv2
    import mediapipe
    import customtkinter
    from PIL import Image
except ImportError as err:
    print(f"\n[ERROR] Missing required library: {err}")
    print("\nPlease ensure you are running within the virtual environment:")
    print("    .\\venv\\Scripts\\activate")
    print("    pip install -r requirements.txt")
    print("Or run directly using:")
    print("    .\\venv\\Scripts\\python.exe main.py\n")
    sys.exit(1)

from src.camera import Camera
from src.effect_manager import EffectManager
from src.scene_manager import ScenePortalManager, AspectRatioFitMode
from src.region_controller import ActivationGestureRule
from src.app import ImmersiveARStudioApp


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )


def print_banner() -> None:
    banner = r"""
  ___ ___ ___ _____ _   _ ___ ___      _   ___    ___ _____ _   _ ___ ___ ___  
 / __| __/ __|_   _| | | | _ \ __|    /_\ | _ \  / __|_   _| | | |   \_ _/ _ \ 
| (_ | _|\__ \ | | | |_| |   / _|    / _ \|   /  \__ \ | | | |_| | |) | | (_) |
 \___|___|___/ |_|  \___/|_|_\___|  /_/ \_\_|_\  |___/ |_|  \___/|___/___\___/ 
          === TWO-HAND-CONTROLLED FULL-SCENE AR PORTAL ===
    """
    print(banner)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-Hand-Controlled Full-Scene AR Portal (Character + Background Scene inside Hand Quad)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0)"
    )
    parser.add_argument(
        "--scenes-dir",
        type=str,
        default="assets/scenes",
        help="Directory containing user scene images (default: assets/scenes)"
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=["pinch", "framing", "both"],
        default="pinch",
        help="Activation gesture rule: 'pinch' (dual pinch), 'framing', 'both' (default: pinch)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for photos and recordings (default: output)"
    )
    parser.add_argument(
        "--scan-cameras",
        action="store_true",
        help="Scan and list working webcam device indices"
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List all discovered complete scene images in assets/scenes/"
    )
    parser.add_argument(
        "--list-effects",
        action="store_true",
        help="List all active hand-controlled effects"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    print_banner()

    if args.scan_cameras:
        print("Scanning for connected webcams (indices 0..4)...")
        found = Camera.scan_available_cameras(max_tested=5)
        if found:
            print(f"Connected camera indices detected: {found}")
            print(f"To launch with a specific camera: python main.py --camera {found[0]}")
        else:
            print("No webcams detected. Check connection and Windows privacy permissions.")
        return

    if args.list_scenes:
        sm = ScenePortalManager(scenes_dir=args.scenes_dir)
        print(f"\nDiscovered {len(sm.scene_items)} Complete Scene Images in {args.scenes_dir}:")
        for idx, item in enumerate(sm.scene_items):
            print(f"  [{idx:02d}] {item.name} ({item.width}x{item.height}, Alpha: {item.has_alpha}) - {item.file_path.name}")
        return

    if args.list_effects:
        em = EffectManager(effects_dir="assets/effects", textures_dir="assets/textures")
        print(f"\nDiscovered {len(em.effects)} Hand-Controlled Effects:")
        for idx, eff in enumerate(em.effects):
            print(f"  [{idx:02d}] {eff.display_name} ({eff.effect_type}) - {eff.category}")
        return

    # Map activation rule string
    rule_map = {
        "pinch": ActivationGestureRule.DUAL_PINCH,
        "framing": ActivationGestureRule.FRAMING,
        "both": ActivationGestureRule.BOTH_HANDS
    }
    activation_rule = rule_map.get(args.activation, ActivationGestureRule.DUAL_PINCH)

    print("Launching Two-Hand-Controlled Full-Scene AR Portal...")
    print(f"Camera index      : {args.camera}")
    print(f"Scenes directory  : {args.scenes_dir}")
    print(f"Activation rule   : {args.activation.upper()} (Thumb & Index)")
    print(f"Output folder     : {args.output}")

    print("\nHow to Use:")
    print("  1. Present both hands to the webcam.")
    print("  2. Pinch thumb and index finger of each hand (or use [G] to switch gesture rule).")
    print("  3. The full scene image appears perspective-warped strictly between your hands.")
    print("  4. Move hands apart to enlarge; bring hands closer to shrink.")
    print("  5. Move hands up/down/left/right to steer the portal.")
    print("  6. Release pinch to close portal. Normal live camera remains 100% untouched outside!")

    print("\nKeyboard Shortcuts:")
    print("  [S] / [Right] : Next Scene")
    print("  [Left]        : Previous Scene")
    print("  [A]           : Cycle Fit Mode (Crop [Fill] / Letterbox / Stretch)")
    print("  [G]           : Cycle Gesture Rule (Dual Pinch / Framing / Both Hands)")
    print("  [O]           : Open File Dialog (Load any custom scene image)")
    print("  [F5]          : Reload Scenes from Disk")
    print("  [D]           : Toggle Debug Mode (Landmarks, Corner Nodes, Telemetry)")
    print("  [F11 / F]     : Toggle Fullscreen")
    print("  [Space]       : Snapshot Photo")
    print("  [R]           : Start/Stop Video Recording")
    print("  [1 - 4]       : Jump to Mode (1: Portal [Default], 2: Effect, 3: Character, 4: Combined)")
    print("  [M]           : Cycle Mode")
    print("  [Esc]         : Exit Fullscreen / Exit Application\n")

    try:
        app = ImmersiveARStudioApp(
            camera_index=args.camera,
            output_dir=args.output,
            scenes_dir=args.scenes_dir,
            activation_rule=activation_rule
        )
        app.run()
    except KeyboardInterrupt:
        print("\n[INFO] Studio closed cleanly.")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Studio closed cleanly.")
        sys.exit(0)
