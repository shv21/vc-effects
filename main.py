"""
Gesture-Controlled AR Experience - Application Launcher
Usage:
    python main.py                  # Launch desktop AR Studio
    python main.py --camera 0       # Select webcam index
    python main.py --scan-cameras   # Discover connected webcams
    python main.py --list-effects   # List available hand-controlled effects
    python main.py --list-scenes    # List available virtual backgrounds
"""

import sys
import os
import subprocess
from pathlib import Path

# Automatic venv delegation:
# If user runs 'py main.py' or 'python main.py' without activating the venv,
# automatically relaunch with .\venv\Scripts\python.exe where dependencies are installed.
ROOT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = ROOT_DIR / "venv" / "Scripts" / "python.exe"

if VENV_PYTHON.exists():
    try:
        current_exe = Path(sys.executable).resolve()
        target_exe = VENV_PYTHON.resolve()
        if current_exe != target_exe:
            exit_code = subprocess.call([str(target_exe)] + sys.argv)
            sys.exit(exit_code)
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
from src.background_compositor import BackgroundCompositor
from src.app import GestureARStudioApp


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
             === HAND-DEFINED REGION & SCENE TRANSFORMATION ===
    """
    print(banner)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gesture-Controlled AR Experience with Hand-Defined Effects & Background Transformation"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0)"
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
        "--list-effects",
        action="store_true",
        help="List all active hand-controlled effects and textures"
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List all discovered virtual backgrounds"
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

    if args.list_effects:
        em = EffectManager(effects_dir="assets/effects", textures_dir="assets/textures")
        print(f"\nDiscovered {len(em.effects)} Hand-Controlled Effects:")
        for idx, eff in enumerate(em.effects):
            print(f"  [{idx:02d}] {eff.display_name} ({eff.effect_type}) - {eff.category}")
        return

    if args.list_scenes:
        bc = BackgroundCompositor(backgrounds_dir="assets/backgrounds")
        print(f"\nDiscovered {len(bc.background_files)} Virtual Scene Backgrounds:")
        for idx, p in enumerate(bc.background_files):
            print(f"  [{idx:02d}] {p.stem.replace('_', ' ').title()} ({p.name})")
        return

    print("Launching Gesture-Controlled AR Studio...")
    print(f"Camera index: {args.camera}")
    print(f"Output folder: {args.output}")
    print("\nPrimary Hands-Free Gestures:")
    print("  👋 Swipe Right / Left : Next / Previous Effect")
    print("  🤏 Dual Pinch         : Lock / Unlock Region")
    print("  ✌️  Peace / V-Sign     : Cycle Scene Mode (A -> B -> C -> D -> E)")
    print("  ✋ Open Palms         : Dynamic Region Follow")
    print("  ✊ Fist               : Freeze Region Position\n")

    app = GestureARStudioApp(camera_index=args.camera, output_dir=args.output)
    app.run()


if __name__ == "__main__":
    main()
