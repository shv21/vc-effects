"""
Spider-Man AR Studio - Main Application Launcher
Usage:
    python main.py                  # Launch desktop GUI
    python main.py --camera 1       # Use camera device 1
    python main.py --scan-cameras   # Discover connected webcams
    python main.py --list-filters   # List all registered filters
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
    except Exception as e:
        # If delegation fails, continue with current interpreter
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
from src.filter_manager import FilterManager
from src.app import ARStudioApp


def setup_logging(verbose: bool = False) -> None:
    """Configures console and file logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )


def print_banner() -> None:
    """Prints a Spider-Man AR Studio ASCII header."""
    banner = r"""
  ___ ___ ___ ___  ___ ___   __  __   _   _  _     _   ___    ___ _____ _   _ ___ ___ ___  
 / __| _ \_ _|   \| __| _ \ |  \/  | /_\ | \| |   /_\ | _ \  / __|_   _| | | |   \_ _/ _ \ 
 \__ \  _/| || |) | _||   / | |\/| |/ _ \| .` |  / _ \|   /  \__ \ | | | |_| | |) | | (_) |
 |___/_| |___|___/|___|_|_\ |_|  |_/_/ \_\_|\_| /_/ \_\_|_\  |___/ |_|  \___/|___/___\___/ 
                     === REAL-TIME FACE FILTER ENGINE ===
    """
    print(banner)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spider-Man AR Studio - Real-Time Custom Face Filters"
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
        help="Directory to save photos and recorded videos (default: output)"
    )
    parser.add_argument(
        "--scan-cameras",
        action="store_true",
        help="Scan and list working webcam device indices"
    )
    parser.add_argument(
        "--list-filters",
        action="store_true",
        help="List all discovered built-in and PNG filters"
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
            print(f"To use a specific camera: python main.py --camera {found[0]}")
        else:
            print("No accessible webcams detected. Check USB connection and Windows privacy permissions.")
        return

    if args.list_filters:
        print("Discovering active filters...")
        fm = FilterManager()
        print(f"\nDiscovered {len(fm.filters)} Filters:")
        for idx, f in enumerate(fm.filters):
            print(f"  [{idx:02d}] {f.display_name} ({f.filter_type}) - {f.category}")
        return

    print("Launching Spider-Man AR Studio GUI...")
    print(f"Camera index: {args.camera}")
    print(f"Output directory: {args.output}")
    print("Shortcuts:")
    print("  [Left/Right] : Previous / Next Filter")
    print("  [Space]      : Capture Snapshot Photo")
    print("  [R]          : Start / Stop Recording Video")
    print("  [0]          : Reset to 'No Filter'")
    print("  [F5]         : Refresh Filters from assets/filters/")
    print("  [ESC]        : Exit Studio\n")

    app = ARStudioApp(camera_index=args.camera, output_dir=args.output)
    app.run()


if __name__ == "__main__":
    main()
