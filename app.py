"""
Convenience entry point forwarding to main.py
Allows running 'python app.py' directly from the root workspace directory.
Automatically delegates to the virtual environment (venv) if executed with system Python.
"""

import sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = ROOT_DIR / "venv" / "Scripts" / "python.exe"

# Automatic delegation to venv if available
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

from main import main

if __name__ == "__main__":
    main()
