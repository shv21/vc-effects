"""
Convenience entry point forwarding to main.py
Allows running 'python app.py' directly from the root workspace directory.
"""

import sys
from pathlib import Path

# Add project root and launch main
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import main

if __name__ == "__main__":
    main()
