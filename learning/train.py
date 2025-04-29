"""
Simple script to run the training process using the modular code.
"""

import os
import sys
import hydra
from pathlib import Path

current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from main import main


if __name__ == "__main__":
    sys.argv[0] = os.path.basename(__file__)

    os.chdir(current_dir)

    main()

