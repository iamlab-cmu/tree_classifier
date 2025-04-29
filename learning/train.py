#!/usr/bin/env python
"""
Simple script to run the training process using the modular code.
"""
import os
import sys
import hydra
from pathlib import Path

# Add the parent directory to sys.path so we can import main
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from main import main


if __name__ == "__main__":
    # Pass command-line arguments to the main function
    sys.argv[0] = os.path.basename(__file__)
    
    # Set configuration search path to be relative to this script
    # This helps hydra find the conf directory
    os.chdir(current_dir)
    
    main() 