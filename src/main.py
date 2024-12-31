#main.py

import os
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent
sys.path.append(str(src_path))

from gui.main_window import MainWindow

def main():
    """Main entry point for the Quantum Stack Encryption application"""
    try:
        app = MainWindow()
        app.run()
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()