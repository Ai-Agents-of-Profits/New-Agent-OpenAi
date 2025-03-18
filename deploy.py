"""
Build and deploy script for the React frontend.
This script builds the React app and moves the build files to the proper location.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

FRONTEND_DIR = Path("frontend")
BUILD_DIR = FRONTEND_DIR / "build"

def main():
    """Build the React app and prepare it for deployment."""
    print("Building React frontend...")
    
    # Check if frontend directory exists
    if not os.path.exists(FRONTEND_DIR):
        print("Frontend directory not found. Please run setup first.")
        return 1
    
    # Build the React app
    os.chdir(FRONTEND_DIR)
    result = subprocess.run(["npm", "run", "build"], check=False)
    os.chdir("..")
    
    if result.returncode != 0:
        print("Build failed. Please check the logs.")
        return 1
    
    print("Build successful!")
    print("Frontend is now ready to be served by the Flask backend.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
