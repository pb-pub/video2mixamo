#!/usr/bin/env python3
"""
Download MediaPipe Pose Landmarker models.

Usage:
    python scripts/download_models.py [model_name]

Models:
    full   - pose_landmarker_full.task  (default, ~13 MB)
    lite   - pose_landmarker_lite.task  (~5 MB)
    heavy  - pose_landmarker_heavy.task (~30 MB)

Examples:
    python scripts/download_models.py          # Download full model
    python scripts/download_models.py lite     # Download lite model
    python scripts/download_models.py all      # Download all models
"""

import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

# Add src to path for config import
SCRIPT_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    MODEL_DIR,
    POSE_LANDMARKER_FULL_URL,
    POSE_LANDMARKER_LITE_URL,
    POSE_LANDMARKER_HEAVY_URL,
)

# Map model names to URLs
MODEL_URLS = {
    "full": ("pose_landmarker_full.task", POSE_LANDMARKER_FULL_URL),
    "lite": ("pose_landmarker_lite.task", POSE_LANDMARKER_LITE_URL),
    "heavy": ("pose_landmarker_heavy.task", POSE_LANDMARKER_HEAVY_URL),
}

# Directory where models are stored
MODELS_PATH = Path(__file__).parent.parent / MODEL_DIR


def download_file(url: str, filepath: Path) -> bool:
    """Download a file from url to filepath with progress indicator."""
    if filepath.exists():
        print(f"✓ Model already exists: {filepath}")
        return True

    print(f"Downloading {filepath.name}...")
    print(f"  From: {url}")
    print(f"  To:   {filepath}")

    try:
        urllib.request.urlretrieve(url, filepath)
        file_size = filepath.stat().st_size / (1024 * 1024)  # MB
        print(f"✓ Downloaded: {filepath.name} ({file_size:.2f} MB)")
        return True
    except urllib.error.URLError as e:
        print(f"✗ Download failed: {e.reason}")
        return False
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def main():
    # Create models directory if it doesn't exist
    MODELS_PATH.mkdir(exist_ok=True)

    # Determine which model(s) to download
    if len(sys.argv) > 1:
        model_arg = sys.argv[1].lower()
    else:
        model_arg = "full"

    if model_arg == "all":
        models_to_download = ["full", "lite", "heavy"]
    elif model_arg in MODEL_URLS:
        models_to_download = [model_arg]
    else:
        print(f"Unknown model: {model_arg}")
        print("Available models: full, lite, heavy, all")
        sys.exit(1)

    # Download selected models
    success_count = 0
    total_count = len(models_to_download)

    for model_name in models_to_download:
        filename, url = MODEL_URLS[model_name]
        filepath = MODELS_PATH / filename
        if download_file(url, filepath):
            success_count += 1
        print()

    # Summary
    print(f"Summary: {success_count}/{total_count} model(s) downloaded successfully")

    if success_count < total_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
