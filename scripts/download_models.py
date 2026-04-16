#!/usr/bin/env python3
"""
Download MediaPipe Pose Landmarker models.

Usage:
    python -m scripts.download_models [model_name]
    python scripts/download_models.py [model_name]  # legacy

Models:
    full   - pose_landmarker_full.task  (default, ~13 MB)
    lite   - pose_landmarker_lite.task  (~5 MB)
    heavy  - pose_landmarker_heavy.task (~30 MB)

Examples:
    python -m scripts.download_models          # Download full model
    python -m scripts.download_models lite     # Download lite model
    python -m scripts.download_models all      # Download all models
"""

import urllib.request
import urllib.error
import sys
from pathlib import Path

# Support both running as module and directly
try:
    from . import config
except (ImportError, ValueError):
    # Fallback for direct script execution (from scripts/ directory)
    SCRIPT_DIR = Path(__file__).parent.parent / "video_to_maximo"
    sys.path.insert(0, str(SCRIPT_DIR))
    import config as config

# Map model names to URLs
MODEL_URLS = {
    "full": ("pose_landmarker_full.task", config.POSE_LANDMARKER_FULL_URL),
    "lite": ("pose_landmarker_lite.task", config.POSE_LANDMARKER_LITE_URL),
    "heavy": ("pose_landmarker_heavy.task", config.POSE_LANDMARKER_HEAVY_URL),
}

# Directory where models are stored
MODELS_PATH = Path(__file__).parent.parent / config.MODEL_DIR


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
