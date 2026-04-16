# Video to Mixamo

Convert video of a person into a BVH motion capture animation using MediaPipe Pose detection.

## Overview

This tool captures video from a webcam or file, detects body pose landmarks using MediaPipe, and exports the results as a BVH (Biovision Hierarchical) animation file that can be imported into 3D animation software like Blender, Mixamo, or Unity.

## Features

- **Video Capture**: Support for webcam or video file input
- **Pose Detection**: Uses MediaPipe Pose Landmarker for accurate body tracking
- **BVH Export**: Generates standard BVH files compatible with most 3D animation tools
- **Model Selection**: Choose between Lite, Full, or Heavy MediaPipe models
- **Temporal Smoothing**: One Euro Filter and Savitzky-Golay filter for smooth animations

## Installation

### From PyPI (when published)

```bash
pip install video2mixamo
```

### From Source

1. Clone the repository:

```bash
git clone https://github.com/pb-pub/video2mixamo.git
cd video-to-maximo
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:

```bash
pip install -e .
```

4. Download a MediaPipe pose model:

```bash
python -m scripts.download_models
```

Options: `full` (default), `lite`, `heavy`, or `all`

## Usage

### Command Line

```bash
# Start webcam capture with preview
python -m video_to_maximo.main

# Use specific camera
python -m video_to_maximo.main --camera 1

# Process video file
python -m video_to_maximo.main --input video.mp4

# Process video and export to specific file
python -m video_to_maximo.main --input video.mp4 --output animation.bvh

# Use custom model
python -m video_to_maximo.main --model models/pose_landmarker_full.task

# Disable smoothing
python -m video_to_maximo.main --no-smooth
```

### Controls (in preview window)

- **R**: Start/Stop recording
- **S**: Stop recording (keep window open)
- **ESC**: Quit without saving

### Python API

```python
from video_to_maximo.capture import VideoCapture
from video_to_maximo.detector import PoseLandmarker
from video_to_maximo.filter import Smoother, FilterConfig
from video_to_maximo.rotation import RotationComputer
from video_to_maximo.exporter_bvh import BVHExporter

# Capture video from webcam
with VideoCapture(camera_id=0) as capture:
    detector = PoseLandmarker(model_path="models/pose_landmarker_full.task")
    smoother = Smoother()
    exporter = BVHExporter()
    
    for frame, timestamp_ms in capture:
        # Detect pose
        result = detector.detect(frame, timestamp_ms)
        
        if result.success:
            # Smooth landmarks
            smoothed = smoother.filter_landmarks(result.pose_world_landmarks, timestamp_ms)
            
            # Compute rotations
            computer = RotationComputer()
            rotations = computer.compute_rotations(smoothed, timestamp_ms)
            
            # Export frame
            # ... add to exporter ...
```

## Project Structure

```
video-to-maximo/
├── video_to_maximo/         # Main package
│   ├── __init__.py
│   ├── capture.py          # Video capture (webcam/file)
│   ├── config.py           # Configuration and constants
│   ├── detector.py         # MediaPipe pose detection
│   ├── exporter_bvh.py     # BVH file generation
│   ├── filter.py           # Temporal smoothing filters
│   ├── main.py             # CLI entry point
│   ├── rotation.py         # Bone rotation computation
│   └── skeleton.py         # Skeleton hierarchy definition
├── scripts/                 # Utility scripts
│   └── download_models.py  # Model download utility
├── models/                  # MediaPipe models (gitignored)
├── output/                  # Generated BVH files
├── pyproject.toml          # Package configuration
├── requirements.txt        # Python dependencies
└── README.md
```

## Dependencies

- **OpenCV**: Video capture and image processing
- **MediaPipe**: Pose detection
- **NumPy**: Array operations
- **SciPy**: Signal processing and smoothing

## Output Format

Exports BVH files with standard hierarchy:
```
HIERARCHY
  ROOT
    ├── Hips
    │   ├── LeftUpLeg
    │   │   └── LeftLeg
    │   │       └── LeftFoot
    │   ├── RightUpLeg
    │   │   └── RightLeg
    │   │       └── RightFoot
    │   ├── Spine
    │   │   ├── Spine1
    │   │   │   └── Spine2
    │   │   │       ├── LeftShoulder
    │   │   │       │   └── LeftArm
    │   │   │       │       └── LeftForeArm
    │   │   │       │           └── LeftHand
    │   │   │       └── RightShoulder
    │   │   │           └── RightArm
    │   │   │               └── RightForeArm
    │   │   │                   └── RightHand
    │   │   └── Neck
    │   │       └── Head
```

## License

MIT