# Video to Maximo

Convert video of a person into a BVH motion capture animation using MediaPipe Pose detection.

## Overview

This tool captures video from a webcam or file, detects body pose landmarks using MediaPipe, and exports the results as a BVH (Biovision Hierarchical) animation file that can be imported into 3D animation software like Blender, Maximo, or Unity.

## Features

- **Video Capture**: Support for webcam or video file input
- **Pose Detection**: Uses MediaPipe Pose Landmarker for accurate body tracking
- **BVH Export**: Generates standard BVH files compatible with most 3D animation tools
- **Model Selection**: Choose between Lite, Full, or Heavy MediaPipe models

## Installation

1. Clone or download this repository

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download a MediaPipe pose model:
```bash
python scripts/download_models.py
```

Options: `full` (default), `lite`, `heavy`, or `all`

## Usage

### Basic Example

```python
from src.capture import VideoCapture
from src.pose import PoseDetector
from src.bvh import BVHWriter

# Capture video from webcam
with VideoCapture(camera_id=0) as capture:
    detector = PoseDetector("models/pose_landmarker_full.task")
    bvh = BVHWriter()
    
    for frame, timestamp_ms in capture:
        # Detect pose
        pose_result = detector.detect(frame)
        
        # Add to BVH animation
        bvh.add_frame(pose_result)
        
        # Break after some frames or when satisfied
        if capture.frame_count > 300:  # 10 seconds at 30fps
            break
    
    # Export to BVH file
    bvh.export("output.bvh")
```

### Command Line (when implemented)

```bash
python -m src.main --video input.mp4 --output output.bvh
python -m src.main --camera 0 --output output.bvh
```

## Project Structure

```
video-to-maximo/
├── src/
│   ├── __init__.py      # Package initialization
│   ├── capture.py       # Video capture (webcam/file)
│   ├── config.py        # Configuration and constants
│   ├── pose.py          # MediaPipe pose detection
│   └── bvh.py           # BVH file generation
├── models/              # MediaPipe models (gitignored)
├── scripts/
│   └── download_models.py  # Model download utility
├── requirements.txt     # Python dependencies
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
ROOT
├── Hips
│   ├── LeftUpperLeg
│   │   └── LeftLowerLeg
│   │       └── LeftFoot
│   ├── RightUpperLeg
│   │   └── RightLowerLeg
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
