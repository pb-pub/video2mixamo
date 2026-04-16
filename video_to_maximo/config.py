# Video to Maximo - Configuration

"""
MediaPipe Pose Landmarker models.
Download URLs for the .task model files.

Models are hosted on Google Cloud Storage at:
https://storage.googleapis.com/mediapipe-models/pose_landmarker/

Each model has a subdirectory structure:
  pose_landmarker_full/float16/latest/pose_landmarker_full.task
  pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
  pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
"""

POSE_LANDMARKER_FULL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
POSE_LANDMARKER_LITE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
POSE_LANDMARKER_HEAVY_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

# Default model to use
DEFAULT_MODEL = "pose_landmarker_full.task"
DEFAULT_MODEL_URL = POSE_LANDMARKER_FULL_URL

# Model directory (gitignored)
MODEL_DIR = "models"

# Default FPS for video capture
DEFAULT_FPS = 30

# Smoothed output BVH frame time (1/30 second)
BVH_FRAME_TIME = 1.0 / 30.0
