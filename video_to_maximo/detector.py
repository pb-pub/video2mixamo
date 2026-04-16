# Video to Mixamo - Pose Detection Module

"""
Pose detection module using MediaPipe PoseLandmarker.

Provides a wrapper around MediaPipe's PoseLandmarker to extract
pose landmarks with visibility scores.

Classes:
    PoseLandmarker: Wrapper around MediaPipe PoseLandmarker
    PoseResult: Data class for detection results
    DetectorError: Custom exception for detector errors
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from .config import MODEL_DIR, DEFAULT_MODEL


@dataclass
class PoseResult:
    """
    Result from pose detection.

    Attributes:
        success: Whether detection was successful
        timestamp_ms: Timestamp of the frame in milliseconds
        pose_landmarks: List of [x, y, z] for each of 33 landmarks
        pose_world_landmarks: List of [x, y, z] in meters (world coordinates)
        visibility: List of visibility scores (0-1) for each landmark
        segmentation_mask: Optional segmentation mask (H, W)
    """

    success: bool
    timestamp_ms: float
    pose_landmarks: Optional[List[List[float]]]  # 33 x [x, y, z]
    pose_world_landmarks: Optional[List[List[float]]]  # 33 x [x, y, z]
    visibility: Optional[List[float]]  # 33 visibility scores
    segmentation_mask: Optional[np.ndarray] = None


class DetectorError(Exception):
    """Custom exception for pose detector errors."""

    pass


class PoseLandmarker:
    """
    Wrapper around MediaPipe PoseLandmarker.

    Usage:
        detector = PoseLandmarker(model_path="models/pose_landmarker_full.task")
        result = detector.detect(frame, timestamp_ms=0)
        if result.success:
            landmarks = result.pose_landmarks
    """

    # MediaPipe landmark indices
    # 0=nose, 11=left_shoulder, 12=right_shoulder, etc.
    NUM_LANDMARKS = 33

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_asset_buffer: Optional[bytes] = None,
        base_options: Optional[BaseOptions] = None,
        running_mode: vision.RunningMode = vision.RunningMode.VIDEO,
        min_pose_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        output_segmentation_masks: bool = False,
    ):
        """
        Initialize the pose landmarker.

        Args:
            model_path: Path to .task model file (required if model_asset_buffer is None)
            model_asset_buffer: Model bytes (required if model_path is None)
            base_options: Optional BaseOptions for custom settings
            running_mode: Running mode (VIDEO, IMAGE, LIVE_STREAM)
            min_pose_detection_confidence: Minimum confidence for detection (default: 0.5)
            min_tracking_confidence: Minimum confidence for tracking (default: 0.5)
            output_segmentation_masks: Whether to output segmentation masks
        """
        self.running_mode = running_mode
        self.min_pose_detection_confidence = min_pose_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Create base options
        if base_options is None:
            base_options = BaseOptions(
                model_asset_buffer=model_asset_buffer, model_asset_path=model_path
            )

        # Create pose landmarker options
        self.options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            min_pose_detection_confidence=self.min_pose_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=output_segmentation_masks,
        )

        # Create the landmarker
        self._landmarker: Optional[vision.PoseLandmarker] = None
        self._create_landmarker()

    def _create_landmarker(self) -> None:
        """Create the pose landmarker instance."""
        if self._landmarker is not None:
            self._landmarker.close()

        self._landmarker = vision.PoseLandmarker.create_from_options(self.options)

    def detect(self, frame: np.ndarray, timestamp_ms: float) -> PoseResult:
        """
        Detect pose in a single frame.

        Args:
            frame: Input frame (H, W, 3) in BGR format
            timestamp_ms: Timestamp in milliseconds

        Returns:
            PoseResult with detection data
        """
        if self._landmarker is None:
            raise DetectorError("Landmarker not initialized")

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        if self.running_mode == vision.RunningMode.VIDEO:
            result = self._landmarker.detect_for_video(mp_image, int(timestamp_ms))
        elif self.running_mode == vision.RunningMode.IMAGE:
            result = self._landmarker.detect(mp_image)
        else:
            raise DetectorError(f"Running mode {self.running_mode} not supported")

        return self._parse_result(result, timestamp_ms)

    def _parse_result(
        self, mp_result: vision.PoseLandmarkerResult, timestamp_ms: float
    ) -> PoseResult:
        """Parse MediaPipe result into PoseResult."""
        if not mp_result.pose_landmarks:
            return PoseResult(
                success=False,
                timestamp_ms=timestamp_ms,
                pose_landmarks=None,
                pose_world_landmarks=None,
                visibility=None,
            )

        # Extract landmarks (first person only)
        landmarks = mp_result.pose_landmarks[0]
        world_landmarks = mp_result.pose_world_landmarks[0]

        pose_landmarks = [[p.x, p.y, p.z] for p in landmarks]
        pose_world_landmarks = [[p.x, p.y, p.z] for p in world_landmarks]
        visibility = [p.visibility for p in landmarks]

        # Get segmentation mask if available
        segmentation_mask = None
        if mp_result.segmentation_masks and self.options.output_segmentation_masks:
            mask_array = mp_result.segmentation_masks[0].numpy_view()
            segmentation_mask = mask_array

        return PoseResult(
            success=True,
            timestamp_ms=timestamp_ms,
            pose_landmarks=pose_landmarks,
            pose_world_landmarks=pose_world_landmarks,
            visibility=visibility,
            segmentation_mask=segmentation_mask,
        )

    def close(self) -> None:
        """Close the landmarker and release resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False


def load_model_from_file(model_path: str) -> bytes:
    """Load model file as bytes."""
    with open(model_path, "rb") as f:
        return f.read()


def get_default_model_path() -> str:
    """Get path to default model file."""
    script_dir = Path(__file__).parent.parent
    model_path = script_dir / MODEL_DIR / DEFAULT_MODEL
    return str(model_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test pose detection")
    parser.add_argument("--model", type=str, help="Path to model file")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--output", type=str, help="Output BVH file (optional)")
    args = parser.parse_args()

    model_path = args.model or get_default_model_path()

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Run: python scripts/download_models.py")
        sys.exit(1)

    print(f"Loading model: {model_path}")

    # Create detector
    with PoseLandmarker(model_path=model_path) as detector:
        print("Detector initialized. Starting webcam...")

        # Open webcam
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Could not open camera {args.camera}")
            sys.exit(1)

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect pose
                result = detector.detect(frame, timestamp_ms=frame_count * 1000 // 30)

                if result.success:
                    print(
                        f"Frame {frame_count}: Detected {len(result.pose_landmarks)} landmarks"
                    )

                    # Draw landmarks on frame
                    for landmark in result.pose_landmarks:
                        x, y, z = landmark
                        # Scale to frame coordinates
                        h, w = frame.shape[:2]
                        cx = int(x * w)
                        cy = int(y * h)
                        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

                cv2.imshow("Pose Detection", frame)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                    break

                frame_count += 1

        finally:
            cap.release()
            cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames")
