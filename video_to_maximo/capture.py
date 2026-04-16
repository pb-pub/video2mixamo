# Video to Mixamo - Video Capture Module

"""
Video capture module supporting webcam and video file input.

Provides a unified iterator interface for frame retrieval:
    for frame, timestamp_ms in VideoCapture(input_source, camera_id=0):
        # Process frame
        pass

Classes:
    VideoCapture: Abstracts webcam and video file capture
    CaptureError: Custom exception for capture-related errors
"""

import time
import cv2
from typing import Iterator, Tuple, Optional
from pathlib import Path


class CaptureError(Exception):
    """Custom exception for video capture errors."""

    pass


class VideoCapture:
    """
    Unified video capture interface for webcam and video files.

    Yields frames as (frame, timestamp_ms) tuples where:
        - frame: numpy array (H, W, 3) in BGR format
        - timestamp_ms: timestamp in milliseconds since start

    For webcam:
        - timestamp_ms is relative to capture start
        - camera_id selects which camera (default 0)

    For video files:
        - timestamp_ms comes from video metadata
        - fps is derived from the video

    Usage:
        # Webcam
        for frame, ts in VideoCapture(camera_id=0):
            process(frame)

        # Video file
        for frame, ts in VideoCapture("input.mp4"):
            process(frame)
    """

    def __init__(
        self,
        input_source: Optional[str] = None,
        camera_id: int = 0,
        fps: Optional[float] = None,
    ):
        """
        Initialize video capture.

        Args:
            input_source: Path to video file, or None for webcam
            camera_id: Camera device ID (used only if input_source is None)
            fps: Target FPS for output (auto-detected if None)
        """
        self.input_source = input_source
        self.camera_id = camera_id
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_opened = False
        self._start_time_ms: Optional[float] = None
        self._frame_count = 0

        if input_source is not None:
            self._init_file_capture(input_source)
        else:
            self._init_camera_capture(camera_id)

    def _init_file_capture(self, filepath: str) -> None:
        """Initialize video file capture."""
        path = Path(filepath)
        if not path.exists():
            raise CaptureError(f"Video file not found: {filepath}")

        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise CaptureError(f"Could not open video file: {filepath}")

        # Auto-detect FPS if not provided
        if self.fps is None:
            detected_fps = self._cap.get(cv2.CAP_PROP_FPS)
            if detected_fps > 0:
                self.fps = detected_fps
            else:
                self.fps = 30.0  # Default fallback

    def _init_camera_capture(self, camera_id: int) -> None:
        """Initialize webcam capture."""
        self._cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise CaptureError(f"Could not open camera {camera_id}")

        # Try to set FPS if specified
        if self.fps is not None:
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Auto-detect FPS
        if self.fps is None:
            detected_fps = self._cap.get(cv2.CAP_PROP_FPS)
            if detected_fps > 0:
                self.fps = detected_fps
            else:
                self.fps = 30.0

    def __iter__(self) -> Iterator[Tuple]:
        """Return iterator for frame capture."""
        return self

    def __next__(self) -> Tuple:
        """Get next frame from capture."""
        if not self._is_opened:
            self._start_capture()

        if self._cap is None:
            raise StopIteration

        ret, frame = self._cap.read()
        if not ret:
            self.stop()
            raise StopIteration

        # Calculate timestamp using elapsed time for monotonic increase
        timestamp_ms = self._get_timestamp()

        self._frame_count += 1
        return frame, timestamp_ms

    def _start_capture(self) -> None:
        """Start capture and record start time."""
        if self._cap is None:
            raise CaptureError("Capture not initialized")

        self._is_opened = True
        self._start_time_ms = time.time() * 1000  # Start time in ms

    def _get_timestamp(self) -> float:
        """Get current timestamp in milliseconds."""
        if self._start_time_ms is None:
            return 0.0

        if self.input_source is not None:
            # For video files, use frame position
            pos_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms > 0:
                return pos_ms
        # For webcam or fallback, use elapsed time from start
        return time.time() * 1000 - self._start_time_ms

    def stop(self) -> None:
        """Stop capture and release resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_opened = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.stop()
        return False

    @property
    def width(self) -> int:
        """Get frame width."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Get frame height."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_count(self) -> int:
        """Get total frames processed."""
        return self._frame_count


def test_webcam(camera_id: int = 0, num_frames: int = 5) -> bool:
    """
    Test webcam capture for a few frames.

    Args:
        camera_id: Camera device ID
        num_frames: Number of frames to capture for testing

    Returns:
        True if capture successful, False otherwise
    """
    try:
        capture = VideoCapture(camera_id=camera_id)
        count = 0
        for frame, ts in capture:
            count += 1
            if count >= num_frames:
                break
        capture.stop()
        return count > 0
    except CaptureError as e:
        print(f"Webcam test failed: {e}")
        return False


def test_video_file(filepath: str, num_frames: int = 5) -> bool:
    """
    Test video file capture for a few frames.

    Args:
        filepath: Path to video file
        num_frames: Number of frames to capture for testing

    Returns:
        True if capture successful, False otherwise
    """
    try:
        capture = VideoCapture(input_source=filepath)
        count = 0
        for frame, ts in capture:
            count += 1
            if count >= num_frames:
                break
        capture.stop()
        return count > 0
    except CaptureError as e:
        print(f"Video file test failed: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Test video file
        filepath = sys.argv[1]
        print(f"Testing video file: {filepath}")
        success = test_video_file(filepath)
        print(f"Result: {'✓ Passed' if success else '✗ Failed'}")
    else:
        # Test webcam
        print("Testing webcam...")
        success = test_webcam(camera_id=0, num_frames=5)
        print(f"Result: {'✓ Passed' if success else '✗ Failed'}")
