#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main CLI entry point for Video to Mixamo.

Usage:
    python main.py                          # Start webcam capture with preview
    python main.py --camera 1               # Use camera 1
    python main.py --input video.mp4        # Process video file
    python main.py --input video.mp4 --output animation.bvh
    python main.py --model models/my_model.task

Controls (in preview window):
    R: Start/Stop recording
    S: Stop recording (keep window open)
    ESC: Quit without saving

Options:
    --input FILE          Input video file (default: use webcam)
    --camera ID           Camera device ID (default: 0)
    --output FILE         Output BVH file (default: auto-generated)
    --format FORMAT       Output format: bvh (default), fbx
    --model FILE          Path to MediaPipe .task model file
    --fps RATE            Frame rate (default: auto-detected or 30)
    --no-preview          Disable live preview window
    --no-smooth           Disable temporal smoothing
    --smooth-cutoff C     One Euro min_cutoff (default: 1.0)
    --smooth-beta B       One Euro beta (default: 0.0)
    --smooth-z-cutoff C   Z-axis cutoff (default: 0.5)
    --auto-download       Auto-download missing model file
    --help                Show this help message

Examples:
    # Webcam capture with preview, press R to record
    python main.py

    # Process video file with auto-generated output name
    python main.py --input myvideo.mp4

    # Full pipeline: video to BVH
    python main.py --input video.mp4 --output animation.bvh

    # Disable smoothing for raw output
    python main.py --no-smooth

    # Custom model path
    python main.py --model models/pose_landmarker_full.task
"""

import argparse
import sys
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple

import cv2

from .capture import VideoCapture, CaptureError
from .detector import PoseLandmarker, DetectorError, PoseResult
from .rotation import RotationComputer
from .exporter_bvh import BVHExporter
from .filter import Smoother, FilterConfig
from .viz3d import Pose3DVisualizer


def get_default_output_path() -> str:
    """Generate default output filename based on timestamp."""
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(output_dir / f"animation_{timestamp}.bvh")


def get_default_model_path() -> str:
    """Get path to default model file."""
    script_dir = Path(__file__).parent.parent
    model_path = script_dir / "models" / "pose_landmarker_full.task"
    return str(model_path)


def check_model_exists(model_path: str) -> bool:
    """Check if model file exists."""
    return os.path.exists(model_path)


def download_model(model_url: str, model_path: str) -> bool:
    """Download model file."""
    import urllib.request
    import urllib.error

    print(f"Downloading model from {model_url}...")
    print(f"Saving to: {model_path}")

    try:
        # Create parent directory if needed
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(model_url, model_path)
        print("Download complete!")
        return True
    except urllib.error.URLError as e:
        print(f"Download failed: {e.reason}")
        return False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert video to Mixamo-compatible BVH animation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Options:")[0] + "\n\nOptions:",
    )

    # Input options
    parser.add_argument(
        "--input", type=str, help="Input video file (default: use webcam)"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--output", type=str, help="Output BVH file (default: auto-generated)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="bvh",
        choices=["bvh", "fbx"],
        help="Output format (default: bvh)",
    )
    parser.add_argument("--model", type=str, help="Path to MediaPipe .task model file")
    parser.add_argument(
        "--fps", type=float, help="Frame rate (default: auto-detected or 30)"
    )

    # Processing options
    parser.add_argument(
        "--no-preview", action="store_true", help="Disable live preview window"
    )
    parser.add_argument(
        "--no-smooth", action="store_true", help="Disable temporal smoothing"
    )
    parser.add_argument(
        "--smooth-cutoff",
        type=float,
        default=1.0,
        help="One Euro min_cutoff for X/Y axes (default: 1.0)",
    )
    parser.add_argument(
        "--smooth-beta", type=float, default=0.0, help="One Euro beta (default: 0.0)"
    )
    parser.add_argument(
        "--smooth-z-cutoff",
        type=float,
        default=0.5,
        help="One Euro min_cutoff for Z axis (default: 0.5)",
    )
    parser.add_argument(
        "--auto-download", action="store_true", help="Auto-download missing model file"
    )

    return parser.parse_args()


class VideoToMixamo:
    """Main application class coordinating the pipeline."""

    def __init__(self, args: argparse.Namespace):
        """Initialize the application."""
        self.args = args

        # Get model path
        self.model_path = args.model or get_default_model_path()

        # Check/download model
        if not check_model_exists(self.model_path):
            if args.auto_download:
                from config import DEFAULT_MODEL_URL

                if download_model(DEFAULT_MODEL_URL, self.model_path):
                    print("Model downloaded successfully")
                else:
                    print("Failed to download model. Please run:")
                    print("  python scripts/download_models.py")
                    sys.exit(1)
            else:
                print(f"Model not found: {self.model_path}")
                print("Run: python scripts/download_models.py")
                sys.exit(1)

        # Initialize components
        self.capture: Optional[VideoCapture] = None
        self.detector: Optional[PoseLandmarker] = None
        self.exporter: Optional[BVHExporter] = None
        self.smoother: Optional[Smoother] = None

        # State
        self.recording = False
        self.frames_captured = 0
        self.recorded_landmarks: List[List[List[float]]] = []
        self.recorded_timestamps: List[float] = []
        self.recorded_visibility: List[Optional[List[float]]] = []
        self._last_result: Optional[PoseResult] = None  # Store last detection result

        # 3-D visualizer (opened on demand with V key)
        self._viz3d = Pose3DVisualizer()

        # Frame counters
        self.total_frames = 0
        self._fps_start_time: float = 0.0
        self._fps_frame_count = 0
        self._current_fps: float = 0.0

    def initialize(self) -> bool:
        """Initialize all components."""
        try:
            # Initialize detector
            self.detector = PoseLandmarker(
                model_path=self.model_path,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            # Initialize smoother if enabled
            if not self.args.no_smooth:
                self.smoother = Smoother(
                    config=FilterConfig(
                        min_cutoff=self.args.smooth_cutoff,
                        beta=self.args.smooth_beta,
                        frequency=self.args.fps or 30.0,
                    ),
                    z_config=FilterConfig(
                        min_cutoff=self.args.smooth_z_cutoff,
                        beta=self.args.smooth_beta,
                        frequency=self.args.fps or 30.0,
                    ),
                )

            # Initialize capture
            self.capture = VideoCapture(
                input_source=self.args.input,
                camera_id=self.args.camera,
                fps=self.args.fps,
            )

            print(
                f"Capture initialized: {self.capture.width}x{self.capture.height} @ {self.capture.fps}fps"
            )

            return True

        except (CaptureError, DetectorError) as e:
            print(f"Initialization failed: {e}")
            return False

    def process_frame(self, frame, timestamp_ms: float) -> bool:
        """Process a single frame."""
        # Detect pose
        result = self.detector.detect(frame, timestamp_ms)

        # Store result for drawing
        self._last_result = result

        if result.success and self.smoother:
            # Smooth landmarks
            landmarks = result.pose_world_landmarks
            if landmarks:
                smoothed_landmarks = self.smoother.filter_landmarks(
                    landmarks, timestamp_ms
                )
                result.pose_world_landmarks = smoothed_landmarks

        # Feed 3-D visualizer if open
        if result.success and self._viz3d.is_running:
            self._viz3d.update_landmarks(
                result.pose_world_landmarks,
                result.visibility,
            )

        return result.success

    def record_frame(self, frame, timestamp_ms: float) -> None:
        """Record a frame for export."""
        self.frames_captured += 1
        self.recorded_timestamps.append(timestamp_ms)

        # Store landmarks if detection was successful
        if self._last_result and self._last_result.success:
            self.recorded_landmarks.append(self._last_result.pose_world_landmarks)
            self.recorded_visibility.append(self._last_result.visibility)
        else:
            # Store None for frames without detection (will be handled in export)
            self.recorded_landmarks.append(None)
            self.recorded_visibility.append(None)

    def start_recording(self) -> None:
        """Start recording mode."""
        self.recording = True
        self.frames_captured = 0
        self.recorded_landmarks = []
        self.recorded_timestamps = []
        self.recorded_visibility = []
        print("Recording started...")

    def stop_recording(self) -> None:
        """Stop recording and export."""
        self.recording = False
        print(f"Recording stopped: {self.frames_captured} frames captured")

        if self.frames_captured == 0:
            print("No frames recorded!")
            return

        # Export to BVH
        output_path = self.args.output or get_default_output_path()

        print(f"Exporting to {output_path}...")

        # Compute rotations from recorded landmarks, skip invalid frames
        rotation_computer = RotationComputer()
        frame_rotations = []
        valid_indices = []

        for i, landmarks in enumerate(self.recorded_landmarks):
            if landmarks is not None and len(landmarks) == 33:
                rotation_result = rotation_computer.compute_rotations(
                    landmarks, self.recorded_timestamps[i], self.recorded_visibility[i]
                )
                if rotation_result.success:
                    frame_rotations.append(rotation_result)
                    valid_indices.append(i)

        print(f"Valid frames: {len(frame_rotations)}/{self.frames_captured}")

        if len(frame_rotations) == 0:
            print("No valid frames to export!")
            return

        # Export BVH
        try:
            self.exporter = BVHExporter()
            self.exporter.export(
                output_path=output_path,
                frame_rotations=frame_rotations,
                fps=self.capture.fps if self.capture else 30.0,
            )
            print(f"Exported successfully: {output_path}")
        except Exception as e:
            print(f"Export failed: {e}")

    def run(self) -> None:
        """Run the main application loop."""
        if not self.initialize():
            sys.exit(1)

        print("Starting video processing...")
        print("Controls:")
        print("  R: Start/Stop recording")
        print("  S: Stop recording (keep window open)")
        print("  V: Toggle 3D skeleton viewer")
        print("  ESC: Quit without saving")

        self._fps_start_time = time.time()

        try:
            for frame, timestamp_ms in self.capture:
                self.total_frames += 1
                self._fps_frame_count += 1

                # Calculate FPS every 30 frames
                if self._fps_frame_count % 30 == 0:
                    elapsed = time.time() - self._fps_start_time
                    if elapsed > 0:
                        self._current_fps = self._fps_frame_count / elapsed

                # Process frame
                success = self.process_frame(frame, timestamp_ms)

                # Draw visualization
                if not self.args.no_preview:
                    self._draw_preview(frame, success, timestamp_ms)

                # Handle recording
                if self.recording:
                    self.record_frame(frame, timestamp_ms)

                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord("r") or key == ord("R"):
                    if self.recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                elif key == ord("s") or key == ord("S"):
                    if self.recording:
                        self.stop_recording()
                        print(
                            "Recording stopped. Press ESC to quit or R to record again."
                        )
                elif key == ord("v") or key == ord("V"):
                    opened = self._viz3d.toggle()
                    print("3D viewer opened" if opened else "3D viewer closed")

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self._cleanup()

    def _get_fps_text(self) -> str:
        """Get FPS display text."""
        return f"FPS: {self._current_fps:.1f}"

    def _draw_preview(self, frame, success: bool, timestamp_ms: float) -> None:
        """Draw preview overlay on frame."""
        # Draw skeleton if pose was detected
        if success and hasattr(self, "_last_result") and self._last_result is not None:
            self._draw_skeleton(frame, self._last_result)

        # Draw status indicators
        h, w = frame.shape[:2]

        # Recording indicator
        if self.recording:
            cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)  # Red circle
            cv2.putText(
                frame, "REC", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

        # Detection status
        if success:
            cv2.putText(
                frame,
                "Pose Detected",
                (w - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "No Pose",
                (w - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # Frame rate info
        fps_text = self._get_fps_text()
        cv2.putText(
            frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        # Frame count
        cv2.putText(
            frame,
            f"Frame: {self.total_frames}",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Instructions
        cv2.putText(
            frame,
            "[R]ecord | [S]top | [V]iz 3D | [ESC] quit",
            (10, h - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.imshow("Video to Mixamo - Press R to record", frame)

    def _draw_skeleton(self, frame, result) -> None:
        """Draw skeleton on frame using detected landmarks."""
        if not result.success or not result.pose_landmarks:
            return

        h, w = frame.shape[:2]
        landmarks = result.pose_landmarks  # List of [x, y, z]
        visibility = result.visibility or []

        # Draw connections (bones) as lines
        connections = [
            # Spine
            (11, 12),  # shoulders
            (11, 9),  # left shoulder to nose area
            (12, 10),  # right shoulder to nose area
            (9, 10),  # between shoulder points
            # Left Arm
            (11, 13),  # left shoulder to elbow
            (13, 15),  # left elbow to wrist
            # Right Arm
            (12, 14),  # right shoulder to elbow
            (14, 16),  # right elbow to wrist
            # Left Leg
            (23, 25),  # left hip to knee
            (25, 27),  # left knee to ankle
            (27, 31),  # left ankle to foot
            # Right Leg
            (24, 26),  # right hip to knee
            (26, 28),  # right knee to ankle
            (28, 32),  # right ankle to foot
            # Hips
            (23, 24),  # between hips
        ]

        # Draw each bone as a line
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = self._landmark_to_pixel(landmarks[start_idx], w, h)
                end_point = self._landmark_to_pixel(landmarks[end_idx], w, h)

                # Draw bone with gradient based on confidence
                conf_start = (
                    visibility[start_idx] if start_idx < len(visibility) else 0.5
                )
                conf_end = visibility[end_idx] if end_idx < len(visibility) else 0.5
                confidence = min(conf_start, conf_end)
                color = self._confidence_color(confidence)
                thickness = max(1, int(confidence * 3))

                cv2.line(frame, start_point, end_point, color, thickness)

        # Draw landmarks as circles
        for i, landmark in enumerate(landmarks):
            if i >= len(landmarks):
                break
            point = self._landmark_to_pixel(landmark, w, h)
            confidence = visibility[i] if i < len(visibility) else 0.5

            # Color based on confidence
            color = self._confidence_color(confidence)

            # Larger circles for key joints
            if i in [11, 12, 23, 24]:  # shoulders/hips
                cv2.circle(frame, point, 6, color, -1)
            elif i in [13, 14, 25, 26]:  # elbows/knees
                cv2.circle(frame, point, 5, color, -1)
            else:
                cv2.circle(frame, point, 3, color, -1)

    def _landmark_to_pixel(self, landmark, w: int, h: int) -> Tuple[int, int]:
        """Convert normalized landmark to pixel coordinates."""
        x = int(landmark[0] * w)
        y = int(landmark[1] * h)
        return (x, y)

    def _confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        """Map confidence to BGR color."""
        if confidence > 0.8:
            return (0, 255, 0)  # Green
        elif confidence > 0.5:
            return (0, 165, 255)  # Orange
        else:
            return (0, 0, 255)  # Red

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self.capture:
            self.capture.stop()
        if self.detector:
            self.detector.close()
        self._viz3d.close()
        cv2.destroyAllWindows()


def main():
    """Main entry point."""
    args = parse_args()

    app = VideoToMixamo(args)
    app.run()


if __name__ == "__main__":
    main()
