# Video to Mixamo - BVH Exporter Module

"""
BVH file exporter module.

Writes BVH (Biovision Hierarchical) files from rotation data.
BVH is a plain-text format widely supported by animation tools.

BVH Format:
    HIERARCHY section: bone hierarchy, offsets, channel declarations
    MOTION section: per-frame rotation data

Coordinate system:
    - Y-up (standard for BVH)
    - Euler angles in ZXY order
    - Rotations in degrees
"""

from typing import Dict, List

import numpy as np

from .rotation import RotationComputer, RotationResult
from .skeleton import Skeleton, get_landmark_indices


class BVHExporter:
    """
    Export pose data to BVH format.

    The BVH file contains:
        - HIERARCHY: bone hierarchy definition
        - MOTION: per-frame rotation and position data

    Coordinate system: Y-up (matching MediaPipe world coordinates)
    """

    def __init__(self, skeleton: Skeleton = None):
        """
        Initialize BVH exporter.

        Args:
            skeleton: Skeleton definition (uses default if None)
        """
        self.skeleton = skeleton or Skeleton()
        self._landmark_indices = get_landmark_indices()

    def export(
        self, output_path: str, frame_rotations: List[RotationResult], fps: float = 30.0
    ) -> None:
        """
        Export rotation data to BVH file.

        Args:
            output_path: Output BVH file path
            frame_rotations: List of RotationResult for each frame
            fps: Frame rate for the output file
        """
        # Validate input
        if not frame_rotations:
            raise ValueError("No rotation data to export")

        print(f"Number of frames: {len(frame_rotations)}")
        print(f"First frame success: {frame_rotations[0].success}")

        if not frame_rotations[0].success:
            raise ValueError("First frame has no valid rotation data")

        # Write BVH file
        with open(output_path, "w") as f:
            self._write_hierarchy(f)
            self._write_motion(f, frame_rotations, fps)

    def _write_hierarchy(self, f) -> None:
        """Write HIERARCHY section."""
        f.write("HIERARCHY\n")
        f.write("\n")

        # Write root bone (Hips)
        root_bone = self.skeleton.get_root()
        self._write_bone_recursive(f, root_bone.name, 0)

    def _write_bone_recursive(self, f, bone_name: str, depth: int) -> int:
        """
        Recursively write bone definition.

        Returns: Number of children
        """
        bone = self.skeleton.get_bone(bone_name)
        if bone is None:
            raise ValueError(f"Unknown bone: {bone_name}")

        indent = "    " * depth

        # Get bone offset from parent (position in T-pose)
        offset = self._get_bone_offset(bone_name)

        f.write(f"{indent}JOINT {bone_name}\n")
        f.write(f"{indent}{{\n")
        f.write(f"{indent}    OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
        f.write(f"{indent}    CHANNELS 6 Xposition Yposition Zposition ")
        f.write("Zrotation Xrotation Yrotation\n")

        # Write children
        children = self.skeleton.get_children(bone_name)
        for child_name in children:
            self._write_bone_recursive(f, child_name, depth + 1)

        # If no children, this is an end site (leaf)
        if not children:
            f.write(f"{indent}    End Site\n")
            f.write(f"{indent}    {{\n")
            # End site offset - use a small offset based on bone length
            end_offset = self._get_end_site_offset(bone_name)
            f.write(
                f"{indent}        OFFSET {end_offset[0]:.6f} {end_offset[1]:.6f} {end_offset[2]:.6f}\n"
            )
            f.write(f"{indent}    }}\n")

        f.write(f"{indent}}}\n")

        return len(children)

    def _get_bone_offset(self, bone_name: str) -> List[float]:
        """
        Get offset for a bone from its parent.

        For root bone, returns [0, 0, 0].
        For other bones, returns the offset from parent position.
        """
        if bone_name == "Hips":
            return [0.0, 0.0, 0.0]

        # Get offsets from a reference T-pose
        # These are approximate human proportions
        offsets = self._get_tpose_offsets()

        if bone_name in offsets:
            return offsets[bone_name]

        # Default: small offset in Y direction
        return [0.0, 0.1, 0.0]

    def _get_tpose_offsets(self) -> Dict[str, List[float]]:
        """
        Get T-pose bone offsets from parent joints.

        These are approximate human proportions.
        """
        return {
            # Root
            "Hips": [0.0, 0.0, 0.0],
            # Spine chain
            "Spine": [0.0, 0.2, 0.0],
            "Spine1": [0.0, 0.15, 0.0],
            "Spine2": [0.0, 0.1, 0.0],
            "Neck": [0.0, 0.15, 0.0],
            "Head": [0.0, 0.15, 0.0],
            # Shoulders
            "RightShoulder": [0.12, 0.1, 0.0],
            "LeftShoulder": [-0.12, 0.1, 0.0],
            # Arms
            "RightArm": [0.15, 0.0, 0.0],
            "LeftArm": [-0.15, 0.0, 0.0],
            "RightForeArm": [0.25, 0.0, 0.0],
            "LeftForeArm": [-0.25, 0.0, 0.0],
            "RightHand": [0.25, 0.0, 0.0],
            "LeftHand": [-0.25, 0.0, 0.0],
            # Legs
            "RightUpLeg": [0.1, -0.25, 0.0],
            "LeftUpLeg": [-0.1, -0.25, 0.0],
            "RightLeg": [0.0, -0.4, 0.0],
            "LeftLeg": [0.0, -0.4, 0.0],
            "RightFoot": [0.0, -0.4, 0.0],
            "LeftFoot": [0.0, -0.4, 0.0],
        }

    def _get_end_site_offset(self, bone_name: str) -> List[float]:
        """Get offset for end sites (leaf bones)."""
        # These are approximate bone lengths
        end_offsets = {
            "RightHand": [0.1, 0.0, 0.0],
            "LeftHand": [-0.1, 0.0, 0.0],
            "RightFoot": [0.0, 0.0, 0.15],
            "LeftFoot": [0.0, 0.0, 0.15],
            "Head": [0.0, 0.1, 0.0],
        }

        return end_offsets.get(bone_name, [0.0, 0.1, 0.0])

    def _write_motion(
        self, f, frame_rotations: List[RotationResult], fps: float
    ) -> None:
        """Write MOTION section."""
        f.write("MOTION\n")
        f.write(f"Frames: {len(frame_rotations)}\n")
        f.write(f"Frame Time: {1.0 / fps:.6f}\n")

        # Get all bone names in hierarchy order
        bone_names = self._get_bone_hierarchy_order()

        for result in frame_rotations:
            if not result.success:
                # Use zeros for failed frames
                values = self._get_empty_frame_values(bone_names)
            else:
                values = self._get_frame_values(result, bone_names)

            # Write frame data
            f.write(" ".join(f"{v:.6f}" for v in values) + "\n")

    def _get_bone_hierarchy_order(self) -> List[str]:
        """Get bone names in hierarchy order (root first, then children)."""
        bone_names = []

        def traverse(name: str):
            bone_names.append(name)
            for child in self.skeleton.get_children(name):
                traverse(child)

        # Start with root
        root = self.skeleton.get_root()
        traverse(root.name)

        return bone_names

    def _get_frame_values(
        self, result: RotationResult, bone_names: List[str]
    ) -> List[float]:
        """Get BVH frame values for a single frame."""
        values = []

        for bone_name in bone_names:
            # Get position for this bone
            if bone_name in result.bone_rotations:
                quat, position = result.bone_rotations[bone_name]
                values.extend(
                    [
                        position[0],  # X position
                        position[1],  # Y position
                        position[2],  # Z position
                    ]
                )

                # Convert quaternion to Euler angles (ZXY order)
                euler = self._quaternion_to_euler_zxy(quat)
                values.extend(euler)
            else:
                # Bone not present in this frame - use zeros
                values.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        return values

    def _get_empty_frame_values(self, bone_names: List[str]) -> List[float]:
        """Get zero values for an empty frame."""
        values = []

        for bone_name in bone_names:
            # Root bone has position
            if (
                bone_name == "Hips"
                or self.skeleton.get_bone(bone_name).parent_name is None
            ):
                values.extend([0.0, 0.0, 0.0])
            else:
                # Non-root bones at origin
                values.extend([0.0, 0.0, 0.0])

            # Rotation (identity = no rotation)
            values.extend([0.0, 0.0, 0.0])

        return values

    def _quaternion_to_euler_zxy(self, rotation_input):
        """
        Convert rotation to Euler angles [z, x, y] in degrees.

        Accepts either a quaternion [w, x, y, z] or a 3x3 rotation matrix.
        """
        # Check if it's a numpy array (rotation matrix)
        if isinstance(rotation_input, np.ndarray):
            return self._rotation_matrix_to_euler_zxy(rotation_input)

        # Otherwise treat as quaternion
        w, x, y, z = rotation_input

        # Convert to rotation matrix then to Euler
        R = self._quat_to_matrix(rotation_input)
        return self._rotation_matrix_to_euler_zxy(R)

    def _quat_to_matrix(self, quat: List[float]) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
        w, x, y, z = quat

        # Ensure quaternion is normalized
        norm = np.sqrt(w * w + x * x + y * y + z * z)
        if norm > 1e-6:
            w, x, y, z = w / norm, x / norm, y / norm, z / norm

        # Rotation matrix from quaternion
        R = np.array(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * w * z,
                    2 * x * z + 2 * w * y,
                ],
                [
                    2 * x * y + 2 * w * z,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * w * x,
                ],
                [
                    2 * x * z - 2 * w * y,
                    2 * y * z + 2 * w * x,
                    1 - 2 * x * x - 2 * y * y,
                ],
            ]
        )

        return R

    def _rotation_matrix_to_euler_zxy(self, R: np.ndarray) -> List[float]:
        """
        Convert rotation matrix to Euler angles [z, x, y] in degrees.

        Uses ZXY intrinsic rotation order.
        """
        # Extract Euler angles from rotation matrix
        # For ZXY order:
        # R = R_z(ψ) * R_x(θ) * R_y(φ)

        # Check for gimbal lock
        sy = R[0, 2]
        cy = np.sqrt(1 - sy * sy)

        if cy > 1e-6:
            # General case
            z = np.arctan2(R[1, 2], R[2, 2])
            x = np.arctan2(-R[0, 2], cy)
            y = np.arctan2(R[0, 1], R[0, 0])
        else:
            # Gimbal lock
            z = np.arctan2(-R[2, 1], R[1, 1])
            x = np.arctan2(-R[0, 2], cy)
            y = 0.0

        # Convert to degrees and return [z, x, y]
        return [np.degrees(z), np.degrees(x), np.degrees(y)]


def test_exporter():
    """Test BVH exporter with sample data."""
    # Create sample rotation results
    computer = RotationComputer()
    exporter = BVHExporter()

    # Simple test: 3 frames of a person raising arms
    # Base landmarks for T-pose (33 landmarks total)
    base_landmarks = [
        [0, 1.7, 0],  # 0: nose
        [0, 1.7, 0],  # 1: left_eye_inner
        [0, 1.7, 0],  # 2: left_eye
        [0, 1.7, 0],  # 3: left_eye_outer
        [0, 1.7, 0],  # 4: right_eye_inner
        [0, 1.7, 0],  # 5: right_eye
        [0, 1.7, 0],  # 6: right_eye_outer
        [0, 1.7, 0],  # 7: left_ear
        [0, 1.7, 0],  # 8: right_ear
        [0, 1.7, 0],  # 9: mouth_left
        [0, 1.7, 0],  # 10: mouth_right
        [0, 1.6, 0],  # 11: left_shoulder
        [0, 1.6, 0],  # 12: right_shoulder
        [0, 1.4, 0],  # 13: left_elbow
        [0, 1.4, 0],  # 14: right_elbow
        [0, 1.2, 0],  # 15: left_wrist
        [0, 1.2, 0],  # 16: right_wrist
        [0, 1.2, 0],  # 17: left_pinky
        [0, 1.2, 0],  # 18: right_pinky
        [0, 1.2, 0],  # 19: left_index
        [0, 1.2, 0],  # 20: right_index
        [0, 1.2, 0],  # 21: left_thumb
        [0, 1.2, 0],  # 22: right_thumb
        [-0.1, 1.0, 0],  # 23: left_hip
        [0.1, 1.0, 0],  # 24: right_hip
        [-0.1, 0.5, 0],  # 25: left_knee
        [0.1, 0.5, 0],  # 26: right_knee
        [-0.1, 0.0, 0],  # 27: left_ankle
        [0.1, 0.0, 0],  # 28: right_ankle
        [-0.1, 0.0, 0.1],  # 29: left_heel
        [0.1, 0.0, 0.1],  # 30: right_heel
        [0, 0.0, 0.1],  # 31: left_foot_index
        [0, 0.0, 0.1],  # 32: right_foot_index
    ]

    # Generate 3 frames
    frame_rotations = []
    for frame_idx in range(3):
        # Slightly move the arms in each frame
        test_landmarks = []
        for i, lm in enumerate(base_landmarks):
            new_lm = list(lm)
            if i in [11, 13, 15]:  # Left arm
                new_lm[0] -= 0.1 * frame_idx  # Move left arm outward
            if i in [12, 14, 16]:  # Right arm
                new_lm[0] += 0.1 * frame_idx  # Move right arm outward
            test_landmarks.append(new_lm)

        result = computer.compute_rotations(
            test_landmarks, timestamp_ms=frame_idx * 1000 // 30
        )
        print(f"Frame {frame_idx}: landmarks count = {len(test_landmarks)}")
        print(f"Frame {frame_idx} success: {result.success}")
        print(f"  Root position: {result.root_position}")
        frame_rotations.append(result)

    # Export to BVH
    output_path = "test_output.bvh"
    exporter.export(output_path, frame_rotations, fps=30.0)

    print(f"Exported BVH to: {output_path}")

    # Print first few lines for verification
    with open(output_path, "r") as f:
        lines = f.readlines()
        print("\nFirst 20 lines of BVH file:")
        for i, line in enumerate(lines[:20]):
            print(f"{i + 1:3d}: {line.rstrip()}")


if __name__ == "__main__":
    test_exporter()
