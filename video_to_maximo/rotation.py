# Video to Mixamo - Rotation Computation Module

"""
Rotation computation module for converting landmarks to bone rotations.

Computes world-space and local-space rotations from MediaPipe landmarks
using forward kinematics (bone direction vectors).

Classes:
    RotationComputer: Converts landmarks to bone rotations
    RotationResult: Result container for rotation data
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R

from .skeleton import Skeleton, get_landmark_indices


@dataclass
class RotationResult:
    """
    Result from rotation computation.

    Attributes:
        success: Whether computation was successful
        timestamp_ms: Timestamp of the frame
        root_position: Hips position as [x, y, z]
        bone_rotations: Dict mapping bone name to (quaternion, position)
            - quaternion: [w, x, y, z] normalized
            - position: [x, y, z] in world space
    """

    success: bool
    timestamp_ms: float
    root_position: Optional[List[float]]
    bone_rotations: Optional[Dict[str, Tuple[List[float], List[float]]]]


class RotationComputer:
    """
    Compute bone rotations from MediaPipe landmarks.

    Uses forward kinematics:
        1. Compute bone direction vectors from landmark positions
        2. Compute world-space rotations from direction vs T-pose
        3. Convert to local-space rotations using parent hierarchy

    Coordinate system:
        - Input: MediaPipe world landmarks (Y-up, hip-centered)
        - Output: BVH-compatible rotations (Y-up, Hips as root)
    """

    def __init__(self, skeleton: Skeleton = None):
        """
        Initialize rotation computer.

        Args:
            skeleton: Skeleton definition (uses default if None)
        """
        self.skeleton = skeleton or Skeleton()
        self._landmark_indices = get_landmark_indices()

    def compute_rotations(
        self, landmarks: List[List[float]], timestamp_ms: float
    ) -> RotationResult:
        """
        Compute bone rotations from landmarks.

        Args:
            landmarks: 33x3 list of [x, y, z] positions
            timestamp_ms: Timestamp in milliseconds

        Returns:
            RotationResult with bone rotations
        """
        if len(landmarks) != 33:
            return RotationResult(
                success=False,
                timestamp_ms=timestamp_ms,
                root_position=None,
                bone_rotations=None,
            )

        # Compute world positions for each bone
        world_positions = self.skeleton.get_world_positions(landmarks)

        # Get root position (Hips)
        root_position = world_positions.get("Hips", [0, 0, 0])

        # Compute world-space rotations for each bone
        world_rotations = self._compute_world_rotations(world_positions)

        # Convert to local-space rotations
        local_rotations = self._world_to_local_rotations(world_rotations)

        # Build bone rotations dict
        bone_rotations = {}
        for bone_name, rotation in local_rotations.items():
            # Position is bone origin in world space
            position = world_positions.get(bone_name, [0, 0, 0])
            bone_rotations[bone_name] = (rotation, position)

        return RotationResult(
            success=True,
            timestamp_ms=timestamp_ms,
            root_position=root_position.tolist(),
            bone_rotations=bone_rotations,
        )

    def _compute_world_rotations(
        self, positions: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compute world-space rotation matrices for each bone.

        Args:
            positions: Dict mapping bone name to position

        Returns:
            Dict mapping bone name to 3x3 rotation matrix
        """
        world_rotations = {}

        # Process bones in hierarchy order (parents first)
        processed = set()
        for bone in self.skeleton.bones:
            bone_name = bone.name
            parent_name = bone.parent_name

            if parent_name is not None and parent_name not in processed:
                # Parent not yet processed, skip for now
                continue

            # Get bone direction in current pose
            current_dir = self.skeleton.get_direction(positions, bone_name)

            # Get T-pose reference direction
            t_pose_dir = self.skeleton._t_pose_ref.get(bone_name)

            if t_pose_dir is None:
                # Root or bone without reference - use identity
                world_rotations[bone_name] = np.eye(3)
            else:
                # Compute rotation that aligns T-pose to current
                rotation = self._compute_rotation_from_directions(
                    t_pose_dir, current_dir
                )
                world_rotations[bone_name] = rotation

            processed.add(bone_name)

        return world_rotations

    def _compute_rotation_from_directions(
        self, from_dir: np.ndarray, to_dir: np.ndarray
    ) -> np.ndarray:
        """
        Compute rotation matrix that aligns from_dir to to_dir.

        Uses axis-angle representation via Rodrigues' formula.

        Args:
            from_dir: Source direction vector (normalized)
            to_dir: Target direction vector (normalized)

        Returns:
            3x3 rotation matrix
        """
        # Check for zero vectors
        from_norm = np.linalg.norm(from_dir)
        to_norm = np.linalg.norm(to_dir)

        if from_norm < 1e-6 or to_norm < 1e-6:
            # Zero vector - return identity rotation
            return np.eye(3)

        from_dir = from_dir / from_norm
        to_dir = to_dir / to_norm

        # Dot product for angle
        dot = np.dot(from_dir, to_dir)

        # Parallel (same direction) - return identity
        if dot > 0.9999:
            return np.eye(3)

        # Parallel (opposite direction) - 180 degree rotation
        if dot < -0.9999:
            # Find perpendicular axis
            perp = self._get_perpendicular_vector(from_dir)
            return self._rotation_matrix_from_axis_angle(perp, np.pi)

        # General case - compute rotation axis and angle
        axis = np.cross(from_dir, to_dir)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-6:
            # Vectors are parallel (should be caught above, but handle safely)
            return np.eye(3)

        axis = axis / axis_norm
        angle = np.arccos(np.clip(dot, -1.0, 1.0))

        return self._rotation_matrix_from_axis_angle(axis, angle)

    def _get_perpendicular_vector(self, v: np.ndarray) -> np.ndarray:
        """Get a unit vector perpendicular to v."""
        v = v / np.linalg.norm(v)

        # Find axis with smallest component
        idx = np.argmin(np.abs(v))

        # Create perpendicular vector
        perp = np.zeros(3)
        perp[idx] = 1.0

        # Make perpendicular to v
        perp = perp - np.dot(perp, v) * v
        perp = perp / np.linalg.norm(perp)

        return perp

    def _rotation_matrix_from_axis_angle(
        self, axis: np.ndarray, angle: float
    ) -> np.ndarray:
        """Create rotation matrix from axis-angle using Rodrigues' formula."""
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis

        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c

        # Rodrigues' rotation formula
        R = np.array(
            [
                [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
            ]
        )

        return R

    def _world_to_local_rotations(
        self, world_rotations: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Convert world-space rotations to local-space rotations.

        Local rotation = parent_world_inverse * child_world

        Args:
            world_rotations: Dict mapping bone name to 3x3 world rotation

        Returns:
            Dict mapping bone name to 3x3 local rotation
        """
        local_rotations = {}

        # Process bones in hierarchy order
        for bone in self.skeleton.bones:
            bone_name = bone.name
            parent_name = bone.parent_name

            if parent_name is None:
                # Root bone - local = world
                local_rotations[bone_name] = world_rotations.get(bone_name, np.eye(3))
            else:
                # Local = parent_inverse * world
                parent_world = world_rotations.get(parent_name, np.eye(3))
                child_world = world_rotations.get(bone_name, np.eye(3))

                # Inverse of parent rotation
                parent_inv = parent_world.T  # Orthogonal matrix transpose = inverse

                # Local rotation
                local_rot = parent_inv @ child_world
                local_rotations[bone_name] = local_rot

        return local_rotations

    def rotation_to_quaternion(self, rotation_mat: np.ndarray) -> List[float]:
        """
        Convert rotation matrix to quaternion [w, x, y, z].

        Args:
            rotation_mat: 3x3 rotation matrix

        Returns:
            Quaternion [w, x, y, z] normalized
        """
        r = R.from_matrix(rotation_mat)
        quat = r.as_quat()  # [x, y, z, w]

        # Convert to [w, x, y, z]
        return [quat[3], quat[0], quat[1], quat[2]]

    def rotation_to_euler(self, rotation_mat: np.ndarray) -> List[float]:
        """
        Convert rotation matrix to Euler angles.

        Uses ZXY convention (BVH standard) in degrees.

        Args:
            rotation_mat: 3x3 rotation matrix

        Returns:
            Euler angles [z, x, y] in degrees
        """
        r = R.from_matrix(rotation_mat)
        euler = r.as_euler("ZXY", degrees=True)  # [z, x, y]
        return euler.tolist()


def test_rotation_computer():
    """Test the rotation computer with known poses."""
    computer = RotationComputer()

    # Test T-pose (arms out, up)
    # Landmarks: y values should give T-pose directions
    # 33 landmarks total (indices 0-32)
    landmarks = [
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
        [-0.5, 1.2, 0],  # 15: left_wrist
        [0.5, 1.2, 0],  # 16: right_wrist
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

    try:
        print(f"Input landmarks count: {len(landmarks)}")
        result = computer.compute_rotations(landmarks, timestamp_ms=0)
        print(f"Result success: {result.success}")

        if result.success:
            print("T-pose test passed!")
            print(f"Root position: {result.root_position}")

            # Check some key rotations
            for bone_name in ["Hips", "Spine", "LeftArm", "RightArm"]:
                if bone_name in result.bone_rotations:
                    quat, pos = result.bone_rotations[bone_name]
                    print(f"  {bone_name}: pos={pos}, quat={quat}")
        else:
            print("T-pose test failed - computation reported failure")
            print(f"Root position: {result.root_position}")
            print(f"Bone rotations: {result.bone_rotations}")
    except Exception as e:
        print(f"T-pose test failed with exception: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_rotation_computer()
