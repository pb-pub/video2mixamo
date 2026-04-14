# Video to Maximo - Skeleton Definition Module

"""
Skeleton hierarchy definition for Mixamo-compatible BVH animation.

Defines the bone parent-child tree, landmark-to-bone mapping,
T-pose reference directions, and BVH-compatible joint names.

Data Structures:
    Bone: Single bone definition with name, parent, and landmarks
    Skeleton: Full skeleton with all bones and T-pose reference
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class Bone:
    """
    Definition of a single bone in the skeleton.
    
    Attributes:
        name: Bone name (BVH-compatible)
        parent_name: Name of parent bone (None for root)
        start_landmark: MediaPipe landmark index for bone start
        end_landmark: MediaPipe landmark index for bone end
        channel_order: BVH channel order (e.g., 'ZXY', 'XYZ')
    """
    name: str
    parent_name: Optional[str]
    start_landmark: int
    end_landmark: int
    channel_order: str = "ZXY"


# Mixamo bone hierarchy with MediaPipe landmark mapping
# MediaPipe landmarks: 0=nose, 11=left_shoulder, 12=right_shoulder, etc.
#
# Mapping strategy:
#   - Hips: midpoint(23, 24) - root position + rotation
#   - Spine chain: midpoint(23,24) -> midpoint(11,12) -> midpoint(9,10) -> 9/10 -> 0
#   - Arms: shoulder -> elbow -> wrist
#   - Legs: hip -> knee -> ankle -> foot_index
#
# Note: MediaPipe doesn't provide all joints (e.g., hands, feet bones).
# We use the best available landmarks for a functional skeleton.

SKELETON_BONES: List[Bone] = [
    # Root bone - Hips (position from midpoint of hips, rotation controls entire body)
    Bone(
        name="Hips",
        parent_name=None,
        start_landmark=23,  # left_hip (used with 24 for position)
        end_landmark=24,    # right_hip
        channel_order="ZXY"
    ),
    
    # Spine chain - distributed across multiple bones
    # Spine: from hips to mid-chest
    Bone(
        name="Spine",
        parent_name="Hips",
        start_landmark=23,  # midpoint of hips
        end_landmark=11,    # midpoint of shoulders
        channel_order="ZXY"
    ),
    
    # Spine1: mid-chest
    Bone(
        name="Spine1",
        parent_name="Spine",
        start_landmark=11,  # shoulders
        end_landmark=9,     # midpoint of nose area
        channel_order="ZXY"
    ),
    
    # Spine2: upper chest/neck base
    Bone(
        name="Spine2",
        parent_name="Spine1",
        start_landmark=9,   # neck area
        end_landmark=9,     # same as start for rotation only
        channel_order="ZXY"
    ),
    
    # Neck
    Bone(
        name="Neck",
        parent_name="Spine2",
        start_landmark=9,   # neck midpoint
        end_landmark=0,     # nose
        channel_order="ZXY"
    ),
    
    # Head
    Bone(
        name="Head",
        parent_name="Neck",
        start_landmark=0,   # nose
        end_landmark=0,     # same point, rotation only
        channel_order="ZXY"
    ),
    
    # Right Arm
    Bone(
        name="RightShoulder",
        parent_name="Spine2",
        start_landmark=12,  # right_shoulder
        end_landmark=12,    # anchor point
        channel_order="ZXY"
    ),
    Bone(
        name="RightArm",
        parent_name="RightShoulder",
        start_landmark=12,  # right_shoulder
        end_landmark=14,    # right_elbow
        channel_order="ZXY"
    ),
    Bone(
        name="RightForeArm",
        parent_name="RightArm",
        start_landmark=14,  # right_elbow
        end_landmark=16,    # right_wrist
        channel_order="ZXY"
    ),
    Bone(
        name="RightHand",
        parent_name="RightForeArm",
        start_landmark=16,  # right_wrist
        end_landmark=16,    # hand tip
        channel_order="ZXY"
    ),
    
    # Left Arm
    Bone(
        name="LeftShoulder",
        parent_name="Spine2",
        start_landmark=11,  # left_shoulder
        end_landmark=11,    # anchor point
        channel_order="ZXY"
    ),
    Bone(
        name="LeftArm",
        parent_name="LeftShoulder",
        start_landmark=11,  # left_shoulder
        end_landmark=13,    # left_elbow
        channel_order="ZXY"
    ),
    Bone(
        name="LeftForeArm",
        parent_name="LeftArm",
        start_landmark=13,  # left_elbow
        end_landmark=15,    # left_wrist
        channel_order="ZXY"
    ),
    Bone(
        name="LeftHand",
        parent_name="LeftForeArm",
        start_landmark=15,  # left_wrist
        end_landmark=15,    # hand tip
        channel_order="ZXY"
    ),
    
    # Right Leg
    Bone(
        name="RightUpLeg",
        parent_name="Hips",
        start_landmark=24,  # right_hip
        end_landmark=26,    # right_knee
        channel_order="ZXY"
    ),
    Bone(
        name="RightLeg",
        parent_name="RightUpLeg",
        start_landmark=26,  # right_knee
        end_landmark=28,    # right_ankle
        channel_order="ZXY"
    ),
    Bone(
        name="RightFoot",
        parent_name="RightLeg",
        start_landmark=28,  # right_ankle
        end_landmark=32,    # right_foot_index
        channel_order="ZXY"
    ),
    
    # Left Leg
    Bone(
        name="LeftUpLeg",
        parent_name="Hips",
        start_landmark=23,  # left_hip
        end_landmark=25,    # left_knee
        channel_order="ZXY"
    ),
    Bone(
        name="LeftLeg",
        parent_name="LeftUpLeg",
        start_landmark=25,  # left_knee
        end_landmark=27,    # left_ankle
        channel_order="ZXY"
    ),
    Bone(
        name="LeftFoot",
        parent_name="LeftLeg",
        start_landmark=27,  # left_ankle
        end_landmark=31,    # left_foot_index
        channel_order="ZXY"
    ),
]


class Skeleton:
    """
    Full skeleton definition with bone hierarchy and T-pose.
    
    The skeleton maps MediaPipe 33 landmarks to a Mixamo-compatible
    hierarchy for animation export.
    
    Key coordinate system notes:
        - MediaPipe world landmarks: Y-up, hip-centered (approximate)
        - BVH: Y-up, origin at Hips
        - We use Y-up throughout for consistency
    """
    
    def __init__(self, bones: List[Bone] = None):
        """Initialize skeleton with bone definitions."""
        self.bones = bones if bones is not None else SKELETON_BONES
        self._bone_map: Dict[str, Bone] = {b.name: b for b in self.bones}
        self._children_map: Dict[str, List[str]] = {}
        self._build_hierarchy()
        self._t_pose_ref: Dict[str, np.ndarray] = self._compute_t_pose_reference()
    
    def _build_hierarchy(self) -> None:
        """Build parent-child mapping."""
        for bone in self.bones:
            if bone.parent_name is None:
                continue
            if bone.parent_name not in self._children_map:
                self._children_map[bone.parent_name] = []
            self._children_map[bone.parent_name].append(bone.name)
    
    def _compute_t_pose_reference(self) -> Dict[str, np.ndarray]:
        """
        Compute reference direction vectors for T-pose.
        
        T-pose: Arms outstretched, palms forward, legs together.
        
        Returns dict mapping bone name to normalized direction vector.
        """
        # Reference directions for T-pose (normalized)
        # These are based on a typical human proportion
        # MediaPipe world coordinates are in meters, hip-centered
        
        t_pose_refs = {}
        
        # Hips: No direction (root)
        t_pose_refs["Hips"] = np.array([0, 0, 0])
        
        # Spine: pointing upward (Y)
        t_pose_refs["Spine"] = np.array([0, 1, 0])
        t_pose_refs["Spine1"] = np.array([0, 1, 0])
        t_pose_refs["Spine2"] = np.array([0, 1, 0])
        
        # Neck: pointing upward (Y)
        t_pose_refs["Neck"] = np.array([0, 1, 0])
        
        # Head: pointing upward (Y)
        t_pose_refs["Head"] = np.array([0, 1, 0])
        
        # Shoulders: pointing outward (X)
        t_pose_refs["RightShoulder"] = np.array([1, 0, 0])
        t_pose_refs["LeftShoulder"] = np.array([-1, 0, 0])
        
        # Arms: pointing down (negative Y)
        t_pose_refs["RightArm"] = np.array([0, -1, 0])
        t_pose_refs["LeftArm"] = np.array([0, -1, 0])
        
        # Forearms: pointing down (negative Y)
        t_pose_refs["RightForeArm"] = np.array([0, -1, 0])
        t_pose_refs["LeftForeArm"] = np.array([0, -1, 0])
        
        # Hands: pointing down (negative Y)
        t_pose_refs["RightHand"] = np.array([0, -1, 0])
        t_pose_refs["LeftHand"] = np.array([0, -1, 0])
        
        # Legs: pointing down (negative Y)
        t_pose_refs["RightUpLeg"] = np.array([0, -1, 0])
        t_pose_refs["LeftUpLeg"] = np.array([0, -1, 0])
        
        # Lower legs: pointing down (negative Y)
        t_pose_refs["RightLeg"] = np.array([0, -1, 0])
        t_pose_refs["LeftLeg"] = np.array([0, -1, 0])
        
        # Feet: pointing forward (Z) and slightly down
        t_pose_refs["RightFoot"] = np.array([0, -0.5, 0.866])  # 30 degrees
        t_pose_refs["LeftFoot"] = np.array([0, -0.5, 0.866])
        
        return t_pose_refs
    
    def get_bone(self, name: str) -> Bone:
        """Get bone by name."""
        return self._bone_map.get(name)
    
    def get_children(self, name: str) -> List[str]:
        """Get child bones of a given bone."""
        return self._children_map.get(name, [])
    
    def get_root(self) -> Bone:
        """Get root bone (Hips)."""
        for bone in self.bones:
            if bone.parent_name is None:
                return bone
        raise ValueError("No root bone found")
    
    def get_world_positions(
        self,
        landmarks: List[List[float]]
    ) -> Dict[str, np.ndarray]:
        """
        Compute world positions for each bone from landmarks.
        
        Args:
            landmarks: 33x3 list of [x, y, z] positions
        
        Returns:
            Dict mapping bone name to position vector
        """
        positions = {}
        
        for bone in self.bones:
            start_idx = bone.start_landmark
            end_idx = bone.end_landmark
            
            start_pos = np.array(landmarks[start_idx])
            
            # For some bones (like shoulders), use same point
            if start_idx == end_idx:
                end_pos = start_pos
            else:
                end_pos = np.array(landmarks[end_idx])
            
            # For Hips, use midpoint of 23 and 24
            if bone.name == "Hips":
                left_hip = np.array(landmarks[23])
                right_hip = np.array(landmarks[24])
                positions[bone.name] = (left_hip + right_hip) / 2
            else:
                # Use midpoint of start and end for bone origin
                positions[bone.name] = (start_pos + end_pos) / 2
        
        return positions
    
    def get_direction(
        self,
        positions: Dict[str, np.ndarray],
        bone_name: str
    ) -> np.ndarray:
        """
        Compute direction vector for a bone from positions.
        
        Args:
            positions: Dict of bone name -> position
            bone_name: Name of bone
        
        Returns:
            Normalized direction vector
        """
        bone = self._bone_map.get(bone_name)
        if bone is None:
            raise ValueError(f"Unknown bone: {bone_name}")
        
        start_pos = positions[bone_name]
        
        # Get child position (bone points toward child)
        children = self._children_map.get(bone_name, [])
        if children:
            child_pos = positions[children[0]]
            direction = child_pos - start_pos
        else:
            # Leaf bone - use reference direction
            direction = self._t_pose_ref.get(bone_name, np.array([0, 0, 0]))
        
        # Normalize
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = direction / norm
        
        return direction
    
    def get_t_pose_rotation(
        self,
        current_direction: np.ndarray,
        bone_name: str
    ) -> np.ndarray:
        """
        Compute rotation from T-pose to current direction.
        
        Args:
            current_direction: Current normalized direction vector
            bone_name: Name of bone
        
        Returns:
            Rotation matrix (3x3)
        """
        t_pose_dir = self._t_pose_ref.get(bone_name)
        if t_pose_dir is None:
            return np.eye(3)
        
        # Compute rotation that aligns t_pose_dir to current_direction
        t_pose_dir = t_pose_dir / np.linalg.norm(t_pose_dir)
        
        # Cross product for rotation axis
        cross = np.cross(t_pose_dir, current_direction)
        cross_norm = np.linalg.norm(cross)
        
        # Parallel vectors - return identity
        if cross_norm < 1e-6:
            return np.eye(3)
        
        # Compute rotation axis and angle
        axis = cross / cross_norm
        dot = np.dot(t_pose_dir, current_direction)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        
        # Create rotation matrix using axis-angle
        # Rodrigues' rotation formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        return R


def get_landmark_indices() -> Dict[str, int]:
    """
    Get MediaPipe landmark index mapping.
    
    Returns dict mapping landmark names to indices.
    """
    return {
        "nose": 0,
        "left_eye_inner": 1,
        "left_eye": 2,
        "left_eye_outer": 3,
        "right_eye_inner": 4,
        "right_eye": 5,
        "right_eye_outer": 6,
        "left_ear": 7,
        "right_ear": 8,
        "mouth_left": 9,
        "mouth_right": 10,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_pinky": 17,
        "right_pinky": 18,
        "left_index": 19,
        "right_index": 20,
        "left_thumb": 21,
        "right_thumb": 22,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
        "left_heel": 29,
        "right_heel": 30,
        "left_foot_index": 31,
        "right_foot_index": 32,
    }


if __name__ == "__main__":
    # Test skeleton
    skeleton = Skeleton()
    
    print("Skeleton bones:")
    for bone in skeleton.bones:
        print(f"  {bone.name}: parent={bone.parent_name}, "
              f"start={bone.start_landmark}, end={bone.end_landmark}")
    
    print("\nLandmark indices:")
    indices = get_landmark_indices()
    for name, idx in sorted(indices.items(), key=lambda x: x[1]):
        print(f"  {idx}: {name}")
    
    print("\nT-pose references:")
    for name, ref in skeleton._t_pose_ref.items():
        print(f"  {name}: {ref}")
