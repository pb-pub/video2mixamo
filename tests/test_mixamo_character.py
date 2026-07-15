"""
Unit tests for JointTree and MixamoCharacter.compute_pose_rotations.

No GLB file is needed: synthetic skeletons are built from SimpleNamespace
nodes that expose the same interface pygltflib Node objects do.
"""

import math
import types

from video_to_maximo.mixamo_character import JointTree, MixamoCharacter, _BONE_LM
from video_to_maximo.quaternion import Quaternion
from video_to_maximo.vector import Vector3


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _node(name, children=None, translation=None, rotation=None):
    return types.SimpleNamespace(
        name=name,
        children=list(children or []),
        translation=list(translation) if translation is not None else None,
        rotation=list(rotation) if rotation is not None else None,
    )


def _landmarks(overrides: dict | None = None) -> list:
    """Return 33 zero-Vector3 landmarks, with optional per-index overrides."""
    lms = [Vector3(0.0, 0.0, 0.0) for _ in range(33)]
    if overrides:
        for idx, (x, y, z) in overrides.items():
            lms[idx] = Vector3(x, y, z)
    return lms


def _make_char(nodes: list, root_idx: int) -> MixamoCharacter:
    """Build a MixamoCharacter with can_animate=True and a synthetic JointTree."""
    char = object.__new__(MixamoCharacter)
    char.can_animate = True
    char.joint_tree = JointTree(nodes, root_idx)
    return char


def _q_close(q1: Quaternion, q2: Quaternion, tol: float = 1e-5) -> bool:
    """True if q1 ≈ q2 or q1 ≈ -q2 (both encode the same rotation)."""
    return all(abs(a - b) < tol for a, b in zip(q1, q2)) or all(
        abs(a + b) < tol for a, b in zip(q1, q2)
    )


def _v3_close(v1: Vector3, v2: Vector3, tol: float = 1e-5) -> bool:
    return (
        abs(v1.x - v2.x) < tol
        and abs(v1.y - v2.y) < tol
        and abs(v1.z - v2.z) < tol
    )


# ---------------------------------------------------------------------------
# Synthetic skeleton fixtures
#
#  SPINE_NODES   : Hips(0) → Spine(1) → Spine1(2)
#  ARM_NODES     : Hips(0) → LeftArm(1) → LeftForeArm(2) → LeftHand(3)
#  SHOULDER_NODES: Hips(0) → LeftShoulder(1) → LeftArm(2)
#
# All rest rotations are identity; all translations point along +Y.
# ---------------------------------------------------------------------------

SPINE_NODES = [
    _node("mixamorig:Hips", children=[1], translation=[0.0, 0.0, 0.0]),
    _node("mixamorig:Spine", children=[2], translation=[0.0, 1.0, 0.0]),
    _node("mixamorig:Spine1", children=[], translation=[0.0, 1.0, 0.0]),
]

ARM_NODES = [
    _node("mixamorig:Hips", children=[1], translation=[0.0, 0.0, 0.0]),
    _node("mixamorig:LeftArm", children=[2], translation=[0.0, 1.0, 0.0]),
    _node("mixamorig:LeftForeArm", children=[3], translation=[0.0, 1.0, 0.0]),
    _node("mixamorig:LeftHand", children=[], translation=[0.0, 1.0, 0.0]),
]

SHOULDER_NODES = [
    _node("mixamorig:Hips", children=[1], translation=[0.0, 0.0, 0.0]),
    _node("mixamorig:LeftShoulder", children=[2], translation=[0.0, 1.0, 0.0]),
    _node("mixamorig:LeftArm", children=[], translation=[0.0, 1.0, 0.0]),
]


# ===========================================================================
# JointTree
# ===========================================================================


class TestJointTree:
    def test_walk_root_is_first(self):
        tree = JointTree(SPINE_NODES, root_idx=0)
        order = [idx for idx, _ in tree.walk()]
        assert order[0] == 0

    def test_walk_visits_all_nodes(self):
        tree = JointTree(SPINE_NODES, root_idx=0)
        order = [idx for idx, _ in tree.walk()]
        assert sorted(order) == [0, 1, 2]

    def test_walk_parent_before_child(self):
        tree = JointTree(SPINE_NODES, root_idx=0)
        order = [idx for idx, _ in tree.walk()]
        assert order.index(1) < order.index(2)  # Spine before Spine1

    def test_world_rotations_all_identity_when_rest_is_identity(self):
        tree = JointTree(SPINE_NODES, root_idx=0)
        for q in tree.world_rotations()[:3]:
            assert _q_close(q, Quaternion.identity())

    def test_world_rotation_accumulates_local(self):
        # 90° around Z at Spine: world rot of Spine should be that same 90°.
        s = math.sqrt(2) / 2
        nodes = [
            _node("mixamorig:Hips", children=[1]),
            _node(
                "mixamorig:Spine",
                children=[2],
                translation=[0.0, 1.0, 0.0],
                rotation=[0.0, 0.0, s, s],  # 90° around +Z
            ),
            _node("mixamorig:Spine1", children=[], translation=[0.0, 1.0, 0.0]),
        ]
        tree = JointTree(nodes, root_idx=0)
        world_rots = tree.world_rotations()
        assert _q_close(world_rots[1], Quaternion.from_list([0.0, 0.0, s, s]))

    def test_bone_direction_equals_normalized_translation_when_rest_is_identity(self):
        # bone_direction(Spine) = direction from Hips to Spine = (0, 1, 0).
        tree = JointTree(SPINE_NODES, root_idx=0)
        d = tree.bone_direction(1, tree.world_rotations())
        assert _v3_close(d, Vector3(0.0, 1.0, 0.0))

    def test_parent_of_root_is_none(self):
        assert JointTree(SPINE_NODES, root_idx=0).parent(0) is None

    def test_parent_of_child_returns_parent_index(self):
        tree = JointTree(SPINE_NODES, root_idx=0)
        assert tree.parent(1) == 0
        assert tree.parent(2) == 1

    def test_find_returns_correct_index(self):
        tree = JointTree(SPINE_NODES, root_idx=0)
        assert tree.find("mixamorig:Spine") == 1
        assert tree.find("mixamorig:Spine1") == 2
        assert tree.find("does_not_exist") is None


# ===========================================================================
# compute_pose_rotations
# ===========================================================================


class TestComputePoseRotations:

    # --- guard: can_animate=False ---

    def test_not_animatable_returns_empty_dict(self):
        char = object.__new__(MixamoCharacter)
        char.can_animate = False
        assert char.compute_pose_rotations(_landmarks()) == {}

    # --- bones that must never appear in the result ---

    def test_hips_always_excluded(self):
        char = _make_char(SPINE_NODES, root_idx=0)
        assert "Hips" not in char.compute_pose_rotations(_landmarks())

    def test_equal_lm_indices_excluded(self):
        # LeftShoulder: lm_a == lm_b == 11 in _BONE_LM → no directional target.
        assert _BONE_LM["LeftShoulder"][0] == _BONE_LM["LeftShoulder"][1]
        char = _make_char(SHOULDER_NODES, root_idx=0)
        assert "LeftShoulder" not in char.compute_pose_rotations(_landmarks())

    def test_leaf_joint_excluded(self):
        # Spine1 has no children, so no bone direction can be defined.
        char = _make_char(SPINE_NODES, root_idx=0)
        assert "Spine1" not in char.compute_pose_rotations(_landmarks())

    def test_zero_length_direction_excluded(self):
        # lm_a and lm_b at the same position → zero vector → no rotation.
        char = _make_char(SPINE_NODES, root_idx=0)
        lms = _landmarks({23: (1.0, 0.5, 0.0), 11: (1.0, 0.5, 0.0)})
        assert "Spine" not in char.compute_pose_rotations(lms)

    def test_unmapped_node_not_in_result(self):
        nodes = [
            _node("mixamorig:Hips", children=[1]),
            _node("mixamorig:UnknownBone", children=[], translation=[0.0, 1.0, 0.0]),
        ]
        char = _make_char(nodes, root_idx=0)
        assert "UnknownBone" not in char.compute_pose_rotations(_landmarks())

    def test_lm_equal_lm_hand_excluded(self):
        assert _BONE_LM["LeftHand"][0] == _BONE_LM["LeftHand"][1]
        char = _make_char(ARM_NODES, root_idx=0)
        assert "LeftHand" not in char.compute_pose_rotations(_landmarks())

    # --- correct rotation values ---

    def test_rest_pose_landmarks_yield_identity_local_rotation(self):
        # In MediaPipe Y-down, the rest "upward" spine direction is (0, -1, 0).
        # GLB conversion flips Y → (0, +1, 0) which matches the rest bone dir.
        # Swing = identity.
        char = _make_char(SPINE_NODES, root_idx=0)
        # _BONE_LM["Spine"] = (23, 11): direction = lm[11] - lm[23]
        lms = _landmarks({23: (0.0, 0.0, 0.0), 11: (0.0, -1.0, 0.0)})
        result = char.compute_pose_rotations(lms)
        assert "Spine" in result
        assert _q_close(result["Spine"], Quaternion.identity())

    def test_90_degree_swing_around_z(self):
        # Target direction (1, 0, 0) in GLB, rest bone direction (0, 1, 0).
        # from_two_vectors((0,1,0), (1,0,0)) = Quaternion(0, 0, -1/√2, 1/√2).
        char = _make_char(SPINE_NODES, root_idx=0)
        lms = _landmarks({23: (0.0, 0.0, 0.0), 11: (1.0, 0.0, 0.0)})
        result = char.compute_pose_rotations(lms)
        assert "Spine" in result
        s = math.sqrt(2) / 2
        assert _q_close(result["Spine"], Quaternion(0.0, 0.0, -s, s))

    def test_rotated_bone_points_at_mp_target(self):
        # Applying the returned local rotation to the child-local direction
        # (0, 1, 0) should yield the GLB target direction.
        # Since Hips world rot = identity, local rot == world rot for Spine.
        char = _make_char(SPINE_NODES, root_idx=0)
        lms = _landmarks({23: (0.0, 0.0, 0.0), 11: (1.0, 0.0, 0.0)})
        result = char.compute_pose_rotations(lms)
        rotated = result["Spine"].rotate(Vector3(0.0, 1.0, 0.0))
        assert _v3_close(rotated, Vector3(1.0, 0.0, 0.0))

    def test_y_negation_flips_mp_down_to_glb_up(self):
        # MP direction (0, +1, 0) = "downward" in world (MP is Y-down).
        # GLB conversion → (0, -1, 0) = downward in GLB.
        # Applying the swing to rest direction (0, 1, 0) should give (0, -1, 0).
        char = _make_char(SPINE_NODES, root_idx=0)
        lms = _landmarks({23: (0.0, 0.0, 0.0), 11: (0.0, 1.0, 0.0)})
        result = char.compute_pose_rotations(lms)
        assert "Spine" in result
        rotated = result["Spine"].rotate(Vector3(0.0, 1.0, 0.0))
        assert _v3_close(rotated, Vector3(0.0, -1.0, 0.0))

    # --- FK propagation ---

    def test_same_target_on_parent_and_child_gives_child_identity(self):
        # LeftArm targets (1, 0, 0); LeftForeArm targets (1, 0, 0) too.
        # The parent's swing already aligns the chain, so the child's local = identity.
        char = _make_char(ARM_NODES, root_idx=0)
        # _BONE_LM["LeftArm"] = (11, 13), _BONE_LM["LeftForeArm"] = (13, 15)
        lms = _landmarks({11: (0.0, 0.0, 0.0), 13: (1.0, 0.0, 0.0), 15: (2.0, 0.0, 0.0)})
        result = char.compute_pose_rotations(lms)
        assert "LeftArm" in result
        assert "LeftForeArm" in result
        assert _q_close(result["LeftForeArm"], Quaternion.identity())

    def test_different_targets_for_arm_and_forearm(self):
        # LeftArm → +X, LeftForeArm → +Z.
        # Composing arm_world * forearm_local and rotating the child's local dir
        # (0, 1, 0) should produce (0, 0, 1) in world space.
        char = _make_char(ARM_NODES, root_idx=0)
        lms = _landmarks(
            {
                11: (0.0, 0.0, 0.0),
                13: (1.0, 0.0, 0.0),   # arm target = (1,0,0) in GLB
                15: (1.0, 0.0, 1.0),   # forearm target = (0,0,1) in GLB
            }
        )
        result = char.compute_pose_rotations(lms)
        assert "LeftArm" in result
        assert "LeftForeArm" in result

        # q_arm_world is the swing that takes (0,1,0) to (1,0,0)
        q_arm_world = Quaternion.from_two_vectors(Vector3(0, 1, 0), Vector3(1, 0, 0))
        rotated = (q_arm_world * result["LeftForeArm"]).rotate(Vector3(0.0, 1.0, 0.0))
        assert _v3_close(rotated, Vector3(0.0, 0.0, 1.0))

    def test_fk_pose_world_equals_parent_times_local(self):
        # Verify the invariant: pose_world[child] == pose_world[parent] * local[child].
        # We reconstruct pose_world from the returned local rotations and check the
        # final bone direction.
        char = _make_char(ARM_NODES, root_idx=0)
        lms = _landmarks(
            {
                11: (0.0, 0.0, 0.0),
                13: (1.0, 0.0, 0.0),
                15: (1.0, 0.0, 1.0),
            }
        )
        result = char.compute_pose_rotations(lms)

        # Hips = identity (root, skipped)
        q_hips = Quaternion.identity()
        q_arm_world = q_hips * result["LeftArm"]
        q_fore_world = q_arm_world * result["LeftForeArm"]

        # LeftForeArm targets (0, 0, 1) → its bone direction should be (0, 0, 1)
        bone_dir = q_fore_world.rotate(Vector3(0.0, 1.0, 0.0))
        assert _v3_close(bone_dir, Vector3(0.0, 0.0, 1.0))


# ===========================================================================
# Non-identity rest rotations
# ===========================================================================


class TestNonIdentityRestRotations:
    """Verify that the FK and swing rotation work when the skeleton is not in
    the all-identity rest pose (as real Mixamo GLBs typically are)."""

    def _spine_with_rest_rot(self, rest_quat_xyzw: list):
        """Return a char whose Spine has the given rest rotation [x,y,z,w]."""
        nodes = [
            _node("mixamorig:Hips", children=[1]),
            _node(
                "mixamorig:Spine",
                children=[2],
                translation=[0.0, 1.0, 0.0],
                rotation=rest_quat_xyzw,
            ),
            _node("mixamorig:Spine1", children=[], translation=[0.0, 1.0, 0.0]),
        ]
        return _make_char(nodes, root_idx=0)

    def test_world_rotations_3_level_non_identity(self):
        # 3-level chain: each joint has a 90°-Z rest rotation.
        # world[0]=id, world[1]=90°Z, world[2]=90°Z*90°Z=180°Z.
        s = math.sqrt(2) / 2
        q90z = [0.0, 0.0, s, s]
        nodes = [
            _node("mixamorig:Hips", children=[1]),
            _node("mixamorig:Spine", children=[2], translation=[0, 1, 0], rotation=q90z),
            _node("mixamorig:Spine1", children=[], translation=[0, 1, 0], rotation=q90z),
        ]
        tree = JointTree(nodes, root_idx=0)
        w = tree.world_rotations()
        # world[1] = id * 90°Z = 90°Z
        assert _q_close(w[1], Quaternion.from_list(q90z))
        # world[2] = 90°Z * 90°Z = 180°Z = Quaternion(0,0,1,0)
        q180z = Quaternion(0.0, 0.0, 1.0, 0.0)
        assert _q_close(w[2], q180z)

    def test_rest_bone_dir_reflects_non_identity_parent_rotation(self):
        # With Spine rest-rotation = 90°Z, child translation (0,1,0) in local frame
        # becomes (-1,0,0) in world space (90°Z rotates +Y → -X).
        s = math.sqrt(2) / 2
        char = self._spine_with_rest_rot([0.0, 0.0, s, s])
        rest_world_rots = char.joint_tree.world_rotations()
        # rest_world_rots[Spine] = Quaternion(0,0,s,s); rotate (0,1,0) → (-1,0,0)
        child_dir = rest_world_rots[1].rotate(Vector3(0.0, 1.0, 0.0))
        assert _v3_close(child_dir, Vector3(-1.0, 0.0, 0.0))

    def test_compute_pose_swing_accounts_for_rest_rotation(self):
        # Rest bone direction = (-1,0,0) (from 90°Z rest).
        # MP target: lm[11]-lm[23] = (1,0,0) in MP → GLB = (1,0,0) (+X).
        # Swing must rotate (-1,0,0) → (+1,0,0).
        # Applying q_hips(id) * result["Spine"] to child_local (0,1,0) must give (1,0,0).
        s = math.sqrt(2) / 2
        char = self._spine_with_rest_rot([0.0, 0.0, s, s])
        lms = _landmarks({23: (0.0, 0.0, 0.0), 11: (1.0, 0.0, 0.0)})
        result = char.compute_pose_rotations(lms)
        assert "Spine" in result
        # Hips world = identity → Spine world = result["Spine"]
        bone_dir = result["Spine"].rotate(Vector3(0.0, 1.0, 0.0))
        assert _v3_close(bone_dir, Vector3(1.0, 0.0, 0.0))

    def test_rest_pose_target_with_non_identity_rest_rot(self):
        # The rest direction of a 90°Z Spine is (-1,0,0).
        # If we feed landmarks whose GLB direction is (-1,0,0) (= rest direction),
        # the swing is identity and the returned local rotation should equal
        # the rest local rotation (i.e. identity swing → pose_local = rest_local).
        #
        # GLB direction (-1,0,0): MP (−1,0,0) since Y component is 0.
        # → lm[11] - lm[23] = (-1, 0, 0).
        s = math.sqrt(2) / 2
        char = self._spine_with_rest_rot([0.0, 0.0, s, s])
        lms = _landmarks({23: (0.0, 0.0, 0.0), 11: (-1.0, 0.0, 0.0)})
        result = char.compute_pose_rotations(lms)
        assert "Spine" in result
        # Swing is identity → pose_world[Spine] = id * rest_world[Spine] = rest_world[Spine].
        # pose_local = q_parent.inv() * rest_world = id.inv() * rest_world = rest_local.
        rest_local = Quaternion.from_list([0.0, 0.0, s, s])
        assert _q_close(result["Spine"], rest_local)

    def test_non_unit_child_translation_is_normalized(self):
        # When the child translation is not unit-length, it must be normalized
        # before being used as the local bone direction.
        # Use translation (0, 5, 0): normalize → (0,1,0), same as (0,1,0).
        nodes = [
            _node("mixamorig:Hips", children=[1]),
            _node("mixamorig:Spine", children=[2], translation=[0.0, 1.0, 0.0]),
            _node("mixamorig:Spine1", children=[], translation=[0.0, 5.0, 0.0]),
        ]
        char_long = _make_char(nodes, root_idx=0)
        char_unit = _make_char(SPINE_NODES, root_idx=0)
        lms = _landmarks({23: (0.0, 0.0, 0.0), 11: (1.0, 0.0, 0.0)})
        # Both should produce the same local rotation.
        r_long = char_long.compute_pose_rotations(lms)
        r_unit = char_unit.compute_pose_rotations(lms)
        assert _q_close(r_long["Spine"], r_unit["Spine"])


# ===========================================================================
# Branching tree (multiple children at the same parent)
# ===========================================================================


class TestBranchingTree:
    """Verify parent-map correctness and independent FK chains in a Y-shaped tree.

    Hierarchy::
        Hips(0)
        ├── Spine(1) → Spine1(2)
        └── LeftUpLeg(3) → LeftLeg(4)
    """

    BRANCH_NODES = [
        _node("mixamorig:Hips", children=[1, 3], translation=[0.0, 0.0, 0.0]),
        _node("mixamorig:Spine", children=[2], translation=[0.0, 1.0, 0.0]),
        _node("mixamorig:Spine1", children=[], translation=[0.0, 1.0, 0.0]),
        _node("mixamorig:LeftUpLeg", children=[4], translation=[-0.5, -1.0, 0.0]),
        _node("mixamorig:LeftLeg", children=[], translation=[0.0, -1.0, 0.0]),
    ]

    def test_parent_map_for_all_nodes(self):
        tree = JointTree(self.BRANCH_NODES, root_idx=0)
        assert tree.parent(0) is None  # root
        assert tree.parent(1) == 0     # Spine ← Hips
        assert tree.parent(2) == 1     # Spine1 ← Spine
        assert tree.parent(3) == 0     # LeftUpLeg ← Hips
        assert tree.parent(4) == 3     # LeftLeg ← LeftUpLeg

    def test_walk_visits_all_5_nodes(self):
        tree = JointTree(self.BRANCH_NODES, root_idx=0)
        visited = [idx for idx, _ in tree.walk()]
        assert sorted(visited) == [0, 1, 2, 3, 4]

    def test_walk_all_parents_before_their_children(self):
        tree = JointTree(self.BRANCH_NODES, root_idx=0)
        order = {idx: pos for pos, (idx, _) in enumerate(tree.walk())}
        assert order[0] < order[1]  # Hips before Spine
        assert order[1] < order[2]  # Spine before Spine1
        assert order[0] < order[3]  # Hips before LeftUpLeg
        assert order[3] < order[4]  # LeftUpLeg before LeftLeg

    def test_world_rotations_independent_branches(self):
        # Both branches start from Hips (identity world rot).
        # With all identity local rots, all world rots should be identity.
        tree = JointTree(self.BRANCH_NODES, root_idx=0)
        w = tree.world_rotations()
        for q in w:
            assert _q_close(q, Quaternion.identity())

    def test_compute_pose_two_independent_chains(self):
        # Spine targets +X; LeftUpLeg targets +Y.
        # Both chains are independent — each should get the correct swing.
        char = _make_char(self.BRANCH_NODES, root_idx=0)
        # _BONE_LM["Spine"] = (23, 11)   → direction lm[11]-lm[23]
        # _BONE_LM["LeftUpLeg"] = (23, 25) → direction lm[25]-lm[23]
        lms = _landmarks({
            23: (0.0, 0.0, 0.0),
            11: (1.0, 0.0, 0.0),   # Spine target: +X in GLB
            25: (0.0, -1.0, 0.0),  # LeftUpLeg target: MP (0,-1,0) → GLB (0,+1,0)
        })
        result = char.compute_pose_rotations(lms)
        assert "Spine" in result
        assert "LeftUpLeg" in result

        # Spine: world rot = identity (Hips) * result["Spine"].
        # Spine's first child translation = (0,1,0); bone_dir should be +X.
        spine_bone_dir = result["Spine"].rotate(Vector3(0.0, 1.0, 0.0))
        assert _v3_close(spine_bone_dir, Vector3(1.0, 0.0, 0.0))

        # LeftUpLeg: first child translation = (0,-1,0); bone_dir should be +Y.
        # rest_bone_dir was (0,-1,0) in world space (identity parent, (0,-1,0) local).
        # Swing rotates (0,-1,0) → (0,+1,0); applying result["LeftUpLeg"] to (0,-1,0) normalized.
        leg_child_local = Vector3(0.0, -1.0, 0.0).normalize()
        leg_bone_dir = result["LeftUpLeg"].rotate(leg_child_local)
        assert _v3_close(leg_bone_dir, Vector3(0.0, 1.0, 0.0))

    def test_spine_rotation_does_not_affect_leg_chain(self):
        # Rotating Spine should not change LeftUpLeg's result.
        char = _make_char(self.BRANCH_NODES, root_idx=0)
        lms_base = _landmarks({23: (0.0,0.0,0.0), 11: (0.0,-1.0,0.0), 25: (0.0,-1.0,0.0)})
        lms_rotspine = _landmarks({23: (0.0,0.0,0.0), 11: (1.0,0.0,0.0), 25: (0.0,-1.0,0.0)})
        r_base = char.compute_pose_rotations(lms_base)
        r_rotspine = char.compute_pose_rotations(lms_rotspine)
        if "LeftUpLeg" in r_base and "LeftUpLeg" in r_rotspine:
            assert _q_close(r_base["LeftUpLeg"], r_rotspine["LeftUpLeg"])


# ===========================================================================
# apply_pose_rotations — LBS mesh deformation
# ===========================================================================


def _make_single_bone_char():
    """
    Two-node skeleton: Root(0) → Bone(1, translation=(0,1,0)).
    One bind-pose vertex at GLB (0,2,0), 100% weighted to Bone (joint 0 → node 1).

    Inverse bind matrix for Bone = inverse of T(0,1,0) = T(0,-1,0).
    """
    import numpy as np

    nodes = [
        _node("Root", children=[1], translation=[0.0, 0.0, 0.0]),
        _node("Bone", children=[], translation=[0.0, 1.0, 0.0]),
    ]
    char = object.__new__(MixamoCharacter)
    char.can_animate = True
    char.joint_tree = JointTree(nodes, root_idx=0)
    char._joint_nodes = [1]  # skin joint index 0  →  node index 1
    char._glb_verts = np.array([[0.0, 2.0, 0.0]], dtype=np.float32)
    char._skin_joints = np.array([[0, 0, 0, 0]], dtype=np.uint16)
    char._skin_weights = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    # inv_bind = inverse of the 4×4 world transform of Bone in bind pose = T(0,-1,0)
    char._inv_bind_matrices = np.array(
        [[[1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0], [0, 0, 0, 1]]],
        dtype=np.float32,
    )
    char.vertices = np.zeros((1, 3), dtype=np.float32)
    char.faces = np.zeros((0, 3), dtype=np.int32)
    return char


class TestApplyPoseRotations:
    def test_not_animatable_returns_rest_vertices(self):
        import numpy as np

        char = object.__new__(MixamoCharacter)
        char.can_animate = False
        char.vertices = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = char.apply_pose_rotations({})
        np.testing.assert_array_equal(result, char.vertices)

    def test_rest_pose_empty_dict_returns_bind_verts_in_display_space(self):
        import numpy as np

        char = _make_single_bone_char()
        result = char.apply_pose_rotations({})
        # GLB (0,2,0) → display (disp_x=-X, disp_y=-Z, disp_z=+Y) = (0,0,2)
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result[0], [0.0, 0.0, 2.0], atol=1e-4)

    def test_90deg_z_rotation_moves_vertex_correctly(self):
        """
        90°Z rotation at Bone (world pos (0,1,0)):
          bind vertex (0,2,0) is 1 unit along the bone (+Y from joint).
          After 90°Z rotation the bone points in the -X direction,
          so the vertex lands at GLB (-1,1,0) → display (1,0,1).
        """
        import numpy as np

        char = _make_single_bone_char()
        s = math.sqrt(2) / 2
        q_90z = Quaternion(0.0, 0.0, s, s)
        result = char.apply_pose_rotations({"Bone": q_90z})
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result[0], [1.0, 0.0, 1.0], atol=1e-4)

    def test_output_dtype_is_float32(self):
        import numpy as np

        char = _make_single_bone_char()
        result = char.apply_pose_rotations({})
        assert result.dtype == np.float32

    def test_lbs_blend_two_joints(self):
        """
        Two joints with equal 50/50 weight on one vertex.
        Joint 0 (node 1) stays at rest; joint 1 (node 2) rotates 90°Z.
        Expected: average of the two deformed positions.
        """
        import numpy as np

        nodes = [
            _node("Root", children=[1, 2], translation=[0.0, 0.0, 0.0]),
            _node("BoneA", children=[], translation=[0.0, 1.0, 0.0]),
            _node("BoneB", children=[], translation=[0.0, 1.0, 0.0]),
        ]
        inv_bind_A = np.array(
            [[1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        # BoneB has the same bind transform as BoneA
        inv_bind_B = inv_bind_A.copy()

        char = object.__new__(MixamoCharacter)
        char.can_animate = True
        char.joint_tree = JointTree(nodes, root_idx=0)
        char._joint_nodes = [1, 2]  # joint 0→node 1, joint 1→node 2
        char._glb_verts = np.array([[0.0, 2.0, 0.0]], dtype=np.float32)
        char._skin_joints = np.array([[0, 1, 0, 0]], dtype=np.uint16)
        char._skin_weights = np.array([[0.5, 0.5, 0.0, 0.0]], dtype=np.float32)
        char._inv_bind_matrices = np.stack([inv_bind_A, inv_bind_B])
        char.vertices = np.zeros((1, 3), dtype=np.float32)
        char.faces = np.zeros((0, 3), dtype=np.int32)

        s = math.sqrt(2) / 2
        q_90z = Quaternion(0.0, 0.0, s, s)
        # BoneA at rest → vertex stays at GLB (0,2,0)
        # BoneB rotated 90°Z → vertex moves to GLB (-1,1,0)
        # Blend 50/50: GLB (-0.5, 1.5, 0) → display (0.5, 0, 1.5)
        result = char.apply_pose_rotations({"BoneB": q_90z})
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result[0], [0.5, 0.0, 1.5], atol=1e-4)
