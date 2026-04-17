"""
Mixamo character GLB loader with real-time Linear Blend Skinning (LBS).

Pipeline
--------
1. Load  — parse the GLB file; extract mesh, skeleton, and rest-pose bone
           direction vectors for every joint.
2. Diff  — per frame: for each mapped joint, measure the delta rotation
           from its rest-pose direction to the current MediaPipe direction.
3. Apply — propagate delta rotations through the joint hierarchy and build
           per-joint skin matrices.
4. Draw  — apply LBS to the bind-pose mesh; return vertices in display space.

Coordinate systems
------------------
  MediaPipe world landmarks : X=right, Y=DOWN, Z=toward-camera  (Y-down)
  GLB / Mixamo bind pose    : X=right, Y=UP,   Z=forward         (Y-up)
  pyqtgraph display         : X=right, Y=depth, Z=UP             (Z-up)

Conversions::
    glb  = [mp_X, -mp_Y, mp_Z]       flip Y: Y-down → Y-up
    disp = [-glb_X, -glb_Z, +glb_Y]  mirror X, swap depth/up
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

from video_to_maximo.quaternion import Quaternion
from video_to_maximo.vector import Vector3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_N_COMP_MAP = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}

_BONE_LM: Dict[str, Tuple[int, int]] = {
    "Hips": (23, 24),  # root: position = midpoint(23,24)
    "Spine": (23, 11),  # hip-mid → shoulder-mid  (1/3 rotation)
    "Spine1": (23, 11),  # (1/3)
    "Spine2": (23, 11),  # (1/3)
    "Neck": (11, 0),  # shoulder-mid → nose
    "Head": (0, 0),  # no direction — inherit
    "LeftShoulder": (11, 11),  # no direction — inherit
    "LeftArm": (11, 13),  # shoulder → elbow
    "LeftForeArm": (13, 15),  # elbow → wrist
    "LeftHand": (15, 15),
    "RightShoulder": (12, 12),
    "RightArm": (12, 14),
    "RightForeArm": (14, 16),
    "RightHand": (16, 16),
    "LeftUpLeg": (23, 25),  # hip → knee
    "LeftLeg": (25, 27),  # knee → ankle
    "LeftFoot": (27, 31),  # ankle → foot_index
    "RightUpLeg": (24, 26),
    "RightLeg": (26, 28),
    "RightFoot": (28, 32),
}


def _read_accessor(gltf, blob: bytes, acc_idx: int) -> np.ndarray:
    """Read a glTF accessor from *blob* into a numpy array (always 2-D)."""
    acc = gltf.accessors[acc_idx]
    bv = gltf.bufferViews[acc.bufferView]

    dtype = _DTYPE_MAP[acc.componentType]
    n_comp = _N_COMP_MAP[acc.type]
    count = acc.count
    itemsize = np.dtype(dtype).itemsize * n_comp

    bv_offset = bv.byteOffset or 0
    acc_offset = acc.byteOffset or 0
    stride = bv.byteStride or itemsize

    if stride == itemsize:
        start = bv_offset + acc_offset
        raw = blob[start : start + count * itemsize]
        return np.frombuffer(raw, dtype=dtype).reshape(count, n_comp).copy()

    # Interleaved buffer view — slower path
    result = np.zeros((count, n_comp), dtype=dtype)
    for i in range(count):
        off = bv_offset + acc_offset + i * stride
        result[i] = np.frombuffer(blob[off : off + itemsize], dtype=dtype)
    return result


# ---------------------------------------------------------------------------
# JointTree — thin wrapper over the flat gltf.nodes list
# ---------------------------------------------------------------------------


class JointTree:
    """
    Read-only view over the flat pygltflib Node list, rooted at a named joint.

    The underlying node list is kept as-is; no data is copied.  Navigation
    methods delegate back to each node's ``children`` index list.

    Attributes
    ----------
    root_idx : int  — index of the root joint ("mixamorig:Hips") in the node list.
    """

    def __init__(self, nodes: list, root_idx: int) -> None:
        self._nodes = nodes
        self.root_idx = root_idx
        # Build a reverse parent map in O(N) so parent() is O(1).
        self._parent: dict[int, int] = {}
        for i, node in enumerate(nodes):
            for child in node.children or []:
                self._parent[child] = i
        # Rest-pose local rotation per node (one Quaternion per node).
        # Identity is used for nodes with no rotation set.
        self.rest_rotations: List[Quaternion] = [
            Quaternion.from_list(n.rotation)
            if n.rotation is not None
            else Quaternion.identity()
            for n in nodes
        ]

    # --- accessors ---

    def node(self, idx: int):
        """Return the raw pygltflib Node at *idx*."""
        return self._nodes[idx]

    def name(self, idx: int) -> str:
        """Return the name of the node at *idx* (empty string if unnamed)."""
        return self._nodes[idx].name or ""

    def children(self, idx: int) -> List[int]:
        """Return child node indices of *idx*."""
        return list(self._nodes[idx].children or [])

    def parent(self, idx: int) -> int | None:
        """Return parent node index of *idx*, or None for the root."""
        return self._parent.get(idx)

    def rest_rotation(self, idx: int) -> Quaternion:
        """Return the rest-pose rotation for node *idx*."""
        return self.rest_rotations[idx]

    def find(self, name: str) -> int | None:
        """Return the index of the first node whose name equals *name*."""
        for i, n in enumerate(self._nodes):
            if n.name == name:
                return i
        return None

    # --- traversal ---

    def walk(self, start: int | None = None):
        """Depth-first generator yielding *(node_idx, depth)* from *start*.

        Default start is ``root_idx``.
        """
        stack = [(self.root_idx if start is None else start, 0)]
        while stack:
            idx, depth = stack.pop()
            yield idx, depth
            for child in reversed(self.children(idx)):
                stack.append((child, depth + 1))

    # --- forward kinematics ---

    def world_rotations(self) -> List[Quaternion]:
        """Return world-space rotation for every node via FK (root → leaf).

        Accumulates ``q_world[child] = q_world[parent] * q_local[child]``
        using the DFS order of ``walk()``, which always yields a parent
        before its children. Nodes outside the joint subtree keep identity.
        """
        result: List[Quaternion] = [Quaternion.identity()] * len(self._nodes)
        for idx, _ in self.walk():
            p = self.parent(idx)
            result[idx] = (
                self.rest_rotations[idx]
                if p is None
                else result[p] * self.rest_rotations[idx]
            )
        return result

    def bone_direction(self, idx: int, world_rots: List[Quaternion]) -> Vector3:
        """World-space direction from parent joint to *idx* (the bone vector).

        Uses the node's translation (parent-local offset) rotated by the
        parent's world rotation.  Falls back to (0,1,0) for the root or
        zero-length translations.
        """
        t = self._nodes[idx].translation or [0.0, 0.0, 0.0]
        local = Vector3(t[0], t[1], t[2])
        length = local.length()
        if length < 1e-8:
            return Vector3(0.0, 1.0, 0.0)
        local_dir = local / length
        p = self.parent(idx)
        if p is None:
            return local_dir
        return world_rots[p].rotate(local_dir)

    # --- display ---

    def print_tree(self, start: int | None = None) -> None:
        """Print the joint hierarchy as an indented tree."""
        for idx, depth in self.walk(start):
            print("  " * depth + self.name(idx))

    def __repr__(self) -> str:
        return (
            f"JointTree(root={self.name(self.root_idx)!r}, "
            f"total_nodes={len(self._nodes)})"
        )


# ---------------------------------------------------------------------------
# MixamoCharacter
# ---------------------------------------------------------------------------


class MixamoCharacter:
    """
    Mixamo character loaded from a GLB file.

    Attributes
    ----------
    vertices : (N,3) float32  — rest-pose mesh vertices in display space (Z-up).
    faces    : (F,3) int32    — triangle face indices (constant).
    can_animate : bool        — True when LBS skinning data is available.
    """

    def __init__(self, glb_path: str | Path) -> None:
        self.glb_path = Path(glb_path)
        self.can_animate: bool = False
        # Populated by _load()
        self.vertices: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.faces: np.ndarray = np.empty((0, 3), dtype=np.int32)
        self._load()

    # ------------------------------------------------------------------
    # Step 1 — Load
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Parse the GLB and populate mesh + optional LBS skinning data."""
        try:
            from pygltflib import GLTF2
        except ImportError as exc:
            raise ImportError(
                "pygltflib is required to load GLB files.\n"
                "  Install with: pip install pygltflib"
            ) from exc

        if not self.glb_path.exists():
            raise FileNotFoundError(f"GLB file not found: {self.glb_path}")

        gltf = GLTF2().load(str(self.glb_path))

        # GLB embedded binary chunk; external URI buffers are not supported.
        blob: bytes = gltf.binary_blob() or b""

        all_verts: List[np.ndarray] = []
        all_faces: List[np.ndarray] = []
        all_joints: List[np.ndarray] = []
        all_weights: List[np.ndarray] = []
        vertex_offset = 0

        for mesh in gltf.meshes:
            for prim in mesh.primitives:
                if prim.attributes.POSITION is None:
                    continue

                verts = _read_accessor(gltf, blob, prim.attributes.POSITION)  # (N,3)

                if prim.indices is not None:
                    idx = _read_accessor(gltf, blob, prim.indices).reshape(-1)
                    faces = idx.reshape(-1, 3).astype(np.int32) + vertex_offset
                else:
                    n = len(verts)
                    faces = np.arange(n, dtype=np.int32).reshape(-1, 3) + vertex_offset

                all_verts.append(verts.astype(np.float32))
                all_faces.append(faces)

                has_skin = (
                    prim.attributes.JOINTS_0 is not None
                    and prim.attributes.WEIGHTS_0 is not None
                )
                if has_skin:
                    j = _read_accessor(gltf, blob, prim.attributes.JOINTS_0)
                    w = _read_accessor(gltf, blob, prim.attributes.WEIGHTS_0)
                    all_joints.append(j)
                    all_weights.append(w)

                vertex_offset += len(verts)

        if not all_verts:
            raise ValueError(f"No mesh geometry found in {self.glb_path}")

        glb_verts = np.concatenate(all_verts, axis=0)  # (N,3) Y-up GLB space
        self.faces = np.concatenate(all_faces, axis=0).astype(np.int32)  # (F,3)

        # GLB (Y-up) → display (Z-up): disp = [-X, -Z, +Y]
        self.vertices = np.column_stack(
            [-glb_verts[:, 0], -glb_verts[:, 2], glb_verts[:, 1]]
        ).astype(np.float32)

        # ---- LBS skinning data (Steps 2-4, used when animating) ----
        skin_complete = all_joints and len(all_joints) == len(all_verts) and gltf.skins
        if skin_complete:
            self._skin_joints = np.concatenate(all_joints, axis=0).astype(
                np.uint16
            )  # (N,4)
            self._skin_weights = np.concatenate(all_weights, axis=0).astype(
                np.float32
            )  # (N,4)
            self._glb_verts = glb_verts  # bind-pose verts kept for LBS deformation

            skin = gltf.skins[0]
            self._joint_nodes: List[int] = skin.joints

            if skin.inverseBindMatrices is not None:
                ibm = _read_accessor(gltf, blob, skin.inverseBindMatrices)  # (J,16)
                self._inv_bind_matrices = ibm.reshape(-1, 4, 4).astype(np.float32)
            else:
                n_j = len(self._joint_nodes)
                self._inv_bind_matrices = np.tile(
                    np.eye(4, dtype=np.float32), (n_j, 1, 1)
                )

            self._gltf_nodes = gltf.nodes
            self.joint_tree = self._build_joint_tree()
            self.can_animate = True

    def _build_joint_tree(self) -> JointTree:
        """Locate 'mixamorig:Hips' and return a JointTree rooted there."""
        root_name = "mixamorig:Hips"
        root_idx = next(
            (i for i, n in enumerate(self._gltf_nodes) if n.name == root_name),
            None,
        )
        if root_idx is None:
            available = [n.name for n in self._gltf_nodes if n.name]
            raise ValueError(
                f"Root joint {root_name!r} not found in GLB.\n"
                f"  Available node names: {available}"
            )
        return JointTree(self._gltf_nodes, root_idx)

    # ------------------------------------------------------------------
    # Step 2 — Compute rotations from MediaPipe landmarks
    # ------------------------------------------------------------------

    def compute_pose_rotations(
        self, landmarks: List[Vector3]
    ) -> Dict[str, Quaternion]:
        """Compute per-joint local rotations from MediaPipe world landmarks.

        Processes joints in DFS root-to-leaf order so every parent's pose
        world rotation is ready before its children are evaluated.

        For each bone mapped in ``_BONE_LM``:

        * Extract the two landmark positions (``lm_a``, ``lm_b``) and build a
          direction vector in GLB Y-up space (MediaPipe is Y-down, so Y is
          negated).
        * Identify the bone direction as the vector from this joint to its
          first child, rotated by the rest-pose world rotation of this joint.
        * Compute the shortest-arc (swing-only) quaternion from the rest bone
          direction to the target direction.
        * Apply the swing on top of the rest-pose world rotation to preserve
          the rest-pose twist around the bone axis.
        * Convert back to a local rotation relative to the parent's accumulated
          *pose* world rotation.

        Bones with ``lm_a == lm_b`` (no directional target) and Hips (whose
        ``_BONE_LM`` entry encodes position, not direction) are skipped; their
        rest local rotation is inherited unchanged via FK propagation.

        Parameters
        ----------
        landmarks : list of Vector3
            MediaPipe world landmarks (Y-down, X-right, Z-toward-camera).
            Must contain at least 33 entries (indices 0-32 used by _BONE_LM).

        Returns
        -------
        dict[str, Quaternion]
            Maps Mixamo bone name (no ``"mixamorig:"`` prefix) to the new
            local rotation quaternion.  Bones without a directional target are
            omitted; callers should fall back to the rest local rotation.
        """
        if not self.can_animate:
            return {}

        # Rest-pose world rotations (constant, computed once per call).
        rest_world_rots = self.joint_tree.world_rotations()

        n = len(self.joint_tree._nodes)
        pose_world_rots: List[Quaternion] = [Quaternion.identity()] * n
        result: Dict[str, Quaternion] = {}

        for idx, _ in self.joint_tree.walk():
            p = self.joint_tree.parent(idx)
            q_parent = pose_world_rots[p] if p is not None else Quaternion.identity()
            rest_local = self.joint_tree.rest_rotations[idx]

            # Default propagation: parent pose * rest local (no change from rest).
            pose_world_rots[idx] = q_parent * rest_local

            node_name = self.joint_tree.name(idx)
            bone_key = node_name.replace("mixamorig:", "")

            if bone_key not in _BONE_LM:
                continue

            lm_a, lm_b = _BONE_LM[bone_key]

            # Hips entry encodes root position (midpoint of lm_a/lm_b), not a
            # bone direction — skip rotation for this joint.
            if bone_key == "Hips" or lm_a == lm_b:
                continue

            pos_a = landmarks[lm_a]
            pos_b = landmarks[lm_b]

            # MediaPipe Y-down → GLB Y-up: negate Y component.
            glb_dir = Vector3(
                pos_b.x - pos_a.x,
                -(pos_b.y - pos_a.y),
                pos_b.z - pos_a.z,
            )
            if glb_dir.length() < 1e-8:
                continue
            target_dir = glb_dir.normalize()

            # Bone direction = this joint → its first child, in rest world space.
            children = self.joint_tree.children(idx)
            if not children:
                continue  # Leaf joint — no outgoing bone to align.

            child_t = self.joint_tree.node(children[0]).translation or [0.0, 0.0, 0.0]
            child_local = Vector3(child_t[0], child_t[1], child_t[2])
            if child_local.length() < 1e-8:
                continue
            child_local_dir = child_local.normalize()

            rest_bone_world_dir = rest_world_rots[idx].rotate(child_local_dir)

            # Shortest-arc swing from rest direction to target direction.
            q_swing = Quaternion.from_two_vectors(rest_bone_world_dir, target_dir)

            # Swing is applied in world space; rest-pose twist is preserved.
            pose_world_rots[idx] = q_swing * rest_world_rots[idx]

            # Local rotation relative to the parent's accumulated pose rotation.
            result[bone_key] = q_parent.inverse() * pose_world_rots[idx]

        return result

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def preview(self) -> None:
        """Open an interactive 3D window showing the character in rest/T-pose.

        Blocks until the window is closed or Ctrl-C is pressed.
        """
        from video_to_maximo.viz3d import Pose3DVisualizer

        print(
            f"[MixamoCharacter] Rest-pose preview\n"
            f"  File     : {self.glb_path}\n"
            f"  Vertices : {len(self.vertices):,}\n"
            f"  Faces    : {len(self.faces):,}\n"
            f"  LBS data : {'available' if self.can_animate else 'not found'}\n"
            f"  Close the window or press Ctrl-C to exit."
        )

        viz = Pose3DVisualizer(character_mesh=(self.vertices, self.faces))
        viz.open()

        try:
            while viz.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            viz.close()
