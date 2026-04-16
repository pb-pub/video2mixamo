"""
Mixamo character GLB loader with real-time Linear Blend Skinning (LBS).

Loads a GLB/glTF2 file exported from Mixamo, extracts the skinned mesh
and skeleton, and deforms the mesh each frame from MediaPipe world landmarks.

Coordinate systems
------------------
* MediaPipe world landmarks: X=right, Y=DOWN, Z=toward-camera  (Y-down)
* GLB / Mixamo bind pose:    X=right, Y=UP,   Z=forward         (Y-up)
* pyqtgraph display:         X=right, Y=depth, Z=UP             (Z-up)

Conversions::

    glb  = [ mp_X,  -mp_Y,  mp_Z ]   (flip Y to go Y-down → Y-up)
    disp = [-glb_X, -glb_Z, +glb_Y]  (mirror X, swap depth/up)

Usage::

    char = MixamoCharacter("character.glb")
    # Static T-pose:
    char.vertices, char.faces        # display-space, passed to viz3d once

    # Animated (per frame, when can_animate is True):
    verts = char.compute_skinned_vertices(landmarks)  # (N,3) float32
    viz.update_mesh(verts, char.faces)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe landmark index → Mixamo bone mapping
# ---------------------------------------------------------------------------
# Keys are bare bone names (no "mixamorig:" prefix).
# Values: (start_lm, end_lm) — direction lm[end]-lm[start] defines bone axis.
# start == end → no rotation change (bone inherits parent).
# All indices are MediaPipe Pose landmark indices (0-32).
# ---------------------------------------------------------------------------
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

_SPINE_BONES = {"Spine", "Spine1", "Spine2"}


class MixamoCharacter:
    """
    Mixamo character loaded from a GLB file.

    Attributes
    ----------
    vertices : (N,3) float32
        T-pose mesh vertices in pyqtgraph display space.
    faces : (F,3) int32
        Triangle face indices (constant).
    can_animate : bool
        True if skin data was found and LBS is available.
    """

    def __init__(self, glb_path: str | Path) -> None:
        self.glb_path = Path(glb_path)

        # Static T-pose mesh (display space)
        self.vertices: np.ndarray
        self.faces: np.ndarray
        self.can_animate: bool = False

        # Skinning data (populated when can_animate)
        self._bind_verts: np.ndarray  # (N,3) GLB space
        self._joint_idx: np.ndarray  # (N,4) int32
        self._joint_w: np.ndarray  # (N,4) float32
        self._inv_bind: np.ndarray  # (J,4,4)
        self._bind_world: np.ndarray  # (J,4,4)
        self._jnames: List[str]  # bare bone names
        self._jparent: List[int]  # -1 = root
        self._jlocal_t: np.ndarray  # (J,3) bind-pose local translations
        self._bind_dir: Dict[int, np.ndarray]
        self._topo: List[int]

        self._load()

    # ------------------------------------------------------------------
    # Static mesh (always available)
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            from pygltflib import GLTF2
        except ImportError as exc:
            raise ImportError("pygltflib required: pip install pygltflib") from exc

        gltf = GLTF2().load(str(self.glb_path))
        blob: Optional[bytes] = gltf.binary_blob()
        if blob is None:
            raise ValueError(f"{self.glb_path.name}: no binary buffer.")

        self._load_mesh(gltf, blob)

        if gltf.skins:
            try:
                self._load_skeleton(gltf, blob)
                self.can_animate = True
            except Exception as exc:
                print(
                    f"[MixamoCharacter] Skeleton load failed ({exc}); static display only."
                )

        status = "animatable" if self.can_animate else "static only"
        print(
            f"[MixamoCharacter] {self.glb_path.name}: "
            f"{len(self.vertices):,} verts, {len(self.faces):,} faces — {status}"
        )

    def _load_mesh(self, gltf, blob: bytes) -> None:
        """Collect all mesh primitives and merge into one static T-pose mesh."""
        all_v: List[np.ndarray] = []
        all_f: List[np.ndarray] = []
        all_ji: List[np.ndarray] = []
        all_jw: List[np.ndarray] = []
        v_off = 0

        for mesh in gltf.meshes:
            for prim in mesh.primitives:
                a = prim.attributes
                pos_idx = getattr(a, "POSITION", None)
                if pos_idx is None:
                    continue

                v = self._acc(gltf, blob, pos_idx).astype(np.float32)

                if prim.indices is not None:
                    f = (
                        self._acc(gltf, blob, prim.indices)
                        .reshape(-1, 3)
                        .astype(np.int32)
                    )
                else:
                    f = np.arange(len(v), dtype=np.int32).reshape(-1, 3)

                all_v.append(v)
                all_f.append(f + v_off)
                v_off += len(v)

                j0 = getattr(a, "JOINTS_0", None)
                w0 = getattr(a, "WEIGHTS_0", None)
                if j0 is not None and w0 is not None:
                    all_ji.append(self._acc(gltf, blob, j0).astype(np.int32))
                    all_jw.append(self._acc(gltf, blob, w0).astype(np.float32))
                else:
                    N = len(v)
                    all_ji.append(np.zeros((N, 4), dtype=np.int32))
                    all_jw.append(np.ones((N, 4), dtype=np.float32) * 0.25)

        if not all_v:
            raise ValueError(f"{self.glb_path.name}: no POSITION found.")

        verts = np.concatenate(all_v, axis=0)
        faces = np.concatenate(all_f, axis=0)
        self._bind_verts = verts.astype(np.float64)
        self._joint_idx = np.concatenate(all_ji, axis=0)
        self._joint_w = np.concatenate(all_jw, axis=0).astype(np.float64)
        self.faces = faces

        # T-pose display vertices: GLB Y-up → display Z-up
        self.vertices = _to_display(verts)

    # ------------------------------------------------------------------
    # Skeleton (only when skin present)
    # ------------------------------------------------------------------

    def _load_skeleton(self, gltf, blob: bytes) -> None:
        skin = gltf.skins[0]

        # ---- joint names ------------------------------------------------
        node_to_j: Dict[int, int] = {}
        self._jnames = []
        for ji, ni in enumerate(skin.joints):
            raw = gltf.nodes[ni].name or f"joint_{ji}"
            name = raw.removeprefix("mixamorig:").removeprefix("mixamorig_")
            self._jnames.append(name)
            node_to_j[ni] = ji

        # ---- parent map -------------------------------------------------
        node_parent: Dict[int, int] = {}
        for ni, node in enumerate(gltf.nodes):
            for ci in node.children or []:
                node_parent[ci] = ni

        self._jparent = []
        for ni in skin.joints:
            pni = node_parent.get(ni, -1)
            self._jparent.append(node_to_j.get(pni, -1))

        # ---- inverse bind matrices (column-major in glTF → transpose) ---
        raw = self._acc(gltf, blob, skin.inverseBindMatrices)
        self._inv_bind = raw.astype(np.float64).reshape(-1, 4, 4).transpose(0, 2, 1)

        J = len(self._jnames)
        self._bind_world = np.array([np.linalg.inv(m) for m in self._inv_bind])

        # ---- local translations (bone offsets from parent) --------------
        self._jlocal_t = np.zeros((J, 3), dtype=np.float64)
        for j in range(J):
            p = self._jparent[j]
            if p >= 0:
                local = np.linalg.inv(self._bind_world[p]) @ self._bind_world[j]
                self._jlocal_t[j] = local[:3, 3]
            else:
                self._jlocal_t[j] = self._bind_world[j][:3, 3]

        # ---- topological order ------------------------------------------
        self._topo = _topo_sort(self._jparent)

        # ---- bind-pose directions (joint → first child) -----------------
        children: Dict[int, List[int]] = {}
        for j, p in enumerate(self._jparent):
            if p >= 0:
                children.setdefault(p, []).append(j)

        self._bind_dir = {}
        for j in range(J):
            kids = children.get(j, [])
            if kids:
                d = self._bind_world[kids[0]][:3, 3] - self._bind_world[j][:3, 3]
            else:
                p = self._jparent[j]
                d = (
                    self._bind_dir[p].copy()
                    if p >= 0 and p in self._bind_dir
                    else np.array([0.0, 1.0, 0.0])
                )
            n = np.linalg.norm(d)
            self._bind_dir[j] = d / n if n > 1e-8 else np.array([0.0, 1.0, 0.0])

        print(
            f"[MixamoCharacter]   {J} joints, "
            f"root bind pos={self._bind_world[self._topo[0]][:3, 3]}"
        )

    # ------------------------------------------------------------------
    # Runtime: LBS animation
    # ------------------------------------------------------------------

    def compute_skinned_vertices(self, landmarks: List[List[float]]) -> np.ndarray:
        """
        Deform the mesh using MediaPipe landmarks + LBS.

        Parameters
        ----------
        landmarks : 33×[x,y,z] MediaPipe world landmarks (Y-down, metres).

        Returns
        -------
        np.ndarray (N,3) float32 — vertices in pyqtgraph display space.
        """
        # Convert MediaPipe (Y-down) → GLB space (Y-up) for direction vectors.
        # Only Y is flipped; X and Z stay the same.
        lm = np.asarray(landmarks, dtype=np.float64)
        lg = lm.copy()
        lg[:, 1] = -lm[:, 1]  # flip Y: MediaPipe Y-down → GLB Y-up

        J = len(self._jnames)
        world_R = np.zeros((J, 3, 3))
        world_mat = np.tile(np.eye(4), (J, 1, 1)).astype(np.float64)

        # Root stays at its bind-pose world position.
        # MediaPipe always reports hips at (0,0,0), so we can't use it for
        # absolute positioning — it would shift the entire skeleton away from
        # the bind-pose origin and collapse all skin matrices into a "ball".
        root_bind_pos = self._bind_world[self._topo[0]][:3, 3]

        # Spine direction (shared across 3 spine bones)
        sp_d = (lg[11] + lg[12]) / 2.0 - (lg[23] + lg[24]) / 2.0
        sp_n = np.linalg.norm(sp_d)
        sp_dir = sp_d / sp_n if sp_n > 1e-8 else None

        for j in self._topo:
            name = self._jnames[j]
            p = self._jparent[j]
            Rbw = self._bind_world[j][:3, :3].copy()
            db = self._bind_dir[j]
            mapping = _BONE_LM.get(name)

            if mapping and mapping[0] != mapping[1]:
                sl, el = mapping
                if name in _SPINE_BONES:
                    if sp_dir is not None:
                        Rf = _rot_between(db, sp_dir)
                        Rcur = _slerp(np.eye(3), Rf, 1.0 / 3.0) @ Rbw
                    else:
                        Rcur = Rbw
                else:
                    dv = lg[el] - lg[sl]
                    n = np.linalg.norm(dv)
                    Rcur = (_rot_between(db, dv / n) @ Rbw) if n > 1e-8 else Rbw
            else:
                Rcur = Rbw

            world_R[j] = Rcur

            if p < 0:
                # Root bone: use bind-pose position, only rotation is driven
                world_mat[j, :3, :3] = Rcur
                world_mat[j, :3, 3] = root_bind_pos
                world_mat[j, 3, 3] = 1.0
            else:
                Rloc = world_R[p].T @ Rcur
                lm4 = np.eye(4)
                lm4[:3, :3] = Rloc
                lm4[:3, 3] = self._jlocal_t[j]
                world_mat[j] = world_mat[p] @ lm4

        # Skin matrices
        skin = np.einsum("jab,jbc->jac", world_mat, self._inv_bind)

        # LBS
        N = len(self._bind_verts)
        vh = np.hstack([self._bind_verts, np.ones((N, 1))])
        res = np.zeros((N, 4))
        for k in range(4):
            ji = self._joint_idx[:, k]
            w = self._joint_w[:, k, None]
            res += w * np.einsum("nij,nj->ni", skin[ji], vh)

        return _to_display(res[:, :3].astype(np.float32))

    # ------------------------------------------------------------------
    # Accessor helper
    # ------------------------------------------------------------------

    @staticmethod
    def _acc(gltf, blob: bytes, idx: int) -> np.ndarray:
        """Read a glTF accessor into a numpy array."""
        _DT = {
            5120: np.int8,
            5121: np.uint8,
            5122: np.int16,
            5123: np.uint16,
            5125: np.uint32,
            5126: np.float32,
        }
        _SZ = {
            "SCALAR": 1,
            "VEC2": 2,
            "VEC3": 3,
            "VEC4": 4,
            "MAT2": 4,
            "MAT3": 9,
            "MAT4": 16,
        }
        a = gltf.accessors[idx]
        bv = gltf.bufferViews[a.bufferView]
        dtype = _DT[a.componentType]
        n_comp = _SZ[a.type]
        count = a.count
        offset = (bv.byteOffset or 0) + (a.byteOffset or 0)
        stride = bv.byteStride
        if not stride:
            data = np.frombuffer(blob, dtype=dtype, count=count * n_comp, offset=offset)
        else:
            rows = [
                np.frombuffer(
                    blob, dtype=dtype, count=n_comp, offset=offset + i * stride
                )
                for i in range(count)
            ]
            data = np.concatenate(rows)
        return data.reshape(count, n_comp) if n_comp > 1 else data


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _to_display(v: np.ndarray) -> np.ndarray:
    """
    GLB space (Y-up) → pyqtgraph display space (Z-up).

    display X = −GLB X   (mirror)
    display Y = −GLB Z   (depth)
    display Z = +GLB Y   (Y-up → Z-up, no negation)
    """
    out = np.empty(v.shape, dtype=np.float32)
    out[:, 0] = -v[:, 0]
    out[:, 1] = -v[:, 2]
    out[:, 2] = v[:, 1]
    return out


def _topo_sort(parents: List[int]) -> List[int]:
    """Return joint indices, parents before children."""
    n = len(parents)
    visited = [False] * n
    order: List[int] = []

    def visit(j: int) -> None:
        if visited[j]:
            return
        visited[j] = True
        p = parents[j]
        if p >= 0:
            visit(p)
        order.append(j)

    for j in range(n):
        visit(j)
    return order


def _rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix rotating unit vector a onto unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    dot = float(np.dot(a, b))
    if dot >= 0.99999:
        return np.eye(3)
    if dot <= -0.99999:
        perp = np.array([0.0, 0.0, 1.0])
        if abs(a[2]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    axis = np.cross(a, b)
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + dot))


def _slerp(R0: np.ndarray, R1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two rotation matrices."""
    from scipy.spatial.transform import Rotation, Slerp

    rots = Rotation.from_matrix(np.stack([R0, R1]))
    return Slerp([0.0, 1.0], rots)(t).as_matrix()
