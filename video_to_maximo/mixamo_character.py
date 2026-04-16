"""
Mixamo character GLB loader — static bind-pose mesh display.

Loads a GLB file and extracts the mesh geometry (vertices + faces) for
display in the 3D visualizer. No skinning or animation — purely static.

Usage::

    char = MixamoCharacter("character.glb")
    # Pass once to Pose3DVisualizer:
    viz = Pose3DVisualizer(character_mesh=(char.vertices, char.faces))
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np


class MixamoCharacter:
    """
    Loads a GLB file and extracts bind-pose mesh geometry.

    Attributes
    ----------
    vertices : np.ndarray, shape (N, 3), float32
        Vertex positions in pyqtgraph display space (Z-up).
    faces : np.ndarray, shape (F, 3), int32
        Triangle face indices.
    """

    def __init__(self, glb_path: str | Path) -> None:
        self.glb_path = Path(glb_path)
        self.vertices: np.ndarray   # (N, 3) float32, display space
        self.faces: np.ndarray      # (F, 3) int32
        self._load()

    def _load(self) -> None:
        try:
            from pygltflib import GLTF2
        except ImportError as exc:
            raise ImportError(
                "pygltflib is required for character loading.\n"
                "  pip install pygltflib"
            ) from exc

        gltf = GLTF2().load(str(self.glb_path))
        blob: Optional[bytes] = gltf.binary_blob()
        if blob is None:
            raise ValueError(f"{self.glb_path.name}: no binary buffer found.")

        # Collect ALL mesh primitives that have a POSITION attribute and merge them.
        # Mixamo characters typically export as multiple primitives
        # (body, hair, clothes …) so we must gather all of them.
        all_verts: List[np.ndarray] = []
        all_faces: List[np.ndarray] = []
        vertex_offset = 0

        for mesh in gltf.meshes:
            for prim in mesh.primitives:
                pos_idx = getattr(prim.attributes, "POSITION", None)
                if pos_idx is None:
                    continue

                verts = self._read_accessor(gltf, blob, pos_idx).astype(np.float32)

                if prim.indices is not None:
                    idx = self._read_accessor(gltf, blob, prim.indices)
                    faces = idx.reshape(-1, 3).astype(np.int32)
                else:
                    n = len(verts)
                    faces = np.arange(n, dtype=np.int32).reshape(-1, 3)

                all_verts.append(verts)
                all_faces.append(faces + vertex_offset)
                vertex_offset += len(verts)

        if not all_verts:
            raise ValueError(
                f"{self.glb_path.name}: no mesh with POSITION attribute found."
            )

        verts_all = np.concatenate(all_verts, axis=0)   # (N_total, 3)
        faces_all = np.concatenate(all_faces, axis=0)   # (F_total, 3)

        # Coordinate transform: GLB is Y-up (head = +Y, feet = −Y).
        # viz3d landmark transform: display_Z = −mediapipe_Y  (because MediaPipe Y
        # is DOWN, so negating makes Z-up).  The GLB Y is already UP, so we must
        # NOT negate it — otherwise the character appears upside-down.
        #
        #   display X = −GLB X   (mirror left/right to match landmark view)
        #   display Y = −GLB Z   (depth axis)
        #   display Z = +GLB Y   (Y-up → Z-up, no negation)
        out = np.empty_like(verts_all)
        out[:, 0] = -verts_all[:, 0]
        out[:, 1] = -verts_all[:, 2]
        out[:, 2] =  verts_all[:, 1]   # ← positive, not negative
        self.vertices = out
        self.faces = faces_all

        print(
            f"[MixamoCharacter] {self.glb_path.name}: "
            f"{len(all_verts)} primitive(s), "
            f"{len(self.vertices):,} vertices, {len(self.faces):,} faces"
        )

    @staticmethod
    def _read_accessor(gltf, blob: bytes, accessor_idx: int) -> np.ndarray:
        """Read a glTF accessor into a numpy array."""
        _DTYPE = {
            5120: np.int8, 5121: np.uint8,
            5122: np.int16, 5123: np.uint16,
            5125: np.uint32, 5126: np.float32,
        }
        _SIZE = {
            "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
            "MAT2": 4, "MAT3": 9, "MAT4": 16,
        }
        acc = gltf.accessors[accessor_idx]
        bv = gltf.bufferViews[acc.bufferView]

        dtype = _DTYPE[acc.componentType]
        n_comp = _SIZE[acc.type]
        count = acc.count
        offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)
        stride = bv.byteStride

        if stride is None or stride == 0:
            data = np.frombuffer(
                blob, dtype=dtype, count=count * n_comp, offset=offset
            )
        else:
            rows: List[np.ndarray] = []
            for i in range(count):
                row = np.frombuffer(
                    blob, dtype=dtype, count=n_comp, offset=offset + i * stride
                )
                rows.append(row)
            data = np.concatenate(rows)

        if n_comp > 1:
            data = data.reshape(count, n_comp)
        return data
