"""
Unit quaternion for 3-D rotations.

Convention: [x, y, z, w] — matches glTF / Mixamo storage order.
All operations assume unit quaternions; call ``normalize()`` after
constructing from raw data when in doubt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from video_to_maximo.vector import Vector3


@dataclass
class Quaternion:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)

    # --- constructors ---

    @classmethod
    def identity(cls) -> Quaternion:
        return cls(0.0, 0.0, 0.0, 1.0)

    @classmethod
    def from_list(cls, xyzw: list) -> Quaternion:
        """Construct from a [x, y, z, w] list (glTF convention)."""
        return cls(xyzw[0], xyzw[1], xyzw[2], xyzw[3])

    @classmethod
    def from_two_vectors(cls, v_from: Vector3, v_to: Vector3) -> Quaternion:
        """Shortest-arc quaternion that rotates *v_from* onto *v_to*.

        Inputs need not be pre-normalized.  Handles the 180° edge case
        (opposite vectors) by choosing an arbitrary perpendicular axis.
        """
        a = v_from.normalize()
        b = v_to.normalize()
        d = a.dot(b)

        if d < -1.0 + 1e-8:
            # Vectors are opposite — 180° rotation around any perpendicular axis.
            perp = Vector3(0.0, 0.0, 1.0) if abs(a.x) < 0.9 else Vector3(0.0, 1.0, 0.0)
            axis = a.cross(perp).normalize()
            return cls(axis.x, axis.y, axis.z, 0.0)

        c = a.cross(b)
        return cls(c.x, c.y, c.z, 1.0 + d).normalize()

    # --- arithmetic ---

    def __mul__(self, other: Quaternion) -> Quaternion:
        """Hamilton product: self ⊗ other."""
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        return Quaternion(x, y, z, w)

    def conjugate(self) -> Quaternion:
        return Quaternion(-self.x, -self.y, -self.z, self.w)

    def norm(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)

    def normalize(self) -> Quaternion:
        n = self.norm()
        if n < 1e-10:
            return Quaternion.identity()
        return Quaternion(self.x / n, self.y / n, self.z / n, self.w / n)

    def inverse(self) -> Quaternion:
        """For unit quaternions inverse == conjugate (no norm division needed)."""
        return self.conjugate()

    def dot(self, other: Quaternion) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w

    def slerp(self, other: Quaternion, t: float) -> Quaternion:
        """Spherical linear interpolation — always takes the shortest arc."""
        d = self.dot(other)
        if d < 0.0:
            other = Quaternion(-other.x, -other.y, -other.z, -other.w)
            d = -d
        if d > 0.9995:
            return Quaternion(
                self.x + t * (other.x - self.x),
                self.y + t * (other.y - self.y),
                self.z + t * (other.z - self.z),
                self.w + t * (other.w - self.w),
            ).normalize()
        theta_0 = math.acos(d)
        theta = theta_0 * t
        sin_t = math.sin(theta)
        sin_t0 = math.sin(theta_0)
        s0 = math.cos(theta) - d * sin_t / sin_t0
        s1 = sin_t / sin_t0
        return Quaternion(
            s0 * self.x + s1 * other.x,
            s0 * self.y + s1 * other.y,
            s0 * self.z + s1 * other.z,
            s0 * self.w + s1 * other.w,
        ).normalize()

    def rotate(self, v: Vector3) -> Vector3:
        """Rotate *v* by this quaternion using the optimized sandwich product.

        Equivalent to q ⊗ (0,v) ⊗ q⁻¹ but avoids two full quaternion
        multiplications (15 muls instead of 28).
        """
        qx, qy, qz = self.x, self.y, self.z
        vx, vy, vz = v.x, v.y, v.z
        # t = 2 * cross(q.xyz, v)
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)
        # result = v + w*t + cross(q.xyz, t)
        return Vector3(
            vx + self.w * tx + (qy * tz - qz * ty),
            vy + self.w * ty + (qz * tx - qx * tz),
            vz + self.w * tz + (qx * ty - qy * tx),
        )

    # --- conversion ---

    def to_matrix(self) -> np.ndarray:
        """Return a 4×4 float32 rotation matrix (row-major, right-multiply)."""
        x, y, z, w = self.x, self.y, self.z, self.w
        x2, y2, z2 = x + x, y + y, z + z
        xx, yy, zz = x * x2, y * y2, z * z2
        xy, xz, yz = x * y2, x * z2, y * z2
        wx, wy, wz = w * x2, w * y2, w * z2
        return np.array(
            [
                [1.0 - (yy + zz),       xy - wz,       xz + wy,  0.0],
                [      xy + wz,   1.0 - (xx + zz),     yz - wx,  0.0],
                [      xz - wy,         yz + wx,  1.0 - (xx + yy), 0.0],
                [      0.0,             0.0,             0.0,      1.0],
            ],
            dtype=np.float32,
        )

    def to_list(self) -> list:
        return [self.x, self.y, self.z, self.w]

    def to_tuple(self) -> tuple:
        return (self.x, self.y, self.z, self.w)

    # --- sequence protocol (mirrors Vector3) ---

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> float:
        return (self.x, self.y, self.z, self.w)[index]

    def __iter__(self):
        return iter((self.x, self.y, self.z, self.w))

    def __repr__(self) -> str:
        return f"Quaternion(x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f}, w={self.w:.4f})"
