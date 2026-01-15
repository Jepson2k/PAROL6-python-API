"""Numba-compatible SE3 utilities.

This module provides pure-numba implementations of SE3 operations
for use in performance-critical code paths. SE3 transforms are
represented as 4x4 numpy arrays (homogeneous transformation matrices).
"""

import numpy as np
from numba import njit  # type: ignore[import-untyped]


@njit(cache=True)
def se3_identity(out: np.ndarray) -> None:
    """Set out to identity SE3 (4x4)."""
    out[:] = 0.0
    out[0, 0] = 1.0
    out[1, 1] = 1.0
    out[2, 2] = 1.0
    out[3, 3] = 1.0


@njit(cache=True)
def se3_from_trans(x: float, y: float, z: float, out: np.ndarray) -> None:
    """Create SE3 from translation only (identity rotation)."""
    se3_identity(out)
    out[0, 3] = x
    out[1, 3] = y
    out[2, 3] = z


@njit(cache=True)
def se3_rx(angle: float, out: np.ndarray) -> None:
    """Create SE3 with rotation about X axis (no translation)."""
    c = np.cos(angle)
    s = np.sin(angle)
    se3_identity(out)
    out[1, 1] = c
    out[1, 2] = -s
    out[2, 1] = s
    out[2, 2] = c


@njit(cache=True)
def se3_ry(angle: float, out: np.ndarray) -> None:
    """Create SE3 with rotation about Y axis (no translation)."""
    c = np.cos(angle)
    s = np.sin(angle)
    se3_identity(out)
    out[0, 0] = c
    out[0, 2] = s
    out[2, 0] = -s
    out[2, 2] = c


@njit(cache=True)
def se3_rz(angle: float, out: np.ndarray) -> None:
    """Create SE3 with rotation about Z axis (no translation)."""
    c = np.cos(angle)
    s = np.sin(angle)
    se3_identity(out)
    out[0, 0] = c
    out[0, 1] = -s
    out[1, 0] = s
    out[1, 1] = c


@njit(cache=True)
def se3_mul(A: np.ndarray, B: np.ndarray, out: np.ndarray) -> None:
    """SE3 multiplication: out = A @ B (4x4 matrix multiply)."""
    for i in range(4):
        for j in range(4):
            out[i, j] = (
                A[i, 0] * B[0, j]
                + A[i, 1] * B[1, j]
                + A[i, 2] * B[2, j]
                + A[i, 3] * B[3, j]
            )


@njit(cache=True)
def se3_copy(src: np.ndarray, dst: np.ndarray) -> None:
    """Copy SE3 matrix."""
    for i in range(4):
        for j in range(4):
            dst[i, j] = src[i, j]
