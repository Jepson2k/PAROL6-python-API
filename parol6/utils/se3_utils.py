"""Fast SE3/SO3 utilities using sophuspy.

This module provides wrapper functions for sophuspy to replace spatialmath.
sophuspy is 10-25x faster with zero timing spikes, critical for real-time control.
"""

import numpy as np
import sophuspy as sp
from scipy.spatial.transform import Rotation

__all__ = [
    "se3_from_rpy",
    "se3_from_trans",
    "se3_from_matrix",
    "se3_rpy",
    "se3_interp",
    "se3_angdist",
    "se3_rx",
    "se3_ry",
    "se3_rz",
    "so3_rpy",
    "so3_from_rpy",
    "so3_rx",
    "so3_ry",
    "so3_rz",
]


def se3_from_rpy(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    degrees: bool = False,
) -> sp.SE3:
    """Create SE3 from position and RPY angles.

    Args:
        x, y, z: Translation components
        roll, pitch, yaw: Rotation angles (xyz order)
        degrees: If True, angles are in degrees
    """
    if degrees:
        roll, pitch, yaw = np.radians([roll, pitch, yaw])
    R = Rotation.from_euler("XYZ", [roll, pitch, yaw]).as_matrix()
    return sp.SE3(R, [x, y, z])


def se3_from_trans(x: float, y: float, z: float) -> sp.SE3:
    """Create SE3 from translation only (identity rotation)."""
    return sp.SE3(np.eye(3), [x, y, z])


def se3_from_matrix(matrix: np.ndarray) -> sp.SE3:
    """Create SE3 from 4x4 homogeneous transformation matrix."""
    return sp.SE3(matrix[:3, :3], matrix[:3, 3])


def se3_rpy(se3: sp.SE3, degrees: bool = False) -> np.ndarray:
    """Extract RPY angles from SE3.

    Args:
        se3: SE3 transformation
        degrees: If True, return angles in degrees

    Returns:
        Array of [roll, pitch, yaw] in xyz order
    """
    R = se3.rotationMatrix()
    rpy = Rotation.from_matrix(R).as_euler("XYZ")
    return np.degrees(rpy) if degrees else rpy


def se3_interp(se3_1: sp.SE3, se3_2: sp.SE3, s: float) -> sp.SE3:
    """Fast SE3 interpolation using Lie algebra.

    Args:
        se3_1: Start pose
        se3_2: End pose
        s: Interpolation factor [0, 1]

    Returns:
        Interpolated SE3 pose
    """
    delta = se3_1.inverse() * se3_2
    return se3_1 * sp.SE3.exp(delta.log() * s)


def se3_angdist(se3_1: sp.SE3, se3_2: sp.SE3) -> float:
    """Calculate angular distance between two SE3 poses.

    Args:
        se3_1: First pose
        se3_2: Second pose

    Returns:
        Angular distance in radians
    """
    R_rel = se3_1.rotationMatrix().T @ se3_2.rotationMatrix()
    return float(Rotation.from_matrix(R_rel).magnitude())


def se3_rx(angle: float, degrees: bool = False) -> sp.SE3:
    """Create SE3 with rotation about X axis (no translation)."""
    if degrees:
        angle = np.radians(angle)
    R = Rotation.from_euler("x", angle).as_matrix()
    return sp.SE3(R, [0, 0, 0])


def se3_ry(angle: float, degrees: bool = False) -> sp.SE3:
    """Create SE3 with rotation about Y axis (no translation)."""
    if degrees:
        angle = np.radians(angle)
    R = Rotation.from_euler("y", angle).as_matrix()
    return sp.SE3(R, [0, 0, 0])


def se3_rz(angle: float, degrees: bool = False) -> sp.SE3:
    """Create SE3 with rotation about Z axis (no translation)."""
    if degrees:
        angle = np.radians(angle)
    R = Rotation.from_euler("z", angle).as_matrix()
    return sp.SE3(R, [0, 0, 0])


def so3_rpy(rotation_matrix: np.ndarray, degrees: bool = False) -> np.ndarray:
    """Extract RPY angles from 3x3 rotation matrix.

    Args:
        rotation_matrix: 3x3 rotation matrix
        degrees: If True, return angles in degrees

    Returns:
        Array of [roll, pitch, yaw] in xyz order
    """
    rpy = Rotation.from_matrix(rotation_matrix).as_euler("XYZ")
    return np.degrees(rpy) if degrees else rpy


def so3_from_rpy(
    roll: float, pitch: float, yaw: float, degrees: bool = False
) -> sp.SO3:
    """Create SO3 from RPY angles.

    Args:
        roll, pitch, yaw: Rotation angles (xyz order)
        degrees: If True, angles are in degrees
    """
    if degrees:
        roll, pitch, yaw = np.radians([roll, pitch, yaw])
    R = Rotation.from_euler("XYZ", [roll, pitch, yaw]).as_matrix()
    return sp.SO3(R)


def so3_rx(angle: float, degrees: bool = False) -> sp.SO3:
    """Create SO3 rotation about X axis."""
    if degrees:
        angle = np.radians(angle)
    R = Rotation.from_euler("x", angle).as_matrix()
    return sp.SO3(R)


def so3_ry(angle: float, degrees: bool = False) -> sp.SO3:
    """Create SO3 rotation about Y axis."""
    if degrees:
        angle = np.radians(angle)
    R = Rotation.from_euler("y", angle).as_matrix()
    return sp.SO3(R)


def so3_rz(angle: float, degrees: bool = False) -> sp.SO3:
    """Create SO3 rotation about Z axis."""
    if degrees:
        angle = np.radians(angle)
    R = Rotation.from_euler("z", angle).as_matrix()
    return sp.SO3(R)
