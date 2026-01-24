"""Numba-compatible SE3 utilities.

This module provides pure-numba implementations of SE3 operations
for use in performance-critical code paths. SE3 transforms are
represented as 4x4 numpy arrays (homogeneous transformation matrices).
"""

import numpy as np
from numba import njit  # type: ignore[import-untyped]


@njit(cache=True)
def arrays_equal_6(a: np.ndarray, b: np.ndarray) -> bool:
    """Fast 6-element array comparison (avoids np.array_equal dispatch overhead)."""
    for i in range(6):
        if a[i] != b[i]:
            return False
    return True


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


@njit(cache=True)
def so3_from_rpy(roll: float, pitch: float, yaw: float, out: np.ndarray) -> None:
    """Create 3x3 rotation matrix from XYZ euler angles (intrinsic).

    Matches scipy.spatial.transform.Rotation.from_euler('XYZ', ...).

    Args:
        roll: Rotation about X axis (radians)
        pitch: Rotation about Y axis (radians)
        yaw: Rotation about Z axis (radians)
        out: 3x3 output array
    """
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # R = Rx(roll) @ Ry(pitch) @ Rz(yaw) for intrinsic XYZ
    out[0, 0] = cp * cy
    out[0, 1] = -cp * sy
    out[0, 2] = sp
    out[1, 0] = sr * sp * cy + cr * sy
    out[1, 1] = cr * cy - sr * sp * sy
    out[1, 2] = -sr * cp
    out[2, 0] = sr * sy - cr * sp * cy
    out[2, 1] = cr * sp * sy + sr * cy
    out[2, 2] = cr * cp


@njit(cache=True)
def se3_from_rpy(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    out: np.ndarray,
) -> None:
    """Create 4x4 SE3 from position and XYZ euler angles.

    Matches scipy.spatial.transform.Rotation.from_euler('XYZ', ...).

    Args:
        x, y, z: Translation components
        roll, pitch, yaw: Rotation angles (radians, XYZ intrinsic)
        out: 4x4 output array
    """
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # Rotation part: R = Rx(roll) @ Ry(pitch) @ Rz(yaw) for intrinsic XYZ
    out[0, 0] = cp * cy
    out[0, 1] = -cp * sy
    out[0, 2] = sp
    out[1, 0] = sr * sp * cy + cr * sy
    out[1, 1] = cr * cy - sr * sp * sy
    out[1, 2] = -sr * cp
    out[2, 0] = sr * sy - cr * sp * cy
    out[2, 1] = cr * sp * sy + sr * cy
    out[2, 2] = cr * cp

    # Translation part
    out[0, 3] = x
    out[1, 3] = y
    out[2, 3] = z

    # Bottom row
    out[3, 0] = 0.0
    out[3, 1] = 0.0
    out[3, 2] = 0.0
    out[3, 3] = 1.0


@njit(cache=True)
def so3_rpy(R: np.ndarray, out: np.ndarray) -> None:
    """Extract XYZ euler angles from 3x3 rotation matrix.

    Matches scipy.spatial.transform.Rotation.as_euler('XYZ', ...).
    For R = Rx(roll) @ Ry(pitch) @ Rz(yaw).

    Args:
        R: 3x3 rotation matrix
        out: 3-element output array [roll, pitch, yaw] in radians
    """
    # Clamp to avoid numerical issues with arcsin
    sp = R[0, 2]
    if sp > 1.0:
        sp = 1.0
    elif sp < -1.0:
        sp = -1.0
    out[1] = np.arcsin(sp)  # pitch
    # Use atan2 which handles near-zero denominators safely
    out[0] = np.arctan2(-R[1, 2], R[2, 2])  # roll
    out[2] = np.arctan2(-R[0, 1], R[0, 0])  # yaw


@njit(cache=True)
def se3_rpy(T: np.ndarray, out: np.ndarray) -> None:
    """Extract XYZ euler angles from 4x4 SE3 matrix.

    Matches scipy.spatial.transform.Rotation.as_euler('XYZ', ...).
    For R = Rx(roll) @ Ry(pitch) @ Rz(yaw).

    Args:
        T: 4x4 SE3 matrix
        out: 3-element output array [roll, pitch, yaw] in radians
    """
    # Clamp to avoid numerical issues with arcsin
    sp = T[0, 2]
    if sp > 1.0:
        sp = 1.0
    elif sp < -1.0:
        sp = -1.0
    out[1] = np.arcsin(sp)  # pitch
    # Use atan2 which handles near-zero denominators safely
    out[0] = np.arctan2(-T[1, 2], T[2, 2])  # roll
    out[2] = np.arctan2(-T[0, 1], T[0, 0])  # yaw


@njit(cache=True)
def se3_inverse(T: np.ndarray, out: np.ndarray) -> None:
    """Compute inverse of SE3 transformation.

    For SE3: T^-1 = [R^T | -R^T * t]

    Args:
        T: 4x4 SE3 matrix
        out: 4x4 output array
    """
    # R^T (transpose of rotation)
    out[0, 0] = T[0, 0]
    out[0, 1] = T[1, 0]
    out[0, 2] = T[2, 0]
    out[1, 0] = T[0, 1]
    out[1, 1] = T[1, 1]
    out[1, 2] = T[2, 1]
    out[2, 0] = T[0, 2]
    out[2, 1] = T[1, 2]
    out[2, 2] = T[2, 2]

    # -R^T * t
    tx = T[0, 3]
    ty = T[1, 3]
    tz = T[2, 3]
    out[0, 3] = -(out[0, 0] * tx + out[0, 1] * ty + out[0, 2] * tz)
    out[1, 3] = -(out[1, 0] * tx + out[1, 1] * ty + out[1, 2] * tz)
    out[2, 3] = -(out[2, 0] * tx + out[2, 1] * ty + out[2, 2] * tz)

    # Bottom row
    out[3, 0] = 0.0
    out[3, 1] = 0.0
    out[3, 2] = 0.0
    out[3, 3] = 1.0


@njit(cache=True)
def so3_log(R: np.ndarray, out: np.ndarray) -> None:
    """SO3 matrix logarithm (rotation matrix to axis-angle vector).

    Args:
        R: 3x3 rotation matrix
        out: 3-element output array (axis-angle vector omega)
    """
    # Compute rotation angle from trace: trace(R) = 1 + 2*cos(theta)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    cos_theta = (trace - 1.0) / 2.0

    # Clamp to handle numerical errors
    if cos_theta > 1.0:
        cos_theta = 1.0
    elif cos_theta < -1.0:
        cos_theta = -1.0

    theta = np.arccos(cos_theta)

    if theta < 1e-10:
        # Near identity: omega ≈ 0
        out[0] = 0.0
        out[1] = 0.0
        out[2] = 0.0
    elif theta > np.pi - 1e-10:
        # Near 180 degrees: use eigenvector of R corresponding to eigenvalue 1
        # Find the column of (R + I) with largest norm
        diag0 = R[0, 0] + 1.0
        diag1 = R[1, 1] + 1.0
        diag2 = R[2, 2] + 1.0

        if diag0 >= diag1 and diag0 >= diag2:
            # Use first column
            v0 = R[0, 0] + 1.0
            v1 = R[1, 0]
            v2 = R[2, 0]
        elif diag1 >= diag2:
            # Use second column
            v0 = R[0, 1]
            v1 = R[1, 1] + 1.0
            v2 = R[2, 1]
        else:
            # Use third column
            v0 = R[0, 2]
            v1 = R[1, 2]
            v2 = R[2, 2] + 1.0

        norm = np.sqrt(v0 * v0 + v1 * v1 + v2 * v2)
        if norm > 1e-10:
            out[0] = theta * v0 / norm
            out[1] = theta * v1 / norm
            out[2] = theta * v2 / norm
        else:
            out[0] = 0.0
            out[1] = 0.0
            out[2] = theta
    else:
        # General case: omega = theta / (2*sin(theta)) * (R - R^T)
        k = theta / (2.0 * np.sin(theta))
        out[0] = k * (R[2, 1] - R[1, 2])
        out[1] = k * (R[0, 2] - R[2, 0])
        out[2] = k * (R[1, 0] - R[0, 1])


@njit(cache=True)
def so3_exp(omega: np.ndarray, out: np.ndarray) -> None:
    """SO3 matrix exponential (axis-angle vector to rotation matrix).

    Uses Rodrigues' formula: R = I + sin(θ)/θ * [ω]× + (1-cos(θ))/θ² * [ω]×²

    Args:
        omega: 3-element axis-angle vector
        out: 3x3 output rotation matrix
    """
    theta_sq = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]
    theta = np.sqrt(theta_sq)

    if theta < 1e-10:
        # Near zero: R ≈ I + [ω]×
        out[0, 0] = 1.0
        out[0, 1] = -omega[2]
        out[0, 2] = omega[1]
        out[1, 0] = omega[2]
        out[1, 1] = 1.0
        out[1, 2] = -omega[0]
        out[2, 0] = -omega[1]
        out[2, 1] = omega[0]
        out[2, 2] = 1.0
    else:
        # Rodrigues' formula
        c = np.cos(theta)
        s = np.sin(theta)
        k1 = s / theta
        k2 = (1.0 - c) / theta_sq

        # Skew-symmetric matrix [ω]×
        wx, wy, wz = omega[0], omega[1], omega[2]

        # [ω]×² components
        wxx = wx * wx
        wyy = wy * wy
        wzz = wz * wz
        wxy = wx * wy
        wxz = wx * wz
        wyz = wy * wz

        out[0, 0] = 1.0 - k2 * (wyy + wzz)
        out[0, 1] = -k1 * wz + k2 * wxy
        out[0, 2] = k1 * wy + k2 * wxz
        out[1, 0] = k1 * wz + k2 * wxy
        out[1, 1] = 1.0 - k2 * (wxx + wzz)
        out[1, 2] = -k1 * wx + k2 * wyz
        out[2, 0] = -k1 * wy + k2 * wxz
        out[2, 1] = k1 * wx + k2 * wyz
        out[2, 2] = 1.0 - k2 * (wxx + wyy)


@njit(cache=True)
def _compute_V_matrix(omega: np.ndarray, V: np.ndarray) -> None:
    """Compute the V matrix for SE3 log/exp.

    V = I + (1-cos(θ))/θ² * [ω]× + (θ - sin(θ))/θ³ * [ω]×²

    Args:
        omega: 3-element axis-angle vector
        V: 3x3 output matrix
    """
    theta_sq = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]
    theta = np.sqrt(theta_sq)

    if theta < 1e-10:
        # Near zero: V ≈ I
        V[0, 0] = 1.0
        V[0, 1] = 0.0
        V[0, 2] = 0.0
        V[1, 0] = 0.0
        V[1, 1] = 1.0
        V[1, 2] = 0.0
        V[2, 0] = 0.0
        V[2, 1] = 0.0
        V[2, 2] = 1.0
    else:
        c = np.cos(theta)
        s = np.sin(theta)
        k1 = (1.0 - c) / theta_sq
        k2 = (theta - s) / (theta_sq * theta)

        wx, wy, wz = omega[0], omega[1], omega[2]
        wxx = wx * wx
        wyy = wy * wy
        wzz = wz * wz
        wxy = wx * wy
        wxz = wx * wz
        wyz = wy * wz

        V[0, 0] = 1.0 - k2 * (wyy + wzz)
        V[0, 1] = -k1 * wz + k2 * wxy
        V[0, 2] = k1 * wy + k2 * wxz
        V[1, 0] = k1 * wz + k2 * wxy
        V[1, 1] = 1.0 - k2 * (wxx + wzz)
        V[1, 2] = -k1 * wx + k2 * wyz
        V[2, 0] = -k1 * wy + k2 * wxz
        V[2, 1] = k1 * wx + k2 * wyz
        V[2, 2] = 1.0 - k2 * (wxx + wyy)


@njit(cache=True)
def _compute_V_inv_matrix(omega: np.ndarray, V_inv: np.ndarray) -> None:
    """Compute the inverse V matrix for SE3 log.

    V^-1 = I - 0.5*[ω]× + (1/θ² - (1+cos(θ))/(2θ sin(θ))) * [ω]×²

    Args:
        omega: 3-element axis-angle vector
        V_inv: 3x3 output matrix
    """
    theta_sq = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]
    theta = np.sqrt(theta_sq)

    if theta < 1e-10:
        # Near zero: V^-1 ≈ I
        V_inv[0, 0] = 1.0
        V_inv[0, 1] = 0.0
        V_inv[0, 2] = 0.0
        V_inv[1, 0] = 0.0
        V_inv[1, 1] = 1.0
        V_inv[1, 2] = 0.0
        V_inv[2, 0] = 0.0
        V_inv[2, 1] = 0.0
        V_inv[2, 2] = 1.0
    else:
        c = np.cos(theta)
        s = np.sin(theta)
        k1 = 0.5
        k2 = (1.0 / theta_sq) - (1.0 + c) / (2.0 * theta * s)

        wx, wy, wz = omega[0], omega[1], omega[2]
        wxx = wx * wx
        wyy = wy * wy
        wzz = wz * wz
        wxy = wx * wy
        wxz = wx * wz
        wyz = wy * wz

        V_inv[0, 0] = 1.0 - k2 * (wyy + wzz)
        V_inv[0, 1] = k1 * wz + k2 * wxy
        V_inv[0, 2] = -k1 * wy + k2 * wxz
        V_inv[1, 0] = -k1 * wz + k2 * wxy
        V_inv[1, 1] = 1.0 - k2 * (wxx + wzz)
        V_inv[1, 2] = k1 * wx + k2 * wyz
        V_inv[2, 0] = k1 * wy + k2 * wxz
        V_inv[2, 1] = -k1 * wx + k2 * wyz
        V_inv[2, 2] = 1.0 - k2 * (wxx + wyy)


@njit(cache=True)
def se3_log(T: np.ndarray, out: np.ndarray) -> None:
    """SE3 matrix logarithm (SE3 to 6D twist vector).

    The twist vector is [v, ω] where v is linear and ω is angular.

    Args:
        T: 4x4 SE3 matrix
        out: 6-element output array [vx, vy, vz, ωx, ωy, ωz]
    """
    # Extract rotation and compute omega
    omega = np.zeros(3, dtype=np.float64)
    R = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            R[i, j] = T[i, j]
    so3_log(R, omega)

    out[3] = omega[0]
    out[4] = omega[1]
    out[5] = omega[2]

    # Compute v = V^-1 * t
    V_inv = np.zeros((3, 3), dtype=np.float64)
    _compute_V_inv_matrix(omega, V_inv)

    tx = T[0, 3]
    ty = T[1, 3]
    tz = T[2, 3]

    out[0] = V_inv[0, 0] * tx + V_inv[0, 1] * ty + V_inv[0, 2] * tz
    out[1] = V_inv[1, 0] * tx + V_inv[1, 1] * ty + V_inv[1, 2] * tz
    out[2] = V_inv[2, 0] * tx + V_inv[2, 1] * ty + V_inv[2, 2] * tz


@njit(cache=True)
def se3_exp(twist: np.ndarray, out: np.ndarray) -> None:
    """SE3 matrix exponential (6D twist to SE3).

    Args:
        twist: 6-element twist vector [vx, vy, vz, ωx, ωy, ωz]
        out: 4x4 output SE3 matrix
    """
    omega = np.zeros(3, dtype=np.float64)
    omega[0] = twist[3]
    omega[1] = twist[4]
    omega[2] = twist[5]

    # Compute rotation matrix
    R = np.zeros((3, 3), dtype=np.float64)
    so3_exp(omega, R)

    for i in range(3):
        for j in range(3):
            out[i, j] = R[i, j]

    # Compute translation: t = V * v
    V = np.zeros((3, 3), dtype=np.float64)
    _compute_V_matrix(omega, V)

    vx = twist[0]
    vy = twist[1]
    vz = twist[2]

    out[0, 3] = V[0, 0] * vx + V[0, 1] * vy + V[0, 2] * vz
    out[1, 3] = V[1, 0] * vx + V[1, 1] * vy + V[1, 2] * vz
    out[2, 3] = V[2, 0] * vx + V[2, 1] * vy + V[2, 2] * vz

    # Bottom row
    out[3, 0] = 0.0
    out[3, 1] = 0.0
    out[3, 2] = 0.0
    out[3, 3] = 1.0


@njit(cache=True)
def se3_interp(T1: np.ndarray, T2: np.ndarray, s: float, out: np.ndarray) -> None:
    """Interpolate between two SE3 transforms using Lie algebra.

    Computes: T1 * exp(s * log(T1^-1 * T2))

    Args:
        T1: 4x4 start SE3 matrix
        T2: 4x4 end SE3 matrix
        s: Interpolation factor [0, 1]
        out: 4x4 output SE3 matrix
    """
    # Compute T1^-1
    T1_inv = np.zeros((4, 4), dtype=np.float64)
    se3_inverse(T1, T1_inv)

    # Compute delta = T1^-1 * T2
    delta = np.zeros((4, 4), dtype=np.float64)
    se3_mul(T1_inv, T2, delta)

    # Compute log(delta)
    twist = np.zeros(6, dtype=np.float64)
    se3_log(delta, twist)

    # Scale twist by s
    for i in range(6):
        twist[i] *= s

    # Compute exp(s * twist)
    delta_scaled = np.zeros((4, 4), dtype=np.float64)
    se3_exp(twist, delta_scaled)

    # Compute T1 * exp(s * twist)
    se3_mul(T1, delta_scaled, out)


@njit(cache=True)
def se3_angdist(T1: np.ndarray, T2: np.ndarray) -> float:
    """Compute angular distance between two SE3 transforms.

    Args:
        T1: 4x4 first SE3 matrix
        T2: 4x4 second SE3 matrix

    Returns:
        Angular distance in radians
    """
    # R_rel = R1^T @ R2
    # Compute trace(R_rel) = sum of diagonal elements
    trace = 0.0
    for i in range(3):
        for j in range(3):
            trace += T1[j, i] * T2[j, i]

    # Angular distance from trace: trace(R_rel) = 1 + 2*cos(theta)
    cos_theta = (trace - 1.0) / 2.0

    # Clamp to handle numerical errors
    if cos_theta > 1.0:
        cos_theta = 1.0
    elif cos_theta < -1.0:
        cos_theta = -1.0

    return np.arccos(cos_theta)


# =============================================================================
# Workspace variants (zero internal allocation)
# =============================================================================


@njit(cache=True)
def se3_log_ws(
    T: np.ndarray,
    out: np.ndarray,
    omega_ws: np.ndarray,
    R_ws: np.ndarray,
    V_inv_ws: np.ndarray,
) -> None:
    """SE3 matrix logarithm with external workspace (zero internal allocation).

    Args:
        T: 4x4 SE3 matrix
        out: 6-element output array [vx, vy, vz, ωx, ωy, ωz]
        omega_ws: Workspace buffer for axis-angle (3,)
        R_ws: Workspace buffer for rotation matrix (3,3)
        V_inv_ws: Workspace buffer for V inverse matrix (3,3)
    """
    # Extract rotation
    for i in range(3):
        for j in range(3):
            R_ws[i, j] = T[i, j]

    # Compute omega
    so3_log(R_ws, omega_ws)

    out[3] = omega_ws[0]
    out[4] = omega_ws[1]
    out[5] = omega_ws[2]

    # Compute v = V^-1 * t
    _compute_V_inv_matrix(omega_ws, V_inv_ws)

    tx = T[0, 3]
    ty = T[1, 3]
    tz = T[2, 3]

    out[0] = V_inv_ws[0, 0] * tx + V_inv_ws[0, 1] * ty + V_inv_ws[0, 2] * tz
    out[1] = V_inv_ws[1, 0] * tx + V_inv_ws[1, 1] * ty + V_inv_ws[1, 2] * tz
    out[2] = V_inv_ws[2, 0] * tx + V_inv_ws[2, 1] * ty + V_inv_ws[2, 2] * tz


@njit(cache=True)
def se3_exp_ws(
    twist: np.ndarray,
    out: np.ndarray,
    omega_ws: np.ndarray,
    R_ws: np.ndarray,
    V_ws: np.ndarray,
) -> None:
    """SE3 matrix exponential with external workspace (zero internal allocation).

    Args:
        twist: 6-element twist vector [vx, vy, vz, ωx, ωy, ωz]
        out: 4x4 output SE3 matrix
        omega_ws: Workspace buffer for axis-angle (3,)
        R_ws: Workspace buffer for rotation matrix (3,3)
        V_ws: Workspace buffer for V matrix (3,3)
    """
    omega_ws[0] = twist[3]
    omega_ws[1] = twist[4]
    omega_ws[2] = twist[5]

    # Compute rotation matrix
    so3_exp(omega_ws, R_ws)

    for i in range(3):
        for j in range(3):
            out[i, j] = R_ws[i, j]

    # Compute translation: t = V * v
    _compute_V_matrix(omega_ws, V_ws)

    vx = twist[0]
    vy = twist[1]
    vz = twist[2]

    out[0, 3] = V_ws[0, 0] * vx + V_ws[0, 1] * vy + V_ws[0, 2] * vz
    out[1, 3] = V_ws[1, 0] * vx + V_ws[1, 1] * vy + V_ws[1, 2] * vz
    out[2, 3] = V_ws[2, 0] * vx + V_ws[2, 1] * vy + V_ws[2, 2] * vz

    # Bottom row
    out[3, 0] = 0.0
    out[3, 1] = 0.0
    out[3, 2] = 0.0
    out[3, 3] = 1.0


@njit(cache=True)
def se3_interp_ws(
    T1: np.ndarray,
    T2: np.ndarray,
    s: float,
    out: np.ndarray,
    T1_inv_ws: np.ndarray,
    delta_ws: np.ndarray,
    twist_ws: np.ndarray,
    delta_scaled_ws: np.ndarray,
    omega_ws: np.ndarray,
    R_ws: np.ndarray,
    V_ws: np.ndarray,
) -> None:
    """Interpolate between SE3 transforms with external workspace (zero internal allocation).

    Computes: T1 * exp(s * log(T1^-1 * T2))

    Args:
        T1: 4x4 start SE3 matrix
        T2: 4x4 end SE3 matrix
        s: Interpolation factor [0, 1]
        out: 4x4 output SE3 matrix
        T1_inv_ws: Workspace buffer for T1 inverse (4,4)
        delta_ws: Workspace buffer for delta transform (4,4)
        twist_ws: Workspace buffer for twist vector (6,)
        delta_scaled_ws: Workspace buffer for scaled delta (4,4)
        omega_ws: Workspace buffer for axis-angle (3,)
        R_ws: Workspace buffer for rotation matrix (3,3)
        V_ws: Workspace buffer for V matrix (3,3)
    """
    # Compute T1^-1
    se3_inverse(T1, T1_inv_ws)

    # Compute delta = T1^-1 * T2
    se3_mul(T1_inv_ws, T2, delta_ws)

    # Compute log(delta) using workspace variant
    se3_log_ws(delta_ws, twist_ws, omega_ws, R_ws, V_ws)

    # Scale twist by s
    for i in range(6):
        twist_ws[i] *= s

    # Compute exp(s * twist) using workspace variant
    se3_exp_ws(twist_ws, delta_scaled_ws, omega_ws, R_ws, V_ws)

    # Compute T1 * exp(s * twist)
    se3_mul(T1, delta_scaled_ws, out)
