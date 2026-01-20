"""Unit tests for se3_utils.py numba-accelerated SE3 operations."""

import numpy as np
from numpy.testing import assert_allclose

from parol6.utils.se3_utils import (
    se3_identity,
    se3_from_trans,
    se3_rx,
    se3_ry,
    se3_rz,
    se3_mul,
    se3_copy,
    so3_from_rpy,
    se3_from_rpy,
    so3_rpy,
    se3_rpy,
    se3_inverse,
    so3_log,
    so3_exp,
    se3_log,
    se3_exp,
    se3_interp,
    se3_angdist,
)


class TestBasicSE3Operations:
    """Tests for basic SE3 matrix operations."""

    def test_se3_identity(self):
        out = np.zeros((4, 4), dtype=np.float64)
        se3_identity(out)
        assert_allclose(out, np.eye(4))

    def test_se3_from_trans(self):
        out = np.zeros((4, 4), dtype=np.float64)
        se3_from_trans(1.0, 2.0, 3.0, out)

        expected = np.eye(4)
        expected[:3, 3] = [1.0, 2.0, 3.0]
        assert_allclose(out, expected)

    def test_se3_rx(self):
        out = np.zeros((4, 4), dtype=np.float64)
        se3_rx(np.pi / 2, out)

        # 90 degree rotation about X
        assert_allclose(out[0, 0], 1.0)
        assert_allclose(out[1, 1], 0.0, atol=1e-10)
        assert_allclose(out[1, 2], -1.0)
        assert_allclose(out[2, 1], 1.0)
        assert_allclose(out[2, 2], 0.0, atol=1e-10)

    def test_se3_ry(self):
        out = np.zeros((4, 4), dtype=np.float64)
        se3_ry(np.pi / 2, out)

        # 90 degree rotation about Y
        assert_allclose(out[0, 0], 0.0, atol=1e-10)
        assert_allclose(out[0, 2], 1.0)
        assert_allclose(out[1, 1], 1.0)
        assert_allclose(out[2, 0], -1.0)
        assert_allclose(out[2, 2], 0.0, atol=1e-10)

    def test_se3_rz(self):
        out = np.zeros((4, 4), dtype=np.float64)
        se3_rz(np.pi / 2, out)

        # 90 degree rotation about Z
        assert_allclose(out[0, 0], 0.0, atol=1e-10)
        assert_allclose(out[0, 1], -1.0)
        assert_allclose(out[1, 0], 1.0)
        assert_allclose(out[1, 1], 0.0, atol=1e-10)
        assert_allclose(out[2, 2], 1.0)

    def test_se3_mul(self):
        A = np.zeros((4, 4), dtype=np.float64)
        B = np.zeros((4, 4), dtype=np.float64)
        out = np.zeros((4, 4), dtype=np.float64)

        se3_from_trans(1.0, 0.0, 0.0, A)
        se3_from_trans(0.0, 2.0, 0.0, B)
        se3_mul(A, B, out)

        # Composition of translations
        assert_allclose(out[:3, 3], [1.0, 2.0, 0.0])
        assert_allclose(out[:3, :3], np.eye(3))

    def test_se3_mul_rotation_composition(self):
        Rx = np.zeros((4, 4), dtype=np.float64)
        Ry = np.zeros((4, 4), dtype=np.float64)
        out = np.zeros((4, 4), dtype=np.float64)

        se3_rx(np.pi / 4, Rx)
        se3_ry(np.pi / 4, Ry)
        se3_mul(Ry, Rx, out)

        # Verify result is orthonormal
        R = out[:3, :3]
        assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_se3_copy(self):
        src = np.random.rand(4, 4)
        dst = np.zeros((4, 4), dtype=np.float64)
        se3_copy(src, dst)
        assert_allclose(dst, src)


class TestSO3FromRPY:
    """Tests for SO3 rotation matrix from RPY angles."""

    def test_identity(self):
        out = np.zeros((3, 3), dtype=np.float64)
        so3_from_rpy(0.0, 0.0, 0.0, out)
        assert_allclose(out, np.eye(3), atol=1e-10)

    def test_roll_90(self):
        out = np.zeros((3, 3), dtype=np.float64)
        so3_from_rpy(np.pi / 2, 0.0, 0.0, out)

        # Roll 90 about X
        assert_allclose(out[0, 0], 1.0)
        assert_allclose(out[1, 1], 0.0, atol=1e-10)
        assert_allclose(out[1, 2], -1.0)
        assert_allclose(out[2, 1], 1.0)

    def test_pitch_90(self):
        out = np.zeros((3, 3), dtype=np.float64)
        so3_from_rpy(0.0, np.pi / 2, 0.0, out)

        # Pitch 90 about Y
        assert_allclose(out[0, 0], 0.0, atol=1e-10)
        assert_allclose(out[0, 2], 1.0)
        assert_allclose(out[2, 0], -1.0)

    def test_yaw_90(self):
        out = np.zeros((3, 3), dtype=np.float64)
        so3_from_rpy(0.0, 0.0, np.pi / 2, out)

        # Yaw 90 about Z
        assert_allclose(out[0, 0], 0.0, atol=1e-10)
        assert_allclose(out[0, 1], -1.0)
        assert_allclose(out[1, 0], 1.0)

    def test_orthonormal(self):
        out = np.zeros((3, 3), dtype=np.float64)
        so3_from_rpy(0.3, 0.5, 0.7, out)

        # R @ R^T = I
        assert_allclose(out @ out.T, np.eye(3), atol=1e-10)
        # det(R) = 1
        assert_allclose(np.linalg.det(out), 1.0, atol=1e-10)


class TestSE3FromRPY:
    """Tests for SE3 from position and RPY angles."""

    def test_identity(self):
        out = np.zeros((4, 4), dtype=np.float64)
        se3_from_rpy(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, out)
        assert_allclose(out, np.eye(4), atol=1e-10)

    def test_translation_only(self):
        out = np.zeros((4, 4), dtype=np.float64)
        se3_from_rpy(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, out)

        assert_allclose(out[:3, :3], np.eye(3), atol=1e-10)
        assert_allclose(out[:3, 3], [1.0, 2.0, 3.0])
        assert_allclose(out[3, :], [0.0, 0.0, 0.0, 1.0])

    def test_rotation_only(self):
        out = np.zeros((4, 4), dtype=np.float64)
        se3_from_rpy(0.0, 0.0, 0.0, np.pi / 4, np.pi / 6, np.pi / 3, out)

        # Translation should be zero
        assert_allclose(out[:3, 3], [0.0, 0.0, 0.0], atol=1e-10)

        # Rotation should be orthonormal
        R = out[:3, :3]
        assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_full_transform(self):
        out = np.zeros((4, 4), dtype=np.float64)
        se3_from_rpy(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, out)

        # Check structure
        assert_allclose(out[3, :], [0.0, 0.0, 0.0, 1.0])
        assert_allclose(out[:3, 3], [0.1, 0.2, 0.3])

        # Rotation orthonormal
        R = out[:3, :3]
        assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


class TestSO3RPYExtraction:
    """Tests for extracting RPY angles from rotation matrix."""

    def test_identity(self):
        R = np.eye(3)
        out = np.zeros(3, dtype=np.float64)
        so3_rpy(R, out)
        assert_allclose(out, [0.0, 0.0, 0.0], atol=1e-10)

    def test_roundtrip(self):
        """Test so3_from_rpy -> so3_rpy roundtrip."""
        original = np.array([0.3, 0.5, 0.7])
        R = np.zeros((3, 3), dtype=np.float64)
        out = np.zeros(3, dtype=np.float64)

        so3_from_rpy(original[0], original[1], original[2], R)
        so3_rpy(R, out)

        assert_allclose(out, original, atol=1e-10)

    def test_various_angles(self):
        """Test multiple angle combinations."""
        test_cases = [
            [0.1, 0.2, 0.3],
            [-0.5, 0.3, -0.2],
            [np.pi / 6, np.pi / 4, np.pi / 3],
            [-np.pi / 4, -np.pi / 6, -np.pi / 5],
        ]
        for angles in test_cases:
            R = np.zeros((3, 3), dtype=np.float64)
            out = np.zeros(3, dtype=np.float64)

            so3_from_rpy(angles[0], angles[1], angles[2], R)
            so3_rpy(R, out)

            assert_allclose(out, angles, atol=1e-10)

    def test_gimbal_lock_positive(self):
        """Test near +90 degree pitch (gimbal lock)."""
        R = np.zeros((3, 3), dtype=np.float64)
        out = np.zeros(3, dtype=np.float64)

        # Create rotation with pitch near 90 degrees
        so3_from_rpy(0.0, np.pi / 2 - 1e-6, 0.0, R)
        so3_rpy(R, out)

        # Pitch should be near pi/2
        assert_allclose(out[1], np.pi / 2, atol=1e-4)

    def test_gimbal_lock_negative(self):
        """Test near -90 degree pitch (gimbal lock)."""
        R = np.zeros((3, 3), dtype=np.float64)
        out = np.zeros(3, dtype=np.float64)

        # Create rotation with pitch near -90 degrees
        so3_from_rpy(0.0, -np.pi / 2 + 1e-6, 0.0, R)
        so3_rpy(R, out)

        # Pitch should be near -pi/2
        assert_allclose(out[1], -np.pi / 2, atol=1e-4)


class TestSE3RPYExtraction:
    """Tests for extracting RPY angles from SE3 matrix."""

    def test_identity(self):
        T = np.eye(4)
        out = np.zeros(3, dtype=np.float64)
        se3_rpy(T, out)
        assert_allclose(out, [0.0, 0.0, 0.0], atol=1e-10)

    def test_roundtrip(self):
        """Test se3_from_rpy -> se3_rpy roundtrip."""
        original = np.array([0.3, 0.5, 0.7])
        T = np.zeros((4, 4), dtype=np.float64)
        out = np.zeros(3, dtype=np.float64)

        se3_from_rpy(1.0, 2.0, 3.0, original[0], original[1], original[2], T)
        se3_rpy(T, out)

        assert_allclose(out, original, atol=1e-10)


class TestSE3Inverse:
    """Tests for SE3 inverse."""

    def test_identity_inverse(self):
        T = np.eye(4)
        out = np.zeros((4, 4), dtype=np.float64)
        se3_inverse(T, out)
        assert_allclose(out, np.eye(4), atol=1e-10)

    def test_translation_inverse(self):
        T = np.zeros((4, 4), dtype=np.float64)
        out = np.zeros((4, 4), dtype=np.float64)

        se3_from_trans(1.0, 2.0, 3.0, T)
        se3_inverse(T, out)

        assert_allclose(out[:3, :3], np.eye(3), atol=1e-10)
        assert_allclose(out[:3, 3], [-1.0, -2.0, -3.0])

    def test_inverse_composition(self):
        """T @ T^-1 should be identity."""
        T = np.zeros((4, 4), dtype=np.float64)
        T_inv = np.zeros((4, 4), dtype=np.float64)
        result = np.zeros((4, 4), dtype=np.float64)

        se3_from_rpy(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, T)
        se3_inverse(T, T_inv)
        se3_mul(T, T_inv, result)

        assert_allclose(result, np.eye(4), atol=1e-10)

    def test_rotation_inverse(self):
        T = np.zeros((4, 4), dtype=np.float64)
        out = np.zeros((4, 4), dtype=np.float64)

        se3_rx(np.pi / 4, T)
        se3_inverse(T, out)

        # For pure rotation, inverse is transpose
        assert_allclose(out[:3, :3], T[:3, :3].T, atol=1e-10)


class TestSO3LogExp:
    """Tests for SO3 logarithm and exponential."""

    def test_identity_log(self):
        R = np.eye(3)
        out = np.zeros(3, dtype=np.float64)
        so3_log(R, out)
        assert_allclose(out, [0.0, 0.0, 0.0], atol=1e-10)

    def test_identity_exp(self):
        omega = np.array([0.0, 0.0, 0.0])
        out = np.zeros((3, 3), dtype=np.float64)
        so3_exp(omega, out)
        assert_allclose(out, np.eye(3), atol=1e-10)

    def test_roundtrip_small_angle(self):
        """Test log -> exp roundtrip with small rotation."""
        omega = np.array([0.1, 0.2, 0.3])
        R = np.zeros((3, 3), dtype=np.float64)
        out = np.zeros(3, dtype=np.float64)

        so3_exp(omega, R)
        so3_log(R, out)

        assert_allclose(out, omega, atol=1e-10)

    def test_roundtrip_medium_angle(self):
        """Test log -> exp roundtrip with medium rotation."""
        omega = np.array([0.5, 0.6, 0.7])
        R = np.zeros((3, 3), dtype=np.float64)
        out = np.zeros(3, dtype=np.float64)

        so3_exp(omega, R)
        so3_log(R, out)

        assert_allclose(out, omega, atol=1e-10)

    def test_roundtrip_large_angle(self):
        """Test log -> exp roundtrip with large rotation (near 180)."""
        # Angle near 180 degrees about some axis
        axis = np.array([1.0, 0.5, 0.3])
        axis = axis / np.linalg.norm(axis)
        theta = 3.0  # ~172 degrees
        omega = theta * axis

        R = np.zeros((3, 3), dtype=np.float64)
        out = np.zeros(3, dtype=np.float64)

        so3_exp(omega, R)
        so3_log(R, out)

        assert_allclose(out, omega, atol=1e-6)

    def test_exp_produces_orthonormal(self):
        """Verify exp produces orthonormal matrix."""
        omega = np.array([0.7, 0.8, 0.9])
        R = np.zeros((3, 3), dtype=np.float64)

        so3_exp(omega, R)

        assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_rotation_x(self):
        """Test rotation about X axis."""
        theta = np.pi / 4
        omega = np.array([theta, 0.0, 0.0])
        R = np.zeros((3, 3), dtype=np.float64)

        so3_exp(omega, R)

        # Compare to known Rx
        c, s = np.cos(theta), np.sin(theta)
        expected = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        assert_allclose(R, expected, atol=1e-10)


class TestSE3LogExp:
    """Tests for SE3 logarithm and exponential."""

    def test_identity_log(self):
        T = np.eye(4)
        out = np.zeros(6, dtype=np.float64)
        se3_log(T, out)
        assert_allclose(out, np.zeros(6), atol=1e-10)

    def test_identity_exp(self):
        twist = np.zeros(6, dtype=np.float64)
        out = np.zeros((4, 4), dtype=np.float64)
        se3_exp(twist, out)
        assert_allclose(out, np.eye(4), atol=1e-10)

    def test_pure_translation_log(self):
        T = np.zeros((4, 4), dtype=np.float64)
        out = np.zeros(6, dtype=np.float64)

        se3_from_trans(1.0, 2.0, 3.0, T)
        se3_log(T, out)

        # For pure translation, twist is [v, 0]
        assert_allclose(out[:3], [1.0, 2.0, 3.0], atol=1e-10)
        assert_allclose(out[3:], [0.0, 0.0, 0.0], atol=1e-10)

    def test_pure_translation_exp(self):
        twist = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        out = np.zeros((4, 4), dtype=np.float64)

        se3_exp(twist, out)

        assert_allclose(out[:3, :3], np.eye(3), atol=1e-10)
        assert_allclose(out[:3, 3], [1.0, 2.0, 3.0], atol=1e-10)

    def test_roundtrip(self):
        """Test log -> exp roundtrip."""
        T = np.zeros((4, 4), dtype=np.float64)
        twist = np.zeros(6, dtype=np.float64)
        T_recovered = np.zeros((4, 4), dtype=np.float64)

        se3_from_rpy(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, T)
        se3_log(T, twist)
        se3_exp(twist, T_recovered)

        assert_allclose(T_recovered, T, atol=1e-10)

    def test_roundtrip_pure_rotation(self):
        """Test log -> exp roundtrip with pure rotation."""
        T = np.zeros((4, 4), dtype=np.float64)
        twist = np.zeros(6, dtype=np.float64)
        T_recovered = np.zeros((4, 4), dtype=np.float64)

        se3_rx(0.7, T)
        se3_log(T, twist)
        se3_exp(twist, T_recovered)

        assert_allclose(T_recovered, T, atol=1e-10)


class TestSE3Interp:
    """Tests for SE3 interpolation."""

    def test_interp_endpoints(self):
        """Test interpolation at endpoints."""
        T1 = np.zeros((4, 4), dtype=np.float64)
        T2 = np.zeros((4, 4), dtype=np.float64)
        out = np.zeros((4, 4), dtype=np.float64)

        se3_from_rpy(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, T1)
        se3_from_rpy(1.0, 2.0, 3.0, 0.5, 0.6, 0.7, T2)

        # s=0 should give T1
        se3_interp(T1, T2, 0.0, out)
        assert_allclose(out, T1, atol=1e-10)

        # s=1 should give T2
        se3_interp(T1, T2, 1.0, out)
        assert_allclose(out, T2, atol=1e-10)

    def test_interp_midpoint_translation(self):
        """Test interpolation midpoint for pure translation."""
        T1 = np.zeros((4, 4), dtype=np.float64)
        T2 = np.zeros((4, 4), dtype=np.float64)
        out = np.zeros((4, 4), dtype=np.float64)

        se3_from_trans(0.0, 0.0, 0.0, T1)
        se3_from_trans(2.0, 4.0, 6.0, T2)

        se3_interp(T1, T2, 0.5, out)

        # Midpoint translation should be [1, 2, 3]
        assert_allclose(out[:3, 3], [1.0, 2.0, 3.0], atol=1e-10)
        assert_allclose(out[:3, :3], np.eye(3), atol=1e-10)

    def test_interp_preserves_SE3(self):
        """Verify interpolation produces valid SE3."""
        T1 = np.zeros((4, 4), dtype=np.float64)
        T2 = np.zeros((4, 4), dtype=np.float64)
        out = np.zeros((4, 4), dtype=np.float64)

        se3_from_rpy(0.1, 0.2, 0.3, 0.1, 0.2, 0.3, T1)
        se3_from_rpy(0.4, 0.5, 0.6, 0.6, 0.5, 0.4, T2)

        for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
            se3_interp(T1, T2, s, out)

            # Check rotation is orthonormal
            R = out[:3, :3]
            assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
            assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

            # Check bottom row
            assert_allclose(out[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-10)


class TestSE3Angdist:
    """Tests for angular distance between SE3 transforms."""

    def test_same_rotation(self):
        T1 = np.zeros((4, 4), dtype=np.float64)
        T2 = np.zeros((4, 4), dtype=np.float64)

        se3_from_rpy(0.0, 0.0, 0.0, 0.3, 0.4, 0.5, T1)
        se3_from_rpy(1.0, 2.0, 3.0, 0.3, 0.4, 0.5, T2)

        # Same rotation, different translation -> angle = 0
        dist = se3_angdist(T1, T2)
        assert_allclose(dist, 0.0, atol=1e-10)

    def test_90_degree_rotation(self):
        T1 = np.eye(4)
        T2 = np.zeros((4, 4), dtype=np.float64)

        se3_rx(np.pi / 2, T2)

        dist = se3_angdist(T1, T2)
        assert_allclose(dist, np.pi / 2, atol=1e-10)

    def test_180_degree_rotation(self):
        T1 = np.eye(4)
        T2 = np.zeros((4, 4), dtype=np.float64)

        se3_rx(np.pi, T2)

        dist = se3_angdist(T1, T2)
        assert_allclose(dist, np.pi, atol=1e-10)

    def test_symmetric(self):
        T1 = np.zeros((4, 4), dtype=np.float64)
        T2 = np.zeros((4, 4), dtype=np.float64)

        se3_from_rpy(0.1, 0.2, 0.3, 0.1, 0.2, 0.3, T1)
        se3_from_rpy(0.4, 0.5, 0.6, 0.6, 0.5, 0.4, T2)

        dist1 = se3_angdist(T1, T2)
        dist2 = se3_angdist(T2, T1)

        assert_allclose(dist1, dist2, atol=1e-10)


class TestNumericalStability:
    """Tests for numerical edge cases."""

    def test_very_small_rotation(self):
        """Test log/exp with very small rotation."""
        omega = np.array([1e-12, 1e-12, 1e-12])
        R = np.zeros((3, 3), dtype=np.float64)
        out = np.zeros(3, dtype=np.float64)

        so3_exp(omega, R)
        so3_log(R, out)

        # Should be essentially identity
        assert_allclose(R, np.eye(3), atol=1e-10)

    def test_near_180_rotation(self):
        """Test log/exp near 180 degree rotation."""
        # 179.9 degrees about X axis
        theta = np.pi - 0.001
        omega = np.array([theta, 0.0, 0.0])
        R = np.zeros((3, 3), dtype=np.float64)
        out = np.zeros(3, dtype=np.float64)

        so3_exp(omega, R)
        so3_log(R, out)

        # Recovered angle should match
        assert_allclose(np.linalg.norm(out), theta, atol=1e-4)

    def test_interp_near_singularity(self):
        """Test interpolation when rotations are nearly opposite."""
        T1 = np.eye(4)
        T2 = np.zeros((4, 4), dtype=np.float64)
        out = np.zeros((4, 4), dtype=np.float64)

        se3_rx(np.pi - 0.01, T2)

        # Should still produce valid SE3
        se3_interp(T1, T2, 0.5, out)

        R = out[:3, :3]
        assert_allclose(R @ R.T, np.eye(3), atol=1e-8)
