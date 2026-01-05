"""Unit tests for parol6.motion trajectory pipeline."""

import numpy as np
import pytest

from parol6.config import INTERVAL_S
from parol6.motion import JointPath, ProfileType, Trajectory, TrajectoryBuilder


class TestJointPath:
    """Tests for JointPath dataclass."""

    def test_interpolate_two_points(self):
        """Interpolate between two joint configurations."""
        start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        end = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        n_samples = 11

        path = JointPath.interpolate(start, end, n_samples)

        assert len(path) == n_samples
        assert np.allclose(path.positions[0], start)
        assert np.allclose(path.positions[-1], end)
        # Midpoint should be average
        assert np.allclose(path.positions[5], (start + end) / 2)

    def test_interpolate_minimum_samples(self):
        """Interpolate with n_samples < 2 should produce 2 samples."""
        start = np.zeros(6)
        end = np.ones(6)

        path = JointPath.interpolate(start, end, n_samples=1)
        assert len(path) == 2

        path = JointPath.interpolate(start, end, n_samples=0)
        assert len(path) == 2

    def test_sample_at_boundaries(self):
        """Sample at s=0 and s=1 should return exact endpoints."""
        start = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        end = np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        path = JointPath.interpolate(start, end, n_samples=10)

        assert np.allclose(path.sample(0.0), start)
        assert np.allclose(path.sample(1.0), end)

    def test_sample_clamps_out_of_bounds(self):
        """Sample at s<0 or s>1 should clamp to boundaries."""
        start = np.zeros(6)
        end = np.ones(6)
        path = JointPath.interpolate(start, end, n_samples=10)

        assert np.allclose(path.sample(-0.5), start)
        assert np.allclose(path.sample(1.5), end)

    def test_append_concatenates_paths(self):
        """Append should concatenate two paths."""
        path1 = JointPath.interpolate(np.zeros(6), np.ones(6), n_samples=5)
        path2 = JointPath.interpolate(np.ones(6), np.ones(6) * 2, n_samples=5)

        combined = path1.append(path2)

        assert len(combined) == 10
        assert np.allclose(combined.positions[:5], path1.positions)
        assert np.allclose(combined.positions[5:], path2.positions)


class TestTrajectoryBuilder:
    """Tests for TrajectoryBuilder."""

    @pytest.fixture
    def simple_joint_path(self) -> JointPath:
        """Create a simple joint path for testing."""
        start = np.array([0.0, -1.57, 3.14, 0.0, 0.0, 0.0])  # ~0, -90deg, 180deg
        end = np.array([0.5, -1.0, 2.5, 0.5, 0.0, 0.5])
        return JointPath.interpolate(start, end, n_samples=50)

    def test_build_linear_profile(self, simple_joint_path):
        """LINEAR profile should produce uniformly spaced trajectory."""
        builder = TrajectoryBuilder(
            joint_path=simple_joint_path,
            profile=ProfileType.LINEAR,
            duration=1.0,  # 1 second
            dt=INTERVAL_S,
        )

        trajectory = builder.build()

        assert isinstance(trajectory, Trajectory)
        assert len(trajectory) > 0
        assert trajectory.duration >= 1.0
        # Steps should be int32
        assert trajectory.steps.dtype == np.int32

    def test_build_quintic_profile(self, simple_joint_path):
        """QUINTIC profile should produce smooth trajectory."""
        builder = TrajectoryBuilder(
            joint_path=simple_joint_path,
            profile=ProfileType.QUINTIC,
            duration=3.0,  # Long duration to stay within velocity limits
            dt=INTERVAL_S,
        )

        trajectory = builder.build()

        assert isinstance(trajectory, Trajectory)
        assert len(trajectory) > 0
        assert trajectory.duration >= 3.0

    def test_build_trapezoid_profile(self, simple_joint_path):
        """TRAPEZOID profile should produce trajectory with plateau."""
        builder = TrajectoryBuilder(
            joint_path=simple_joint_path,
            profile=ProfileType.TRAPEZOID,
            duration=3.0,  # Long duration to stay within velocity limits
            dt=INTERVAL_S,
        )

        trajectory = builder.build()

        assert isinstance(trajectory, Trajectory)
        assert len(trajectory) > 0
        assert trajectory.duration >= 3.0

    def test_build_ruckig_profile(self, simple_joint_path):
        """RUCKIG profile should produce jerk-limited trajectory."""
        builder = TrajectoryBuilder(
            joint_path=simple_joint_path,
            profile=ProfileType.RUCKIG,
            dt=INTERVAL_S,
        )

        trajectory = builder.build()

        assert isinstance(trajectory, Trajectory)
        assert len(trajectory) > 0
        assert trajectory.duration > 0

    def test_velocity_percent_scaling(self, simple_joint_path):
        """Lower velocity_percent should increase duration."""
        # Use TOPPRA which is time-optimal and respects velocity limits
        builder_100 = TrajectoryBuilder(
            joint_path=simple_joint_path,
            profile=ProfileType.TOPPRA,
            velocity_percent=100.0,
        )
        builder_50 = TrajectoryBuilder(
            joint_path=simple_joint_path,
            profile=ProfileType.TOPPRA,
            velocity_percent=50.0,
        )

        traj_100 = builder_100.build()
        traj_50 = builder_50.build()

        # At 50% velocity, duration should be longer
        assert traj_50.duration >= traj_100.duration

    def test_single_point_path(self):
        """Single-point path should produce single-step trajectory."""
        path = JointPath(positions=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))

        builder = TrajectoryBuilder(
            joint_path=path,
            profile=ProfileType.LINEAR,
        )

        trajectory = builder.build()

        assert len(trajectory) == 1
        assert trajectory.duration == 0.0


class TestTrajectory:
    """Tests for Trajectory dataclass."""

    def test_len_returns_step_count(self):
        """len() should return number of steps."""
        steps = np.zeros((100, 6), dtype=np.int32)
        traj = Trajectory(steps=steps, duration=1.0)

        assert len(traj) == 100

    def test_getitem_returns_step(self):
        """Indexing should return individual step."""
        steps = np.arange(60, dtype=np.int32).reshape(10, 6)
        traj = Trajectory(steps=steps, duration=1.0)

        assert np.array_equal(traj[0], steps[0])
        assert np.array_equal(traj[5], steps[5])
        assert np.array_equal(traj[-1], steps[-1])
