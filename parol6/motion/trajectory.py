"""
Unified trajectory generation pipeline using TOPP-RA for time-optimal path parameterization.

This module provides the shared trajectory infrastructure for all motion commands.
Path generation (geometry) stays in command files; this handles time parameterization.

Pipeline:
  1. Command generates Cartesian poses (for cart commands) or joint targets
  2. JointPath holds uniformly-sampled joint positions
  3. TrajectoryBuilder applies TOPP-RA + motion profile to produce Trajectory
  4. Trajectory contains motor steps ready for tick-by-tick execution
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray
from ruckig import InputParameter, OutputParameter, Result, Ruckig

import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.config import INTERVAL_S, LIMITS, rad_to_steps
from parol6.utils.ik import solve_ik
from parol6.utils.se3_utils import se3_from_rpy

if TYPE_CHECKING:
    import sophuspy as sp

# Silence toppra's verbose debug output
logging.getLogger("toppra").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


class ProfileType(Enum):
    """Available trajectory profile types for motion planning."""

    TOPPRA = "toppra"  # Time-optimal path following (default)
    RUCKIG = "ruckig"  # Point-to-point jerk-limited (can't follow Cartesian paths)
    QUINTIC = "quintic"  # Quintic polynomial (C² smooth, predictable shape)
    TRAPEZOID = "trapezoid"  # Trapezoidal velocity profile
    SCURVE = "scurve"  # S-curve (jerk-limited) velocity profile
    LINEAR = "linear"  # Direct linear interpolation (no smoothing)

    @classmethod
    def from_string(cls, name: str) -> ProfileType:
        """Convert string to ProfileType, case-insensitive."""
        name_upper = name.upper()
        if name_upper == "NONE":
            return cls.LINEAR
        try:
            return cls[name_upper]
        except KeyError:
            logger.warning("Unknown profile type '%s', using TOPPRA", name)
            return cls.TOPPRA


@dataclass
class JointPath:
    """
    Joint-space path uniformly sampled in path space.

    This is the common abstraction for all motion commands. Cartesian commands
    solve IK to produce this; joint commands interpolate directly.

    Attributes:
        positions: (N, 6) array of joint angles in radians
    """

    positions: NDArray[np.float64]  # (N, 6) joint angles in radians

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> NDArray[np.float64]:
        return self.positions[idx]

    @classmethod
    def from_poses(
        cls,
        poses: NDArray[np.float64] | list[sp.SE3],
        seed_q: NDArray[np.float64],
        quiet_logging: bool = True,
    ) -> JointPath:
        """
        Solve IK for poses with seeded chain.

        Each IK solve uses the previous solution as seed, maintaining continuity.

        Args:
            poses: Either (N, 6) array of [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
                   or list of SE3 poses
            seed_q: Initial joint angles for IK seeding (radians)
            quiet_logging: Suppress IK logging

        Returns:
            JointPath with solved joint positions

        Raises:
            IKError: If any pose is unreachable
        """
        from parol6.utils.errors import IKError

        # Convert to list of SE3 if NDArray
        if isinstance(poses, np.ndarray):
            se3_poses = [
                se3_from_rpy(
                    p[0] / 1000.0,  # x mm -> m
                    p[1] / 1000.0,  # y mm -> m
                    p[2] / 1000.0,  # z mm -> m
                    p[3],  # rx degrees
                    p[4],  # ry degrees
                    p[5],  # rz degrees
                    degrees=True,
                )
                for p in poses
            ]
        else:
            se3_poses = poses

        n_poses = len(se3_poses)
        positions = np.empty((n_poses, 6), dtype=np.float64)
        q_prev = np.asarray(seed_q, dtype=np.float64)

        for i, target_se3 in enumerate(se3_poses):
            ik_result = solve_ik(
                PAROL6_ROBOT.robot,
                target_se3,
                q_prev,
                quiet_logging=quiet_logging,
            )

            if not ik_result.success or ik_result.q is None:
                error_str = f"Cartesian path point {i}/{n_poses} is unreachable."
                if ik_result.violations:
                    error_str += f" Reason: {ik_result.violations}"
                raise IKError(error_str)

            q_curr = np.asarray(ik_result.q, dtype=np.float64)
            positions[i] = q_curr
            q_prev = q_curr

        return cls(positions=positions)

    @classmethod
    def interpolate(
        cls,
        start_rad: NDArray[np.float64],
        end_rad: NDArray[np.float64],
        n_samples: int,
    ) -> JointPath:
        """
        Direct joint-space linear interpolation (for MovePose/MoveJoint).

        Args:
            start_rad: Starting joint angles in radians
            end_rad: Ending joint angles in radians
            n_samples: Number of samples (minimum 2)

        Returns:
            JointPath with interpolated positions
        """
        n_samples = max(2, n_samples)
        start = np.asarray(start_rad, dtype=np.float64)
        end = np.asarray(end_rad, dtype=np.float64)

        # Vectorized interpolation using broadcasting
        t = np.linspace(0, 1, n_samples).reshape(-1, 1)
        positions = start + t * (end - start)

        return cls(positions=positions)

    def append(self, other: JointPath) -> JointPath:
        """
        Concatenate paths (for path blending).

        Args:
            other: Path to append

        Returns:
            New JointPath with concatenated positions
        """
        combined = np.concatenate([self.positions, other.positions], axis=0)
        return JointPath(positions=combined)

    def sample(self, s: float) -> NDArray[np.float64]:
        """
        Sample path at normalized position s in [0, 1].

        Uses linear interpolation between path points.

        Args:
            s: Path position from 0 (start) to 1 (end)

        Returns:
            Interpolated joint position
        """
        s = np.clip(s, 0.0, 1.0)
        n = len(self.positions)
        if n < 2:
            return self.positions[0].copy()

        idx_float = s * (n - 1)
        idx_lo = int(idx_float)
        idx_hi = min(idx_lo + 1, n - 1)
        frac = idx_float - idx_lo

        return self.positions[idx_lo] * (1 - frac) + self.positions[idx_hi] * frac

    def sample_many(self, s_values: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Vectorized sampling at multiple path positions.

        Args:
            s_values: Array of path positions from 0 (start) to 1 (end)

        Returns:
            (N, 6) array of interpolated joint positions
        """
        s_values = np.clip(s_values, 0.0, 1.0)
        n = len(self.positions)
        if n < 2:
            return np.tile(self.positions[0], (len(s_values), 1))

        idx_float = s_values * (n - 1)
        idx_lo = idx_float.astype(np.intp)
        idx_hi = np.minimum(idx_lo + 1, n - 1)
        frac = (idx_float - idx_lo).reshape(-1, 1)

        return self.positions[idx_lo] * (1 - frac) + self.positions[idx_hi] * frac


@dataclass
class Trajectory:
    """
    Ready-to-execute trajectory with motor steps at control rate.

    Precomputed trajectories are sent directly to the controller without smoothing.
    StreamingExecutor is only used for online targets (jogging/streaming).

    Attributes:
        steps: (M, 6) motor steps at each control tick
        duration: Actual duration in seconds
    """

    steps: NDArray[np.int32]  # (M, 6) motor steps
    duration: float  # seconds

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, idx: int) -> NDArray[np.int32]:
        return self.steps[idx]


class TrajectoryBuilder:
    """
    Converts JointPath to executable Trajectory.

    Uses TOPP-RA to compute maximum allowable path speed, then applies
    the selected motion profile (clamped to TOPP-RA limits).

    All limits come from PAROL6_ROBOT config - no hardcoded fallbacks.
    """

    def __init__(
        self,
        joint_path: JointPath,
        profile: ProfileType | str,
        velocity_percent: float | None = None,
        accel_percent: float | None = None,
        jerk_percent: float | None = None,
        duration: float | None = None,
        dt: float = INTERVAL_S,
        cart_vel_limit: float | None = None,
        cart_acc_limit: float | None = None,
    ):
        """
        Initialize trajectory builder.

        Args:
            joint_path: Path in joint space
            profile: Motion profile to apply
            velocity_percent: Scale joint velocity limits (0-100), default 100
            accel_percent: Scale joint acceleration limits (0-100), default 100
            jerk_percent: Scale jerk limits (0-100), default 100
            duration: Override duration (stretches profile if longer than TOPP-RA min)
            dt: Control loop time step
            cart_vel_limit: Cartesian linear velocity limit in m/s (for Cartesian commands)
            cart_acc_limit: Cartesian linear acceleration limit in m/s² (for Cartesian commands)
        """
        self.joint_path = joint_path
        self.profile = (
            ProfileType.from_string(profile) if isinstance(profile, str) else profile
        )
        self.velocity_percent = (
            velocity_percent if velocity_percent is not None else 100.0
        )
        self.accel_percent = accel_percent if accel_percent is not None else 100.0
        self.jerk_percent = jerk_percent if jerk_percent is not None else 100.0
        self.duration = duration
        self.dt = dt
        self.cart_vel_limit = cart_vel_limit
        self.cart_acc_limit = cart_acc_limit

        # Joint limits at full (hard upper bounds) - from centralized config
        self.v_max = LIMITS.joint.hard.velocity * (self.velocity_percent / 100.0)
        self.a_max = LIMITS.joint.hard.acceleration * (self.accel_percent / 100.0)
        self.j_max = LIMITS.joint.hard.jerk * (self.jerk_percent / 100.0)

        # Pre-compute limit arrays for TOPP-RA (avoids allocation per build() call)
        self._vlim = np.column_stack([-self.v_max, self.v_max])
        self._alim = np.column_stack([-self.a_max, self.a_max])

    def build(self) -> Trajectory:
        """
        Generate time-parameterized trajectory.

        Uses TOPP-RA to compute time-optimal trajectory, then samples it directly
        at the control rate. No interpolation of the original joint path is needed
        since TOPP-RA's trajectory already provides smooth, continuous positions.

        For RUCKIG profile: Uses Ruckig for point-to-point motion (ignores path waypoints)
        For other profiles: Uses TOPP-RA trajectory directly

        Returns:
            Trajectory ready for execution
        """
        if len(self.joint_path) < 2:
            # Trivial path - single point
            steps = np.array(
                [rad_to_steps(self.joint_path.positions[0])],
                dtype=np.int32,
            )
            return Trajectory(steps=steps, duration=0.0)

        # Route to appropriate trajectory builder based on profile
        if self.profile == ProfileType.RUCKIG:
            # Point-to-point jerk-limited motion (doesn't follow path waypoints)
            return self._build_ruckig_trajectory()
        elif self.profile == ProfileType.LINEAR:
            # Simple linear interpolation - no velocity smoothing
            return self._build_simple_trajectory()
        elif self.profile == ProfileType.QUINTIC:
            # Quintic polynomial timing
            return self._build_quintic_trajectory()
        elif self.profile == ProfileType.TRAPEZOID:
            # Trapezoidal velocity profile along path
            return self._build_trapezoid_trajectory()
        elif self.profile == ProfileType.SCURVE:
            # S-curve (jerk-limited) velocity profile along path
            return self._build_scurve_trajectory()
        else:
            # TOPPRA is default - time-optimal path following
            return self._build_toppra_trajectory()

    def _build_toppra_trajectory(self) -> Trajectory:
        """
        Build trajectory using TOPP-RA's time-optimal path parameterization.

        Uses cubic spline interpolation through waypoints and computes
        time-optimal velocity profile respecting joint limits and optional
        Cartesian velocity limits.
        """
        positions = self.joint_path.positions
        n_points = len(positions)

        # Uniform parameterization for spline knots
        ss_waypoints = np.linspace(0.0, 1.0, n_points)

        path = ta.SplineInterpolator(ss_waypoints, positions)

        # Use pre-computed limit arrays for constraints
        joint_vel_constraint = constraint.JointVelocityConstraint(self._vlim)
        joint_acc_constraint = constraint.JointAccelerationConstraint(self._alim)
        constraints = [joint_vel_constraint, joint_acc_constraint]

        # Add Cartesian velocity constraint if specified
        if self.cart_vel_limit is not None and self.cart_vel_limit > 0:
            cart_constraint = self._build_cart_vel_constraint(path, ss_waypoints)
            if cart_constraint is not None:
                constraints.append(cart_constraint)

        try:
            # Use evenly-spaced gridpoints - TOPPRA docs recommend "at least a few times
            # the number of waypoints". Auto-selection can cluster points near
            # discontinuities, causing TOPPRAsd to produce incorrect durations.
            n_gridpoints = n_points * 3
            gridpoints = np.linspace(0.0, 1.0, n_gridpoints)

            # Use TOPPRAsd if duration is specified, otherwise time-optimal TOPPRA
            if self.duration is not None and self.duration > 0:
                instance = algo.TOPPRAsd(constraints, path, gridpoints=gridpoints)
                instance.set_desired_duration(self.duration)
                jnt_traj = instance.compute_trajectory()
                if jnt_traj is not None:
                    duration = self.duration
                    logger.debug(
                        "TrajectoryBuilder: TOPPRAsd target_duration=%.3f, path_len=%d",
                        duration,
                        n_points,
                    )
                else:
                    # Fall back to time-optimal if TOPPRAsd fails
                    logger.warning("TOPPRAsd failed, trying time-optimal TOPPRA")
                    instance = algo.TOPPRA(constraints, path, gridpoints=gridpoints)
                    jnt_traj = instance.compute_trajectory()
            else:
                instance = algo.TOPPRA(constraints, path, gridpoints=gridpoints)
                jnt_traj = instance.compute_trajectory()

            if jnt_traj is None:
                logger.warning("TOPP-RA failed, falling back to simple trajectory")
                return self._build_simple_trajectory()

            duration = float(jnt_traj.duration)

            logger.debug(
                "TrajectoryBuilder: TOPP-RA duration=%.3f, path_len=%d",
                duration,
                n_points,
            )

            # Sample trajectory at control rate, including endpoint (vectorized)
            n_output = max(2, int(np.floor(duration / self.dt)) + 1)
            times = np.arange(n_output - 1) * self.dt
            trajectory_rad = np.empty((n_output, 6), dtype=np.float64)
            trajectory_rad[:-1] = jnt_traj(times)
            trajectory_rad[-1] = jnt_traj(duration)

            logger.debug(
                "TrajectoryBuilder: output_samples=%d, duration=%.3f",
                len(trajectory_rad),
                duration,
            )

            # Convert to motor steps (vectorized)
            steps = cast(NDArray[np.int32], rad_to_steps(trajectory_rad))

            return Trajectory(steps=steps, duration=duration)

        except Exception as e:
            logger.warning("TOPP-RA error: %s. Using fallback.", e)
            return self._build_simple_trajectory()

    def _build_simple_trajectory(self) -> Trajectory:
        """
        Build trajectory with simple linear interpolation.

        Iteratively extends duration if joint velocity/acceleration limits
        are violated, using the same approach as QUINTIC/TRAPEZOID/SCURVE.
        """
        user_duration = self.duration if self.duration and self.duration > 0 else None
        if user_duration:
            duration = user_duration
        else:
            duration = self._estimate_simple_duration()

        max_iterations = 10
        for iteration in range(max_iterations):
            n_output = max(2, int(np.ceil(duration / self.dt)))

            # Vectorized path sampling
            s_values = np.linspace(0.0, 1.0, n_output)
            trajectory_rad = self.joint_path.sample_many(s_values)

            # Check if joint limits are satisfied
            slowdown = self._compute_slowdown_factor(trajectory_rad, duration)
            if slowdown <= 1.0:
                break

            # Extend duration and retry
            new_duration = duration * slowdown * 1.05  # 5% margin
            if user_duration:
                logger.warning(
                    "LINEAR: Extending duration from %.3fs to %.3fs to respect joint limits",
                    user_duration,
                    new_duration,
                )
            duration = new_duration
        else:
            raise ValueError(
                f"LINEAR: Could not satisfy joint limits after {max_iterations} iterations. "
                f"Path may be too aggressive for current joint limits."
            )

        # Convert to motor steps (vectorized)
        steps = cast(NDArray[np.int32], rad_to_steps(trajectory_rad))

        return Trajectory(steps=steps, duration=duration)

    def _compute_slowdown_factor(
        self,
        trajectory_rad: NDArray[np.float64],
        duration: float,
    ) -> float:
        """
        Compute slowdown factor needed to bring trajectory within joint limits.

        Returns 1.0 if trajectory is valid, >1.0 if it needs to be slower.

        Args:
            trajectory_rad: (N, 6) joint positions in radians at each sample
            duration: Total trajectory duration in seconds

        Returns:
            Slowdown factor (multiply duration by this to fix violations)
        """
        n_samples = len(trajectory_rad)
        if n_samples < 2:
            return 1.0

        # Actual time spacing between samples
        actual_dt = duration / (n_samples - 1)

        slowdown = 1.0

        # Compute velocities via finite difference using actual sample spacing
        velocities = np.diff(trajectory_rad, axis=0) / actual_dt
        max_vel = np.max(np.abs(velocities), axis=0)

        # Check velocity limits - slowdown is linear with velocity
        vel_ratios = max_vel / self.v_max
        max_vel_ratio = float(np.max(vel_ratios))
        if max_vel_ratio > 1.0:
            slowdown = max(slowdown, max_vel_ratio)

        # Compute accelerations via finite difference
        if len(velocities) > 1:
            accelerations = np.diff(velocities, axis=0) / actual_dt
            max_acc = np.max(np.abs(accelerations), axis=0)

            # Acceleration scales with 1/t², so slowdown factor is sqrt
            acc_ratios = max_acc / self.a_max
            max_acc_ratio = float(np.max(acc_ratios))
            if max_acc_ratio > 1.0:
                slowdown = max(slowdown, np.sqrt(max_acc_ratio))

        return slowdown

    def _is_cartesian_path(self) -> bool:
        """Check if this is a Cartesian path (has Cartesian velocity limits set)."""
        return self.cart_vel_limit is not None and self.cart_vel_limit > 0

    def _compute_joint_duration_trapezoid(self) -> float:
        """
        Compute duration for joint paths using trapezoidal profile.

        For each joint, uses InterpolatePy to compute the minimum duration
        for its displacement given its velocity/acceleration limits.
        Returns the maximum (slowest joint determines overall duration).
        """
        from interpolatepy.trapezoidal import (
            TrajectoryParams as TrapParams,
            TrapezoidalTrajectory,
        )

        positions = self.joint_path.positions
        if len(positions) < 2:
            return self.dt * 2

        total_delta = positions[-1] - positions[0]
        max_duration = 0.0

        for j in range(6):
            delta = abs(total_delta[j])
            if delta < 1e-6:
                continue

            params = TrapParams(
                q0=0.0,
                q1=delta,
                v0=0.0,
                v1=0.0,
                vmax=self.v_max[j],
                amax=self.a_max[j],
            )
            _, duration = TrapezoidalTrajectory.generate_trajectory(params)
            max_duration = max(max_duration, duration)

        return max(max_duration, self.dt * 2)

    def _compute_joint_duration_scurve(self) -> float:
        """
        Compute duration for joint paths using S-curve profile.

        For each joint, uses InterpolatePy's DoubleSTrajectory to compute
        the minimum duration given velocity/acceleration/jerk limits.
        Returns the maximum (slowest joint determines overall duration).
        """
        from interpolatepy import DoubleSTrajectory, StateParams, TrajectoryBounds

        positions = self.joint_path.positions
        if len(positions) < 2:
            return self.dt * 2

        total_delta = positions[-1] - positions[0]
        max_duration = 0.0

        for j in range(6):
            delta = abs(total_delta[j])
            if delta < 1e-6:
                continue

            state = StateParams(q_0=0.0, q_1=delta, v_0=0.0, v_1=0.0)
            bounds = TrajectoryBounds(
                v_bound=self.v_max[j],
                a_bound=self.a_max[j],
                j_bound=self.j_max[j],
            )
            traj = DoubleSTrajectory(state, bounds)
            max_duration = max(max_duration, traj.get_duration())

        return max(max_duration, self.dt * 2)

    def _compute_cartesian_duration_trapezoid(self) -> float:
        """Compute duration for Cartesian paths using trapezoidal profile."""
        from interpolatepy.trapezoidal import (
            TrajectoryParams as TrapParams,
            TrapezoidalTrajectory,
        )

        v_max = self.cart_vel_limit if self.cart_vel_limit else 0.1
        a_max = self.cart_acc_limit if self.cart_acc_limit else v_max * 2

        # Profile for unit path (s: 0 to 1)
        params = TrapParams(q0=0.0, q1=1.0, v0=0.0, v1=0.0, vmax=v_max, amax=a_max)
        _, duration = TrapezoidalTrajectory.generate_trajectory(params)
        return duration

    def _compute_cartesian_duration_scurve(self) -> float:
        """Compute duration for Cartesian paths using S-curve profile."""
        from interpolatepy import DoubleSTrajectory, StateParams, TrajectoryBounds

        v_max = self.cart_vel_limit if self.cart_vel_limit else 0.1
        a_max = self.cart_acc_limit if self.cart_acc_limit else v_max * 2
        j_max = a_max * 4  # Conservative jerk for Cartesian

        state = StateParams(q_0=0.0, q_1=1.0, v_0=0.0, v_1=0.0)
        bounds = TrajectoryBounds(v_bound=v_max, a_bound=a_max, j_bound=j_max)
        traj = DoubleSTrajectory(state, bounds)
        return traj.get_duration()

    def _build_quintic_trajectory(self) -> Trajectory:
        """
        Build trajectory with quintic polynomial velocity profile.

        For Cartesian paths: falls back to TOPPRA since simple profiles can't
        handle the non-uniform joint movements from IK (especially near singularities).
        For Joint paths: computes per-joint durations, uses slowest joint's timing.

        Uses InterpolatePy's order-5 polynomial which provides zero velocity
        and acceleration at endpoints (C² smooth).

        Iteratively extends duration if joint limits are violated.
        """
        from interpolatepy import BoundaryCondition, PolynomialTrajectory, TimeInterval

        # For Cartesian paths, fall back to TOPPRA - simple profiles can't handle
        # the non-uniform joint movements from IK (especially near singularities)
        if self._is_cartesian_path():
            logger.debug("QUINTIC: Falling back to TOPPRA for Cartesian path")
            return self._build_toppra_trajectory()

        # Compute initial duration for joint paths
        user_duration = self.duration if self.duration and self.duration > 0 else None
        if user_duration:
            duration = user_duration
        else:
            duration = self._compute_joint_duration_trapezoid()

        max_iterations = 10
        for iteration in range(max_iterations):
            # Create quintic trajectory from s=0 to s=1 over duration
            bc_start = BoundaryCondition(position=0.0, velocity=0.0, acceleration=0.0)
            bc_end = BoundaryCondition(position=1.0, velocity=0.0, acceleration=0.0)
            interval = TimeInterval(start=0.0, end=duration)
            traj = PolynomialTrajectory.order_5_trajectory(bc_start, bc_end, interval)

            n_output = max(2, int(np.ceil(duration / self.dt)))
            times = np.linspace(0.0, duration, n_output)

            # Evaluate trajectory - returns (pos, vel, acc, jerk), we only need position
            s_values = np.array([traj(t)[0] for t in times], dtype=np.float64)
            trajectory_rad = self.joint_path.sample_many(s_values)

            # Check if joint limits are satisfied
            slowdown = self._compute_slowdown_factor(trajectory_rad, duration)
            if slowdown <= 1.0:
                break

            # Extend duration and retry
            new_duration = duration * slowdown * 1.05  # 5% extra margin
            if user_duration:
                logger.warning(
                    "QUINTIC: Extending duration from %.3fs to %.3fs to respect joint limits",
                    user_duration,
                    new_duration,
                )
            duration = new_duration
        else:
            raise ValueError(
                f"QUINTIC: Could not satisfy joint limits after {max_iterations} iterations. "
                f"Path may be too aggressive for current joint limits."
            )

        # Convert to motor steps
        steps = cast(NDArray[np.int32], rad_to_steps(trajectory_rad))

        return Trajectory(steps=steps, duration=duration)

    def _build_trapezoid_trajectory(self) -> Trajectory:
        """
        Build trajectory with trapezoidal velocity profile.

        For Cartesian paths: falls back to TOPPRA since simple profiles can't
        handle the non-uniform joint movements from IK (especially near singularities).
        For Joint paths: computes per-joint durations, uses slowest joint's timing.

        Iteratively extends duration if joint limits are violated.
        """
        from interpolatepy.trapezoidal import (
            TrajectoryParams as TrapParams,
            TrapezoidalTrajectory,
        )

        # For Cartesian paths, fall back to TOPPRA - simple profiles can't handle
        # the non-uniform joint movements from IK (especially near singularities)
        if self._is_cartesian_path():
            logger.debug("TRAPEZOID: Falling back to TOPPRA for Cartesian path")
            return self._build_toppra_trajectory()

        # Compute initial duration for joint paths
        user_duration = self.duration if self.duration and self.duration > 0 else None
        if user_duration:
            duration = user_duration
        else:
            duration = self._compute_joint_duration_trapezoid()

        max_iterations = 10
        for iteration in range(max_iterations):
            # Create trapezoidal profile for path parameter s (0 to 1)
            params = TrapParams(
                q0=0.0,
                q1=1.0,
                v0=0.0,
                v1=0.0,
                vmax=2.0 / duration,
                amax=4.0 / (duration * duration),
            )
            traj_fn, profile_duration = TrapezoidalTrajectory.generate_trajectory(
                params
            )

            # Scale times to fit our desired duration
            time_scale = profile_duration / duration if duration > 0 else 1.0

            n_output = max(2, int(np.ceil(duration / self.dt)))
            times = np.linspace(0.0, duration, n_output)

            # Evaluate profile at scaled times to get s values
            s_values = np.array(
                [traj_fn(t * time_scale)[0] for t in times], dtype=np.float64
            )
            trajectory_rad = self.joint_path.sample_many(s_values)

            # Check if joint limits are satisfied
            slowdown = self._compute_slowdown_factor(trajectory_rad, duration)
            if slowdown <= 1.0:
                break

            # Extend duration and retry
            new_duration = duration * slowdown * 1.05  # 5% extra margin
            if user_duration:
                logger.warning(
                    "TRAPEZOID: Extending duration from %.3fs to %.3fs to respect joint limits",
                    user_duration,
                    new_duration,
                )
            duration = new_duration
        else:
            raise ValueError(
                f"TRAPEZOID: Could not satisfy joint limits after {max_iterations} iterations. "
                f"Path may be too aggressive for current joint limits."
            )

        # Convert to motor steps
        steps = cast(NDArray[np.int32], rad_to_steps(trajectory_rad))

        return Trajectory(steps=steps, duration=duration)

    def _build_scurve_trajectory(self) -> Trajectory:
        """
        Build trajectory with S-curve (jerk-limited) velocity profile.

        For Cartesian paths: falls back to TOPPRA since simple profiles can't
        handle the non-uniform joint movements from IK (especially near singularities).
        For Joint paths: computes per-joint durations using jerk limits.

        Uses InterpolatePy's DoubleSTrajectory for smooth jerk-limited motion.
        Similar to RUCKIG but follows the path instead of point-to-point.

        Iteratively extends duration if joint limits are violated.
        """
        from interpolatepy import DoubleSTrajectory, StateParams, TrajectoryBounds

        # For Cartesian paths, fall back to TOPPRA - simple profiles can't handle
        # the non-uniform joint movements from IK (especially near singularities)
        if self._is_cartesian_path():
            logger.debug("SCURVE: Falling back to TOPPRA for Cartesian path")
            return self._build_toppra_trajectory()

        # Compute initial duration for joint paths
        user_duration = self.duration if self.duration and self.duration > 0 else None
        if user_duration:
            duration = user_duration
        else:
            duration = self._compute_joint_duration_scurve()

        max_iterations = 10
        for iteration in range(max_iterations):
            # Create S-curve profile for path parameter s (0 to 1)
            state = StateParams(q_0=0.0, q_1=1.0, v_0=0.0, v_1=0.0)
            bounds = TrajectoryBounds(
                v_bound=2.0 / duration,
                a_bound=4.0 / (duration * duration),
                j_bound=16.0 / (duration * duration * duration),
            )
            traj = DoubleSTrajectory(state, bounds)
            profile_duration = traj.get_duration()

            # Scale times to fit our desired duration
            time_scale = profile_duration / duration if duration > 0 else 1.0

            n_output = max(2, int(np.ceil(duration / self.dt)))
            times = np.linspace(0.0, duration, n_output)

            # Evaluate profile at scaled times to get s values
            s_values = np.array(
                [traj.evaluate(t * time_scale)[0] for t in times], dtype=np.float64
            )
            trajectory_rad = self.joint_path.sample_many(s_values)

            # Check if joint limits are satisfied
            slowdown = self._compute_slowdown_factor(trajectory_rad, duration)
            if slowdown <= 1.0:
                break

            # Extend duration and retry
            new_duration = duration * slowdown * 1.05  # 5% extra margin
            if user_duration:
                logger.warning(
                    "SCURVE: Extending duration from %.3fs to %.3fs to respect joint limits",
                    user_duration,
                    new_duration,
                )
            duration = new_duration
        else:
            raise ValueError(
                f"SCURVE: Could not satisfy joint limits after {max_iterations} iterations. "
                f"Path may be too aggressive for current joint limits."
            )

        # Convert to motor steps
        steps = cast(NDArray[np.int32], rad_to_steps(trajectory_rad))

        return Trajectory(steps=steps, duration=duration)

    def _build_cart_vel_constraint(
        self, path: ta.SplineInterpolator, ss_waypoints: NDArray
    ) -> constraint.JointVelocityConstraintVarying | None:
        """
        Build Cartesian velocity constraint for TOPP-RA using path-tangent method.

        Uses the path tangent (dq/ds) to compute accurate Cartesian velocity limits.
        At each path point s:
        - cart_vel = J_lin @ q_dot = J_lin @ (dq/ds * s_dot)
        - ||cart_vel|| = ||J_lin @ dq/ds|| * |s_dot|
        - For ||cart_vel|| <= v_max: |s_dot| <= v_max / ||J_lin @ dq/ds||

        This is more accurate than the column-norm method as it considers the
        actual direction of motion along the path.

        Args:
            path: The spline path through joint space
            ss_waypoints: Path parameter values at each waypoint

        Returns:
            JointVelocityConstraintVarying with path-dependent limits, or None if error
        """
        if self.cart_vel_limit is None or self.cart_vel_limit <= 0:
            return None

        try:
            robot = PAROL6_ROBOT.robot

            # cart_vel_limit is already in m/s (SI units)
            v_max_m_s = self.cart_vel_limit
            # Use scaled joint limits (respects user's velocity_percent)
            v_max_joint = self.v_max

            # Pre-allocate buffer for velocity limits (avoids per-call allocation)
            vlim_buffer = np.empty((6, 2), dtype=np.float64)

            def vlim_func(s: float) -> NDArray:
                """Compute velocity limits at path position s using path tangent."""
                q = path(s)
                dq_ds = path(s, 1)  # Path tangent (first derivative)

                # Get the linear part of the Jacobian (first 3 rows)
                assert robot is not None
                J_lin = robot.jacob0(q)[:3, :]

                # Cartesian velocity per unit s_dot along path tangent
                cart_vel_per_sdot = np.linalg.norm(J_lin @ dq_ds)

                if cart_vel_per_sdot < 1e-6:
                    # Near-zero path tangent (at waypoint or singular), use joint limits
                    vlim_buffer[:, 0] = -v_max_joint
                    vlim_buffer[:, 1] = v_max_joint
                    return vlim_buffer

                # Maximum s_dot to satisfy Cartesian velocity constraint
                max_sdot = v_max_m_s / cart_vel_per_sdot

                # The Cartesian constraint limits s_dot, not individual joint velocities.
                # We scale ALL joint velocity limits uniformly by the ratio of
                # (Cartesian-limited s_dot) / (fastest achievable s_dot from joint limits).
                #
                # This ensures the path velocity respects the Cartesian limit while
                # keeping joints at their relative proportions.
                abs_dq_ds = np.abs(dq_ds)

                # Compute s_dot limit from each joint's velocity limit
                with np.errstate(divide="ignore", invalid="ignore"):
                    s_dot_per_joint = np.where(
                        abs_dq_ds > 1e-9,
                        v_max_joint / abs_dq_ds,
                        np.inf,
                    )

                # The binding joint limit determines max achievable s_dot
                s_dot_from_joints = float(np.min(s_dot_per_joint))

                # If Cartesian constraint is more restrictive, scale down all limits
                if max_sdot < s_dot_from_joints and s_dot_from_joints > 0:
                    scale = max_sdot / s_dot_from_joints
                    q_dot_max = v_max_joint * scale
                else:
                    # Cartesian constraint is not binding, use joint limits
                    q_dot_max = v_max_joint

                vlim_buffer[:, 0] = -q_dot_max
                vlim_buffer[:, 1] = q_dot_max

                return vlim_buffer

            return constraint.JointVelocityConstraintVarying(vlim_func)

        except Exception as e:
            logger.warning("Failed to build Cartesian velocity constraint: %s", e)
            return None

    def _build_ruckig_trajectory(self) -> Trajectory:
        """
        Build trajectory using Ruckig for jerk-limited point-to-point motion.

        Note: This does NOT follow the path waypoints - it goes directly from
        start to end. Use TOPP-RA profiles for path-following motion.
        """
        n_dofs = 6
        gen = Ruckig(n_dofs, self.dt)
        inp = InputParameter(n_dofs)
        out = OutputParameter(n_dofs)

        start_pos = self.joint_path.positions[0]
        end_pos = self.joint_path.positions[-1]

        # Ruckig requires Python lists for input parameters
        inp.current_position = start_pos.tolist()
        inp.current_velocity = [0.0] * n_dofs
        inp.current_acceleration = [0.0] * n_dofs
        inp.target_position = end_pos.tolist()
        inp.target_velocity = [0.0] * n_dofs
        inp.target_acceleration = [0.0] * n_dofs
        inp.max_velocity = self.v_max.tolist()
        inp.max_acceleration = self.a_max.tolist()
        inp.max_jerk = self.j_max.tolist()

        # Pre-allocate buffer (estimate max iterations from simple duration + margin)
        est_duration = self._estimate_simple_duration()
        max_iters = int(est_duration / self.dt) + 500  # generous margin
        trajectory_rad = np.empty((max_iters, n_dofs), dtype=np.float64)

        count = 0
        result = Result.Working

        while result == Result.Working:
            result = gen.update(inp, out)
            if count < max_iters:
                trajectory_rad[count] = out.new_position
            count += 1
            out.pass_to_input(inp)

        if result == Result.Error:
            logger.warning("Ruckig failed, falling back to simple trajectory")
            return self._build_simple_trajectory()

        actual_duration = out.trajectory.duration

        # Trim to actual size
        trajectory_rad = trajectory_rad[:count]

        # Convert to motor steps (vectorized)
        steps = cast(NDArray[np.int32], rad_to_steps(trajectory_rad))

        return Trajectory(steps=steps, duration=actual_duration)

    def _estimate_simple_duration(self) -> float:
        """Estimate duration from path length and velocity limits."""
        positions = self.joint_path.positions
        if len(positions) < 2:
            return self.dt * 2

        # Vectorized: compute all segment deltas at once
        deltas = np.diff(positions, axis=0)  # (N-1, 6)
        # Time for each segment is max of per-joint times
        segment_times = np.max(np.abs(deltas) / self.v_max, axis=1)  # (N-1,)
        total_arc = np.sum(segment_times)

        return max(total_arc, self.dt * 2)


def build_cartesian_trajectory(
    start_pose: NDArray | list[float],
    end_pose: NDArray | list[float],
    seed_q: NDArray[np.float64],
    profile: ProfileType | str,
    n_samples: int = 100,
    velocity_percent: float | None = None,
    accel_percent: float | None = None,
    duration: float | None = None,
    dt: float = INTERVAL_S,
) -> Trajectory:
    """
    Convenience function to build trajectory for straight-line Cartesian motion.

    Args:
        start_pose: [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
        end_pose: [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
        seed_q: Current joint angles for IK seeding
        profile: Motion profile to apply
        n_samples: Number of Cartesian waypoints to generate
        velocity_percent: Scale velocity limits (0-100)
        accel_percent: Scale acceleration limits (0-100)
        duration: Override duration
        dt: Control loop time step

    Returns:
        Trajectory ready for execution
    """
    start = np.asarray(start_pose, dtype=np.float64)
    end = np.asarray(end_pose, dtype=np.float64)

    # Generate Cartesian waypoints (vectorized)
    t = np.linspace(0.0, 1.0, n_samples).reshape(-1, 1)
    poses = start + t * (end - start)

    # Solve IK for all poses
    joint_path = JointPath.from_poses(poses, seed_q)

    # Build trajectory
    builder = TrajectoryBuilder(
        joint_path=joint_path,
        profile=profile,
        velocity_percent=velocity_percent,
        accel_percent=accel_percent,
        duration=duration,
        dt=dt,
    )

    return builder.build()


def build_joint_trajectory(
    start_rad: NDArray[np.float64],
    end_rad: NDArray[np.float64],
    profile: ProfileType | str,
    n_samples: int = 50,
    velocity_percent: float | None = None,
    accel_percent: float | None = None,
    duration: float | None = None,
    dt: float = INTERVAL_S,
) -> Trajectory:
    """
    Convenience function to build trajectory for joint-space motion.

    Args:
        start_rad: Starting joint angles in radians
        end_rad: Ending joint angles in radians
        profile: Motion profile to apply
        n_samples: Number of joint waypoints
        velocity_percent: Scale velocity limits (0-100)
        accel_percent: Scale acceleration limits (0-100)
        duration: Override duration
        dt: Control loop time step

    Returns:
        Trajectory ready for execution
    """
    joint_path = JointPath.interpolate(start_rad, end_rad, n_samples)

    builder = TrajectoryBuilder(
        joint_path=joint_path,
        profile=profile,
        velocity_percent=velocity_percent,
        accel_percent=accel_percent,
        duration=duration,
        dt=dt,
    )

    return builder.build()
