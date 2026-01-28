"""
Cartesian Movement Commands
Contains commands for Cartesian space movements: CartesianJog, MovePose, MoveCart, MoveCartRelTrf
"""

import logging
import time
from abc import abstractmethod
from typing import cast

import numpy as np
from numba import njit  # type: ignore[import-untyped]

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.config import (
    CART_ANG_JOG_MIN,
    CART_LIN_JOG_MIN,
    INTERVAL_S,
    LIMITS,
    PATH_SAMPLES,
    rad_to_steps,
    steps_to_rad,
)
from parol6.motion import JointPath, TrajectoryBuilder
from parol6.protocol.wire import (
    CartJogCmd,
    CmdType,
    MoveCartCmd,
    MoveCartRelTrfCmd,
)
from parol6.server.command_registry import register_command
from parol6.server.state import ControllerState, get_fkine_se3
from parol6.utils.ik import AXIS_MAP, solve_ik
from parol6.utils.se3_utils import (
    se3_exp_ws,
    se3_from_rpy,
    se3_interp,
    se3_mul,
)

from .base import (
    ExecutionStatus,
    MotionCommand,
    TrajectoryMoveCommandBase,
)

logger = logging.getLogger(__name__)


@njit(cache=True)
def _apply_velocity_delta_wrf_jit(
    R: np.ndarray,
    smoothed_vel: np.ndarray,
    dt: float,
    current_pose: np.ndarray,
    vel_lin: np.ndarray,
    vel_ang: np.ndarray,
    world_twist: np.ndarray,
    delta: np.ndarray,
    out: np.ndarray,
    omega_ws: np.ndarray,
    R_ws: np.ndarray,
    V_ws: np.ndarray,
) -> None:
    """Apply smoothed velocity delta in World Reference Frame.

    Transforms body-frame velocity to world-frame and left-multiplies.
    WRF: target = delta @ current (world-frame delta applied first)

    Args:
        R: 3x3 rotation matrix (reference pose rotation for WRF)
        smoothed_vel: 6D body-frame velocity [vx, vy, vz, wx, wy, wz]
        dt: Time step
        current_pose: Current pose as 4x4 SE3
        vel_lin: Workspace buffer for linear velocity (3,)
        vel_ang: Workspace buffer for angular velocity (3,)
        world_twist: Workspace buffer for world-frame twist (6,)
        delta: Workspace buffer for delta transform (4x4)
        out: Output pose (4x4 SE3)
        omega_ws: Workspace buffer for axis-angle (3,)
        R_ws: Workspace buffer for rotation matrix (3,3)
        V_ws: Workspace buffer for V matrix (3,3)
    """
    # Transform velocity to world frame: R @ vel
    for i in range(3):
        vel_lin[i] = (
            R[i, 0] * smoothed_vel[0]
            + R[i, 1] * smoothed_vel[1]
            + R[i, 2] * smoothed_vel[2]
        )
        vel_ang[i] = (
            R[i, 0] * smoothed_vel[3]
            + R[i, 1] * smoothed_vel[4]
            + R[i, 2] * smoothed_vel[5]
        )

    # Build world-frame twist scaled by dt
    world_twist[0] = vel_lin[0] * dt
    world_twist[1] = vel_lin[1] * dt
    world_twist[2] = vel_lin[2] * dt
    world_twist[3] = vel_ang[0] * dt
    world_twist[4] = vel_ang[1] * dt
    world_twist[5] = vel_ang[2] * dt

    # Exponential map and apply (world frame = left multiply)
    se3_exp_ws(world_twist, delta, omega_ws, R_ws, V_ws)
    se3_mul(delta, current_pose, out)


@njit(cache=True)
def _apply_velocity_delta_trf_jit(
    smoothed_vel: np.ndarray,
    dt: float,
    current_pose: np.ndarray,
    body_twist: np.ndarray,
    delta: np.ndarray,
    out: np.ndarray,
    omega_ws: np.ndarray,
    R_ws: np.ndarray,
    V_ws: np.ndarray,
) -> None:
    """Apply smoothed velocity delta in Tool Reference Frame.

    Uses body-frame velocity directly and right-multiplies.
    TRF: target = current @ delta (body-frame delta applied in tool frame)

    Args:
        smoothed_vel: 6D body-frame velocity [vx, vy, vz, wx, wy, wz]
        dt: Time step
        current_pose: Current pose as 4x4 SE3
        body_twist: Workspace buffer for body-frame twist (6,)
        delta: Workspace buffer for delta transform (4x4)
        out: Output pose (4x4 SE3)
        omega_ws: Workspace buffer for axis-angle (3,)
        R_ws: Workspace buffer for rotation matrix (3,3)
        V_ws: Workspace buffer for V matrix (3,3)
    """
    # Build body-frame twist scaled by dt (no transformation needed)
    body_twist[0] = smoothed_vel[0] * dt
    body_twist[1] = smoothed_vel[1] * dt
    body_twist[2] = smoothed_vel[2] * dt
    body_twist[3] = smoothed_vel[3] * dt
    body_twist[4] = smoothed_vel[4] * dt
    body_twist[5] = smoothed_vel[5] * dt

    # Exponential map and apply (tool frame = right multiply)
    se3_exp_ws(body_twist, delta, omega_ws, R_ws, V_ws)
    se3_mul(current_pose, delta, out)


class CartesianMoveCommandBase(TrajectoryMoveCommandBase):
    """Base class for Cartesian move commands with straight-line path following."""

    streamable = True

    __slots__ = (
        "initial_pose",
        "target_pose",
        "_ik_stopping",
        "_duration",
    )

    def __init__(self):
        super().__init__()
        self.initial_pose: np.ndarray | None = None
        self.target_pose: np.ndarray | None = None
        self._duration: float | None = None

    @abstractmethod
    def _compute_target_pose(self, state: "ControllerState") -> None:
        """Compute self.target_pose from parsed parameters. Called during setup."""
        ...

    def do_setup(self, state: "ControllerState") -> None:
        """Set up the move - compute target pose and pre-compute trajectory if non-streaming."""
        self.initial_pose = get_fkine_se3(state)
        self._compute_target_pose(state)
        self._streaming_initialized = False  # Track first-tick initialization
        self._ik_stopping = False  # Track graceful stop on IK failure

        if state.stream_mode:
            return

        # Non-streaming: pre-compute trajectory
        self._precompute_trajectory(state)

    def _precompute_trajectory(self, state: "ControllerState") -> None:
        """Pre-compute joint trajectory that follows straight-line Cartesian path."""
        assert self.initial_pose is not None and self.target_pose is not None
        assert self.p is not None

        steps_to_rad(state.Position_in, self._q_rad_buf)
        current_rad = self._q_rad_buf

        duration = self.p.duration if self.p.duration > 0.0 else None
        vel_pct = self.p.speed_pct if self.p.speed_pct > 0.0 else 100.0
        acc_pct = self.p.accel_pct

        cart_poses = []
        interp_buf = np.zeros((4, 4), dtype=np.float64)
        for i in range(PATH_SAMPLES):
            s = i / (PATH_SAMPLES - 1)
            se3_interp(self.initial_pose, self.target_pose, s, interp_buf)
            cart_poses.append(interp_buf.copy())

        joint_path = JointPath.from_poses(cart_poses, current_rad, quiet_logging=True)

        # SI units (m/s, m/s²) - trajectory builder uses SI directly
        cart_vel_max = LIMITS.cart.hard.velocity.linear * (vel_pct / 100.0)
        cart_acc_max = LIMITS.cart.hard.acceleration.linear * (acc_pct / 100.0)

        builder = TrajectoryBuilder(
            joint_path=joint_path,
            profile=state.motion_profile,
            velocity_percent=vel_pct if duration is None else None,
            accel_percent=acc_pct,
            duration=duration,
            dt=INTERVAL_S,
            cart_vel_limit=cart_vel_max,
            cart_acc_limit=cart_acc_max,
        )

        trajectory = builder.build()
        self.trajectory_steps = trajectory.steps
        self._duration = trajectory.duration

        self.log_debug(
            "  -> Pre-computed Cartesian path: profile=%s, steps=%d, duration=%.3fs",
            state.motion_profile,
            len(self.trajectory_steps),
            float(self._duration),
        )

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Execute one tick - handles both streaming and non-streaming modes."""
        if state.stream_mode:
            # Streaming mode uses CartesianStreamingExecutor for straight-line TCP motion
            cse = state.cartesian_streaming_executor
            if cse is None:
                logger.warning("[MOVECART] No cartesian_streaming_executor available")
                self.is_finished = True
                return ExecutionStatus.completed("MOVECART: no executor")

            # Get current joint position for IK seed
            steps_to_rad(state.Position_in, self._q_rad_buf)

            # Initialize on first tick, or if executor not active (streaming interrupted)
            if not self._streaming_initialized or not cse.active:
                assert self.initial_pose is not None and self.target_pose is not None
                assert self.p is not None
                # Only sync pose if not already active to preserve velocity
                if not cse.active:
                    cse.sync_pose(self.initial_pose)
                vel_pct = self.p.speed_pct if self.p.speed_pct > 0.0 else 100.0
                acc_pct = self.p.accel_pct
                cse.set_limits(vel_pct, acc_pct)
                cse.set_pose_target(self.target_pose)
                self._streaming_initialized = True

            # Get smoothed pose from Cartesian executor (straight-line interpolation)
            smoothed_pose, _vel, finished = cse.tick()

            # Solve IK for the smoothed Cartesian pose
            ik_solution = solve_ik(PAROL6_ROBOT.robot, smoothed_pose, self._q_rad_buf)
            if not ik_solution.success or ik_solution.q is None:
                if not self._ik_stopping:
                    logger.warning(
                        f"[MOVECART] IK failed - initiating graceful stop: "
                        f"pos={smoothed_pose[:3, 3]}"
                    )
                    cse.stop()
                    self._ik_stopping = True
                else:
                    # Still failing, check if we've stopped decelerating
                    if np.dot(_vel, _vel) < 1e-8:
                        # Sync CSE to actual robot pose now that we've stopped
                        cse.sync_pose(get_fkine_se3(state))
                        self.is_finished = True
                        return ExecutionStatus.completed(
                            f"{self.__class__.__name__}: IK limit reached"
                        )
                return ExecutionStatus.executing(f"{self.__class__.__name__} stopping")

            # IK succeeded - if we were stopping, recover by resuming motion
            if self._ik_stopping:
                logger.info("[MOVECART] IK recovered - resuming motion")
                assert self.target_pose is not None
                cse.set_pose_target(self.target_pose)
                self._ik_stopping = False

            # Send joint position to robot
            rad_to_steps(ik_solution.q, self._steps_buf)
            self.set_move_position(state, self._steps_buf)

            if finished:
                self.log_info("%s (streaming) finished.", self.__class__.__name__)
                # Deactivate executor so next command properly syncs pose
                cse.active = False
                self.is_finished = True
                return ExecutionStatus.completed(f"{self.__class__.__name__} complete")

            return ExecutionStatus.executing(self.__class__.__name__)

        # Non-streaming: use inherited trajectory execution
        return super().execute_step(state)


@register_command(CmdType.CARTJOG)
class CartesianJogCommand(MotionCommand):
    """
    A non-blocking command to jog the robot's end-effector in Cartesian space.
    """

    PARAMS_TYPE = CartJogCmd
    streamable = True

    __slots__ = (
        "axis_vectors",
        "is_rotation",
        "_ik_stopping",
        "_jog_initialized",
        "_axis_index",
        "_axis_sign",
        # Pre-allocated buffers (allocated once in __init__, reused across streaming)
        "_world_twist_buf",
        "_vel_lin_buf",
        "_vel_ang_buf",
        "_delta_se3_buf",
        "_target_pose_buf",
        "_omega_ws",
        "_R_ws",
        "_V_ws",
    )

    # Class-level rate limiting for IK warnings (shared across instances)
    _last_ik_warn_time: float = 0.0
    _IK_WARN_INTERVAL: float = 1.0  # Log at most once per second

    def __init__(self):
        super().__init__()
        self.axis_vectors = None
        self.is_rotation = False
        self._ik_stopping = False
        self._jog_initialized = False
        self._axis_index = 0
        self._axis_sign = 1.0

        self._world_twist_buf = np.zeros(6, dtype=np.float64)
        self._vel_lin_buf = np.zeros(3, dtype=np.float64)
        self._vel_ang_buf = np.zeros(3, dtype=np.float64)
        self._delta_se3_buf = np.zeros((4, 4), dtype=np.float64)
        self._target_pose_buf = np.zeros((4, 4), dtype=np.float64)
        self._omega_ws = np.zeros(3, dtype=np.float64)
        self._R_ws = np.zeros((3, 3), dtype=np.float64)
        self._V_ws = np.zeros((3, 3), dtype=np.float64)

    def do_setup(self, state: "ControllerState") -> None:
        """Set the end time when the command actually starts."""
        assert self.p is not None

        # Axis is validated by struct pattern constraint, but we still need to look it up
        axis_key = self.p.axis
        self.axis_vectors = AXIS_MAP[axis_key]
        self.is_rotation = any(self.axis_vectors[1])

        self.start_timer(self.p.duration)
        self._jog_initialized = False
        self._ik_stopping = False

        if self.is_rotation:
            vec = self.axis_vectors[1]
        else:
            vec = self.axis_vectors[0]

        for i, v in enumerate(vec):
            if v != 0:
                self._axis_index = i
                self._axis_sign = float(np.sign(v))
                break

    def _compute_target_pose_from_velocity(
        self, state: "ControllerState", smoothed_vel: np.ndarray
    ) -> None:
        """Compute target pose from smoothed velocity."""
        assert self.p is not None
        cse = state.cartesian_streaming_executor
        assert cse is not None
        current_pose = get_fkine_se3(state)

        if self.p.frame == "WRF":
            # WRF: transform velocity to world frame and left-multiply
            assert cse.reference_pose is not None
            R = cse.reference_pose[:3, :3]
            _apply_velocity_delta_wrf_jit(
                R,
                smoothed_vel,
                cse.dt,
                current_pose,
                self._vel_lin_buf,
                self._vel_ang_buf,
                self._world_twist_buf,
                self._delta_se3_buf,
                self._target_pose_buf,
                self._omega_ws,
                self._R_ws,
                self._V_ws,
            )
        else:
            # TRF: use body-frame velocity directly and right-multiply
            _apply_velocity_delta_trf_jit(
                smoothed_vel,
                cse.dt,
                current_pose,
                self._world_twist_buf,
                self._delta_se3_buf,
                self._target_pose_buf,
                self._omega_ws,
                self._R_ws,
                self._V_ws,
            )

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Execute one tick of Cartesian jogging."""
        assert self.p is not None

        cse = state.cartesian_streaming_executor
        if cse is None:
            logger.warning("[CARTJOG] No cartesian_streaming_executor available")
            self.is_finished = True
            return ExecutionStatus.completed("CARTJOG: no executor")

        steps_to_rad(state.Position_in, self._q_rad_buf)

        # Initialize only if not already active (preserve velocity across streaming)
        if not cse.active:
            cse.sync_pose(get_fkine_se3(state))
            cse.set_limits(100.0, self.p.accel_pct)

        # Handle timer expiry - stop smoothly
        if self.timer_expired():
            cse.set_jog_velocity_1dof(self._axis_index, 0.0, self.is_rotation)
            _smoothed_pose, smoothed_vel, finished = cse.tick()

            if not finished and np.dot(smoothed_vel, smoothed_vel) > 1e-8:
                self._compute_target_pose_from_velocity(state, smoothed_vel)
                ik_result = solve_ik(
                    PAROL6_ROBOT.robot, self._target_pose_buf, self._q_rad_buf
                )
                if ik_result.success and ik_result.q is not None:
                    rad_to_steps(ik_result.q, self._steps_buf)
                    self.set_move_position(state, self._steps_buf)
                return ExecutionStatus.executing("CARTJOG (stopping)")

            self.is_finished = True
            self.stop_and_idle(state)
            return ExecutionStatus.completed("CARTJOG complete")

        # Compute target velocity based on speed percentage
        if self.is_rotation:
            cart_ang_max = np.rad2deg(LIMITS.cart.jog.velocity.angular)
            vel_deg_s = self.linmap_pct(
                self.p.speed_pct, CART_ANG_JOG_MIN, cart_ang_max
            )
            velocity = np.radians(vel_deg_s) * self._axis_sign
        else:
            cart_lin_max = LIMITS.cart.jog.velocity.linear * 1000
            vel_mm_s = self.linmap_pct(self.p.speed_pct, CART_LIN_JOG_MIN, cart_lin_max)
            velocity = (vel_mm_s / 1000.0) * self._axis_sign

        # Set target velocity (WRF transforms to body frame, TRF uses body directly)
        if self.p.frame == "WRF":
            cse.set_jog_velocity_1dof_wrf(self._axis_index, velocity, self.is_rotation)
        else:
            cse.set_jog_velocity_1dof(self._axis_index, velocity, self.is_rotation)

        _smoothed_pose, smoothed_vel, _finished = cse.tick()
        self._compute_target_pose_from_velocity(state, smoothed_vel)

        ik_result = solve_ik(PAROL6_ROBOT.robot, self._target_pose_buf, self._q_rad_buf)
        if not ik_result.success or ik_result.q is None:
            if not self._ik_stopping:
                now = time.monotonic()
                if (
                    now - CartesianJogCommand._last_ik_warn_time
                    > CartesianJogCommand._IK_WARN_INTERVAL
                ):
                    logger.warning(
                        f"[CARTJOG] IK failed - initiating graceful stop: pos={self._target_pose_buf[:3, 3]}"
                    )
                    CartesianJogCommand._last_ik_warn_time = now
                cse.stop()
                self._ik_stopping = True
            else:
                # Still failing, check if we've stopped decelerating
                if np.dot(smoothed_vel, smoothed_vel) < 1e-8:
                    # Sync CSE to actual robot pose now that we've stopped
                    # This allows recovery by jogging in a different direction
                    cse.sync_pose(get_fkine_se3(state))
                    self.is_finished = True
                    return ExecutionStatus.completed("CARTJOG: IK limit reached")
            return ExecutionStatus.executing("CARTJOG (IK stop)")

        # IK succeeded - if we were stopping, recover by resuming jogging
        if self._ik_stopping:
            logger.info("[CARTJOG] IK recovered - resuming jog")
            # Sync to actual robot pose before resuming (CSE drifted during stop)
            cse.sync_pose(get_fkine_se3(state))
            self._ik_stopping = False
            # Re-apply the jog velocity to resume motion
            if self.p.frame == "WRF":
                cse.set_jog_velocity_1dof_wrf(
                    self._axis_index, velocity, self.is_rotation
                )
            else:
                cse.set_jog_velocity_1dof(self._axis_index, velocity, self.is_rotation)

        rad_to_steps(ik_result.q, self._steps_buf)
        self.set_move_position(state, self._steps_buf)

        return ExecutionStatus.executing("CARTJOG")


@register_command(CmdType.MOVECART)
class MoveCartCommand(CartesianMoveCommandBase):
    """Move the robot's end-effector in a straight line to an absolute Cartesian pose."""

    PARAMS_TYPE = MoveCartCmd

    __slots__ = ()

    def __init__(self):
        super().__init__()

    def _compute_target_pose(self, state: "ControllerState") -> None:
        """Compute absolute target pose from parsed coordinates."""
        assert self.p is not None
        pose = self.p.pose
        self.target_pose = np.zeros((4, 4), dtype=np.float64)
        se3_from_rpy(
            pose[0] / 1000.0,
            pose[1] / 1000.0,
            pose[2] / 1000.0,
            np.radians(pose[3]),
            np.radians(pose[4]),
            np.radians(pose[5]),
            self.target_pose,
        )


@register_command(CmdType.MOVECARTRELTRF)
class MoveCartRelTrfCommand(CartesianMoveCommandBase):
    """Move the robot's end-effector relative to current position in Tool Reference Frame."""

    PARAMS_TYPE = MoveCartRelTrfCmd

    __slots__ = ()

    def __init__(self):
        super().__init__()

    def _compute_target_pose(self, state: "ControllerState") -> None:
        """Compute target pose from current pose + TRF delta."""
        assert self.p is not None
        deltas = self.p.deltas
        delta_se3 = np.zeros((4, 4), dtype=np.float64)
        se3_from_rpy(
            deltas[0] / 1000.0,
            deltas[1] / 1000.0,
            deltas[2] / 1000.0,
            np.radians(deltas[3]),
            np.radians(deltas[4]),
            np.radians(deltas[5]),
            delta_se3,
        )
        # Post-multiply for tool-relative motion
        self.target_pose = cast(np.ndarray, self.initial_pose) @ delta_se3
