"""
Cartesian Movement Commands
Contains commands for Cartesian space movements: CartesianJog, MovePose, MoveCart, MoveCartRelTrf
"""

import logging
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, cast

import numpy as np
import sophuspy as sp

if TYPE_CHECKING:
    from numpy.typing import NDArray

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.config import (
    CART_ANG_JOG_MIN,
    CART_LIN_JOG_MIN,
    DEFAULT_ACCEL_PERCENT,
    INTERVAL_S,
    LIMITS,
    PATH_SAMPLES,
    rad_to_steps,
    steps_to_rad,
)
from parol6.motion import JointPath, TrajectoryBuilder
from parol6.server.command_registry import register_command
from parol6.server.state import ControllerState, get_fkine_se3
from parol6.utils.ik import AXIS_MAP, solve_ik
from parol6.utils.se3_utils import (
    se3_from_rpy,
    se3_interp,
)

from .base import (
    ExecutionStatus,
    MotionCommand,
    TrajectoryMoveCommandBase,
    parse_opt_float,
)

logger = logging.getLogger(__name__)


class CartesianMoveCommandBase(TrajectoryMoveCommandBase):
    """Base class for Cartesian move commands with straight-line path following.

    Subclasses must implement:
    - do_match(): Parse command parameters
    - _compute_target_pose(): Set self.target_pose from parsed parameters

    Supports both streaming mode (real-time IK each tick) and pre-computed trajectories.
    """

    streamable = True

    __slots__ = (
        "duration",
        "velocity_percent",
        "accel_percent",
        "initial_pose",
        "target_pose",
        "_ik_stopping",
    )

    def __init__(self):
        super().__init__()
        self.duration: float | None = None
        self.velocity_percent: float | None = None
        self.accel_percent: float = DEFAULT_ACCEL_PERCENT
        self.initial_pose: sp.SE3 | None = None
        self.target_pose: sp.SE3 | None = None

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

        current_rad = np.asarray(steps_to_rad(state.Position_in), dtype=np.float64)

        profile_str = state.cartesian_motion_profile
        vel_pct = self.velocity_percent if self.velocity_percent is not None else 100.0
        acc_pct = self.accel_percent if self.accel_percent is not None else 100.0

        cart_poses = []
        for i in range(PATH_SAMPLES):
            s = i / (PATH_SAMPLES - 1)
            cart_poses.append(se3_interp(self.initial_pose, self.target_pose, s))

        joint_path = JointPath.from_poses(cart_poses, current_rad, quiet_logging=True)

        # SI units (m/s, m/s²) - trajectory builder uses SI directly
        cart_vel_max = LIMITS.cart.hard.velocity.linear * (vel_pct / 100.0)
        cart_acc_max = LIMITS.cart.hard.acceleration.linear * (acc_pct / 100.0)

        builder = TrajectoryBuilder(
            joint_path=joint_path,
            profile=profile_str,
            velocity_percent=vel_pct,
            accel_percent=acc_pct,
            duration=self.duration,
            dt=INTERVAL_S,
            cart_vel_limit=cart_vel_max,
            cart_acc_limit=cart_acc_max,
        )

        trajectory = builder.build()
        self.trajectory_steps = trajectory.steps
        self.duration = trajectory.duration

        self.log_debug(
            "  -> Pre-computed Cartesian path: profile=%s, steps=%d, duration=%.3fs",
            profile_str,
            len(self.trajectory_steps),
            float(self.duration),
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
            current_q_rad = cast("NDArray[np.float64]", steps_to_rad(state.Position_in))

            # Initialize on first tick, or if executor not active (streaming interrupted)
            if not self._streaming_initialized or not cse.active:
                # Only sync pose if not already active to preserve velocity
                if not cse.active:
                    cse.sync_pose(self.initial_pose)
                vel_pct = (
                    self.velocity_percent
                    if self.velocity_percent is not None
                    else 100.0
                )
                acc_pct = (
                    self.accel_percent if self.accel_percent is not None else 100.0
                )
                cse.set_limits(vel_pct, acc_pct)
                cse.set_pose_target(self.target_pose)
                self._streaming_initialized = True

            # Get smoothed pose from Cartesian executor (straight-line interpolation)
            smoothed_pose, _vel, finished = cse.tick()

            # Solve IK for the smoothed Cartesian pose
            ik_solution = solve_ik(PAROL6_ROBOT.robot, smoothed_pose, current_q_rad)
            if not ik_solution.success or ik_solution.q is None:
                if not self._ik_stopping:
                    logger.warning(
                        f"[MOVECART] IK failed - initiating graceful stop: "
                        f"pos={smoothed_pose.translation()}"
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
                cse.set_pose_target(self.target_pose)
                self._ik_stopping = False

            # Send joint position to robot
            steps = cast("NDArray[np.int32]", rad_to_steps(ik_solution.q))
            self.set_move_position(state, steps)

            if finished:
                self.log_info("%s (streaming) finished.", self.__class__.__name__)
                self.is_finished = True
                return ExecutionStatus.completed(f"{self.__class__.__name__} complete")

            return ExecutionStatus.executing(self.__class__.__name__)

        # Non-streaming: use inherited trajectory execution
        return super().execute_step(state)


@register_command("CARTJOG")
class CartesianJogCommand(MotionCommand):
    """
    A non-blocking command to jog the robot's end-effector in Cartesian space.
    """

    streamable = True

    __slots__ = (
        "frame",
        "axis",
        "speed_percentage",
        "accel_percent",
        "duration",
        "axis_vectors",
        "is_rotation",
        "_ik_stopping",
        "_world_twist_buf",
        "_vel_lin_buf",
        "_vel_ang_buf",
    )

    # Class-level rate limiting for IK warnings (shared across instances)
    _last_ik_warn_time: float = 0.0
    _IK_WARN_INTERVAL: float = 1.0  # Log at most once per second

    def __init__(self):
        super().__init__()
        self.frame = None
        self.axis = None
        self.speed_percentage: float = 50.0
        self.accel_percent: float = 100.0
        self.duration: float = 1.5
        self.axis_vectors = None
        self.is_rotation = False

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
        """
        Parse CARTJOG command parameters.

        Format: CARTJOG|frame|axis|speed_pct|duration[|accel_pct]
        Example: CARTJOG|WRF|+X|50|2.0 or CARTJOG|WRF|+X|50|2.0|80
        """
        if len(parts) < 5 or len(parts) > 6:
            return (
                False,
                "CARTJOG requires 4-5 parameters: frame, axis, speed, duration[, accel]",
            )

        self.frame = parts[1].upper()
        self.axis = parts[2]
        self.speed_percentage = float(parts[3])
        self.duration = float(parts[4])
        if len(parts) == 6:
            self.accel_percent = float(parts[5])

        if self.frame not in ["WRF", "TRF"]:
            return (False, f"Invalid frame: {self.frame}. Must be WRF or TRF")

        if self.axis not in AXIS_MAP:
            return (False, f"Invalid axis: {self.axis}")

        self.axis_vectors = AXIS_MAP[self.axis]
        self.is_rotation = any(self.axis_vectors[1])

        self.is_valid = True
        return (True, None)

    def do_setup(self, state: "ControllerState") -> None:
        """Set the end time when the command actually starts."""
        self.start_timer(float(self.duration))
        self._jog_initialized = (
            False  # Track whether cartesian executor has been synced
        )
        self._ik_stopping = False  # Track graceful stop on IK failure

        # Parse axis index and sign from axis_vectors
        # axis_vectors is ([x,y,z], [rx,ry,rz]) where exactly one component is ±1
        if self.is_rotation:
            vec = self.axis_vectors[1]  # Rotation vector
        else:
            vec = self.axis_vectors[0]  # Linear vector

        # Find which axis (0=X, 1=Y, 2=Z)
        self._axis_index = 0
        self._axis_sign = 1.0
        for i, v in enumerate(vec):
            if v != 0:
                self._axis_index = i
                self._axis_sign = float(np.sign(v))
                break

        # Pre-allocate buffers for hot path (avoids allocations per tick)
        self._world_twist_buf = np.zeros(6, dtype=np.float64)
        self._vel_lin_buf = np.zeros(3, dtype=np.float64)
        self._vel_ang_buf = np.zeros(3, dtype=np.float64)

    def _apply_smoothed_velocity(
        self, state: "ControllerState", smoothed_vel: np.ndarray
    ) -> sp.SE3:
        """Apply smoothed velocity to actual current pose.

        Converts body-frame velocity to world-frame and applies as delta.
        Returns the target pose for IK solving.
        """
        cse = state.cartesian_streaming_executor
        assert cse is not None
        current_pose = get_fkine_se3(state)

        # WRF: use reference_pose rotation (velocity was transformed TO body frame using it)
        # TRF: use current_pose rotation (velocity is in tool frame)
        if self.frame == "WRF":
            assert cse.reference_pose is not None
            R = cse.reference_pose.rotationMatrix()
        else:
            R = current_pose.rotationMatrix()

        np.dot(R, smoothed_vel[:3], out=self._vel_lin_buf)
        np.dot(R, smoothed_vel[3:], out=self._vel_ang_buf)

        # World-frame delta requires LEFT multiplication (reuse pre-allocated buffer)
        self._world_twist_buf[:3] = self._vel_lin_buf
        self._world_twist_buf[3:] = self._vel_ang_buf
        self._world_twist_buf *= cse.dt
        delta_se3 = sp.SE3.exp(self._world_twist_buf)
        return delta_se3 * current_pose

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Execute one tick of Cartesian jogging using Cartesian-space Ruckig.

        Unlike JogCommand which has a velocity-based fallback when streaming
        is unavailable, CartesianJogCommand requires CartesianStreamingExecutor
        because:
        1. Cartesian jogging needs smooth interpolation in SE3 space
        2. Each tick requires IK solve (Cartesian pose → joint positions)
        3. No direct "Cartesian velocity" command exists at the motor level
        """
        cse = state.cartesian_streaming_executor
        if cse is None:
            logger.warning("[CARTJOG] No cartesian_streaming_executor available")
            self.is_finished = True
            return ExecutionStatus.completed("CARTJOG: no executor")

        q_current = cast("NDArray[np.float64]", steps_to_rad(state.Position_in))

        # Initialize only if not already active (preserve velocity across streaming)
        if not cse.active:
            cse.sync_pose(get_fkine_se3(state))
            cse.set_limits(100.0, self.accel_percent)

        # Handle timer expiry - stop smoothly
        if self.timer_expired():
            cse.set_jog_velocity_1dof(self._axis_index, 0.0, self.is_rotation)
            _smoothed_pose, smoothed_vel, finished = cse.tick()

            if not finished and np.dot(smoothed_vel, smoothed_vel) > 1e-8:
                target_pose = self._apply_smoothed_velocity(state, smoothed_vel)
                ik_result = solve_ik(PAROL6_ROBOT.robot, target_pose, q_current)
                if ik_result.success and ik_result.q is not None:
                    steps = cast("NDArray[np.int32]", rad_to_steps(ik_result.q))
                    self.set_move_position(state, steps)
                return ExecutionStatus.executing("CARTJOG (stopping)")

            self.is_finished = True
            self.stop_and_idle(state)
            return ExecutionStatus.completed("CARTJOG complete")

        # Compute target velocity based on speed percentage
        if self.is_rotation:
            cart_ang_max = np.rad2deg(LIMITS.cart.jog.velocity.angular)
            vel_deg_s = self.linmap_pct(
                self.speed_percentage, CART_ANG_JOG_MIN, cart_ang_max
            )
            velocity = np.radians(vel_deg_s) * self._axis_sign
        else:
            cart_lin_max = LIMITS.cart.jog.velocity.linear * 1000  # m/s → mm/s
            vel_mm_s = self.linmap_pct(
                self.speed_percentage, CART_LIN_JOG_MIN, cart_lin_max
            )
            velocity = (vel_mm_s / 1000.0) * self._axis_sign

        # Set target velocity (WRF transforms to body frame, TRF uses body directly)
        if self.frame == "WRF":
            cse.set_jog_velocity_1dof_wrf(self._axis_index, velocity, self.is_rotation)
        else:
            cse.set_jog_velocity_1dof(self._axis_index, velocity, self.is_rotation)

        _smoothed_pose, smoothed_vel, _finished = cse.tick()
        target_pose = self._apply_smoothed_velocity(state, smoothed_vel)

        ik_result = solve_ik(PAROL6_ROBOT.robot, target_pose, q_current)
        if not ik_result.success or ik_result.q is None:
            if not self._ik_stopping:
                now = time.monotonic()
                if (
                    now - CartesianJogCommand._last_ik_warn_time
                    > CartesianJogCommand._IK_WARN_INTERVAL
                ):
                    logger.warning(
                        f"[CARTJOG] IK failed - initiating graceful stop: pos={target_pose.translation()}"
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
            self._ik_stopping = False
            # Re-apply the jog velocity to resume motion
            if self.frame == "WRF":
                cse.set_jog_velocity_1dof_wrf(
                    self._axis_index, velocity, self.is_rotation
                )
            else:
                cse.set_jog_velocity_1dof(self._axis_index, velocity, self.is_rotation)

        steps = cast("NDArray[np.int32]", rad_to_steps(ik_result.q))
        self.set_move_position(state, steps)

        return ExecutionStatus.executing("CARTJOG")


@register_command("MOVECART")
class MoveCartCommand(CartesianMoveCommandBase):
    """Move the robot's end-effector in a straight line to an absolute Cartesian pose."""

    __slots__ = ("pose",)

    def __init__(self):
        super().__init__()
        self.pose: list[float] | None = None

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
        """
        Parse MOVECART command parameters.

        Format: MOVECART|x|y|z|rx|ry|rz|duration|speed|accel
        Example: MOVECART|100|200|300|0|0|0|2.0|None|50
        """
        if len(parts) != 10:
            return (
                False,
                "MOVECART requires 9 parameters: x, y, z, rx, ry, rz, duration, speed, accel",
            )

        self.pose = [float(parts[i]) for i in range(1, 7)]
        self.duration = parse_opt_float(parts[7])
        self.velocity_percent = parse_opt_float(parts[8])
        self.accel_percent = parse_opt_float(parts[9], DEFAULT_ACCEL_PERCENT)

        if self.duration is None and self.velocity_percent is None:
            return (False, "MOVECART requires either duration or velocity_percent")

        if self.duration is not None and self.velocity_percent is not None:
            logger.info(
                "  -> INFO: Both duration and velocity_percent provided. Using duration."
            )
            self.velocity_percent = None

        self.log_debug("Parsed MoveCart: %s, accel=%s%%", self.pose, self.accel_percent)
        self.is_valid = True
        return (True, None)

    def _compute_target_pose(self, state: "ControllerState") -> None:
        """Compute absolute target pose from parsed coordinates."""
        pose = cast(list[float], self.pose)
        self.target_pose = se3_from_rpy(
            pose[0] / 1000.0,
            pose[1] / 1000.0,
            pose[2] / 1000.0,
            pose[3],
            pose[4],
            pose[5],
            degrees=True,
        )


@register_command("MOVECARTRELTRF")
class MoveCartRelTrfCommand(CartesianMoveCommandBase):
    """Move the robot's end-effector relative to current position in Tool Reference Frame."""

    __slots__ = ("deltas",)

    def __init__(self):
        super().__init__()
        self.deltas: list[float] | None = None

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
        """
        Parse MOVECARTRELTRF command parameters.

        Format: MOVECARTRELTRF|dx|dy|dz|rx|ry|rz|duration|speed|accel
        Example: MOVECARTRELTRF|10|0|0|0|0|0|NONE|50|100
        """
        if len(parts) != 10:
            return (
                False,
                "MOVECARTRELTRF requires 9 parameters: dx, dy, dz, rx, ry, rz, duration, speed, accel",
            )

        self.deltas = [float(parts[i]) for i in range(1, 7)]
        self.duration = parse_opt_float(parts[7])
        self.velocity_percent = parse_opt_float(parts[8])
        self.accel_percent = parse_opt_float(parts[9], DEFAULT_ACCEL_PERCENT)

        if self.duration is None and self.velocity_percent is None:
            return (
                False,
                "MOVECARTRELTRF requires either duration or velocity_percent",
            )

        if self.duration is not None and self.velocity_percent is not None:
            logger.info(
                "  -> INFO: Both duration and velocity_percent provided. Using duration."
            )
            self.velocity_percent = None

        self.log_debug(
            "Parsed MoveCartRelTrf: deltas=%s, accel=%s%%",
            self.deltas,
            self.accel_percent,
        )
        self.is_valid = True
        return (True, None)

    def _compute_target_pose(self, state: "ControllerState") -> None:
        """Compute target pose from current pose + TRF delta."""
        deltas = cast(list[float], self.deltas)
        delta_se3 = se3_from_rpy(
            deltas[0] / 1000.0,
            deltas[1] / 1000.0,
            deltas[2] / 1000.0,
            deltas[3],
            deltas[4],
            deltas[5],
            degrees=True,
        )
        # Post-multiply for tool-relative motion
        self.target_pose = cast(sp.SE3, self.initial_pose) * delta_se3
