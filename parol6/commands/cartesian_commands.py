"""
Cartesian Movement Commands
Contains commands for Cartesian space movements: CartesianJog, MovePose, and MoveCart
"""

import logging
import time
from typing import cast

import numpy as np
import sophuspy as sp

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.config import DEFAULT_ACCEL_PERCENT, INTERVAL_S, TRACE, TRACE_ENABLED
from parol6.protocol.wire import CommandCode
from parol6.server.command_registry import register_command
from parol6.server.state import ControllerState, get_fkine_se3
from parol6.utils.errors import IKError
from parol6.utils.ik import AXIS_MAP, fast_quintic_scaling, quintic_scaling, solve_ik
from parol6.utils.se3_utils import (
    se3_angdist,
    se3_from_rpy,
    se3_from_trans,
    se3_interp,
    se3_rx,
    se3_ry,
    se3_rz,
)

from .base import ExecutionStatus, MotionCommand, MotionProfile

logger = logging.getLogger(__name__)


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
        "duration",
        "axis_vectors",
        "is_rotation",
    )

    def __init__(self):
        """
        Initializes the Cartesian jog command.
        Parameters are parsed in do_match() method.
        """
        super().__init__()

        # Parameters (set in do_match())
        self.frame = None
        self.axis = None
        self.speed_percentage: float = 50.0
        self.duration: float = 1.5

        # Runtime state
        self.axis_vectors = None
        self.is_rotation = False

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
        """
        Parse CARTJOG command parameters.

        Format: CARTJOG|frame|axis|speed_pct|duration
        Example: CARTJOG|WRF|+X|50|2.0

        Args:
            parts: Pre-split message parts

        Returns:
            Tuple of (can_handle, error_message)
        """
        if len(parts) != 5:
            return (
                False,
                "CARTJOG requires 4 parameters: frame, axis, speed, duration",
            )

        # Parse parameters
        self.frame = parts[1].upper()
        self.axis = parts[2]
        self.speed_percentage = float(parts[3])
        self.duration = float(parts[4])

        # Validate frame
        if self.frame not in ["WRF", "TRF"]:
            return (False, f"Invalid frame: {self.frame}. Must be WRF or TRF")

        # Validate axis
        if self.axis not in AXIS_MAP:
            return (False, f"Invalid axis: {self.axis}")

        # Store axis vectors for execution
        self.axis_vectors = AXIS_MAP[self.axis]
        self.is_rotation = any(self.axis_vectors[1])

        self.is_valid = True
        return (True, None)

    def do_setup(self, state: "ControllerState") -> None:
        """Set the end time when the command actually starts."""
        self.start_timer(float(self.duration))

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        # --- A. Check for completion ---
        if self._t_end is None:
            # Initialize timer if missing (stream update or late init)
            self.start_timer(
                max(0.1, self.duration if self.duration is not None else 0.1)
            )
        if self.timer_expired():
            self.is_finished = True
            self.stop_and_idle(state)
            return ExecutionStatus.completed("CARTJOG complete")

        # --- B. Calculate Target Pose using clean vector math ---
        state.Command_out = CommandCode.JOG

        q_current = np.asarray(
            PAROL6_ROBOT.ops.steps_to_rad(state.Position_in), dtype=float
        )
        T_current = get_fkine_se3()

        if T_current is None:
            return ExecutionStatus.executing("Waiting for valid pose")
        if self.axis_vectors is None:
            return ExecutionStatus.executing("Waiting for axis vectors")

        linear_speed_ms = self.linmap_pct(
            self.speed_percentage, self.CART_LIN_JOG_MIN, self.CART_LIN_JOG_MAX
        )
        angular_speed_degs = self.linmap_pct(
            self.speed_percentage, self.CART_ANG_JOG_MIN, self.CART_ANG_JOG_MAX
        )

        delta_linear = linear_speed_ms * INTERVAL_S
        delta_angular_rad = np.deg2rad(angular_speed_degs * INTERVAL_S)

        # Create the small incremental transformation (delta_pose)
        trans_vec = np.array(self.axis_vectors[0]) * delta_linear
        rot_vec = np.array(self.axis_vectors[1]) * delta_angular_rad

        # Build delta transformation
        R_current = T_current.rotationMatrix()
        t_current = T_current.translation()

        if not self.is_rotation:
            if self.frame == "WRF":
                new_t = t_current + trans_vec
            else:  # TRF
                new_t = t_current + (R_current @ trans_vec)
            target_pose = sp.SE3(R_current, new_t)
        else:
            if rot_vec[0] != 0:  # RX rotation
                delta_pose = se3_rx(rot_vec[0]) * se3_from_trans(*trans_vec)
            elif rot_vec[1] != 0:  # RY rotation
                delta_pose = se3_ry(rot_vec[1]) * se3_from_trans(*trans_vec)
            elif rot_vec[2] != 0:  # RZ rotation
                delta_pose = se3_rz(rot_vec[2]) * se3_from_trans(*trans_vec)
            else:
                delta_pose = se3_from_trans(*trans_vec)
            # Apply the transformation in the correct reference frame
            if self.frame == "WRF":
                # Pre-multiply to apply the change in the World Reference Frame
                target_pose = delta_pose * T_current
            else:  # TRF
                # Post-multiply to apply the change in the Tool Reference Frame
                target_pose = T_current * delta_pose

        # --- C. Solve IK and Calculate Velocities ---
        var = solve_ik(PAROL6_ROBOT.robot, target_pose, q_current, jogging=True)

        if var.success and var.q is not None:
            q = np.asarray(var.q, dtype=float)
            q_velocities = (q - q_current) / INTERVAL_S
            sps = PAROL6_ROBOT.ops.speed_rad_to_steps(q_velocities)
            np.copyto(state.Speed_out, np.asarray(sps), casting="no")
        else:
            raise IKError("IK Warning: Could not find solution for jog step. Stopping.")

        # --- D. Speed Scaling using base class helper ---
        scaled_speeds = self.scale_speeds_to_joint_max(state.Speed_out)
        np.copyto(state.Speed_out, scaled_speeds)

        return ExecutionStatus.executing("CARTJOG")


@register_command("MOVEPOSE")
class MovePoseCommand(MotionCommand):
    """
    A non-blocking command to move the robot to a specific Cartesian pose.
    The movement itself is a joint-space interpolation.
    """

    __slots__ = (
        "command_step",
        "trajectory_steps",
        "pose",
        "duration",
        "velocity_percent",
        "accel_percent",
        "trajectory_type",
    )

    def __init__(self, pose=None, duration=None):
        super().__init__()
        self.command_step = 0
        self.trajectory_steps: np.ndarray = np.empty((0, 6), dtype=np.int32)

        # Parameters (set in do_match())
        self.pose = pose
        self.duration = duration
        self.velocity_percent = None
        self.accel_percent = DEFAULT_ACCEL_PERCENT
        self.trajectory_type = "trapezoid"

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
        """
        Parse MOVEPOSE command parameters.

        Format: MOVEPOSE|x|y|z|rx|ry|rz|duration|speed|accel|
        Example: MOVEPOSE|100|200|300|0|0|0|None|50|50

        Args:
            parts: Pre-split message parts

        Returns:
            Tuple of (can_handle, error_message)
        """
        if len(parts) != 10:
            return (
                False,
                "MOVEPOSE requires 9 parameters: x, y, z, rx, ry, rz, duration, speed, accel",
            )

        # Parse pose (6 values)
        self.pose = [float(parts[i]) for i in range(1, 7)]

        # Parse duration and speed
        self.duration = None if parts[7].upper() == "NONE" else float(parts[7])
        self.velocity_percent = None if parts[8].upper() == "NONE" else float(parts[8])

        # Parse acceleration
        self.accel_percent = float(parts[9]) if parts[9].upper() != "NONE" else DEFAULT_ACCEL_PERCENT

        self.log_debug("Parsed MovePose: %s, accel=%s%%", self.pose, self.accel_percent)
        self.is_valid = True
        return (True, None)

    def do_setup(self, state: "ControllerState") -> None:
        """Calculates the full trajectory just-in-time before execution."""
        self.log_trace("  -> Preparing trajectory for MovePose to %s...", self.pose)

        initial_pos_rad = np.asarray(
            PAROL6_ROBOT.ops.steps_to_rad(state.Position_in), dtype=float
        )
        pose = cast(list[float], self.pose)
        # Position in mm, angles in degrees
        target_pose = se3_from_rpy(
            pose[0] / 1000.0,
            pose[1] / 1000.0,
            pose[2] / 1000.0,
            pose[3],
            pose[4],
            pose[5],
            degrees=True,
        )

        ik_solution = solve_ik(PAROL6_ROBOT.robot, target_pose, initial_pos_rad)

        if not ik_solution.success:
            error_str = "An intermediate point on the path is unreachable."
            if ik_solution.violations:
                error_str += (
                    f" Reason: Path violates joint limits: {ik_solution.violations}"
                )
            raise IKError(error_str)

        target_pos_rad = ik_solution.q

        if self.duration and self.duration > 0:
            if self.velocity_percent is not None:
                self.log_trace(
                    "  -> INFO: Both duration and velocity were provided. Using duration."
                )
            initial_pos_steps = state.Position_in
            target_pos_steps = np.asarray(
                PAROL6_ROBOT.ops.rad_to_steps(target_pos_rad), dtype=np.int32
            )
            dur = float(self.duration)
            self.trajectory_steps = MotionProfile.from_duration_steps(
                initial_pos_steps, target_pos_steps, dur, dt=INTERVAL_S
            )

        elif self.velocity_percent is not None:
            initial_pos_steps = state.Position_in
            target_pos_steps = np.asarray(
                PAROL6_ROBOT.ops.rad_to_steps(target_pos_rad), dtype=np.int32
            )
            accel_percent = (
                float(self.accel_percent)
                if self.accel_percent is not None
                else float(DEFAULT_ACCEL_PERCENT)
            )
            self.trajectory_steps = MotionProfile.from_velocity_percent(
                initial_pos_steps,
                target_pos_steps,
                float(self.velocity_percent),
                accel_percent,
                dt=INTERVAL_S,
            )
            self.log_trace("  -> Command is valid (velocity profile).")
        else:
            self.log_trace("  -> Using conservative values for MovePose.")
            command_len = 200
            initial_pos_steps = state.Position_in
            target_pos_steps = np.asarray(
                PAROL6_ROBOT.ops.rad_to_steps(target_pos_rad), dtype=np.int32
            )
            total_dur = float(command_len) * INTERVAL_S
            self.trajectory_steps = MotionProfile.from_duration_steps(
                initial_pos_steps, target_pos_steps, total_dur, dt=INTERVAL_S
            )

        if len(self.trajectory_steps) == 0:
            raise IKError(
                "Trajectory calculation resulted in no steps. Command is invalid."
            )
        logger.log(
            TRACE, " -> Trajectory prepared with %s steps.", len(self.trajectory_steps)
        )

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        if self.command_step >= len(self.trajectory_steps):
            logger.info(f"{type(self).__name__} finished.")
            self.is_finished = True
            self.stop_and_idle(state)
            return ExecutionStatus.completed("MOVEPOSE complete")
        else:
            self.set_move_position(state, self.trajectory_steps[self.command_step])
            self.command_step += 1
            return ExecutionStatus.executing("MovePose")


@register_command("MOVECART")
class MoveCartCommand(MotionCommand):
    """
    A non-blocking command to move the robot's end-effector in a straight line
    in Cartesian space, completing the move in an exact duration.

    It works by:
    1. Pre-validating the final target pose.
    2. Interpolating the pose in Cartesian space in real-time.
    3. Solving Inverse Kinematics for each intermediate step to ensure path validity.
    """

    streamable = True

    # Cartesian acceleration limits (m/s²)
    CART_LIN_ACC_MIN: float = PAROL6_ROBOT.cart.acc.linear.min
    CART_LIN_ACC_MAX: float = PAROL6_ROBOT.cart.acc.linear.max
    # Angular acceleration limits (deg/s²) - derived from linear limits scaled appropriately
    # Using a reasonable ratio based on typical robot arm kinematics
    CART_ANG_ACC_MIN: float = 1.0    # deg/s²
    CART_ANG_ACC_MAX: float = 120.0  # deg/s²

    @staticmethod
    def _trapezoidal_duration(distance: float, v_max: float, a_max: float) -> float:
        """
        Calculate the duration for a trapezoidal velocity profile move.

        For a move where max velocity is reached (long moves):
            t = distance/v_max + v_max/a_max

        For short moves where max velocity isn't reached (triangular profile):
            t = 2 * sqrt(distance/a_max)

        Args:
            distance: Total distance to travel (positive)
            v_max: Maximum velocity (must be positive)
            a_max: Maximum acceleration (must be positive)

        Returns:
            Duration in seconds for the move
        """
        if distance <= 0 or v_max <= 0 or a_max <= 0:
            return 0.0

        # Distance needed to reach v_max and decelerate back to 0
        # (triangular profile distance = v_max^2 / a_max)
        d_accel = (v_max * v_max) / a_max

        if distance >= d_accel:
            # Long move: trapezoid profile (accel + cruise + decel)
            # Time = accel_time + cruise_time + decel_time
            # accel_time = decel_time = v_max / a_max
            # cruise_time = (distance - d_accel) / v_max
            t_accel = v_max / a_max
            t_cruise = (distance - d_accel) / v_max
            return 2 * t_accel + t_cruise
        else:
            # Short move: triangular profile (never reaches v_max)
            # t = 2 * sqrt(distance / a_max)
            return 2.0 * np.sqrt(distance / a_max)

    __slots__ = (
        "pose",
        "duration",
        "velocity_percent",
        "accel_percent",
        "start_time",
        "initial_pose",
        "target_pose",
        "_s_offset",
    )

    def __init__(self):
        super().__init__()

        # Parameters (set in do_match())
        self.pose = None
        self.duration = None
        self.velocity_percent = None
        self.accel_percent = DEFAULT_ACCEL_PERCENT

        # Runtime state
        self.start_time = None
        self.initial_pose: sp.SE3 | None = None
        self.target_pose: sp.SE3 | None = None
        self._s_offset = 0.0  # Progress offset for streaming (phase-preserving quintic)

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
        """
        Parse MOVECART command parameters.

        Format: MOVECART|x|y|z|rx|ry|rz|duration|speed|accel
        Example: MOVECART|100|200|300|0|0|0|2.0|None|50

        Args:
            parts: Pre-split message parts

        Returns:
            Tuple of (can_handle, error_message)
        """
        if len(parts) != 10:
            return (
                False,
                "MOVECART requires 9 parameters: x, y, z, rx, ry, rz, duration, speed, accel",
            )

        # Parse pose (6 values)
        self.pose = [float(parts[i]) for i in range(1, 7)]

        # Parse duration and speed
        self.duration = None if parts[7].upper() == "NONE" else float(parts[7])
        self.velocity_percent = None if parts[8].upper() == "NONE" else float(parts[8])

        # Parse acceleration
        self.accel_percent = float(parts[9]) if parts[9].upper() != "NONE" else DEFAULT_ACCEL_PERCENT

        # Validate that at least one timing parameter is given
        if self.duration is None and self.velocity_percent is None:
            return (False, "MOVECART requires either duration or velocity_percent")

        if self.duration is not None and self.velocity_percent is not None:
            logger.info(
                "  -> INFO: Both duration and velocity_percent provided. Using duration."
            )
            self.velocity_percent = None  # Prioritize duration

        self.log_debug("Parsed MoveCart: %s, accel=%s%%", self.pose, self.accel_percent)
        self.is_valid = True
        return (True, None)

    def do_setup(self, state: "ControllerState") -> None:
        """Captures the initial state and validates the path just before execution.

        In stream mode, when called on an in-progress command, blends smoothly
        to the new target instead of restarting the trajectory.
        """
        pose = cast(list[float], self.pose)

        # Construct new target pose (position in mm, angles in degrees)
        new_target = se3_from_rpy(
            pose[0] / 1000.0,
            pose[1] / 1000.0,
            pose[2] / 1000.0,
            pose[3],
            pose[4],
            pose[5],
            degrees=True,
        )

        # Check if this is a stream update (command already in progress)
        # Only blend when streaming is enabled AND command is mid-execution
        # _t0 is set on first execute_step, so if it's set we're mid-execution
        is_stream_update = (
            state.stream_mode  # Only blend when streaming is enabled
            and self._t0 is not None
            and self.initial_pose is not None
            and self.target_pose is not None
            and not self.is_finished
        )

        if state.stream_mode and not is_stream_update:
            self.log_debug("  -> Stream update check failed: _t0=%s, initial=%s, target=%s, finished=%s",
                          self._t0 is not None, self.initial_pose is not None,
                          self.target_pose is not None, self.is_finished)

        if is_stream_update:
            # STREAM UPDATE: Update target while preserving motion continuity
            self.log_debug("  -> Stream blend: updating target")

            # Capture current interpolated position as new start point
            dur = float(self.duration or 0.0)
            if dur > 0:
                s = self.progress01(dur)
                # Use fast quintic for faster response
                s_scaled = fast_quintic_scaling(float(s), compression=0.3)
                # sophuspy's native Lie algebra interpolation is fast
                self.initial_pose = se3_interp(
                    cast(sp.SE3, self.initial_pose),
                    cast(sp.SE3, self.target_pose),
                    s_scaled,
                )
            else:
                self.initial_pose = get_fkine_se3()

            # Update target to new destination
            self.target_pose = new_target
            self.is_finished = False

            # Recalculate duration based on remaining distance
            if self.velocity_percent is not None:
                tp = cast(sp.SE3, self.target_pose)
                ip = cast(sp.SE3, self.initial_pose)
                linear_distance = float(np.linalg.norm(tp.translation() - ip.translation()))
                angular_distance_rad = se3_angdist(ip, tp)
                angular_distance_deg = np.rad2deg(angular_distance_rad)

                target_linear_speed = self.linmap_pct(
                    self.velocity_percent, self.CART_LIN_JOG_MIN, self.CART_LIN_JOG_MAX
                )
                target_angular_speed_deg = self.linmap_pct(
                    self.velocity_percent, self.CART_ANG_JOG_MIN, self.CART_ANG_JOG_MAX
                )
                target_linear_accel = self.linmap_pct(
                    self.accel_percent, self.CART_LIN_ACC_MIN, self.CART_LIN_ACC_MAX
                )
                target_angular_accel_deg = self.linmap_pct(
                    self.accel_percent, self.CART_ANG_ACC_MIN, self.CART_ANG_ACC_MAX
                )

                time_linear = self._trapezoidal_duration(
                    linear_distance, target_linear_speed, target_linear_accel
                )
                time_angular = self._trapezoidal_duration(
                    angular_distance_deg, target_angular_speed_deg, target_angular_accel_deg
                )
                # Use minimum duration for responsiveness
                self.duration = max(time_linear, time_angular, 0.1)

            # Preserve progress offset and reset timer
            old_s = self.progress01(dur) if dur > 0 and self._t0 else 0.0
            self._s_offset = min(float(old_s), 0.5)
            self._t0 = time.perf_counter()
        else:
            # FRESH START: Original behavior - reset all streaming state
            self._t0 = None  # Reset timer for fresh start
            self._s_offset = 0.0  # Reset phase offset
            self.initial_pose = get_fkine_se3()
            self.target_pose = new_target

            if self.velocity_percent is not None:
                # Calculate the total distance for translation and rotation
                tp = cast(sp.SE3, self.target_pose)
                ip = cast(sp.SE3, self.initial_pose)
                linear_distance = float(np.linalg.norm(tp.translation() - ip.translation()))
                angular_distance_rad = se3_angdist(ip, tp)
                angular_distance_deg = np.rad2deg(angular_distance_rad)

                target_linear_speed = self.linmap_pct(
                    self.velocity_percent, self.CART_LIN_JOG_MIN, self.CART_LIN_JOG_MAX
                )
                target_angular_speed_deg = self.linmap_pct(
                    self.velocity_percent, self.CART_ANG_JOG_MIN, self.CART_ANG_JOG_MAX
                )

                # Get acceleration from accel_percent
                target_linear_accel = self.linmap_pct(
                    self.accel_percent, self.CART_LIN_ACC_MIN, self.CART_LIN_ACC_MAX
                )
                target_angular_accel_deg = self.linmap_pct(
                    self.accel_percent, self.CART_ANG_ACC_MIN, self.CART_ANG_ACC_MAX
                )

                # Use trapezoidal profile duration (accounts for accel/decel phases)
                time_linear = self._trapezoidal_duration(
                    linear_distance, target_linear_speed, target_linear_accel
                )
                time_angular = self._trapezoidal_duration(
                    angular_distance_deg, target_angular_speed_deg, target_angular_accel_deg
                )

                # The total duration is the longer of the two times to ensure synchronization
                calculated_duration = max(time_linear, time_angular)

                # Use minimum duration to keep command alive for streaming updates.
                # This must be longer than the typical command update rate (50ms at 20Hz)
                # to ensure commands overlap and can be blended smoothly.
                calculated_duration = max(calculated_duration, 0.2)
                if calculated_duration == 0.2:
                    self.log_debug("  -> Using minimum duration %.3fs for streaming", calculated_duration)

                self.duration = calculated_duration
                self.log_debug("  -> Calculated MoveCart duration: %.2fs", self.duration)

            self.log_debug("  -> Command is valid and ready for execution.")
            if self.duration and float(self.duration) > 0.0:
                self.start_timer(float(self.duration))

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        dur = float(self.duration or 0.0)
        if dur <= 0.0:
            self.is_finished = True
            self.stop_and_idle(state)
            return ExecutionStatus.completed("MOVECART complete")
        s_raw = self.progress01(dur)
        # Apply s_offset for streaming (phase-preserving quintic)
        s_offset = getattr(self, '_s_offset', 0.0)
        if s_offset > 0:
            # Map raw progress to continue from where we left off on quintic curve
            s = s_offset + s_raw * (1.0 - s_offset)
        else:
            s = s_raw

        # Use fast quintic in stream mode for faster ramps
        if state.stream_mode:
            s_scaled = fast_quintic_scaling(float(s))
        else:
            s_scaled = quintic_scaling(float(s))

        assert self.initial_pose is not None and self.target_pose is not None
        # Use sophuspy's fast Lie algebra interpolation
        current_target_pose = se3_interp(self.initial_pose, self.target_pose, s_scaled)

        current_q_rad = np.asarray(
            PAROL6_ROBOT.ops.steps_to_rad(state.Position_in), dtype=float
        )
        ik_solution = solve_ik(PAROL6_ROBOT.robot, current_target_pose, current_q_rad)

        if not ik_solution.success:
            error_str = "An intermediate point on the path is unreachable."
            if ik_solution.violations:
                error_str += (
                    f" Reason: Path violates joint limits: {ik_solution.violations}"
                )
            raise IKError(error_str)

        current_pos_rad = ik_solution.q

        # Send only the target position and let the firmware's P-controller handle speed.
        # Set feed-forward velocity to zero for smooth P-control.
        steps = PAROL6_ROBOT.ops.rad_to_steps(current_pos_rad)
        self.set_move_position(state, np.asarray(steps))

        if s >= 1.0:
            actual_elapsed = (
                (time.perf_counter() - self._t0) if self._t0 is not None else dur
            )
            self.log_info("MoveCart finished in ~%.2fs.", actual_elapsed)
            self.is_finished = True
            return ExecutionStatus.completed("MOVECART complete")

        return ExecutionStatus.executing("MoveCart")
