"""
Joint Movement Commands
Contains commands for joint-space movements with unified trajectory execution.

Uses unified motion pipeline with TOPP-RA for time-optimal path parameterization.
All commands inherit from JointMoveCommandBase which uses MotionExecutor for
jerk-limited smoothing during execution.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.commands.base import TrajectoryMoveCommandBase, parse_opt_float
from parol6.config import DEFAULT_ACCEL_PERCENT, INTERVAL_S, LIMITS, steps_to_rad
from parol6.motion import JointPath, TrajectoryBuilder
from parol6.server.command_registry import register_command
from parol6.utils.errors import IKError
from parol6.utils.ik import solve_ik
from parol6.utils.se3_utils import se3_from_rpy

if TYPE_CHECKING:
    from parol6.server.state import ControllerState

logger = logging.getLogger(__name__)


class JointMoveCommandBase(TrajectoryMoveCommandBase):
    """Base class for joint-space trajectory commands.

    Subclasses must implement:
    - do_match(): Parse command parameters
    - _get_target_rad(): Return target joint positions in radians

    This base class provides:
    - do_setup(): Builds trajectory via JointPath.interpolate + TrajectoryBuilder
    - execute_step(): Inherited from TrajectoryMoveCommandBase (uses MotionExecutor)
    """

    __slots__ = ("duration", "velocity_percent", "accel_percent")

    def __init__(self) -> None:
        super().__init__()
        self.duration: float | None = None
        self.velocity_percent: float | None = None
        self.accel_percent: float = DEFAULT_ACCEL_PERCENT

    @abstractmethod
    def _get_target_rad(self, state: ControllerState) -> np.ndarray:
        """Return target joint positions in radians."""
        ...

    def do_setup(self, state: ControllerState) -> None:
        """Build trajectory from current position to target using unified motion pipeline."""
        current_rad = np.asarray(
            steps_to_rad(state.Position_in), dtype=np.float64
        )
        target_rad = self._get_target_rad(state)

        profile = state.motion_profile
        accel_pct = float(self.accel_percent) if self.accel_percent else DEFAULT_ACCEL_PERCENT

        joint_path = JointPath.interpolate(current_rad, target_rad, n_samples=50)
        builder = TrajectoryBuilder(
            joint_path=joint_path,
            profile=profile,
            velocity_percent=self.velocity_percent,
            accel_percent=accel_pct,
            duration=self.duration,
            dt=INTERVAL_S,
        )

        trajectory = builder.build()
        self.trajectory_steps = trajectory.steps

        if len(self.trajectory_steps) == 0:
            raise ValueError("Trajectory calculation resulted in no steps.")

        self.log_trace(
            "  -> Using profile: %s, duration: %.3fs, steps: %d",
            profile,
            trajectory.duration,
            len(self.trajectory_steps),
        )


@register_command("MOVEJOINT")
class MoveJointCommand(JointMoveCommandBase):
    """Move the robot's joints to a specific configuration."""

    __slots__ = ("target_angles", "target_radians")

    def __init__(self) -> None:
        super().__init__()
        self.target_angles: np.ndarray | None = None
        self.target_radians: np.ndarray | None = None

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
        """
        Parse MOVEJOINT command parameters.

        Format: MOVEJOINT|j1|j2|j3|j4|j5|j6|duration|speed|accel
        Example: MOVEJOINT|0|45|90|-45|30|0|None|50|50
        """
        if len(parts) != 10:
            return (
                False,
                "MOVEJOINT requires 9 parameters: 6 joint angles, duration, speed, accel",
            )

        self.target_angles = np.asarray(
            [float(parts[i]) for i in range(1, 7)], dtype=float
        )

        self.duration = parse_opt_float(parts[7])
        self.velocity_percent = parse_opt_float(parts[8])
        self.accel_percent = parse_opt_float(parts[9], DEFAULT_ACCEL_PERCENT)

        self.target_radians = np.deg2rad(self.target_angles)
        for i in range(6):
            min_rad, max_rad = LIMITS.joint.position.rad[i]
            if not (min_rad <= self.target_radians[i] <= max_rad):
                return (
                    False,
                    f"Joint {i + 1} target ({self.target_angles[i]} deg) is out of range",
                )

        self.log_debug(
            "Parsed MoveJoint: %s, accel=%s%%", self.target_angles, self.accel_percent
        )
        self.is_valid = True
        return (True, None)

    def _get_target_rad(self, state: ControllerState) -> np.ndarray:
        """Return target joint positions in radians."""
        return np.asarray(self.target_radians, dtype=np.float64)


@register_command("MOVEPOSE")
class MovePoseCommand(JointMoveCommandBase):
    """Move the robot to a specific Cartesian pose via joint-space interpolation.

    Uses IK to find the target joint configuration, then interpolates in joint space.
    This is different from MoveCart which follows a straight-line Cartesian path.
    """

    __slots__ = ("pose",)

    def __init__(self, pose: list[float] | None = None, duration: float | None = None) -> None:
        super().__init__()
        self.pose: list[float] | None = pose
        self.duration = duration

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
        """
        Parse MOVEPOSE command parameters.

        Format: MOVEPOSE|x|y|z|rx|ry|rz|duration|speed|accel
        Example: MOVEPOSE|100|200|300|0|0|0|None|50|50
        """
        if len(parts) != 10:
            return (
                False,
                "MOVEPOSE requires 9 parameters: x, y, z, rx, ry, rz, duration, speed, accel",
            )

        self.pose = [float(parts[i]) for i in range(1, 7)]
        self.duration = parse_opt_float(parts[7])
        self.velocity_percent = parse_opt_float(parts[8])
        self.accel_percent = parse_opt_float(parts[9], DEFAULT_ACCEL_PERCENT)

        self.log_debug("Parsed MovePose: %s, accel=%s%%", self.pose, self.accel_percent)
        self.is_valid = True
        return (True, None)

    def _get_target_rad(self, state: ControllerState) -> np.ndarray:
        """Solve IK for target pose and return joint positions in radians."""
        current_rad = np.asarray(
            steps_to_rad(state.Position_in), dtype=np.float64
        )

        assert self.pose is not None
        target_pose = se3_from_rpy(
            self.pose[0] / 1000.0,
            self.pose[1] / 1000.0,
            self.pose[2] / 1000.0,
            self.pose[3],
            self.pose[4],
            self.pose[5],
            degrees=True,
        )

        ik_solution = solve_ik(PAROL6_ROBOT.robot, target_pose, current_rad)
        if not ik_solution.success:
            error_str = "Target pose is unreachable."
            if ik_solution.violations:
                error_str += f" Reason: {ik_solution.violations}"
            raise IKError(error_str)

        return np.asarray(ik_solution.q, dtype=np.float64)
