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
from parol6.commands.base import TrajectoryMoveCommandBase
from parol6.config import (
    INTERVAL_S,
    LIMITS,
    speed_rad_to_steps,
    steps_to_rad,
)
from parol6.motion import JointPath, TrajectoryBuilder
from parol6.protocol.wire import CmdType, MoveJointCmd, MovePoseCmd
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
    - _get_target_rad(): Return target joint positions in radians

    This base class provides:
    - do_setup(): Builds trajectory via JointPath.interpolate + TrajectoryBuilder
    - execute_step(): Inherited from TrajectoryMoveCommandBase (uses MotionExecutor)
    """

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def _get_target_rad(
        self, state: ControllerState, current_rad: np.ndarray
    ) -> np.ndarray:
        """Return target joint positions in radians.

        Args:
            state: Controller state
            current_rad: Current joint positions in radians (for IK seed if needed)
        """
        ...

    def do_setup(self, state: ControllerState) -> None:
        """Build trajectory from current position to target using unified motion pipeline."""
        assert self.p is not None

        steps_to_rad(state.Position_in, self._q_rad_buf)
        target_rad = self._get_target_rad(state, self._q_rad_buf)
        current_rad = self._q_rad_buf

        profile = state.motion_profile
        duration = self.p.duration if self.p.duration > 0.0 else None
        vel_pct = self.p.speed_pct if self.p.speed_pct > 0.0 else 100.0
        accel_pct = self.p.accel_pct

        joint_path = JointPath.interpolate(current_rad, target_rad, n_samples=50)
        builder = TrajectoryBuilder(
            joint_path=joint_path,
            profile=profile,
            velocity_percent=vel_pct if duration is None else None,
            accel_percent=accel_pct,
            duration=duration,
            dt=INTERVAL_S,
        )

        trajectory = builder.build()
        self.trajectory_steps = trajectory.steps

        if len(self.trajectory_steps) == 0:
            raise ValueError("Trajectory calculation resulted in no steps.")

        self._is_cartesian = False
        v_max_rad = LIMITS.joint.hard.velocity * (vel_pct / 100.0)
        a_max_rad = LIMITS.joint.hard.acceleration * (accel_pct / 100.0)
        speed_rad_to_steps(v_max_rad, self._steps_buf)
        self._v_max_steps = np.abs(self._steps_buf.astype(np.float64))
        speed_rad_to_steps(a_max_rad, self._steps_buf)
        self._a_max_steps = np.abs(self._steps_buf.astype(np.float64))

        self.log_trace(
            "  -> Using profile: %s, duration: %.3fs, steps: %d",
            profile,
            trajectory.duration,
            len(self.trajectory_steps),
        )


@register_command(CmdType.MOVEJOINT)
class MoveJointCommand(JointMoveCommandBase):
    """Move the robot's joints to a specific configuration."""

    PARAMS_TYPE = MoveJointCmd

    __slots__ = ()

    def _get_target_rad(
        self, state: ControllerState, current_rad: np.ndarray
    ) -> np.ndarray:
        """Return target joint positions in radians."""
        assert self.p is not None
        return np.deg2rad(self.p.angles)


@register_command(CmdType.MOVEPOSE)
class MovePoseCommand(JointMoveCommandBase):
    """Move the robot to a specific Cartesian pose via joint-space interpolation.

    Uses IK to find the target joint configuration, then interpolates in joint space.
    This is different from MoveCart which follows a straight-line Cartesian path.
    """

    PARAMS_TYPE = MovePoseCmd

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__()

    def _get_target_rad(
        self, state: ControllerState, current_rad: np.ndarray
    ) -> np.ndarray:
        """Solve IK for target pose and return joint positions in radians."""
        assert self.p is not None
        pose = self.p.pose

        target_pose = np.zeros((4, 4), dtype=np.float64)
        se3_from_rpy(
            pose[0] / 1000.0,
            pose[1] / 1000.0,
            pose[2] / 1000.0,
            np.radians(pose[3]),
            np.radians(pose[4]),
            np.radians(pose[5]),
            target_pose,
        )

        ik_solution = solve_ik(PAROL6_ROBOT.robot, target_pose, current_rad)
        if not ik_solution.success:
            error_str = "Target pose is unreachable."
            if ik_solution.violations:
                error_str += f" Reason: {ik_solution.violations}"
            raise IKError(error_str)

        return np.asarray(ik_solution.q, dtype=np.float64)
