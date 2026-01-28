"""
Query commands that return immediate status information.
"""

from typing import TYPE_CHECKING

import numpy as np

from parol6 import config as cfg
from parol6.commands.base import ExecutionStatus, QueryCommand
from parol6.protocol.wire import (
    CmdType,
    GetAnglesCmd,
    GetCurrentActionCmd,
    GetGcodeStatusCmd,
    GetGripperCmd,
    GetIOCmd,
    GetLoopStatsCmd,
    GetPoseCmd,
    GetProfileCmd,
    GetQueueCmd,
    GetSpeedsCmd,
    GetStatusCmd,
    GetToolCmd,
    PingCmd,
    QueryType,
)
from parol6.server.command_registry import register_command
from parol6.server.state import get_fkine_flat_mm, get_fkine_matrix
from parol6.server.status_cache import get_cache
from parol6.server.transports import is_simulation_mode
from parol6.tools import list_tools

if TYPE_CHECKING:
    from parol6.server.state import ControllerState


@register_command(CmdType.GET_POSE)
class GetPoseCommand(QueryCommand):
    """Get current robot pose matrix in specified frame (WRF or TRF)."""

    PARAMS_TYPE = GetPoseCmd

    __slots__ = ()

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Execute immediately and return pose data with translation in mm."""
        assert self.p is not None
        frame = self.p.frame or "WRF"
        if frame == "TRF":
            # Get current pose as 4x4 matrix (translation in meters)
            T = get_fkine_matrix(state)

            # Compute inverse to express world in tool frame: T^(-1) = [R^T | -R^T * t]
            T_inv = np.linalg.inv(T)

            # Convert translation to mm
            T_inv[0:3, 3] *= 1000.0

            # Flatten row-major (same format as WRF)
            pose_flat = T_inv.reshape(-1)
        else:
            # WRF: use existing implementation
            pose_flat = get_fkine_flat_mm(state)

        self.reply(QueryType.POSE, pose_flat)

        self.finish()
        return ExecutionStatus.completed("Pose sent")


@register_command(CmdType.GET_ANGLES)
class GetAnglesCommand(QueryCommand):
    """Get current joint angles in degrees."""

    PARAMS_TYPE = GetAnglesCmd

    __slots__ = ()

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Execute immediately and return angle data."""
        cfg.steps_to_rad(state.Position_in, self._q_rad_buf)
        self.reply(QueryType.ANGLES, np.rad2deg(self._q_rad_buf))

        self.finish()
        return ExecutionStatus.completed("Angles sent")


@register_command(CmdType.GET_IO)
class GetIOCommand(QueryCommand):
    """Get current I/O status."""

    PARAMS_TYPE = GetIOCmd

    __slots__ = ()

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Execute immediately and return I/O data."""
        self.reply(QueryType.IO, state.InOut_in[:5])

        self.finish()
        return ExecutionStatus.completed("I/O sent")


@register_command(CmdType.GET_GRIPPER)
class GetGripperCommand(QueryCommand):
    """Get current gripper status."""

    PARAMS_TYPE = GetGripperCmd

    __slots__ = ()

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Execute immediately and return gripper data."""
        self.reply(QueryType.GRIPPER, state.Gripper_data_in)

        self.finish()
        return ExecutionStatus.completed("Gripper sent")


@register_command(CmdType.GET_SPEEDS)
class GetSpeedsCommand(QueryCommand):
    """Get current joint speeds."""

    PARAMS_TYPE = GetSpeedsCmd

    __slots__ = ()

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Execute immediately and return speed data."""
        self.reply(QueryType.SPEEDS, state.Speed_in)

        self.finish()
        return ExecutionStatus.completed("Speeds sent")


@register_command(CmdType.GET_STATUS)
class GetStatusCommand(QueryCommand):
    """Get aggregated robot status (pose, angles, I/O, gripper) from cache."""

    PARAMS_TYPE = GetStatusCmd

    __slots__ = ()

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Execute immediately and return cached aggregated status (binary)."""
        # Always refresh cache from current state before replying
        cache = get_cache()
        cache.update_from_state(state)
        # [pose, angles, speeds, io, gripper]
        self.reply(
            QueryType.STATUS,
            [
                cache.pose,
                cache.angles_deg,
                cache.speeds,
                cache.io,
                cache.gripper,
            ],
        )

        self.finish()
        return ExecutionStatus.completed("Status sent")


@register_command(CmdType.GET_GCODE_STATUS)
class GetGcodeStatusCommand(QueryCommand):
    """Get GCODE interpreter status."""

    PARAMS_TYPE = GetGcodeStatusCmd

    __slots__ = ()

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Execute immediately and return GCODE status."""
        if self.gcode_interpreter:
            s = self.gcode_interpreter.get_status()
            # [is_running, is_paused, current_line, total_lines, state]
            gcode_status = [
                s.get("is_running", False),
                s.get("is_paused", False),
                s.get("current_line"),
                s.get("total_lines", 0),
                s.get("state", {}),
            ]
        else:
            # [is_running, is_paused, current_line, total_lines, state]
            gcode_status = [False, False, None, 0, {}]

        self.reply(QueryType.GCODE_STATUS, gcode_status)

        self.finish()
        return ExecutionStatus.completed("GCODE status sent")


@register_command(CmdType.GET_LOOP_STATS)
class GetLoopStatsCommand(QueryCommand):
    """Return control-loop metrics (no ACK dependency)."""

    PARAMS_TYPE = GetLoopStatsCmd

    __slots__ = ()

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        target_hz = 1.0 / max(cfg.INTERVAL_S, 1e-9)
        mean_hz = (1.0 / state.mean_period_s) if state.mean_period_s > 0.0 else 0.0
        # [target_hz, loop_count, overrun_count, mean_period_s, std_period_s,
        #  min_period_s, max_period_s, p95_period_s, p99_period_s, mean_hz]
        payload = [
            target_hz,
            state.loop_count,
            state.overrun_count,
            state.mean_period_s,
            state.std_period_s,
            state.min_period_s,
            state.max_period_s,
            state.p95_period_s,
            state.p99_period_s,
            mean_hz,
        ]
        self.reply(QueryType.LOOP_STATS, payload)
        self.finish()
        return ExecutionStatus.completed("Loop stats sent")


@register_command(CmdType.PING)
class PingCommand(QueryCommand):
    """Respond to ping requests."""

    PARAMS_TYPE = PingCmd

    __slots__ = ()

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Execute immediately and return PONG with serial connectivity bit (0 in simulator mode)."""
        # Check if we're in simulator mode
        sim = is_simulation_mode()

        # In simulator mode, report SERIAL=0 (not real hardware)
        # Otherwise, check if we've observed fresh serial frames recently
        if sim:
            serial_connected = 0
        else:
            serial_connected = 1 if get_cache().age_s() <= cfg.STATUS_STALE_S else 0

        self.reply(QueryType.PING, serial_connected)

        self.finish()
        return ExecutionStatus.completed("PONG")


@register_command(CmdType.GET_TOOL)
class GetToolCommand(QueryCommand):
    """Get current tool configuration and available tools."""

    PARAMS_TYPE = GetToolCmd

    __slots__ = ()

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Execute immediately and return current tool info."""
        # [tool, available]
        payload = [state.current_tool, list_tools()]
        self.reply(QueryType.TOOL, payload)

        self.finish()
        return ExecutionStatus.completed("Tool info sent")


@register_command(CmdType.GET_CURRENT_ACTION)
class GetCurrentActionCommand(QueryCommand):
    """Get the current executing action/command and its state."""

    PARAMS_TYPE = GetCurrentActionCmd

    __slots__ = ()

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Execute immediately and return current action info."""
        # [current, state, next]
        payload = [state.action_current, state.action_state, state.action_next]
        self.reply(QueryType.CURRENT_ACTION, payload)

        self.finish()
        return ExecutionStatus.completed("Current action info sent")


@register_command(CmdType.GET_QUEUE)
class GetQueueCommand(QueryCommand):
    """Get the list of queued non-streamable commands."""

    PARAMS_TYPE = GetQueueCmd

    __slots__ = ()

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Execute immediately and return queue info."""
        # Just the queue list (size is len of array)
        self.reply(QueryType.QUEUE, state.queue_nonstreamable)

        self.finish()
        return ExecutionStatus.completed("Queue info sent")


@register_command(CmdType.GET_PROFILE)
class GetProfileCommand(QueryCommand):
    """
    Query the current motion profile.

    Format: [CmdType.GET_PROFILE]
    Response: [RESPONSE, PROFILE, profile_type]
    """

    PARAMS_TYPE = GetProfileCmd

    __slots__ = ()

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Return the current motion profile."""
        profile = state.motion_profile
        self.reply(QueryType.PROFILE, profile)

        self.finish()
        return ExecutionStatus.completed(
            f"Current motion profile: {profile}",
            details={"profile": profile},
        )
