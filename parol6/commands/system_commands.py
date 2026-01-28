"""
System control commands that can execute regardless of controller enable state.

These commands control the overall state of the robot controller (enable/disable, stop, etc.)
and can execute even when the controller is disabled.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from parol6.commands.base import ExecutionStatus, SystemCommand
from parol6.config import save_com_port
from parol6.protocol.wire import (
    CmdType,
    DisableCmd,
    EnableCmd,
    SetIOCmd,
    SetPortCmd,
    SetProfileCmd,
    SimulatorCmd,
    StopCmd,
    StreamCmd,
)
from parol6.protocol.wire import CommandCode
from parol6.server.command_registry import register_command

if TYPE_CHECKING:
    from parol6.server.state import ControllerState

logger = logging.getLogger(__name__)


@register_command(CmdType.STOP)
class StopCommand(SystemCommand):
    """Emergency stop command - immediately stops all motion."""

    PARAMS_TYPE = StopCmd

    __slots__ = ()

    def execute_step(self, state: ControllerState) -> ExecutionStatus:
        """Execute stop - set all speeds to zero and command to IDLE."""
        logger.info("STOP command executed")
        state.Speed_out.fill(0)
        state.Command_out = CommandCode.IDLE

        self.finish()
        return ExecutionStatus.completed("Robot stopped")


@register_command(CmdType.ENABLE)
class EnableCommand(SystemCommand):
    """Enable the robot controller."""

    PARAMS_TYPE = EnableCmd

    __slots__ = ()

    def execute_step(self, state: ControllerState) -> ExecutionStatus:
        """Execute enable - set controller to enabled state."""
        logger.info("ENABLE command executed")
        state.enabled = True
        state.disabled_reason = ""
        state.Command_out = CommandCode.ENABLE

        self.finish()
        return ExecutionStatus.completed("Controller enabled")


@register_command(CmdType.DISABLE)
class DisableCommand(SystemCommand):
    """Disable the robot controller."""

    PARAMS_TYPE = DisableCmd

    __slots__ = ()

    def execute_step(self, state: ControllerState) -> ExecutionStatus:
        """Execute disable - zero speeds and set controller to disabled state."""
        logger.info("DISABLE command executed")
        state.Speed_out.fill(0)
        state.enabled = False
        state.disabled_reason = "User requested disable"
        state.Command_out = CommandCode.DISABLE

        self.finish()
        return ExecutionStatus.completed("Controller disabled")


@register_command(CmdType.SET_IO)
class SetIOCommand(SystemCommand):
    """Set a digital I/O port state."""

    PARAMS_TYPE = SetIOCmd

    __slots__ = ()

    def execute_step(self, state: ControllerState) -> ExecutionStatus:
        """Execute set port - update I/O port state."""
        assert self.p is not None

        logger.info(f"SET_IO: Setting port {self.p.port_index} to {self.p.value}")

        state.InOut_out[self.p.port_index] = self.p.value

        self.finish()
        return ExecutionStatus.completed(
            f"Port {self.p.port_index} set to {self.p.value}"
        )


@register_command(CmdType.SET_PORT)
class SetSerialPortCommand(SystemCommand):
    """Set the serial COM port used by the controller."""

    PARAMS_TYPE = SetPortCmd

    __slots__ = ()

    def execute_step(self, state: ControllerState) -> ExecutionStatus:
        """Persist the serial port selection; controller may reconnect based on this."""
        assert self.p is not None

        ok = save_com_port(self.p.port_str)
        if not ok:
            self.fail("Failed to save COM port")
            return ExecutionStatus.failed("Failed to save COM port")

        self.finish()
        # Include details so the controller can reconnect immediately
        return ExecutionStatus.completed(
            "Serial port saved", details={"serial_port": self.p.port_str}
        )


@register_command(CmdType.STREAM)
class StreamCommand(SystemCommand):
    """Toggle stream mode for real-time jogging."""

    PARAMS_TYPE = StreamCmd

    __slots__ = ()

    def execute_step(self, state: ControllerState) -> ExecutionStatus:
        """Execute stream mode toggle."""
        assert self.p is not None
        # The controller will handle the actual stream mode setting
        # This is just a placeholder that sets a flag
        logger.info(f"STREAM: Setting stream mode to {self.p.on}")

        state.stream_mode = self.p.on

        self.finish()
        return ExecutionStatus.completed(
            f"Stream mode {'enabled' if self.p.on else 'disabled'}"
        )


@register_command(CmdType.SIMULATOR)
class SimulatorCommand(SystemCommand):
    """Toggle simulator (fake serial) mode on/off."""

    PARAMS_TYPE = SimulatorCmd

    __slots__ = ()

    def execute_step(self, state: ControllerState) -> ExecutionStatus:
        """Execute simulator toggle by setting env var and returning details to trigger reconfiguration."""
        assert self.p is not None

        os.environ["PAROL6_FAKE_SERIAL"] = "1" if self.p.on else "0"
        logger.info(f"SIMULATOR command executed: {'ON' if self.p.on else 'OFF'}")

        self.finish()
        return ExecutionStatus.completed(
            f"Simulator {'ON' if self.p.on else 'OFF'}",
            details={"simulator_mode": "on" if self.p.on else "off"},
        )


# Valid motion profile types
VALID_PROFILES = frozenset(("TOPPRA", "RUCKIG", "QUINTIC", "TRAPEZOID", "LINEAR"))


@register_command(CmdType.SET_PROFILE)
class SetProfileCommand(SystemCommand):
    """
    Set the motion profile for all moves.

    Format: [CmdType.SET_PROFILE, profile_type]

    Profile Types:
        TOPPRA    - Time-optimal path parameterization (default)
        RUCKIG    - Time-optimal jerk-limited (point-to-point only, joint moves only)
        QUINTIC   - C² smooth polynomial trajectories
        TRAPEZOID - Linear segments with parabolic blends
        LINEAR    - Direct interpolation (no smoothing)

    Note: RUCKIG is point-to-point and cannot follow Cartesian paths.
    Cartesian moves will use TOPPRA when RUCKIG is set.
    """

    PARAMS_TYPE = SetProfileCmd

    __slots__ = ()

    def do_setup(self, state: ControllerState) -> None:
        """Validate profile name against VALID_PROFILES."""
        assert self.p is not None
        profile = self.p.profile.upper()
        if profile not in VALID_PROFILES:
            valid_list = ", ".join(sorted(VALID_PROFILES))
            raise ValueError(
                f"Invalid profile '{self.p.profile}'. Valid profiles: {valid_list}"
            )

    def execute_step(self, state: ControllerState) -> ExecutionStatus:
        """Execute profile change."""
        assert self.p is not None
        profile = self.p.profile.upper()

        old_profile = state.motion_profile
        state.motion_profile = profile
        logger.info(
            f"SETPROFILE: Changed motion profile from {old_profile} to {profile}"
        )

        self.finish()
        return ExecutionStatus.completed(
            f"Motion profile set to {profile}",
            details={"profile": profile, "previous": old_profile},
        )
