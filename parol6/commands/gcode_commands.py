"""
GCODE command wrappers for robot control.

These commands integrate the GCODE interpreter with the robot command system.
"""

from typing import TYPE_CHECKING

from parol6.commands.base import CommandBase, ExecutionStatus
from parol6.gcode import GcodeInterpreter
from parol6.protocol.wire import (
    CmdType,
    Command,
    GcodeCmd,
    GcodePauseCmd,
    GcodeProgramCmd,
    GcodeResumeCmd,
    GcodeStopCmd,
)
from parol6.server.command_registry import register_command
from parol6.server.state import ControllerState, get_fkine_matrix

if TYPE_CHECKING:
    from parol6.server.state import ControllerState


@register_command(CmdType.GCODE)
class GcodeCommand(CommandBase):
    """Execute a single GCODE line."""

    PARAMS_TYPE = GcodeCmd

    __slots__ = (
        "interpreter",
        "generated_commands",
        "current_command_index",
    )
    interpreter: GcodeInterpreter | None
    generated_commands: list[Command]
    current_command_index: int

    def do_setup(self, state: "ControllerState") -> None:
        """Set up GCODE interpreter and parse the line."""
        assert self.p is not None

        # Use injected interpreter or create one
        self.interpreter = (
            self.gcode_interpreter or self.interpreter or GcodeInterpreter()
        )
        assert self.interpreter is not None
        # Update interpreter position with current robot position
        current_pose_matrix = get_fkine_matrix()
        current_xyz = current_pose_matrix[:3, 3]
        self.interpreter.state.update_position(
            {
                "X": current_xyz[0] * 1000,
                "Y": current_xyz[1] * 1000,
                "Z": current_xyz[2] * 1000,
            }
        )
        # Parse and store generated robot Command structs
        self.generated_commands = self.interpreter.parse_line(self.p.line) or []

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Return generated commands for the controller to enqueue."""
        # Report back the list so controller can enqueue them
        details = {}
        if self.generated_commands:
            details["enqueue"] = self.generated_commands
        self.finish()
        return ExecutionStatus.completed("GCODE parsed", details=details)


@register_command(CmdType.GCODE_PROGRAM)
class GcodeProgramCommand(CommandBase):
    """Load and execute a GCODE program."""

    PARAMS_TYPE = GcodeProgramCmd

    __slots__ = ("interpreter",)
    interpreter: GcodeInterpreter | None

    def do_setup(self, state: ControllerState) -> None:
        """Load the GCODE program using the interpreter."""
        assert self.p is not None

        # Use injected interpreter or create one
        self.interpreter = (
            self.gcode_interpreter or self.interpreter or GcodeInterpreter()
        )
        assert self.interpreter is not None

        # Load program lines directly
        if not self.interpreter.load_program(self.p.lines):
            raise RuntimeError("Failed to load GCODE program")

        # Start program execution
        self.interpreter.start_program()

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Signal that the program was loaded; controller will fetch commands."""
        self.finish()
        return ExecutionStatus.completed("GCODE program loaded")


@register_command(CmdType.GCODE_STOP)
class GcodeStopCommand(CommandBase):
    """Stop GCODE program execution."""

    PARAMS_TYPE = GcodeStopCmd

    __slots__ = ()
    is_immediate: bool = True

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Stop the GCODE program."""
        if self.gcode_interpreter:
            self.gcode_interpreter.stop_program()
        self.finish()
        return ExecutionStatus.completed("GCODE stopped")


@register_command(CmdType.GCODE_PAUSE)
class GcodePauseCommand(CommandBase):
    """Pause GCODE program execution."""

    PARAMS_TYPE = GcodePauseCmd

    __slots__ = ()
    is_immediate: bool = True

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Pause the GCODE program."""
        if self.gcode_interpreter:
            self.gcode_interpreter.pause_program()
        self.finish()
        return ExecutionStatus.completed("GCODE paused")


@register_command(CmdType.GCODE_RESUME)
class GcodeResumeCommand(CommandBase):
    """Resume GCODE program execution."""

    PARAMS_TYPE = GcodeResumeCmd

    __slots__ = ()
    is_immediate: bool = True

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Resume the GCODE program."""
        if self.gcode_interpreter:
            self.gcode_interpreter.start_program()  # resumes if already loaded
        self.finish()
        return ExecutionStatus.completed("GCODE resumed")
