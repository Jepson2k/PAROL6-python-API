"""
Utility Commands
Contains utility commands like Delay
"""

import logging

from parol6.commands.base import CommandBase, ExecutionStatus, SystemCommand, parse_float
from parol6.protocol.wire import CommandCode
from parol6.server.command_registry import register_command
from parol6.server.state import ControllerState

logger = logging.getLogger(__name__)


@register_command("DELAY")
class DelayCommand(CommandBase):
    """
    A non-blocking command that pauses execution for a specified duration.
    During the delay, it ensures the robot remains idle by sending the
    appropriate commands.
    """

    __slots__ = ("duration",)

    def __init__(self):
        """
        Initializes the Delay command.
        Parameters are parsed in do_match() method.
        """
        super().__init__()
        self.duration: float | None = None

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
        """
        Parse DELAY command parameters.

        Format: DELAY|duration
        Example: DELAY|2.5
        """
        if len(parts) != 2:
            return (False, "DELAY requires 1 parameter: duration")

        self.duration = parse_float(parts[1])
        if self.duration is None or self.duration <= 0:
            return (False, f"Delay duration must be positive, got {parts[1]}")
        logger.info(f"Parsed Delay command for {self.duration} seconds")
        self.is_valid = True
        return (True, None)

    def setup(self, state: "ControllerState") -> None:
        """Start the delay timer."""
        if self.duration:
            self.start_timer(self.duration)
            logger.info(f"  -> Delay starting for {self.duration} seconds...")

    def tick(self, state: "ControllerState") -> ExecutionStatus:
        """
        Template-method wrapper that centralizes lifecycle/error handling.
        """
        if self.is_finished or not self.is_valid:
            return (
                ExecutionStatus.completed("Already finished")
                if self.is_finished
                else ExecutionStatus.failed("Invalid command")
            )
        try:
            status = self.execute_step(state)
        except Exception as e:
            self.is_valid = False
            self.error_state = True
            self.error_message = str(e)
            self.is_finished = True
            logger.error(f"[DelayCommand] Execution error: {e}")
            return ExecutionStatus.failed("Execution error", error=e)
        return status

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """
        Keep the robot idle during the delay and report status via ExecutionStatus.
        """
        if self.is_finished or not self.is_valid:
            return (
                ExecutionStatus.completed("Already finished")
                if self.is_finished
                else ExecutionStatus.failed("Invalid command")
            )

        # Keep the robot idle during the delay
        state.Command_out = CommandCode.IDLE
        state.Speed_out.fill(0)

        # Check for completion
        if self.timer_expired():
            logger.info(f"Delay finished after {self.duration} seconds.")
            self.is_finished = True
            return ExecutionStatus.completed("Delay complete")

        return ExecutionStatus.executing("Delaying")


@register_command("RESET")
class ResetCommand(SystemCommand):
    """
    Instantly reset controller state to initial values.

    Useful for test isolation - avoids slow homing motion by instantly
    resetting positions, clearing queues, and resetting tool/errors.
    Preserves serial connection and network config.
    """

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
        """Parse RESET command (no parameters)."""
        if len(parts) != 1:
            return (False, "RESET takes no parameters")
        return (True, None)

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Reset state immediately."""
        state.reset()
        logger.debug("RESET command executed")
        return ExecutionStatus.completed("Reset complete")
