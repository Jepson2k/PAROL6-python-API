"""
Utility Commands
Contains utility commands like Delay and Reset
"""

import logging

from parol6.commands.base import (
    CommandBase,
    ExecutionStatus,
    SystemCommand,
)
from parol6.protocol.wire import CmdType, DelayCmd, ResetCmd, ResetLoopStatsCmd
from parol6.protocol.wire import CommandCode
from parol6.server.command_registry import register_command
from parol6.server.state import ControllerState

logger = logging.getLogger(__name__)


@register_command(CmdType.DELAY)
class DelayCommand(CommandBase):
    """
    A non-blocking command that pauses execution for a specified duration.
    """

    PARAMS_TYPE = DelayCmd

    __slots__ = ()

    def __init__(self):
        super().__init__()

    def do_setup(self, state: "ControllerState") -> None:
        """Start the delay timer."""
        assert self.p is not None
        self.start_timer(self.p.seconds)
        logger.info(f"  -> Delay starting for {self.p.seconds} seconds...")

    def tick(self, state: "ControllerState") -> ExecutionStatus:
        """Template-method wrapper that centralizes lifecycle/error handling."""
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
        """Keep the robot idle during the delay and report status via ExecutionStatus."""
        assert self.p is not None

        if self.is_finished or not self.is_valid:
            return (
                ExecutionStatus.completed("Already finished")
                if self.is_finished
                else ExecutionStatus.failed("Invalid command")
            )

        state.Command_out = CommandCode.IDLE
        state.Speed_out.fill(0)

        if self.timer_expired():
            logger.info(f"Delay finished after {self.p.seconds} seconds.")
            self.is_finished = True
            return ExecutionStatus.completed("Delay complete")

        return ExecutionStatus.executing("Delaying")


@register_command(CmdType.RESET)
class ResetCommand(SystemCommand):
    """
    Instantly reset controller state to initial values.
    """

    PARAMS_TYPE = ResetCmd

    __slots__ = ()

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Reset state immediately."""
        state.reset()
        logger.debug("RESET command executed")
        self.is_finished = True
        return ExecutionStatus.completed("Reset complete")


@register_command(CmdType.RESET_LOOP_STATS)
class ResetLoopStatsCommand(SystemCommand):
    """
    Reset control loop timing statistics without affecting controller state.

    Resets: min/max period, overrun count, rolling statistics.
    Preserves: loop_count (uptime), robot state, command queues.
    """

    PARAMS_TYPE = ResetLoopStatsCmd

    __slots__ = ()

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Signal controller to reset loop stats."""
        state.loop_stats_reset_pending = True
        logger.debug("RESET_LOOP_STATS command executed")
        self.is_finished = True
        return ExecutionStatus.completed("Loop stats reset pending")
