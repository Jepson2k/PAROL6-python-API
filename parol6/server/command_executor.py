"""Command queue management and execution."""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from parol6.commands.base import (
    CommandBase,
    CommandContext,
    ExecutionStatus,
    ExecutionStatusCode,
    MotionCommand,
)
from parol6.config import TRACE
from parol6.server.command_registry import create_command_from_parts

if TYPE_CHECKING:
    from parol6.gcode import GcodeInterpreter
    from parol6.server.state import ControllerState, StateManager
    from parol6.server.transports.udp_transport import UDPTransport

logger = logging.getLogger("parol6.server.command_executor")


@dataclass
class QueuedCommand:
    """Represents a command in the queue with metadata."""

    command: CommandBase
    command_id: str | None = None
    address: tuple[str, int] | None = None
    queued_time: float = field(default_factory=time.time)
    activated: bool = False
    initialized: bool = False
    first_tick_logged: bool = False


class CommandExecutor:
    """Manages command queue and execution lifecycle.

    Handles queueing, executing, cancelling, and clearing commands.
    """

    def __init__(
        self,
        state_manager: "StateManager",
        gcode_interpreter: "GcodeInterpreter",
        udp_transport_getter: Callable[[], "UDPTransport | None"],
        send_ack: Callable[[str, str, str, tuple[str, int]], None],
        sync_mock_from_state: Callable[["ControllerState"], None],
        dt: float,
    ):
        """Initialize the command executor.

        Args:
            state_manager: StateManager for accessing controller state.
            gcode_interpreter: GCODE interpreter for fetching commands.
            udp_transport_getter: Callable that returns current UDP transport.
            send_ack: Callback to send ACK messages.
            sync_mock_from_state: Callback to sync mock transport after RESET.
            dt: Loop interval for command context.
        """
        self._state_manager = state_manager
        self._gcode_interpreter = gcode_interpreter
        self._get_udp_transport = udp_transport_getter
        self._send_ack = send_ack
        self._sync_mock_from_state = sync_mock_from_state
        self._dt = dt

        self.command_queue: deque[QueuedCommand] = deque(maxlen=100)
        self.active_command: QueuedCommand | None = None

    def _update_queue_state(self, state: "ControllerState") -> None:
        """Update queue snapshot and next action in state."""
        # Reuse list to avoid allocation (clear + extend pattern)
        state.queue_nonstreamable.clear()
        for qc in self.command_queue:
            if not (
                isinstance(qc.command, MotionCommand)
                and getattr(qc.command, "streamable", False)
            ):
                state.queue_nonstreamable.append(type(qc.command).__name__)
        state.action_next = (
            state.queue_nonstreamable[0] if state.queue_nonstreamable else ""
        )

    def _make_command_context(self, addr: tuple[str, int] | None) -> CommandContext:
        """Create a CommandContext for command execution."""
        return CommandContext(
            udp_transport=self._get_udp_transport(),
            addr=addr,
            gcode_interpreter=self._gcode_interpreter,
            dt=self._dt,
        )

    def queue_command(
        self,
        address: tuple[str, int] | None,
        command: CommandBase,
        command_id: str | None = None,
    ) -> ExecutionStatus:
        """Add a command to the execution queue.

        Args:
            address: Optional (ip, port) for acknowledgments.
            command: The command to queue.
            command_id: Optional ID for tracking.

        Returns:
            ExecutionStatus indicating queue status.
        """
        # Check if queue is full
        if len(self.command_queue) >= 100:
            logger.warning("Command queue full (max 100)")
            if command_id and address:
                self._send_ack(command_id, "FAILED", "Queue full", address)
            return ExecutionStatus.failed("Queue full")

        # Create queued command
        queued_cmd = QueuedCommand(
            command=command, command_id=command_id, address=address
        )

        # Bind dynamic context so the command can reply/inspect interpreter when executed
        command.bind(self._make_command_context(address))

        self.command_queue.append(queued_cmd)

        # Update queue snapshot
        state = self._state_manager.get_state()
        self._update_queue_state(state)

        # Send acknowledgment
        if command_id and address:
            queue_pos = len(self.command_queue)
            self._send_ack(
                command_id, "QUEUED", f"Position {queue_pos} in queue", address
            )

        logger.log(
            TRACE, "Queued command: %s (ID: %s)", type(command).__name__, command_id
        )

        return ExecutionStatus(
            code=ExecutionStatusCode.QUEUED,
            message=f"Command queued at position {len(self.command_queue)}",
        )

    def execute_active_command(self) -> ExecutionStatus | None:
        """Execute one step of the active command from the queue.

        Returns:
            ExecutionStatus of the execution, or None if no active command.
        """
        # Import here to avoid circular import
        from parol6.commands.utility_commands import ResetCommand

        # Check if we need to activate a new command from queue
        if self.active_command is None:
            if not self.command_queue:
                return None
            # Get next command from queue
            self.active_command = self.command_queue.popleft()
            self.active_command.activated = False

        # Execute active command step
        if self.active_command:
            ac = self.active_command
            try:
                state = self._state_manager.get_state()

                # Check if controller is enabled
                if state.enabled:
                    # Perform setup and EXECUTING ACK only once
                    if ac and not getattr(ac, "activated", False):
                        ac.command.setup(state)

                        # Update action tracking
                        state.action_current = type(ac.command).__name__
                        state.action_state = "EXECUTING"

                        # Send executing acknowledgment once
                        if ac.command_id and ac.address:
                            self._send_ack(
                                ac.command_id,
                                "EXECUTING",
                                f"Starting {type(ac.command).__name__}",
                                ac.address,
                            )

                        ac.activated = True
                        logger.log(
                            TRACE,
                            "Activated command: %s (id=%s)",
                            type(ac.command).__name__,
                            ac.command_id,
                        )

                else:
                    # Cancel command due to disabled controller
                    self.cancel_active_command("Controller disabled")
                    return ExecutionStatus(
                        code=ExecutionStatusCode.CANCELLED,
                        message="Controller disabled",
                    )

                # Execute command step
                if not getattr(ac, "first_tick_logged", False):
                    logger.log(TRACE, "tick_start name=%s", type(ac.command).__name__)
                    ac.first_tick_logged = True
                status = ac.command.tick(state)

                # Enqueue any generated commands (e.g., from GCODE parsed in queued mode)
                if (
                    status.details
                    and isinstance(status.details, dict)
                    and "enqueue" in status.details
                ):
                    try:
                        for robot_cmd_str in status.details["enqueue"]:
                            cmd_obj, _ = create_command_from_parts(
                                robot_cmd_str.split("|")
                            )
                            if cmd_obj:
                                # Queue without address/id for generated commands
                                self.queue_command(("127.0.0.1", 0), cmd_obj, None)
                    except Exception as e:
                        logger.error(f"Error enqueuing generated commands: {e}")

                # Check if command is finished
                if status.code == ExecutionStatusCode.COMPLETED:
                    name = type(ac.command).__name__
                    cid, addr = ac.command_id, ac.address
                    logger.log(
                        TRACE,
                        "Command completed: %s (id=%s) at t=%f",
                        name,
                        cid,
                        time.time(),
                    )

                    # Send completion acknowledgment
                    if cid and addr:
                        self._send_ack(cid, "COMPLETED", status.message, addr)

                    # Update action tracking to idle
                    state.action_current = ""
                    state.action_state = "IDLE"
                    self._update_queue_state(state)

                    # Sync mock transport after RESET to ensure position consistency
                    if isinstance(ac.command, ResetCommand):
                        self._sync_mock_from_state(state)

                    self.active_command = None

                elif status.code == ExecutionStatusCode.FAILED:
                    name = type(ac.command).__name__
                    cid, addr = ac.command_id, ac.address
                    logger.debug(
                        f"Command failed: {name} (id={cid}) - {status.message} at t={time.time():.6f}"
                    )

                    # Send failure acknowledgment
                    if cid and addr:
                        self._send_ack(cid, "FAILED", status.message, addr)

                    # Update action tracking to idle
                    state.action_current = ""
                    state.action_state = "IDLE"

                    # Clear queued streamable commands on failure to prevent pileup
                    if isinstance(ac.command, MotionCommand) and getattr(
                        ac.command, "streamable", False
                    ):
                        removed = self.clear_streamable_commands(
                            f"Active streamable command failed: {status.message}"
                        )
                        if removed > 0:
                            logger.info(
                                f"Cleared {removed} queued streamable commands due to active command failure"
                            )

                    self._update_queue_state(state)
                    self.active_command = None

                return status

            except Exception as e:
                logger.error(f"Command execution error: {e}")

                cid = ac.command_id if ac else None
                addr = ac.address if ac else None

                if cid and addr:
                    self._send_ack(cid, "FAILED", f"Execution error: {e!s}", addr)
                self.active_command = None

                return ExecutionStatus.failed(f"Execution error: {e!s}", error=e)

        return None

    def cancel_active_command(self, reason: str = "Cancelled by user") -> None:
        """Cancel the currently active command.

        Args:
            reason: Reason for cancellation.
        """
        if not self.active_command:
            return

        logger.info(
            f"Cancelling active command: {type(self.active_command.command).__name__} - {reason}"
        )

        # Send cancellation acknowledgment
        if self.active_command.command_id and self.active_command.address:
            self._send_ack(
                self.active_command.command_id,
                "CANCELLED",
                reason,
                self.active_command.address,
            )

        # Update action tracking to idle
        state = self._state_manager.get_state()
        state.action_current = ""
        state.action_state = "IDLE"

        self.active_command = None

    def clear_queue(
        self, reason: str = "Queue cleared"
    ) -> list[tuple[str, ExecutionStatus]]:
        """Clear all queued commands.

        Args:
            reason: Reason for clearing the queue.

        Returns:
            List of (command_id, status) for cleared commands.
        """
        cleared = []
        while self.command_queue:
            queued_cmd = self.command_queue.popleft()

            # Send cancellation acknowledgment
            if queued_cmd.command_id and queued_cmd.address:
                self._send_ack(
                    queued_cmd.command_id, "CANCELLED", reason, queued_cmd.address
                )

            # Record cleared command
            if queued_cmd.command_id:
                status = ExecutionStatus(
                    code=ExecutionStatusCode.CANCELLED, message=reason
                )
                cleared.append((queued_cmd.command_id, status))

        logger.info(f"Cleared {len(cleared)} commands from queue: {reason}")

        # Update action tracking
        state = self._state_manager.get_state()
        state.queue_nonstreamable = []
        state.action_next = ""

        return cleared

    def clear_streamable_commands(
        self, reason: str = "Streamable commands cleared"
    ) -> int:
        """Clear all queued streamable motion commands.

        Args:
            reason: Reason for clearing streamable commands.

        Returns:
            Number of commands cleared.
        """
        removed_count = 0

        for queued_cmd in list(self.command_queue):
            if isinstance(queued_cmd.command, MotionCommand) and getattr(
                queued_cmd.command, "streamable", False
            ):
                self.command_queue.remove(queued_cmd)
                removed_count += 1

                if queued_cmd.command_id and queued_cmd.address:
                    self._send_ack(
                        queued_cmd.command_id, "CANCELLED", reason, queued_cmd.address
                    )

        if removed_count > 0:
            logger.debug(
                f"Cleared {removed_count} streamable commands from queue: {reason}"
            )

        return removed_count

    def fetch_gcode_commands(self) -> None:
        """Fetch next command from GCODE interpreter if program is running."""
        if not self._gcode_interpreter.is_running:
            return

        try:
            next_gcode_cmd = self._gcode_interpreter.get_next_command()
            if not next_gcode_cmd:
                return

            command_obj, _ = create_command_from_parts(next_gcode_cmd.split("|"))

            if command_obj:
                self.queue_command(("127.0.0.1", 0), command_obj, None)
                cmd_name = (
                    next_gcode_cmd.split("|")[0]
                    if "|" in next_gcode_cmd
                    else next_gcode_cmd
                )
                logger.debug(f"Queued GCODE command: {cmd_name}")
            else:
                logger.warning(f"Unknown GCODE command generated: {next_gcode_cmd}")

        except Exception as e:
            logger.error(f"Error fetching GCODE commands: {e}")
