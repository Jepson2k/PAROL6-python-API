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
from parol6.config import MAX_COMMAND_QUEUE_SIZE, TRACE
from parol6.protocol.wire import decode_command
from parol6.server.command_registry import create_command_from_struct

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
        sync_mock_from_state: Callable[["ControllerState"], None],
        dt: float,
    ):
        """Initialize the command executor.

        Args:
            state_manager: StateManager for accessing controller state.
            gcode_interpreter: GCODE interpreter for fetching commands.
            udp_transport_getter: Callable that returns current UDP transport.
            sync_mock_from_state: Callback to sync mock transport after RESET.
            dt: Loop interval for command context.
        """
        self._state_manager = state_manager
        self._gcode_interpreter = gcode_interpreter
        self._get_udp_transport = udp_transport_getter
        self._sync_mock_from_state = sync_mock_from_state
        self._dt = dt

        self.command_queue: deque[QueuedCommand] = deque(maxlen=MAX_COMMAND_QUEUE_SIZE)
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

    def execute_immediate(
        self,
        command: CommandBase,
        state: "ControllerState",
        addr: tuple[str, int],
    ) -> ExecutionStatus:
        """Execute a command immediately (system or query).

        Args:
            command: The command to execute.
            state: Controller state.
            addr: Client address for context.

        Returns:
            ExecutionStatus from the command.
        """
        command.bind(self._make_command_context(addr))
        return command.execute_step(state)

    def try_stream_fast_path(
        self,
        data: bytes,
        state: "ControllerState",
        addr: tuple[str, int],
    ) -> bool:
        """Attempt stream fast-path for active streamable command.

        When in stream mode with an active streamable command, this allows
        updating the command's parameters without full command creation/queueing.

        Args:
            data: Raw msgpack-encoded command bytes.
            state: Controller state.
            addr: Client address for context binding.

        Returns:
            True if command was handled via fast-path, False otherwise.
        """
        if not (state.stream_mode and self.active_command):
            return False

        active_inst = self.active_command.command
        if not (isinstance(active_inst, MotionCommand) and active_inst.streamable):
            return False

        # Decode incoming command
        try:
            cmd_struct = decode_command(data)
        except Exception as e:
            logger.debug("Stream fast-path decode failed: %s", e)
            return False

        # Check if struct type matches active command's expected type
        active_params_type = getattr(active_inst, "PARAMS_TYPE", None)
        if active_params_type is None or type(cmd_struct) is not active_params_type:
            return False

        logger.log(
            TRACE,
            "stream_fast_path active=%s incoming=%s",
            type(active_inst).__name__,
            type(cmd_struct).__name__,
        )

        # Assign new params (validation already done by decode)
        active_inst.assign_params(cmd_struct)

        # Re-setup with new params
        try:
            active_inst.bind(self._make_command_context(addr))
            active_inst.setup(state)
            logger.log(TRACE, "stream_fast_path applied")
            return True
        except Exception as e:
            logger.error("Stream fast-path setup failed: %s", e)
            return False

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
        if len(self.command_queue) >= MAX_COMMAND_QUEUE_SIZE:
            logger.warning("Command queue full (max %d)", MAX_COMMAND_QUEUE_SIZE)
            return ExecutionStatus.failed("Queue full")

        # Create queued command
        queued_cmd = QueuedCommand(
            command=command, command_id=command_id, address=address
        )

        # Bind dynamic context so the command can reply/inspect interpreter when executed
        command.bind(self._make_command_context(address))

        self.command_queue.append(queued_cmd)

        # Update queue snapshot
        self._update_queue_state(self._state_manager.get_state())

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
                    # Perform setup only once
                    if not ac.activated:
                        ac.command.setup(state)

                        # Update action tracking
                        state.action_current = type(ac.command).__name__
                        state.action_state = "EXECUTING"

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
                        for cmd_struct in status.details["enqueue"]:
                            cmd_obj, _, err = create_command_from_struct(cmd_struct)
                            if cmd_obj:
                                # Queue without address/id for generated commands
                                self.queue_command(("127.0.0.1", 0), cmd_obj, None)
                            else:
                                logger.warning(
                                    "Failed to create command from struct: %s", err
                                )
                    except Exception as e:
                        logger.error("Error enqueuing generated commands: %s", e)

                # Check if command is finished
                if status.code == ExecutionStatusCode.COMPLETED:
                    logger.log(
                        TRACE,
                        "Command completed: %s (id=%s) at t=%f",
                        type(ac.command).__name__,
                        ac.command_id,
                        time.time(),
                    )

                    # Update action tracking to idle
                    state.action_current = ""
                    state.action_state = "IDLE"
                    self._update_queue_state(state)

                    # Sync mock transport after RESET to ensure position consistency
                    if isinstance(ac.command, ResetCommand):
                        self._sync_mock_from_state(state)

                    self.active_command = None

                elif status.code == ExecutionStatusCode.FAILED:
                    logger.debug(
                        "Command failed: %s (id=%s) - %s at t=%.6f",
                        type(ac.command).__name__,
                        ac.command_id,
                        status.message,
                        time.time(),
                    )

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
                                "Cleared %d queued streamable commands due to active command failure",
                                removed,
                            )

                    self._update_queue_state(state)
                    self.active_command = None

                return status

            except Exception as e:
                logger.error("Command execution error: %s", e)
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
            "Cancelling active command: %s - %s",
            type(self.active_command.command).__name__,
            reason,
        )

        # Update action tracking to idle
        state = self._state_manager.get_state()
        state.action_current = ""
        state.action_state = "IDLE"

        self.active_command = None

    def cancel_active_streamable(self) -> bool:
        """Cancel active command if it's a streamable motion command.

        Returns:
            True if a command was cancelled.
        """
        ac = self.active_command
        if (
            ac
            and isinstance(ac.command, MotionCommand)
            and getattr(ac.command, "streamable", False)
        ):
            self.active_command = None
            return True
        return False

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
            if queued_cmd.command_id:
                status = ExecutionStatus(
                    code=ExecutionStatusCode.CANCELLED, message=reason
                )
                cleared.append((queued_cmd.command_id, status))

        logger.info("Cleared %d commands from queue: %s", len(cleared), reason)

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
        to_remove: list[QueuedCommand] = []

        # First pass: identify commands to remove (no mutation during iteration)
        for queued_cmd in self.command_queue:
            if isinstance(queued_cmd.command, MotionCommand) and getattr(
                queued_cmd.command, "streamable", False
            ):
                to_remove.append(queued_cmd)

        # Second pass: remove
        for queued_cmd in to_remove:
            self.command_queue.remove(queued_cmd)
            removed_count += 1

        if removed_count > 0:
            logger.debug(
                "Cleared %d streamable commands from queue: %s", removed_count, reason
            )

        return removed_count

    def fetch_gcode_commands(self) -> None:
        """Fetch next command from GCODE interpreter if program is running."""
        if not self._gcode_interpreter.is_running:
            return

        try:
            cmd_struct = self._gcode_interpreter.get_next_command()
            if cmd_struct is None:
                return

            # Create command directly from the struct
            command_obj, _, err = create_command_from_struct(cmd_struct)

            if command_obj:
                self.queue_command(("127.0.0.1", 0), command_obj, None)
                logger.debug("Queued GCODE command: %s", type(cmd_struct).__name__)
            else:
                logger.warning(
                    "Unknown GCODE command: %s - %s", type(cmd_struct).__name__, err
                )

        except Exception as e:
            logger.error("Error fetching GCODE commands: %s", e)
