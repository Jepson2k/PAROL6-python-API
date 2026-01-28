"""
Base abstractions and helpers for command implementations.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, ClassVar

import numpy as np

from parol6.config import INTERVAL_S, LIMITS, TRACE
from parol6.protocol.wire import (
    CmdType,
    Command,
    QueryType,
    pack_error,
    pack_response,
)
from parol6.protocol.wire import CommandCode
from parol6.server.state import ControllerState

logger = logging.getLogger(__name__)


class ExecutionStatusCode(Enum):
    """Enumeration for command execution status codes."""

    QUEUED = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class ExecutionStatus:
    """
    Status returned from command execution steps.
    """

    code: ExecutionStatusCode
    message: str
    error: Exception | None = None
    details: dict[str, Any] | None = None
    error_type: str | None = None

    @classmethod
    def executing(
        cls, message: str = "Executing", details: dict[str, Any] | None = None
    ) -> "ExecutionStatus":
        return cls(ExecutionStatusCode.EXECUTING, message, error=None, details=details)

    @classmethod
    def completed(
        cls, message: str = "Completed", details: dict[str, Any] | None = None
    ) -> "ExecutionStatus":
        return cls(ExecutionStatusCode.COMPLETED, message, error=None, details=details)

    @classmethod
    def failed(
        cls,
        message: str,
        error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> "ExecutionStatus":
        et = type(error).__name__ if error is not None else None
        return cls(
            ExecutionStatusCode.FAILED,
            message,
            error=error,
            details=details,
            error_type=et,
        )


# ----- Shared context and small utilities -----


@dataclass
class CommandContext:
    """Shared dynamic execution context for commands."""

    udp_transport: Any = None
    addr: tuple | None = None
    gcode_interpreter: Any = None
    dt: float = INTERVAL_S


class Countdown:
    """Simple count-down timer."""

    def __init__(self, count: int):
        self.count = max(0, int(count))

    def tick(self) -> bool:
        """Decrement and return True when reaches zero."""
        if self.count > 0:
            self.count -= 1
        return self.count == 0


class Debouncer:
    """Simple count-based debouncer."""

    def __init__(self, count: int = 5) -> None:
        self.count_init = max(0, int(count))
        self.count = self.count_init

    def reset(self) -> None:
        self.count = self.count_init

    def tick(self, active: bool) -> bool:
        """
        Returns True exactly once when 'active' stays non-zero for 'count_init' ticks.
        Resets when 'active' becomes False.
        """
        if active:
            if self.count > 0:
                self.count -= 1
            return self.count == 0
        else:
            self.reset()
            return False


class CommandBase(ABC):
    """
    Reusable base for commands with shared lifecycle and safety helpers.

    Commands use typed msgspec structs for parameters. The PARAMS_TYPE class
    variable indicates which struct type this command expects. The validate()
    method receives a pre-validated struct and performs business logic validation.
    """

    # Set by @register_command decorator; used by controller stream fast-path
    _cmd_type: ClassVar[CmdType | None] = None

    # The params struct type this command expects (override in subclass)
    PARAMS_TYPE: ClassVar[type[Command] | None] = None

    __slots__ = (
        "p",
        "is_valid",
        "is_finished",
        "error_state",
        "error_message",
        "udp_transport",
        "addr",
        "gcode_interpreter",
        "_t0",
        "_t_end",
        "_q_rad_buf",
        "_steps_buf",
    )

    def __init__(self) -> None:
        self.p: Command | None = None  # Params struct, set by validate()
        self.is_valid: bool = True
        self.is_finished: bool = False
        self.error_state: bool = False
        self.error_message: str = ""
        self.udp_transport: Any = None
        self.addr: Any = None
        self.gcode_interpreter: Any = None
        self._t0: float | None = None
        self._t_end: float | None = None
        # Pre-allocated buffers for zero-allocation unit conversions
        self._q_rad_buf: np.ndarray = np.zeros(6, dtype=np.float64)
        self._steps_buf: np.ndarray = np.zeros(6, dtype=np.int32)

    # Ensure command objects are usable as dict keys (e.g., in server command_id_map)
    def __hash__(self) -> int:
        # Identity-based hash is appropriate for ephemeral command instances
        return id(self)

    @property
    def name(self) -> str:
        return self._cmd_type.name if self._cmd_type else type(self).__name__

    # Logging helpers (uniform, include command identity)
    def log_trace(self, msg: str, *args: Any) -> None:
        logger.log(TRACE, "[%s] " + msg, self.name, *args)

    def log_debug(self, msg: str, *args: Any) -> None:
        logger.debug("[%s] " + msg, self.name, *args)

    def log_info(self, msg: str, *args: Any) -> None:
        logger.info("[%s] " + msg, self.name, *args)

    def log_warning(self, msg: str, *args: Any) -> None:
        logger.warning("[%s] " + msg, self.name, *args)

    def log_error(self, msg: str, *args: Any) -> None:
        logger.error("[%s] " + msg, self.name, *args)

    @staticmethod
    def stop_and_idle(state: ControllerState) -> None:
        state.Speed_out.fill(0)
        state.Command_out = CommandCode.IDLE

    def bind(self, context: CommandContext) -> None:
        """
        Bind dynamic execution context. Controller should call this prior to setup().
        """
        self.udp_transport = context.udp_transport
        self.addr = context.addr
        self.gcode_interpreter = context.gcode_interpreter

    def assign_params(self, params: Command) -> None:
        """
        Assign pre-validated params struct.

        Called AFTER msgspec has decoded and validated the struct
        (via constraints and __post_init__). No validation here.

        Args:
            params: Pre-validated typed struct from msgspec decode
        """
        self.p = params
        self.is_valid = True

    def do_setup(self, state: ControllerState) -> None:
        """Subclass hook for preparation; override in subclasses."""
        return

    def setup(self, state: ControllerState) -> None:
        """Public setup wrapper providing centralized logging and error handling."""
        self.log_trace("setup start")
        try:
            self.do_setup(state)
            self.log_trace("setup ok")
        except Exception as e:
            # Mark invalid and propagate for higher-level lifecycle logging
            self.fail(f"Setup error: {e}")
            self.log_error("Setup error: %s", e)
            raise

    @abstractmethod
    def tick(self, state: ControllerState) -> ExecutionStatus:
        """
        Template-method wrapper that centralizes lifecycle/error handling and calls do_execute().
        Controllers should prefer tick() over calling execute_step() directly.
        """
        raise NotImplementedError

    @abstractmethod
    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """
        Execute one control-loop step and return an ExecutionStatus.

        Commands MUST interact with state.* arrays/buffers directly (Position_in/out, Speed_out, Command_out, etc.).
        """
        raise NotImplementedError

    # ----- lifecycle helpers -----

    def finish(self) -> None:
        """Mark command as finished."""
        self.is_finished = True

    def fail(self, message: str) -> None:
        """Mark command as invalid/failed with an error message."""
        self.is_valid = False
        self.error_state = True
        self.error_message = message
        self.is_finished = True

    # ---- timing helpers ----
    def start_timer(self, duration_s: float) -> None:
        """Start a timer for the given duration in seconds."""
        self._t_end = time.perf_counter() + max(0.0, duration_s)

    def timer_expired(self) -> bool:
        """Check if the timer has expired."""
        return self._t_end is not None and time.perf_counter() >= self._t_end

    def progress01(self, duration_s: float) -> float:
        """Get progress as a value between 0 and 1."""
        if self._t0 is None:
            self._t0 = time.perf_counter()
        if duration_s <= 0.0:
            return 1.0
        p = (time.perf_counter() - self._t0) / duration_s
        return 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)


class QueryCommand(CommandBase):
    """
    Base class for query commands that execute immediately and bypass the queue.

    Query commands are read-only operations that return information about the robot state.
    They execute immediately without waiting in the command queue.
    """

    def reply(self, query_type: QueryType, value: Any) -> None:
        """Send a query response: [RESPONSE, query_type, value]."""
        if self.udp_transport and self.addr:
            try:
                self.udp_transport.send(pack_response(query_type, value), self.addr)
            except Exception as e:
                self.log_warning("Failed to send reply: %s", e)

    def reply_error(self, message: str) -> None:
        """Send an error response: [ERROR, message]."""
        if self.udp_transport and self.addr:
            try:
                self.udp_transport.send(pack_error(message), self.addr)
            except Exception as e:
                self.log_warning("Failed to send error reply: %s", e)

    def tick(self, state: ControllerState) -> ExecutionStatus:
        """
        Template-method wrapper that centralizes lifecycle/error handling and calls do_execute().
        Controllers should prefer tick() over calling execute_step() directly.
        """
        if self.is_finished or not self.is_valid:
            return (
                ExecutionStatus.completed("Already finished")
                if self.is_finished
                else ExecutionStatus.failed("Invalid command")
            )
        if not self.udp_transport or not self.addr:
            self.fail("Missing UDP transport or address")
            return ExecutionStatus.failed("Missing UDP transport or address")
        try:
            status = self.execute_step(state)
        except Exception as e:
            # Hard failure safeguards
            self.fail(str(e))
            return ExecutionStatus.failed("Execution error", error=e)
        return status


class MotionCommand(CommandBase):
    """
    Base class for motion commands that require the controller to be enabled.

    Motion commands involve robot movement and require the controller to be in an enabled state.
    Some motion commands (like jog commands) can be replaced in stream mode.
    """

    streamable: bool = False

    def __init__(self) -> None:
        super().__init__()

    # ---- mapping ----
    @staticmethod
    def linmap_pct(pct: float, lo: float, hi: float) -> float:
        if pct < 0.0:
            pct = 0.0
        elif pct > 100.0:
            pct = 100.0
        return lo + (hi - lo) * (pct / 100.0)

    @staticmethod
    def limit_hit_mask(pos_steps: np.ndarray, speeds: np.ndarray) -> np.ndarray:
        return ((speeds > 0) & (pos_steps >= LIMITS.joint.position.steps[:, 1])) | (
            (speeds < 0) & (pos_steps <= LIMITS.joint.position.steps[:, 0])
        )

    def fail_and_idle(self, state: ControllerState, message: str) -> None:
        self.fail(message)
        self.stop_and_idle(state)

    def set_move_position(self, state: ControllerState, steps: np.ndarray) -> None:
        """Set position for MOVE command (zero speeds, Command=MOVE)."""
        np.copyto(state.Position_out, steps, casting="no")
        state.Speed_out.fill(0)
        state.Command_out = CommandCode.MOVE

    def tick(self, state: ControllerState) -> ExecutionStatus:
        """
        Template-method wrapper that centralizes lifecycle/error handling and calls do_execute().
        Controllers should prefer tick() over calling execute_step() directly.
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
            # Hard failure safeguards
            self.fail_and_idle(state, str(e))
            self.log_error(str(e))
            return ExecutionStatus.failed("Execution error", error=e)
        return status


class TrajectoryMoveCommandBase(MotionCommand):
    """
    Base class for commands that execute pre-computed trajectories.

    Subclasses pre-compute trajectory_steps in do_setup(). Velocity/acceleration
    limits are enforced during trajectory building via local segment slowdown,
    so execute_step() simply outputs waypoints tick-by-tick.
    """

    __slots__ = ("trajectory_steps", "command_step")

    def __init__(self):
        super().__init__()
        self.trajectory_steps: np.ndarray = np.empty((0, 6), dtype=np.int32)
        self.command_step = 0

    def execute_step(self, state: ControllerState) -> ExecutionStatus:
        """Execute trajectory by outputting pre-computed waypoints."""
        if self.command_step >= len(self.trajectory_steps):
            self.log_info("%s finished.", self.__class__.__name__)
            self.is_finished = True
            self.stop_and_idle(state)
            return ExecutionStatus.completed(f"{self.__class__.__name__} complete")

        target = self.trajectory_steps[self.command_step]
        np.copyto(state.Position_out, target)
        state.Command_out = CommandCode.MOVE
        self.command_step += 1

        return ExecutionStatus.executing(self.__class__.__name__)


class SystemCommand(CommandBase):
    """
    Base class for system control commands that can execute regardless of enable state.

    System commands control the overall state of the robot controller (enable/disable, stop, etc.)
    and can execute even when the controller is disabled.
    """

    def tick(self, state: "ControllerState") -> ExecutionStatus:
        """
        Centralized lifecycle/error handling for system commands.
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
            self.fail(str(e))
            self.log_error(str(e))
            return ExecutionStatus.failed("Execution error", error=e)
        return status
