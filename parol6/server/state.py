from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from parol6.motion import CartesianStreamingExecutor, StreamingExecutor

import numpy as np
import sophuspy as sp

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.config import steps_to_rad
from parol6.utils.se3_utils import se3_from_matrix
from parol6.protocol.wire import CommandCode


@dataclass
class GripperModeResetTracker:
    """Tracks gripper mode for auto-reset functionality."""

    calibration_sent: bool = False  # Flag for calibration mode
    error_clear_sent: bool = False  # Flag for error clear mode


@dataclass
class ControllerState:
    """
    Centralized mutable state for the headless controller.

    Buffers use preallocated NumPy ndarrays for zero-copy, in-place operations.
    """

    # Serial/transport
    ser: Any = None
    last_reconnect_attempt: float = 0.0

    # Safety and control flags
    enabled: bool = True
    soft_error: bool = False
    disabled_reason: str = ""
    e_stop_active: bool = False
    stream_mode: bool = False

    # Motion profiles - separate for joint and Cartesian moves
    # Joint: TOPPRA, RUCKIG, QUINTIC, TRAPEZOID, SCURVE, LINEAR
    # Cartesian: TOPPRA, LINEAR only
    joint_motion_profile: str = "TOPPRA"
    cartesian_motion_profile: str = "TOPPRA"

    # Streaming executors for online motion (jogging/streaming)
    streaming_executor: "StreamingExecutor | None" = None  # Joint-space Ruckig
    cartesian_streaming_executor: "CartesianStreamingExecutor | None" = None  # Cartesian Ruckig

    # Tool configuration (affects kinematics and visualization)
    _current_tool: str = "NONE"

    # I/O buffers and protocol tracking (serial frame parsing state)
    input_byte: int = 0
    start_cond1: int = 0
    start_cond2: int = 0
    start_cond3: int = 0
    good_start: int = 0
    data_len: int = 0
    data_buffer: list[bytes] = field(default_factory=lambda: [b""] * 255)
    data_counter: int = 0

    # Robot telemetry and command buffers - using ndarray for efficiency
    Command_out: CommandCode = CommandCode.IDLE  # The command code to send to firmware

    # int32 joint buffers
    Position_out: np.ndarray = field(
        default_factory=lambda: np.zeros((6,), dtype=np.int32)
    )
    Speed_out: np.ndarray = field(
        default_factory=lambda: np.zeros((6,), dtype=np.int32)
    )
    Gripper_data_out: np.ndarray = field(
        default_factory=lambda: np.zeros((6,), dtype=np.int32)
    )

    Position_in: np.ndarray = field(
        default_factory=lambda: np.zeros((6,), dtype=np.int32)
    )
    Speed_in: np.ndarray = field(default_factory=lambda: np.zeros((6,), dtype=np.int32))
    Timing_data_in: np.ndarray = field(
        default_factory=lambda: np.zeros((1,), dtype=np.int32)
    )
    Gripper_data_in: np.ndarray = field(
        default_factory=lambda: np.zeros((6,), dtype=np.int32)
    )

    # uint8 flag/bitfield buffers
    Affected_joint_out: np.ndarray = field(
        default_factory=lambda: np.zeros((8,), dtype=np.uint8)
    )
    InOut_out: np.ndarray = field(
        default_factory=lambda: np.zeros((8,), dtype=np.uint8)
    )
    InOut_in: np.ndarray = field(default_factory=lambda: np.zeros((8,), dtype=np.uint8))
    Homed_in: np.ndarray = field(default_factory=lambda: np.zeros((8,), dtype=np.uint8))
    Temperature_error_in: np.ndarray = field(
        default_factory=lambda: np.zeros((8,), dtype=np.uint8)
    )
    Position_error_in: np.ndarray = field(
        default_factory=lambda: np.zeros((8,), dtype=np.uint8)
    )

    Timeout_out: int = 0
    XTR_data: int = 0

    # Command queueing and tracking
    command_queue: deque[Any] = field(default_factory=deque)
    incoming_command_buffer: deque[tuple[str, tuple[str, int]]] = field(
        default_factory=deque
    )
    command_id_map: dict[Any, tuple[str, tuple[str, int]]] = field(default_factory=dict)
    active_command: Any = None
    active_command_id: str | None = None
    last_command_time: float = 0.0

    # Action tracking for status broadcast and queries
    action_current: str = ""
    action_state: str = "IDLE"  # IDLE, EXECUTING, COMPLETED, FAILED
    action_next: str = ""
    queue_nonstreamable: list[str] = field(default_factory=list)

    # Network setup and uptime
    ip: str = "127.0.0.1"
    port: int = 5001
    start_time: float = 0.0

    gripper_mode_tracker: GripperModeResetTracker = field(
        default_factory=GripperModeResetTracker
    )

    # Control loop runtime metrics (used by benchmarks/monitoring)
    loop_count: int = 0
    overrun_count: int = 0
    last_period_s: float = 0.0
    ema_period_s: float = 0.0

    # Command frequency metrics
    command_count: int = 0
    last_command_period_s: float = 0.0
    ema_command_period_s: float = 0.0
    command_timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=10))

    # Forward kinematics cache (invalidated when Position_in or current_tool changes)
    _fkine_last_pos_in: np.ndarray = field(
        default_factory=lambda: np.zeros((6,), dtype=np.int32)
    )
    _fkine_last_tool: str = ""
    _fkine_se3: Any = None  # sophuspy SE3 instance
    _fkine_mat: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    _fkine_flat_mm: np.ndarray = field(
        default_factory=lambda: np.zeros((16,), dtype=np.float64)
    )

    def reset(self) -> None:
        """
        Reset robot state to initial values without losing connection state.

        Preserves: ser, ip, port, start_time
        Resets: positions, speeds, I/O, queues, tool, errors, etc.
        """
        # Safety and control flags
        self.enabled = True
        self.soft_error = False
        self.disabled_reason = ""
        self.e_stop_active = False
        self.stream_mode = False
        self.joint_motion_profile = "TOPPRA"
        self.cartesian_motion_profile = "TOPPRA"

        # Tool back to none
        self._current_tool = "NONE"
        PAROL6_ROBOT.apply_tool("NONE")

        # Serial frame parsing state
        self.input_byte = 0
        self.start_cond1 = 0
        self.start_cond2 = 0
        self.start_cond3 = 0
        self.good_start = 0
        self.data_len = 0
        self.data_buffer = [b""] * 255
        self.data_counter = 0

        # Command and telemetry buffers - zero out
        self.Command_out = CommandCode.IDLE
        self.Position_out.fill(0)
        self.Speed_out.fill(0)
        self.Gripper_data_out.fill(0)
        self.Position_in.fill(0)
        self.Speed_in.fill(0)
        self.Timing_data_in.fill(0)
        self.Gripper_data_in.fill(0)
        self.Affected_joint_out.fill(0)
        self.InOut_out.fill(0)
        self.InOut_in.fill(0)
        self.InOut_in[4] = 1  # E-STOP released (0=pressed, 1=released)
        self.Homed_in.fill(0)
        self.Temperature_error_in.fill(0)
        self.Position_error_in.fill(0)
        self.Timeout_out = 0
        self.XTR_data = 0

        # Command queues - clear
        self.command_queue.clear()
        self.incoming_command_buffer.clear()
        self.command_id_map.clear()
        self.active_command = None
        self.active_command_id = None
        self.last_command_time = 0.0

        # Action tracking
        self.action_current = ""
        self.action_state = "IDLE"
        self.action_next = ""
        self.queue_nonstreamable.clear()

        # Gripper mode tracker
        self.gripper_mode_tracker = GripperModeResetTracker()

        # Metrics (keep loop_count for uptime, reset command metrics)
        self.command_count = 0
        self.last_command_period_s = 0.0
        self.ema_command_period_s = 0.0
        self.command_timestamps.clear()

        # Invalidate fkine cache
        self._fkine_se3 = None
        self._fkine_last_pos_in.fill(0)
        self._fkine_last_tool = ""

        logger.debug("Controller state reset (preserving connection)")

    @property
    def current_tool(self) -> str:
        """Get the current tool name."""
        return self._current_tool

    @current_tool.setter
    def current_tool(self, tool_name: str) -> None:
        """Set the current tool and apply it to the robot model."""
        if tool_name != self._current_tool:
            self._current_tool = tool_name
            # Apply tool to robot model (rebuilds with tool as final link)
            PAROL6_ROBOT.apply_tool(tool_name)
            # Invalidate cache
            self._fkine_se3 = None
            logger.info(f"Tool changed to {tool_name}, fkine cache invalidated")


logger = logging.getLogger(__name__)


class StateManager:
    """
    Singleton manager for ControllerState with thread-safe operations.

    This class ensures that all state access is synchronized and provides
    convenience methods for common state operations.
    """

    _instance: StateManager | None = None
    _lock: threading.Lock = threading.Lock()
    _state: ControllerState | None = None

    def __new__(cls) -> StateManager:
        """Ensure singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the state manager (only runs once due to singleton)."""
        if not hasattr(self, "_initialized"):
            self._state = ControllerState()
            self._state_lock = threading.RLock()  # Use RLock for re-entrant locking
            self._initialized = True
            self._init_streaming_executor()
            logger.info("StateManager initialized with NumPy buffers")

    def _init_streaming_executor(self) -> None:
        """Initialize the streaming executors for jogging/streaming."""
        from parol6.config import CONTROL_RATE_HZ
        from parol6.motion import CartesianStreamingExecutor, StreamingExecutor

        if self._state is not None:
            dt = 1.0 / CONTROL_RATE_HZ
            self._state.streaming_executor = StreamingExecutor(
                num_dofs=6,
                dt=dt,
            )
            self._state.cartesian_streaming_executor = CartesianStreamingExecutor(
                dt=dt,
            )

    def get_state(self) -> ControllerState:
        """
        Get the current controller state.

        Note: This returns the actual state object. Modifications to it
        should be done through StateManager methods to ensure thread safety.

        Returns:
            The current ControllerState instance
        """
        with self._state_lock:
            if self._state is None:
                self._state = ControllerState()
            return self._state

    def reset_state(self) -> None:
        """
        Reset the controller state to a fresh instance.

        This is useful at controller startup to ensure buffers are initialized
        to known defaults. Callers must ensure they hold appropriate locks in
        higher layers if concurrent access is possible.
        """
        with self._state_lock:
            self._state = ControllerState()
            self._init_streaming_executor()
            logger.info("Controller state reset")


# Global singleton instance accessor
_state_manager: StateManager | None = None


def get_instance() -> StateManager:
    """
    Get the global StateManager instance.
    """
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager


def get_state() -> ControllerState:
    """
    Convenience function to get the current controller state.
    """
    return get_instance().get_state()


# -----------------------------
# Forward kinematics cache management
# -----------------------------


def invalidate_fkine_cache() -> None:
    """
    Invalidate the fkine cache, forcing recomputation on next access.
    Call this when the robot model changes (e.g., tool change).
    """
    state = get_state()
    state._fkine_se3 = None
    state._fkine_last_tool = ""
    logger.debug("fkine cache invalidated")


def ensure_fkine_updated(state: ControllerState) -> None:
    """
    Ensure the fkine cache is up to date with current Position_in and tool.
    If Position_in or current_tool has changed, recalculate fkine and update cache.

    This function is thread-safe when called with state from get_state().

    Parameters
    ----------
    state : ControllerState
        The controller state to update
    """
    # Check if cache is valid
    pos_changed = not np.array_equal(state.Position_in, state._fkine_last_pos_in)
    tool_changed = state.current_tool != state._fkine_last_tool

    if pos_changed or tool_changed or state._fkine_se3 is None:
        # Recompute fkine
        q = steps_to_rad(state.Position_in)
        assert PAROL6_ROBOT.robot is not None
        T_raw = cast(Any, PAROL6_ROBOT.robot).fkine(q)

        # Cache as 4x4 matrix first (fkine returns spatialmath SE3, extract .A)
        mat = np.asarray(T_raw.A, dtype=np.float64).copy()
        np.copyto(state._fkine_mat, mat)

        # Convert to sophuspy SE3 for fast operations
        state._fkine_se3 = se3_from_matrix(mat)

        # Cache as flattened 16-vector with mm translation
        flat = mat.reshape(-1).copy()
        flat[3] *= 1000.0  # X translation to mm
        flat[7] *= 1000.0  # Y translation to mm
        flat[11] *= 1000.0  # Z translation to mm
        np.copyto(state._fkine_flat_mm, flat)

        # Update cache tracking
        np.copyto(state._fkine_last_pos_in, state.Position_in)
        state._fkine_last_tool = state.current_tool


def get_fkine_se3(state: ControllerState | None = None) -> sp.SE3:
    """
    Get the current end-effector pose as a sophuspy SE3 object.
    Automatically updates cache if needed.

    Returns
    -------
    sp.SE3
        Current end-effector pose
    """
    if state is None:
        state = get_state()
    ensure_fkine_updated(state)
    return state._fkine_se3


def get_fkine_matrix(state: ControllerState | None = None) -> np.ndarray:
    """
    Get the current end-effector pose as a 4x4 homogeneous transformation matrix.
    Automatically updates cache if needed.
    Translation is in meters.

    Returns
    -------
    np.ndarray
        4x4 transformation matrix (translation in meters)
    """
    if state is None:
        state = get_state()
    ensure_fkine_updated(state)
    return state._fkine_mat


def get_fkine_flat_mm(state: ControllerState | None = None) -> np.ndarray:
    """
    Get the current end-effector pose as a flattened 16-element array.
    Automatically updates cache if needed.
    Translation components (indices 3, 7, 11) are in millimeters for compatibility.

    Returns
    -------
    np.ndarray
        Flattened 16-element pose array (translation in mm)
    """
    if state is None:
        state = get_state()
    ensure_fkine_updated(state)
    return state._fkine_flat_mm
