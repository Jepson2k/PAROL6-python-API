"""
Wire protocol for PAROL6 robot communication.

This module contains all protocol definitions:
- Binary serial frame packing/unpacking (firmware communication)
- Msgpack message types and structs (UDP communication)
- Command/response encoding and decoding

Wire format uses msgpack arrays with integer type codes:
- OK:       MsgType.OK (just the integer)
- ERROR:    [MsgType.ERROR, message]
- STATUS:   [MsgType.STATUS, pose, angles, speeds, io, gripper, action_current, action_state, joint_en, cart_en_wrf, cart_en_trf]
- RESPONSE: [MsgType.RESPONSE, query_type, value]
- COMMAND:  [CmdType.XXX, ...params]
"""

import logging
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Annotated, Any, NamedTuple, TypeAlias, Union

import msgspec
import numpy as np
import ormsgpack
from numba import njit  # type: ignore[import-untyped]

from parol6.config import LIMITS
from parol6.tools import TOOL_CONFIGS, list_tools

logger = logging.getLogger(__name__)


# =============================================================================
# Numpy msgpack encoding hooks
# =============================================================================


def _enc_hook(obj: object) -> object:
    """Custom encoder hook for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Convert numpy scalar to Python native type
    raise NotImplementedError(f"Cannot encode {type(obj)}")


# Module-level encoder with numpy support (thread-safe, reusable)
_encoder = msgspec.msgpack.Encoder(enc_hook=_enc_hook)

# Module-level decoder for generic msgpack
_decoder = msgspec.msgpack.Decoder()


# =============================================================================
# Message Types
# =============================================================================


class MsgType(IntEnum):
    """Message type codes for responses."""

    OK = auto()
    ERROR = auto()
    STATUS = auto()
    RESPONSE = auto()


class QueryType(IntEnum):
    """Query type codes for responses."""

    PING = auto()
    STATUS = auto()
    ANGLES = auto()
    POSE = auto()
    IO = auto()
    GRIPPER = auto()
    SPEEDS = auto()
    TOOL = auto()
    QUEUE = auto()
    CURRENT_ACTION = auto()
    LOOP_STATS = auto()
    GCODE_STATUS = auto()
    PROFILE = auto()


class CmdType(IntEnum):
    """Command type codes for incoming commands.

    Wire format: [CmdType.XXX, ...params]
    """

    # Query commands (immediate, read-only)
    PING = auto()
    GET_STATUS = auto()
    GET_ANGLES = auto()
    GET_POSE = auto()
    GET_IO = auto()
    GET_GRIPPER = auto()
    GET_SPEEDS = auto()
    GET_TOOL = auto()
    GET_QUEUE = auto()
    GET_CURRENT_ACTION = auto()
    GET_LOOP_STATS = auto()
    GET_GCODE_STATUS = auto()
    GET_PROFILE = auto()

    # System commands (execute regardless of enable state)
    STOP = auto()
    ENABLE = auto()
    DISABLE = auto()
    SET_IO = auto()
    SET_PORT = auto()
    STREAM = auto()
    SIMULATOR = auto()
    SET_PROFILE = auto()
    RESET = auto()
    RESET_LOOP_STATS = auto()

    # Motion commands
    HOME = auto()
    JOG = auto()
    MULTIJOG = auto()
    CARTJOG = auto()
    MOVEJOINT = auto()
    MOVEPOSE = auto()
    MOVECART = auto()
    MOVECARTRELTRF = auto()
    SET_TOOL = auto()
    DELAY = auto()

    # Gripper commands
    PNEUMATICGRIPPER = auto()
    ELECTRICGRIPPER = auto()

    # GCODE commands
    GCODE = auto()
    GCODE_PROGRAM = auto()
    GCODE_STOP = auto()
    GCODE_PAUSE = auto()
    GCODE_RESUME = auto()

    # Smooth motion commands
    SMOOTH_CIRCLE = auto()
    SMOOTH_ARC_CENTER = auto()
    SMOOTH_ARC_PARAM = auto()
    SMOOTH_SPLINE = auto()


# =============================================================================
# Command Structs - Tagged Union for single-pass decode
# Wire format: [CmdType.XXX, ...fields]
# =============================================================================


class JogCmd(msgspec.Struct, tag=int(CmdType.JOG), array_like=True, frozen=True):
    """JOG: [CmdType.JOG, joint, speed_pct, duration, accel_pct]"""

    joint: Annotated[int, msgspec.Meta(ge=0, le=11)]  # 0-5 positive, 6-11 negative
    speed_pct: Annotated[float, msgspec.Meta(gt=0.0, le=100.0)]
    duration: Annotated[float, msgspec.Meta(gt=0.0)]
    accel_pct: Annotated[float, msgspec.Meta(gt=0.0, le=100.0)] = 100.0


class MultiJogCmd(
    msgspec.Struct, tag=int(CmdType.MULTIJOG), array_like=True, frozen=True
):
    """MULTIJOG: [CmdType.MULTIJOG, joints, speeds, duration]"""

    joints: list[int]
    speeds: list[float]
    duration: Annotated[float, msgspec.Meta(gt=0.0)]

    def __post_init__(self) -> None:
        if len(self.joints) != len(self.speeds):
            raise ValueError("Number of joints must match number of speeds")
        # Check for conflicting joint commands
        base: set[int] = set()
        for j in self.joints:
            b = j % 6
            if b in base:
                raise ValueError(f"Conflicting commands for Joint {b + 1}")
            base.add(b)


class CartJogCmd(
    msgspec.Struct, tag=int(CmdType.CARTJOG), array_like=True, frozen=True
):
    """CARTJOG: [CmdType.CARTJOG, frame, axis, speed_pct, duration, accel_pct]"""

    frame: Annotated[str, msgspec.Meta(pattern=r"^(WRF|TRF)$")]
    axis: Annotated[str, msgspec.Meta(pattern=r"^(X|Y|Z|RX|RY|RZ)[+-]$")]
    speed_pct: Annotated[float, msgspec.Meta(gt=0.0, le=100.0)]
    duration: Annotated[float, msgspec.Meta(gt=0.0)]
    accel_pct: Annotated[float, msgspec.Meta(gt=0.0, le=100.0)] = 100.0


class MoveJointCmd(
    msgspec.Struct, tag=int(CmdType.MOVEJOINT), array_like=True, frozen=True
):
    """MOVEJOINT: [CmdType.MOVEJOINT, angles, duration, speed_pct, accel_pct]"""

    angles: Annotated[list[float], msgspec.Meta(min_length=6, max_length=6)]
    duration: Annotated[float, msgspec.Meta(ge=0.0)] = 0.0
    speed_pct: Annotated[float, msgspec.Meta(ge=0.0, le=100.0)] = 0.0
    accel_pct: Annotated[float, msgspec.Meta(gt=0.0, le=100.0)] = 100.0

    def __post_init__(self) -> None:
        has_duration = self.duration > 0.0
        has_speed = self.speed_pct > 0.0
        if not has_duration and not has_speed:
            raise ValueError("MOVEJOINT requires either duration > 0 or speed_pct > 0")
        if has_duration and has_speed:
            raise ValueError("MOVEJOINT requires only one of duration or speed_pct")

        for i, angle_deg in enumerate(self.angles):
            min_rad, max_rad = LIMITS.joint.position.rad[i]
            angle_rad = np.deg2rad(angle_deg)
            if not (min_rad <= angle_rad <= max_rad):
                raise ValueError(
                    f"Joint {i + 1} target ({angle_deg:.1f} deg) is out of range"
                )


class MovePoseCmd(
    msgspec.Struct, tag=int(CmdType.MOVEPOSE), array_like=True, frozen=True
):
    """MOVEPOSE: [CmdType.MOVEPOSE, pose, duration, speed_pct, accel_pct]"""

    pose: Annotated[list[float], msgspec.Meta(min_length=6, max_length=6)]
    duration: Annotated[float, msgspec.Meta(ge=0.0)] = 0.0
    speed_pct: Annotated[float, msgspec.Meta(ge=0.0, le=100.0)] = 0.0
    accel_pct: Annotated[float, msgspec.Meta(gt=0.0, le=100.0)] = 100.0

    def __post_init__(self) -> None:
        has_duration = self.duration > 0.0
        has_speed = self.speed_pct > 0.0
        if not has_duration and not has_speed:
            raise ValueError("MOVEPOSE requires either duration > 0 or speed_pct > 0")
        if has_duration and has_speed:
            raise ValueError("MOVEPOSE requires only one of duration or speed_pct")


class MoveCartCmd(
    msgspec.Struct, tag=int(CmdType.MOVECART), array_like=True, frozen=True
):
    """MOVECART: [CmdType.MOVECART, pose, duration, speed_pct, accel_pct]"""

    pose: Annotated[list[float], msgspec.Meta(min_length=6, max_length=6)]
    duration: Annotated[float, msgspec.Meta(ge=0.0)] = 0.0
    speed_pct: Annotated[float, msgspec.Meta(ge=0.0, le=100.0)] = 0.0
    accel_pct: Annotated[float, msgspec.Meta(gt=0.0, le=100.0)] = 100.0

    def __post_init__(self) -> None:
        has_duration = self.duration > 0.0
        has_speed = self.speed_pct > 0.0
        if not has_duration and not has_speed:
            raise ValueError("MOVECART requires either duration > 0 or speed_pct > 0")
        if has_duration and has_speed:
            raise ValueError("MOVECART requires only one of duration or speed_pct")


class MoveCartRelTrfCmd(
    msgspec.Struct, tag=int(CmdType.MOVECARTRELTRF), array_like=True, frozen=True
):
    """MOVECARTRELTRF: [CmdType.MOVECARTRELTRF, deltas, duration, speed_pct, accel_pct]"""

    deltas: Annotated[list[float], msgspec.Meta(min_length=6, max_length=6)]
    duration: Annotated[float, msgspec.Meta(ge=0.0)] = 0.0
    speed_pct: Annotated[float, msgspec.Meta(ge=0.0, le=100.0)] = 0.0
    accel_pct: Annotated[float, msgspec.Meta(gt=0.0, le=100.0)] = 100.0

    def __post_init__(self) -> None:
        has_duration = self.duration > 0.0
        has_speed = self.speed_pct > 0.0
        if not has_duration and not has_speed:
            raise ValueError(
                "MOVECARTRELTRF requires either duration > 0 or speed_pct > 0"
            )
        if has_duration and has_speed:
            raise ValueError(
                "MOVECARTRELTRF requires only one of duration or speed_pct"
            )


class HomeCmd(msgspec.Struct, tag=int(CmdType.HOME), array_like=True, frozen=True):
    """HOME: [CmdType.HOME]"""

    pass


class StopCmd(msgspec.Struct, tag=int(CmdType.STOP), array_like=True, frozen=True):
    """STOP: [CmdType.STOP]"""

    pass


class EnableCmd(msgspec.Struct, tag=int(CmdType.ENABLE), array_like=True, frozen=True):
    """ENABLE: [CmdType.ENABLE]"""

    pass


class DisableCmd(
    msgspec.Struct, tag=int(CmdType.DISABLE), array_like=True, frozen=True
):
    """DISABLE: [CmdType.DISABLE]"""

    pass


class ResetCmd(msgspec.Struct, tag=int(CmdType.RESET), array_like=True, frozen=True):
    """RESET: [CmdType.RESET]"""

    pass


class ResetLoopStatsCmd(
    msgspec.Struct, tag=int(CmdType.RESET_LOOP_STATS), array_like=True, frozen=True
):
    """RESET_LOOP_STATS: [CmdType.RESET_LOOP_STATS]

    Reset timing statistics (min/max/overrun counts) without affecting controller state.
    """

    pass


class SetIOCmd(msgspec.Struct, tag=int(CmdType.SET_IO), array_like=True, frozen=True):
    """SET_IO: [CmdType.SET_IO, port_index, value]

    port_index: 0-7 (8-bit I/O port)
    value: 0 or 1
    """

    port_index: Annotated[int, msgspec.Meta(ge=0, le=7)]
    value: Annotated[int, msgspec.Meta(ge=0, le=1)]


class SetPortCmd(
    msgspec.Struct, tag=int(CmdType.SET_PORT), array_like=True, frozen=True
):
    """SET_PORT: [CmdType.SET_PORT, port_str]"""

    port_str: Annotated[str, msgspec.Meta(min_length=1, max_length=256)]


class StreamCmd(msgspec.Struct, tag=int(CmdType.STREAM), array_like=True, frozen=True):
    """STREAM: [CmdType.STREAM, on]"""

    on: bool


class SimulatorCmd(
    msgspec.Struct, tag=int(CmdType.SIMULATOR), array_like=True, frozen=True
):
    """SIMULATOR: [CmdType.SIMULATOR, on]"""

    on: bool


class DelayCmd(msgspec.Struct, tag=int(CmdType.DELAY), array_like=True, frozen=True):
    """DELAY: [CmdType.DELAY, seconds]"""

    seconds: Annotated[float, msgspec.Meta(gt=0.0)]


class SetToolCmd(
    msgspec.Struct, tag=int(CmdType.SET_TOOL), array_like=True, frozen=True
):
    """SET_TOOL: [CmdType.SET_TOOL, tool_name]"""

    tool_name: Annotated[str, msgspec.Meta(min_length=1, max_length=64)]

    def __post_init__(self) -> None:
        name = self.tool_name.strip().upper()
        if name not in TOOL_CONFIGS:
            raise ValueError(f"Unknown tool '{name}'. Available: {list_tools()}")


class SetProfileCmd(
    msgspec.Struct, tag=int(CmdType.SET_PROFILE), array_like=True, frozen=True
):
    """SET_PROFILE: [CmdType.SET_PROFILE, profile]"""

    profile: Annotated[str, msgspec.Meta(min_length=1, max_length=32)]


class PneumaticGripperCmd(
    msgspec.Struct, tag=int(CmdType.PNEUMATICGRIPPER), array_like=True, frozen=True
):
    """PNEUMATICGRIPPER: [CmdType.PNEUMATICGRIPPER, open, port]"""

    open: bool  # True = open, False = close
    port: Annotated[int, msgspec.Meta(ge=1, le=2)]  # Output port 1 or 2


class ElectricGripperCmd(
    msgspec.Struct, tag=int(CmdType.ELECTRICGRIPPER), array_like=True, frozen=True
):
    """ELECTRICGRIPPER: [CmdType.ELECTRICGRIPPER, calibrate, position, speed, current]"""

    calibrate: bool  # True = calibrate mode, False = move mode
    position: Annotated[int, msgspec.Meta(ge=0, le=255)]
    speed: Annotated[int, msgspec.Meta(gt=0, le=255)]
    current: Annotated[int, msgspec.Meta(ge=100, le=1000)]


# Query commands (no params, just the tag)
class PingCmd(msgspec.Struct, tag=int(CmdType.PING), array_like=True, frozen=True):
    """PING: [CmdType.PING]"""

    pass


class GetStatusCmd(
    msgspec.Struct, tag=int(CmdType.GET_STATUS), array_like=True, frozen=True
):
    """GET_STATUS: [CmdType.GET_STATUS]"""

    pass


class GetAnglesCmd(
    msgspec.Struct, tag=int(CmdType.GET_ANGLES), array_like=True, frozen=True
):
    """GET_ANGLES: [CmdType.GET_ANGLES]"""

    pass


class GetPoseCmd(
    msgspec.Struct, tag=int(CmdType.GET_POSE), array_like=True, frozen=True
):
    """GET_POSE: [CmdType.GET_POSE, frame]"""

    frame: Annotated[str, msgspec.Meta(pattern=r"^(WRF|TRF)$")] | None = None


class GetIOCmd(msgspec.Struct, tag=int(CmdType.GET_IO), array_like=True, frozen=True):
    """GET_IO: [CmdType.GET_IO]"""

    pass


class GetGripperCmd(
    msgspec.Struct, tag=int(CmdType.GET_GRIPPER), array_like=True, frozen=True
):
    """GET_GRIPPER: [CmdType.GET_GRIPPER]"""

    pass


class GetSpeedsCmd(
    msgspec.Struct, tag=int(CmdType.GET_SPEEDS), array_like=True, frozen=True
):
    """GET_SPEEDS: [CmdType.GET_SPEEDS]"""

    pass


class GetToolCmd(
    msgspec.Struct, tag=int(CmdType.GET_TOOL), array_like=True, frozen=True
):
    """GET_TOOL: [CmdType.GET_TOOL]"""

    pass


class GetQueueCmd(
    msgspec.Struct, tag=int(CmdType.GET_QUEUE), array_like=True, frozen=True
):
    """GET_QUEUE: [CmdType.GET_QUEUE]"""

    pass


class GetCurrentActionCmd(
    msgspec.Struct, tag=int(CmdType.GET_CURRENT_ACTION), array_like=True, frozen=True
):
    """GET_CURRENT_ACTION: [CmdType.GET_CURRENT_ACTION]"""

    pass


class GetLoopStatsCmd(
    msgspec.Struct, tag=int(CmdType.GET_LOOP_STATS), array_like=True, frozen=True
):
    """GET_LOOP_STATS: [CmdType.GET_LOOP_STATS]"""

    pass


class GetGcodeStatusCmd(
    msgspec.Struct, tag=int(CmdType.GET_GCODE_STATUS), array_like=True, frozen=True
):
    """GET_GCODE_STATUS: [CmdType.GET_GCODE_STATUS]"""

    pass


class GetProfileCmd(
    msgspec.Struct, tag=int(CmdType.GET_PROFILE), array_like=True, frozen=True
):
    """GET_PROFILE: [CmdType.GET_PROFILE]"""

    pass


# GCODE commands
class GcodeCmd(msgspec.Struct, tag=int(CmdType.GCODE), array_like=True, frozen=True):
    """GCODE: [CmdType.GCODE, line]"""

    line: Annotated[str, msgspec.Meta(min_length=1, max_length=1024)]


class GcodeProgramCmd(
    msgspec.Struct, tag=int(CmdType.GCODE_PROGRAM), array_like=True, frozen=True
):
    """GCODE_PROGRAM: [CmdType.GCODE_PROGRAM, lines]"""

    lines: Annotated[list[str], msgspec.Meta(min_length=1, max_length=10000)]


class GcodeStopCmd(
    msgspec.Struct, tag=int(CmdType.GCODE_STOP), array_like=True, frozen=True
):
    """GCODE_STOP: [CmdType.GCODE_STOP]"""

    pass


class GcodePauseCmd(
    msgspec.Struct, tag=int(CmdType.GCODE_PAUSE), array_like=True, frozen=True
):
    """GCODE_PAUSE: [CmdType.GCODE_PAUSE]"""

    pass


class GcodeResumeCmd(
    msgspec.Struct, tag=int(CmdType.GCODE_RESUME), array_like=True, frozen=True
):
    """GCODE_RESUME: [CmdType.GCODE_RESUME]"""

    pass


# Smooth motion commands
class SmoothCircleCmd(
    msgspec.Struct, tag=int(CmdType.SMOOTH_CIRCLE), array_like=True, frozen=True
):
    """SMOOTH_CIRCLE: [CmdType.SMOOTH_CIRCLE, center, radius, plane, frame, center_mode, duration, speed_pct, accel_pct, clockwise]"""

    center: Annotated[list[float], msgspec.Meta(min_length=3, max_length=3)]
    radius: Annotated[float, msgspec.Meta(gt=0.0)]
    plane: Annotated[str, msgspec.Meta(pattern=r"^(XY|XZ|YZ)$")]
    frame: Annotated[str, msgspec.Meta(pattern=r"^(WRF|TRF)$")]
    center_mode: Annotated[str, msgspec.Meta(pattern=r"^(ABSOLUTE|TOOL|RELATIVE)$")]
    duration: Annotated[float, msgspec.Meta(ge=0.0)] | None = None
    speed_pct: Annotated[float, msgspec.Meta(gt=0.0, le=100.0)] | None = None
    accel_pct: Annotated[float, msgspec.Meta(gt=0.0, le=100.0)] = 100.0
    clockwise: bool = False


class SmoothArcCenterCmd(
    msgspec.Struct, tag=int(CmdType.SMOOTH_ARC_CENTER), array_like=True, frozen=True
):
    """SMOOTH_ARC_CENTER: [CmdType.SMOOTH_ARC_CENTER, end_pose, center, frame, duration, speed_pct, accel_pct, clockwise]"""

    end_pose: Annotated[list[float], msgspec.Meta(min_length=6, max_length=6)]
    center: Annotated[list[float], msgspec.Meta(min_length=3, max_length=3)]
    frame: Annotated[str, msgspec.Meta(pattern=r"^(WRF|TRF)$")]
    duration: Annotated[float, msgspec.Meta(ge=0.0)] | None = None
    speed_pct: Annotated[float, msgspec.Meta(gt=0.0, le=100.0)] | None = None
    accel_pct: Annotated[float, msgspec.Meta(gt=0.0, le=100.0)] = 100.0
    clockwise: bool = False


class SmoothArcParamCmd(
    msgspec.Struct, tag=int(CmdType.SMOOTH_ARC_PARAM), array_like=True, frozen=True
):
    """SMOOTH_ARC_PARAM: [CmdType.SMOOTH_ARC_PARAM, end_pose, radius, arc_angle, frame, duration, speed_pct, accel_pct, clockwise]"""

    end_pose: Annotated[list[float], msgspec.Meta(min_length=6, max_length=6)]
    radius: Annotated[float, msgspec.Meta(gt=0.0)]
    arc_angle: float  # degrees, can be negative for direction
    frame: Annotated[str, msgspec.Meta(pattern=r"^(WRF|TRF)$")]
    duration: Annotated[float, msgspec.Meta(ge=0.0)] | None = None
    speed_pct: Annotated[float, msgspec.Meta(gt=0.0, le=100.0)] | None = None
    accel_pct: Annotated[float, msgspec.Meta(gt=0.0, le=100.0)] = 100.0
    clockwise: bool = False


class SmoothSplineCmd(
    msgspec.Struct, tag=int(CmdType.SMOOTH_SPLINE), array_like=True, frozen=True
):
    """SMOOTH_SPLINE: [CmdType.SMOOTH_SPLINE, waypoints, frame, duration, speed_pct, accel_pct]"""

    waypoints: Annotated[list[list[float]], msgspec.Meta(min_length=2)]
    frame: Annotated[str, msgspec.Meta(pattern=r"^(WRF|TRF)$")]
    duration: Annotated[float, msgspec.Meta(ge=0.0)] | None = None
    speed_pct: Annotated[float, msgspec.Meta(gt=0.0, le=100.0)] | None = None
    accel_pct: Annotated[float, msgspec.Meta(gt=0.0, le=100.0)] = 100.0

    def __post_init__(self) -> None:
        # Validate each waypoint has 6 values
        for i, wp in enumerate(self.waypoints):
            if len(wp) != 6:
                raise ValueError(f"Waypoint {i} must have 6 values (x,y,z,rx,ry,rz)")


# =============================================================================
# Auto-generated Command union and STRUCT_TO_CMDTYPE
# =============================================================================


def _collect_command_structs() -> list[type]:
    """Collect all command struct classes from this module."""
    import sys

    module = sys.modules[__name__]
    structs = []
    for name, cls in vars(module).items():
        if not name.endswith("Cmd"):
            continue
        if not isinstance(cls, type):
            continue
        if not issubclass(cls, msgspec.Struct):
            continue
        # Check for tag in struct config
        config = getattr(cls, "__struct_config__", None)
        if config is not None and config.tag is not None:
            structs.append(cls)
    return structs


def _build_struct_to_cmdtype(structs: list[type]) -> dict[type, CmdType]:
    """Auto-generate struct -> CmdType mapping from tagged structs."""
    mapping: dict[type, CmdType] = {}
    for struct_cls in structs:
        config = getattr(struct_cls, "__struct_config__", None)
        if config is None:
            continue
        tag = getattr(config, "tag", None)
        if tag is None:
            continue
        try:
            cmd_type = CmdType(tag)
            mapping[struct_cls] = cmd_type
        except ValueError:
            pass  # Not a valid CmdType tag
    return mapping


# Build at import time
_COMMAND_STRUCTS = _collect_command_structs()
STRUCT_TO_CMDTYPE: dict[type, CmdType] = _build_struct_to_cmdtype(_COMMAND_STRUCTS)

# Build Command union dynamically from collected structs
Command: TypeAlias = Union[tuple(_COMMAND_STRUCTS)]  # type: ignore[valid-type]

# Module-level decoder for single-pass decode
_command_decoder = msgspec.msgpack.Decoder(Command)

# Module-level encoder with numpy support
_command_encoder = msgspec.msgpack.Encoder(enc_hook=_enc_hook)


def decode_command(data: bytes) -> Command:
    """Decode raw bytes to typed command struct.

    Args:
        data: Raw msgpack-encoded command bytes

    Returns:
        Typed command struct

    Raises:
        msgspec.ValidationError: If data is invalid or doesn't match any command type
    """
    return _command_decoder.decode(data)


def encode_command(cmd: Command) -> bytes:
    """Encode a typed command struct to bytes.

    Args:
        cmd: Typed command struct

    Returns:
        Raw msgpack-encoded bytes
    """
    return _command_encoder.encode(cmd)


# =============================================================================
# Response Structs - Tagged Union for single-pass decode
# Wire format: [MsgType.RESPONSE, QueryType.XXX, ...fields]
# =============================================================================


class StatusResultStruct(
    msgspec.Struct, tag=int(QueryType.STATUS), array_like=True, frozen=True
):
    """Aggregate robot status."""

    pose: list[float]
    angles: list[float]
    speeds: list[float]
    io: list[int]
    gripper: list[int]


class LoopStatsResultStruct(
    msgspec.Struct, tag=int(QueryType.LOOP_STATS), array_like=True, frozen=True
):
    """Control loop runtime metrics."""

    target_hz: float
    loop_count: int
    overrun_count: int
    mean_period_s: float
    std_period_s: float
    min_period_s: float
    max_period_s: float
    p95_period_s: float
    p99_period_s: float
    mean_hz: float


class ToolResultStruct(
    msgspec.Struct, tag=int(QueryType.TOOL), array_like=True, frozen=True
):
    """Tool configuration."""

    tool: str
    available: list[str]


class CurrentActionResultStruct(
    msgspec.Struct, tag=int(QueryType.CURRENT_ACTION), array_like=True, frozen=True
):
    """Current executing action."""

    current: str
    state: str
    next: str


class GcodeStatusResultStruct(
    msgspec.Struct, tag=int(QueryType.GCODE_STATUS), array_like=True, frozen=True
):
    """G-code interpreter status."""

    is_running: bool
    is_paused: bool
    current_line: int | None
    total_lines: int
    state: dict[str, Any]


class PingResultStruct(
    msgspec.Struct, tag=int(QueryType.PING), array_like=True, frozen=True
):
    """Ping response with serial connectivity status."""

    serial_connected: int  # 0 or 1


class AnglesResultStruct(
    msgspec.Struct, tag=int(QueryType.ANGLES), array_like=True, frozen=True
):
    """Joint angles response."""

    angles: list[float]


class PoseResultStruct(
    msgspec.Struct, tag=int(QueryType.POSE), array_like=True, frozen=True
):
    """Pose response."""

    pose: list[float]


class IOResultStruct(
    msgspec.Struct, tag=int(QueryType.IO), array_like=True, frozen=True
):
    """I/O status response."""

    io: list[int]


class GripperResultStruct(
    msgspec.Struct, tag=int(QueryType.GRIPPER), array_like=True, frozen=True
):
    """Gripper status response."""

    gripper: list[int]


class SpeedsResultStruct(
    msgspec.Struct, tag=int(QueryType.SPEEDS), array_like=True, frozen=True
):
    """Speeds response."""

    speeds: list[float]


class ProfileResultStruct(
    msgspec.Struct, tag=int(QueryType.PROFILE), array_like=True, frozen=True
):
    """Motion profile response."""

    profile: str


class QueueResultStruct(
    msgspec.Struct, tag=int(QueryType.QUEUE), array_like=True, frozen=True
):
    """Queue status response."""

    queue: list


# Tagged Union for responses
Response = (
    StatusResultStruct
    | LoopStatsResultStruct
    | ToolResultStruct
    | CurrentActionResultStruct
    | GcodeStatusResultStruct
    | PingResultStruct
    | AnglesResultStruct
    | PoseResultStruct
    | IOResultStruct
    | GripperResultStruct
    | SpeedsResultStruct
    | ProfileResultStruct
    | QueueResultStruct
)


# Typed message classes for parsed responses


class OkMsg(NamedTuple):
    """OK response."""

    pass


class ErrorMsg(NamedTuple):
    """Error response with message."""

    message: str


class ResponseMsg(NamedTuple):
    """Query response with type and value."""

    query_type: QueryType
    value: Any


# Union type for all parsed messages
Message = OkMsg | ErrorMsg | ResponseMsg


def parse_message(msg: object) -> Message | None:
    """Parse a raw msgpack message into a typed Message.

    Args:
        msg: Raw unpacked msgpack data

    Returns:
        OkMsg, ErrorMsg, or ResponseMsg, or None if invalid/unknown
    """
    # OK is just the integer
    if msg == MsgType.OK:
        return OkMsg()

    if not isinstance(msg, (list, tuple)) or len(msg) < 2:
        return None

    match msg[0]:
        case MsgType.ERROR:
            return ErrorMsg(str(msg[1]))
        case MsgType.RESPONSE if len(msg) >= 3:
            return ResponseMsg(QueryType(msg[1]), msg[2])

    return None


# =============================================================================
# Generic msgpack encode/decode functions
# =============================================================================


def encode(obj: object) -> bytes:
    """Encode any msgspec struct or Python object to bytes with numpy support."""
    return _encoder.encode(obj)


def decode(data: bytes) -> object:
    """Decode msgpack bytes to a Python object."""
    return _decoder.decode(data)


# Pre-packed common responses (avoid repeated packing)
OK_PACKED = _encoder.encode(MsgType.OK)

# Cache for common error messages (3x faster for repeated errors)
_ERROR_CACHE: dict[str, bytes] = {
    "Unknown command": _encoder.encode((MsgType.ERROR, "Unknown command")),
}


def pack_ok() -> bytes:
    """Pack an OK response."""
    return OK_PACKED


def pack_error(message: str) -> bytes:
    """Pack an error response: [ERROR, message].

    Common error messages are cached for performance.
    """
    cached = _ERROR_CACHE.get(message)
    if cached is not None:
        return cached
    return _encoder.encode((MsgType.ERROR, message))


def pack_response(query_type: QueryType, value: Any) -> bytes:
    """Pack a query response: [RESPONSE, query_type, value]."""
    return _encoder.encode((MsgType.RESPONSE, query_type, value))


def pack_status(
    pose: np.ndarray,
    angles: np.ndarray,
    speeds: np.ndarray,
    io: np.ndarray,
    gripper: np.ndarray,
    action_current: str,
    action_state: str,
    joint_en: np.ndarray,
    cart_en_wrf: np.ndarray,
    cart_en_trf: np.ndarray,
) -> bytes:
    """Pack a status broadcast message.

    Uses ormsgpack with OPT_SERIALIZE_NUMPY for ~80x fewer allocations
    compared to msgspec with enc_hook (reads numpy buffers directly via C API).
    """
    return ormsgpack.packb(
        (
            MsgType.STATUS,
            pose,
            angles,
            speeds,
            io,
            gripper,
            action_current,
            action_state,
            joint_en,
            cart_en_wrf,
            cart_en_trf,
        ),
        option=ormsgpack.OPT_SERIALIZE_NUMPY,
    )


def unpack(data: bytes) -> object:
    """Unpack a msgpack message."""
    return _decoder.decode(data)


def pack_command(cmd_type: CmdType, *params: object) -> bytes:
    """Pack a command as [CmdType, ...params]."""
    return _encoder.encode((cmd_type, *params))


def get_command_type(msg: object) -> tuple[CmdType | None, tuple]:
    """Extract command type and params from a message array.

    Returns (cmd_type, params) or (None, ()) if invalid.
    """
    if not isinstance(msg, (list, tuple)) or len(msg) < 1:
        return None, ()
    try:
        cmd_type = CmdType(msg[0])
        return cmd_type, tuple(msg[1:]) if len(msg) > 1 else ()
    except (ValueError, TypeError):
        return None, ()


def get_command_name(msg: object) -> str | None:
    """Get the command name string from a message array.

    Returns the command name (e.g., "HOME", "JOG") or None if invalid.
    """
    cmd_type, _ = get_command_type(msg)
    if cmd_type is None:
        return None
    return cmd_type.name


def is_ok(msg: object) -> bool:
    """Check if message is OK response."""
    return msg == MsgType.OK


def is_error(msg: object) -> tuple[bool, str]:
    """Check if message is error. Returns (is_error, message)."""
    if isinstance(msg, (list, tuple)) and len(msg) >= 2 and msg[0] == MsgType.ERROR:
        return True, str(msg[1])
    return False, ""


def is_status(msg: object) -> bool:
    """Check if message is a status broadcast."""
    return isinstance(msg, (list, tuple)) and len(msg) >= 1 and msg[0] == MsgType.STATUS


def is_response(msg: object) -> bool:
    """Check if message is a query response."""
    return (
        isinstance(msg, (list, tuple)) and len(msg) >= 1 and msg[0] == MsgType.RESPONSE
    )


def get_response_value(msg: object) -> tuple[QueryType | None, object]:
    """Extract query type and value from response. Returns (query_type, value)."""
    if isinstance(msg, (list, tuple)) and len(msg) >= 3 and msg[0] == MsgType.RESPONSE:
        return QueryType(msg[1]), msg[2]
    return None, None


# =============================================================================
# Status Buffer (for zero-allocation status parsing)
# =============================================================================


@dataclass
class StatusBuffer:
    """Preallocated buffer for zero-allocation status parsing.

    All numeric arrays are numpy for cache-friendly access and potential numba use.
    Use decode_status_bin_into() to fill this buffer without allocating new objects.
    """

    pose: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=np.float64))
    angles: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    speeds: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    io: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.int32))
    gripper: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.int32))
    joint_en: np.ndarray = field(default_factory=lambda: np.ones(12, dtype=np.int32))
    cart_en_wrf: np.ndarray = field(default_factory=lambda: np.ones(12, dtype=np.int32))
    cart_en_trf: np.ndarray = field(default_factory=lambda: np.ones(12, dtype=np.int32))
    action_current: str = ""
    action_state: str = ""

    def copy(self) -> "StatusBuffer":
        """Return a deep copy with all arrays copied."""
        return StatusBuffer(
            pose=self.pose.copy(),
            angles=self.angles.copy(),
            speeds=self.speeds.copy(),
            io=self.io.copy(),
            gripper=self.gripper.copy(),
            joint_en=self.joint_en.copy(),
            cart_en_wrf=self.cart_en_wrf.copy(),
            cart_en_trf=self.cart_en_trf.copy(),
            action_current=self.action_current,
            action_state=self.action_state,
        )


def decode_status_bin_into(data: bytes, buf: StatusBuffer) -> bool:
    """Zero-allocation decode of STATUS message into preallocated buffer.

    Message format: [MsgType.STATUS, pose, angles, speeds, io, gripper,
                     action_current, action_state, joint_en, cart_en_wrf, cart_en_trf]

    Args:
        data: Raw msgpack bytes
        buf: Preallocated StatusBuffer to fill

    Returns:
        True if valid STATUS message, False otherwise.
    """
    try:
        msg = _decoder.decode(data)
        if (
            not isinstance(msg, (list, tuple))
            or len(msg) < 11
            or msg[0] != MsgType.STATUS
        ):
            return False

        # Positional fields: [type, pose, angles, speeds, io, gripper, ac, as, je, cw, ct]
        # numpy slice assignment is 2x faster than element-by-element loop
        buf.pose[:] = msg[1]
        buf.angles[:] = msg[2]
        buf.speeds[:] = msg[3]
        buf.io[:] = msg[4]
        buf.gripper[:] = msg[5]
        buf.action_current = msg[6] if isinstance(msg[6], str) else ""
        buf.action_state = msg[7] if isinstance(msg[7], str) else ""
        buf.joint_en[:] = msg[8]
        buf.cart_en_wrf[:] = msg[9]
        buf.cart_en_trf[:] = msg[10]

        return True
    except Exception:
        return False


# =============================================================================
# Binary serial frame packing/unpacking (firmware communication)
# =============================================================================


class CommandCode(IntEnum):
    """Unified command codes for firmware interface."""

    HOME = 100
    ENABLE = 101
    DISABLE = 102
    JOG = 123
    MOVE = 156
    IDLE = 255


START = b"\xff\xff\xff"
END = b"\x01\x02"
PAYLOAD_LEN = 52  # matches existing firmware expectation


@njit(cache=True)
def split_to_3_bytes(n: int) -> tuple[int, int, int]:
    """
    Convert int to signed 24-bit big-endian (two's complement) encoded bytes (b0,b1,b2).
    """
    n24 = n & 0xFFFFFF
    return ((n24 >> 16) & 0xFF, (n24 >> 8) & 0xFF, n24 & 0xFF)


@njit(cache=True)
def fuse_3_bytes(b0: int, b1: int, b2: int) -> int:
    """
    Convert 3 bytes (big-endian) into a signed 24-bit integer.
    """
    val = (b0 << 16) | (b1 << 8) | b2
    return val - 0x1000000 if (val & 0x800000) else val


@njit(cache=True)
def fuse_2_bytes(b0: int, b1: int) -> int:
    """
    Convert 2 bytes (big-endian) into a signed 16-bit integer.
    """
    val = (b0 << 8) | b1
    return val - 0x10000 if (val & 0x8000) else val


@njit(cache=True)
def _pack_positions(out: np.ndarray, values: np.ndarray, offset: int) -> None:
    for i in range(6):
        v = int(values[i]) & 0xFFFFFF
        j = offset + i * 3
        out[j] = (v >> 16) & 0xFF
        out[j + 1] = (v >> 8) & 0xFF
        out[j + 2] = v & 0xFF


@njit(cache=True)
def _unpack_positions(data: np.ndarray, out: np.ndarray) -> None:
    for i in range(6):
        j = i * 3
        val = (int(data[j]) << 16) | (int(data[j + 1]) << 8) | int(data[j + 2])
        if val >= 0x800000:
            val -= 0x1000000
        out[i] = val


@njit(cache=True)
def _pack_bitfield(arr: np.ndarray) -> int:
    """Pack 8-element array into a single byte (MSB first)."""
    return (
        (int(arr[0] != 0) << 7)
        | (int(arr[1] != 0) << 6)
        | (int(arr[2] != 0) << 5)
        | (int(arr[3] != 0) << 4)
        | (int(arr[4] != 0) << 3)
        | (int(arr[5] != 0) << 2)
        | (int(arr[6] != 0) << 1)
        | int(arr[7] != 0)
    )


@njit(cache=True)
def _unpack_bitfield(byte_val: int, out: np.ndarray) -> None:
    """Unpack a byte into 8 bits (MSB first) into output array."""
    out[0] = (byte_val >> 7) & 1
    out[1] = (byte_val >> 6) & 1
    out[2] = (byte_val >> 5) & 1
    out[3] = (byte_val >> 4) & 1
    out[4] = (byte_val >> 3) & 1
    out[5] = (byte_val >> 2) & 1
    out[6] = (byte_val >> 1) & 1
    out[7] = byte_val & 1


@njit(cache=True)
def pack_tx_frame_into(
    out: memoryview,
    position_out: np.ndarray,
    speed_out: np.ndarray,
    command_code: int,
    affected_joint_out: np.ndarray,
    inout_out: np.ndarray,
    timeout_out: int,
    gripper_data_out: np.ndarray,
) -> None:
    """
    Pack a full TX frame into the provided memoryview without allocations.

    Expects 'out' to be a writable buffer of length >= 56 bytes.
    """
    # Header: 0xFF 0xFF 0xFF + payload length
    out[0] = 0xFF
    out[1] = 0xFF
    out[2] = 0xFF
    out[3] = 52

    # Positions and speeds: JIT-compiled packing
    _pack_positions(out, position_out, 4)
    _pack_positions(out, speed_out, 22)

    # Command
    out[40] = command_code

    # Bitfields
    out[41] = _pack_bitfield(affected_joint_out)
    out[42] = _pack_bitfield(inout_out)

    # Timeout
    out[43] = int(timeout_out) & 0xFF

    # Gripper: position, speed, current as 2 bytes each (big-endian)
    g0 = int(gripper_data_out[0]) & 0xFFFF
    g1 = int(gripper_data_out[1]) & 0xFFFF
    g2 = int(gripper_data_out[2]) & 0xFFFF
    out[44] = (g0 >> 8) & 0xFF
    out[45] = g0 & 0xFF
    out[46] = (g1 >> 8) & 0xFF
    out[47] = g1 & 0xFF
    out[48] = (g2 >> 8) & 0xFF
    out[49] = g2 & 0xFF

    # Gripper command, mode, id
    out[50] = int(gripper_data_out[3]) & 0xFF
    out[51] = int(gripper_data_out[4]) & 0xFF
    out[52] = int(gripper_data_out[5]) & 0xFF

    # CRC placeholder
    out[53] = 228

    # End bytes
    out[54] = 0x01
    out[55] = 0x02


@njit(cache=True)
def unpack_rx_frame_into(
    data: memoryview,
    pos_out: np.ndarray,
    spd_out: np.ndarray,
    homed_out: np.ndarray,
    io_out: np.ndarray,
    temp_out: np.ndarray,
    poserr_out: np.ndarray,
    timing_out: np.ndarray,
    grip_out: np.ndarray,
) -> bool:
    """
    Zero-allocation decode of a 52-byte RX frame payload (memoryview) directly into numpy arrays.
    Expects:
      - pos_out, spd_out: shape (6,), dtype=int32
      - homed_out, io_out, temp_out, poserr_out: shape (8,), dtype=uint8
      - timing_out: shape (1,), dtype=int32
      - grip_out: shape (6,), dtype=int32 [device_id, pos, spd, cur, status, obj]
    """
    if len(data) < 52:
        return False

    _unpack_positions(data, pos_out)
    _unpack_positions(data[18:], spd_out)

    _unpack_bitfield(int(data[36]), homed_out)
    _unpack_bitfield(int(data[37]), io_out)
    _unpack_bitfield(int(data[38]), temp_out)
    _unpack_bitfield(int(data[39]), poserr_out)

    timing_out[0] = fuse_3_bytes(0, int(data[40]), int(data[41]))

    device_id = int(data[44])
    grip_pos = fuse_2_bytes(int(data[45]), int(data[46]))
    grip_spd = fuse_2_bytes(int(data[47]), int(data[48]))
    grip_cur = fuse_2_bytes(int(data[49]), int(data[50]))
    status_byte = int(data[51])

    obj_detection = ((status_byte >> 3) & 1) << 1 | ((status_byte >> 2) & 1)

    grip_out[0] = device_id
    grip_out[1] = grip_pos
    grip_out[2] = grip_spd
    grip_out[3] = grip_cur
    grip_out[4] = status_byte
    grip_out[5] = obj_detection

    return True


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Enums
    "MsgType",
    "QueryType",
    "CmdType",
    "CommandCode",
    # Command structs
    "JogCmd",
    "MultiJogCmd",
    "CartJogCmd",
    "MoveJointCmd",
    "MovePoseCmd",
    "MoveCartCmd",
    "MoveCartRelTrfCmd",
    "HomeCmd",
    "StopCmd",
    "EnableCmd",
    "DisableCmd",
    "ResetCmd",
    "ResetLoopStatsCmd",
    "SetIOCmd",
    "SetPortCmd",
    "StreamCmd",
    "SimulatorCmd",
    "DelayCmd",
    "SetToolCmd",
    "SetProfileCmd",
    "PneumaticGripperCmd",
    "ElectricGripperCmd",
    "PingCmd",
    "GetStatusCmd",
    "GetAnglesCmd",
    "GetPoseCmd",
    "GetIOCmd",
    "GetGripperCmd",
    "GetSpeedsCmd",
    "GetToolCmd",
    "GetQueueCmd",
    "GetCurrentActionCmd",
    "GetLoopStatsCmd",
    "GetGcodeStatusCmd",
    "GetProfileCmd",
    "GcodeCmd",
    "GcodeProgramCmd",
    "GcodeStopCmd",
    "GcodePauseCmd",
    "GcodeResumeCmd",
    "SmoothCircleCmd",
    "SmoothArcCenterCmd",
    "SmoothArcParamCmd",
    "SmoothSplineCmd",
    "Command",
    # Response structs
    "StatusResultStruct",
    "LoopStatsResultStruct",
    "ToolResultStruct",
    "CurrentActionResultStruct",
    "GcodeStatusResultStruct",
    "PingResultStruct",
    "AnglesResultStruct",
    "PoseResultStruct",
    "IOResultStruct",
    "GripperResultStruct",
    "SpeedsResultStruct",
    "ProfileResultStruct",
    "QueueResultStruct",
    "Response",
    # Message types
    "OkMsg",
    "ErrorMsg",
    "ResponseMsg",
    "Message",
    # Encode/decode
    "decode_command",
    "encode_command",
    "STRUCT_TO_CMDTYPE",
    "parse_message",
    "encode",
    "decode",
    "pack_ok",
    "pack_error",
    "pack_response",
    "pack_status",
    "unpack",
    "pack_command",
    "get_command_type",
    "get_command_name",
    "is_ok",
    "is_error",
    "is_status",
    "is_response",
    "get_response_value",
    # Status buffer
    "StatusBuffer",
    "decode_status_bin_into",
    # Binary frame protocol
    "START",
    "END",
    "PAYLOAD_LEN",
    "split_to_3_bytes",
    "fuse_3_bytes",
    "fuse_2_bytes",
    "pack_tx_frame_into",
    "unpack_rx_frame_into",
]
