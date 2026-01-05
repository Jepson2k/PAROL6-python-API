"""
Wire protocol helpers for UDP encoding/decoding.

This module centralizes encoding of command strings and decoding of common
response payloads used by the headless controller.
"""

import logging
from collections.abc import Sequence

# Centralized binary wire protocol helpers (pack/unpack + codes)
from enum import IntEnum
from typing import Literal, cast

import numpy as np

from .types import Axis, Frame, PingResult, StatusAggregate

logger = logging.getLogger(__name__)

# Precomputed bit-unpack lookup table for 0..255 (MSB..LSB)
# Using NumPy ensures fast vectorized selection without per-call allocations.
_BIT_UNPACK = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1, bitorder="big"
)

# Pre-allocated work buffers for unpack_rx_frame_into (avoid per-call allocations)
_UNPACK_BUF_B = np.zeros((12, 3), dtype=np.uint8)
_UNPACK_BUF_POS = np.zeros(6, dtype=np.int32)
_UNPACK_BUF_SPD = np.zeros(6, dtype=np.int32)
_SIGN_THRESHOLD = 1 << 23
_SIGN_ADJUST = 1 << 24
START = b"\xff\xff\xff"
END = b"\x01\x02"
PAYLOAD_LEN = 52  # matches existing firmware expectation

__all__ = [
    "CommandCode",
    "pack_tx_frame_into",
    "unpack_rx_frame_into",
    "encode_move_joint",
    "encode_move_pose",
    "encode_move_cartesian",
    "encode_move_cartesian_rel_trf",
    "encode_jog_joint",
    "encode_cart_jog",
    "encode_gcode",
    "encode_gcode_program_inline",
    "encode_reset",
    "decode_ping",
    "decode_simple",
    "decode_status",
    "split_to_3_bytes",
    "fuse_3_bytes",
    "fuse_2_bytes",
]


class CommandCode(IntEnum):
    """Unified command codes for firmware interface."""

    HOME = 100
    ENABLE = 101
    DISABLE = 102
    JOG = 123
    MOVE = 156
    IDLE = 255


def split_bitfield(byte_val: int) -> list[int]:
    """Split an 8-bit integer into a big-endian list of bits (MSB..LSB)."""
    return [(byte_val >> i) & 1 for i in range(7, -1, -1)]


def fuse_bitfield_2_bytearray(bits: list[int] | Sequence[int]) -> bytes:
    """
    Fuse a big-endian list of 8 bits (MSB..LSB) into a single byte.
    Any truthy value is treated as 1.
    """
    number = 0
    for b in bits[:8]:
        number = (number << 1) | (1 if b else 0)
    return bytes([number])


def split_to_3_bytes(n: int) -> tuple[int, int, int]:
    """
    Convert int to signed 24-bit big-endian (two's complement) encoded bytes (b0,b1,b2).
    """
    n24 = n & 0xFFFFFF
    return ((n24 >> 16) & 0xFF, (n24 >> 8) & 0xFF, n24 & 0xFF)


def fuse_3_bytes(b0: int, b1: int, b2: int) -> int:
    """
    Convert 3 bytes (big-endian) into a signed 24-bit integer.
    """
    val = (b0 << 16) | (b1 << 8) | b2
    return val - 0x1000000 if (val & 0x800000) else val


def fuse_2_bytes(b0: int, b1: int) -> int:
    """
    Convert 2 bytes (big-endian) into a signed 16-bit integer.
    """
    val = (b0 << 8) | b1
    return val - 0x10000 if (val & 0x8000) else val


def pack_tx_frame_into(
    out: memoryview,
    position_out: np.ndarray,
    speed_out: np.ndarray,
    command_code: int | CommandCode,
    affected_joint_out: np.ndarray,
    inout_out: np.ndarray,
    timeout_out: int,
    gripper_data_out: np.ndarray,
) -> None:
    """
    Pack a full TX frame into the provided memoryview without allocations.

    Expects 'out' to be a writable buffer of length >= 56 bytes.
    """
    # Header
    out[0:3] = START
    out[3] = PAYLOAD_LEN

    # Positions: 6 joints * 3 bytes each, big-endian 24-bit
    for i in range(6):
        v = int(position_out[i]) & 0xFFFFFF
        j = 4 + i * 3
        out[j] = (v >> 16) & 0xFF
        out[j + 1] = (v >> 8) & 0xFF
        out[j + 2] = v & 0xFF

    # Speeds: 6 joints * 3 bytes each
    for i in range(6):
        v = int(speed_out[i]) & 0xFFFFFF
        j = 22 + i * 3
        out[j] = (v >> 16) & 0xFF
        out[j + 1] = (v >> 8) & 0xFF
        out[j + 2] = v & 0xFF

    # Command
    out[40] = int(command_code)

    # Bitfields - manual packing avoids numpy allocation
    out[41] = (
        (int(bool(affected_joint_out[0])) << 7)
        | (int(bool(affected_joint_out[1])) << 6)
        | (int(bool(affected_joint_out[2])) << 5)
        | (int(bool(affected_joint_out[3])) << 4)
        | (int(bool(affected_joint_out[4])) << 3)
        | (int(bool(affected_joint_out[5])) << 2)
        | (int(bool(affected_joint_out[6])) << 1)
        | int(bool(affected_joint_out[7]))
    )
    out[42] = (
        (int(bool(inout_out[0])) << 7)
        | (int(bool(inout_out[1])) << 6)
        | (int(bool(inout_out[2])) << 5)
        | (int(bool(inout_out[3])) << 4)
        | (int(bool(inout_out[4])) << 3)
        | (int(bool(inout_out[5])) << 2)
        | (int(bool(inout_out[6])) << 1)
        | int(bool(inout_out[7]))
    )

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


def unpack_rx_frame_into(
    data: memoryview,
    *,
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
    global _UNPACK_BUF_B, _UNPACK_BUF_POS, _UNPACK_BUF_SPD
    try:
        if len(data) < 52:
            logger.warning(
                f"unpack_rx_frame_into: payload too short ({len(data)} bytes)"
            )
            return False

        mv = memoryview(data)

        # Positions (0..17) and speeds (18..35), 3 bytes per value, big-endian signed 24-bit
        # Use pre-allocated buffers to avoid per-call allocations
        np.copyto(_UNPACK_BUF_B, np.frombuffer(mv[:36], dtype=np.uint8).reshape(12, 3))

        # Decode positions: combine 3 bytes into int32 using pre-allocated buffer
        # pos = (b0 << 16) | (b1 << 8) | b2
        _UNPACK_BUF_POS[:] = _UNPACK_BUF_B[:6, 0]
        _UNPACK_BUF_POS <<= 8
        _UNPACK_BUF_POS |= _UNPACK_BUF_B[:6, 1]
        _UNPACK_BUF_POS <<= 8
        _UNPACK_BUF_POS |= _UNPACK_BUF_B[:6, 2]

        # Decode speeds similarly
        _UNPACK_BUF_SPD[:] = _UNPACK_BUF_B[6:, 0]
        _UNPACK_BUF_SPD <<= 8
        _UNPACK_BUF_SPD |= _UNPACK_BUF_B[6:, 1]
        _UNPACK_BUF_SPD <<= 8
        _UNPACK_BUF_SPD |= _UNPACK_BUF_B[6:, 2]

        # Sign-correct 24-bit to int32 (in-place)
        _UNPACK_BUF_POS[_UNPACK_BUF_POS >= _SIGN_THRESHOLD] -= _SIGN_ADJUST
        _UNPACK_BUF_SPD[_UNPACK_BUF_SPD >= _SIGN_THRESHOLD] -= _SIGN_ADJUST

        np.copyto(pos_out, _UNPACK_BUF_POS, casting="no")
        np.copyto(spd_out, _UNPACK_BUF_SPD, casting="no")

        homed_byte = mv[36]
        io_byte = mv[37]
        temp_err_byte = mv[38]
        pos_err_byte = mv[39]
        timing_b0 = mv[40]
        timing_b1 = mv[41]
        # indices 42..43 exist in some variants (timeout/xtr), legacy code ignores

        device_id = mv[44]
        grip_pos_b0, grip_pos_b1 = mv[45], mv[46]
        grip_spd_b0, grip_spd_b1 = mv[47], mv[48]
        grip_cur_b0, grip_cur_b1 = mv[49], mv[50]
        status_byte = mv[51]

        # Bitfields (MSB..LSB) via LUT (no per-call Python loops)
        homed_out[:] = _BIT_UNPACK[int(homed_byte)]
        io_out[:] = _BIT_UNPACK[int(io_byte)]
        temp_out[:] = _BIT_UNPACK[int(temp_err_byte)]
        poserr_out[:] = _BIT_UNPACK[int(pos_err_byte)]

        # Timing (legacy semantics: fuse_3_bytes(0, b0, b1))
        timing_val = fuse_3_bytes(0, int(timing_b0), int(timing_b1))
        timing_out[0] = int(timing_val)

        # Gripper values
        grip_pos = fuse_2_bytes(int(grip_pos_b0), int(grip_pos_b1))
        grip_spd = fuse_2_bytes(int(grip_spd_b0), int(grip_spd_b1))
        grip_cur = fuse_2_bytes(int(grip_cur_b0), int(grip_cur_b1))

        sbits = _BIT_UNPACK[int(status_byte)]
        obj_detection = (int(sbits[4]) << 1) | int(sbits[5])

        grip_out[0] = int(device_id)
        grip_out[1] = int(grip_pos)
        grip_out[2] = int(grip_spd)
        grip_out[3] = int(grip_cur)
        grip_out[4] = int(status_byte)
        grip_out[5] = int(obj_detection)

        return True
    except Exception as e:
        logger.error(f"unpack_rx_frame_into: exception {e}")
        return False


# =========================
# Encoding helpers
# =========================


def _opt(value: object | None, none_token: str = "NONE") -> str:
    """Format an optional value as a string, using none_token for None."""
    return none_token if value is None else str(value)


def encode_move_joint(
    angles: Sequence[float],
    duration: float | None,
    speed: float | None,
    accel: float | None = None,
) -> str:
    """
    MOVEJOINT|j1|j2|j3|j4|j5|j6|DUR|SPD|ACC
    Use "NONE" for omitted duration/speed/accel.
    Note: Validation (requiring one of duration/speed) is left to caller.
    """
    angles_str = "|".join(str(a) for a in angles)
    return f"MOVEJOINT|{angles_str}|{_opt(duration)}|{_opt(speed)}|{_opt(accel)}"


def encode_move_pose(
    pose: Sequence[float],
    duration: float | None,
    speed: float | None,
    accel: float | None = None,
) -> str:
    """
    MOVEPOSE|x|y|z|rx|ry|rz|DUR|SPD|ACC
    Use "NONE" for omitted duration/speed/accel.
    """
    pose_str = "|".join(str(v) for v in pose)
    return f"MOVEPOSE|{pose_str}|{_opt(duration)}|{_opt(speed)}|{_opt(accel)}"


def encode_move_cartesian(
    pose: Sequence[float],
    duration: float | None,
    speed: float | None,
    accel: float | None = None,
) -> str:
    """
    MOVECART|x|y|z|rx|ry|rz|DUR|SPD|ACC
    Use "NONE" for omitted duration/speed/accel.
    """
    pose_str = "|".join(str(v) for v in pose)
    return f"MOVECART|{pose_str}|{_opt(duration)}|{_opt(speed)}|{_opt(accel)}"


def encode_move_cartesian_rel_trf(
    deltas: Sequence[float],  # [dx, dy, dz, rx, ry, rz] in mm/deg relative to TRF
    duration: float | None,
    speed: float | None,
    accel: int | None,
    profile: str | None,
    tracking: str | None,
) -> str:
    """
    MOVECARTRELTRF|dx|dy|dz|rx|ry|rz|DUR|SPD|ACC|PROFILE|TRACKING
    Non-required fields should use "NONE".
    """
    delta_str = "|".join(str(v) for v in deltas)
    prof_str = (profile or "NONE").upper()
    track_str = (tracking or "NONE").upper()
    return (
        f"MOVECARTRELTRF|{delta_str}|{_opt(duration)}|{_opt(speed)}|"
        f"{_opt(accel)}|{prof_str}|{track_str}"
    )


def encode_jog_joint(
    joint_index: int,
    speed_percentage: int,
    duration: float | None,
    distance_deg: float | None,
) -> str:
    """
    JOG|joint_index|speed_pct|DUR|DIST
    duration and distance_deg are optional; use "NONE" if omitted.
    """
    return f"JOG|{joint_index}|{speed_percentage}|{_opt(duration)}|{_opt(distance_deg)}"


def encode_cart_jog(
    frame: Frame,
    axis: Axis,
    speed_percentage: int,
    duration: float,
) -> str:
    """
    CARTJOG|FRAME|AXIS|speed_pct|duration
    """
    return f"CARTJOG|{frame}|{axis}|{speed_percentage}|{duration}"


def encode_gcode(line: str) -> str:
    """
    GCODE|<single_line>
    The caller should ensure that '|' is not present in the line.
    """
    return f"GCODE|{line}"


def encode_gcode_program_inline(lines: Sequence[str]) -> str:
    """
    GCODE_PROGRAM|INLINE|line1;line2;...
    The caller should ensure that '|' is not present inside any line.
    """
    program_str = ";".join(lines)
    return f"GCODE_PROGRAM|INLINE|{program_str}"


def encode_reset() -> str:
    """RESET - instantly reset controller state to initial values."""
    return "RESET"


# =========================
# Decoding helpers
# =========================
def decode_ping(resp: str) -> PingResult | None:
    """Parse PONG response: 'PONG|SERIAL=1' -> PingResult.

    Args:
        resp: Raw ping response string (e.g., 'PONG|SERIAL=1')

    Returns:
        PingResult with serial_connected status, or None if invalid response
    """
    if not resp or not resp.startswith("PONG"):
        return None
    serial_connected = False
    if "SERIAL=" in resp:
        serial_part = resp.split("SERIAL=", 1)[-1].split("|")[0].strip()
        serial_connected = serial_part.startswith("1")
    return {"serial_connected": serial_connected, "raw": resp}


def decode_simple(
    resp: str, expected_prefix: Literal["ANGLES", "IO", "GRIPPER", "SPEEDS", "POSE"]
) -> list[float] | list[int] | None:
    """
    Decode simple prefixed payloads like:
      ANGLES|a0,a1,a2,a3,a4,a5
      IO|in1,in2,out1,out2,estop
      GRIPPER|id,pos,spd,cur,status,obj
      SPEEDS|s0,s1,s2,s3,s4,s5
      POSE|p0,p1,...,p15

    Returns list[float] or list[int] depending on the expected_prefix.
    """
    if not resp:
        logger.debug(
            f"decode_simple: Empty response for expected prefix '{expected_prefix}'"
        )
        return None
    parts = resp.strip().split("|", 1)
    if len(parts) != 2 or parts[0] != expected_prefix:
        logger.warning(
            f"decode_simple: Invalid response format. Expected '{expected_prefix}|...' but got '{resp}'"
        )
        return None
    payload = parts[1]
    tokens = [t for t in payload.split(",") if t != ""]

    # IO and GRIPPER are integer-based; others default to float
    if expected_prefix in ("IO", "GRIPPER"):
        try:
            return [int(t) for t in tokens]
        except ValueError as e:
            logger.error(
                f"decode_simple: Failed to parse integers for {expected_prefix}. Payload: '{payload}', Error: {e}"
            )
            return None
    else:
        try:
            return [float(t) for t in tokens]
        except ValueError as e:
            logger.error(
                f"decode_simple: Failed to parse floats for {expected_prefix}. Payload: '{payload}', Error: {e}"
            )
            return None


def decode_status(resp: str) -> StatusAggregate | None:
    """
    Decode aggregate status:
      STATUS|POSE=p0,p1,...,p15|ANGLES=a0,...,a5|SPEEDS=s0,...,s5|IO=in1,in2,out1,out2,estop|GRIPPER=id,pos,spd,cur,status,obj|
             ACTION_CURRENT=...|ACTION_STATE=...

    Returns a dict matching StatusAggregate or None on parse failure.
    """
    if not resp or not resp.startswith("STATUS|"):
        return None

    # Split top-level sections after "STATUS|"
    sections = resp.split("|")[1:]
    result: dict[str, object] = {
        "pose": None,
        "angles": None,
        "speeds": None,
        "io": None,
        "gripper": None,
        "action_current": None,
        "action_state": None,
        "joint_en": None,
        "cart_en_wrf": None,
        "cart_en_trf": None,
    }
    for sec in sections:
        if sec.startswith("POSE="):
            vals = [float(x) for x in sec[len("POSE=") :].split(",") if x]
            result["pose"] = vals
        elif sec.startswith("ANGLES="):
            vals = [float(x) for x in sec[len("ANGLES=") :].split(",") if x]
            result["angles"] = vals
        elif sec.startswith("SPEEDS="):
            vals = [float(x) for x in sec[len("SPEEDS=") :].split(",") if x]
            result["speeds"] = vals
        elif sec.startswith("IO="):
            vals = [int(x) for x in sec[len("IO=") :].split(",") if x]
            result["io"] = vals
        elif sec.startswith("GRIPPER="):
            vals = [int(x) for x in sec[len("GRIPPER=") :].split(",") if x]
            result["gripper"] = vals
        elif sec.startswith("ACTION_CURRENT="):
            result["action_current"] = sec[len("ACTION_CURRENT=") :]
        elif sec.startswith("ACTION_STATE="):
            result["action_state"] = sec[len("ACTION_STATE=") :]
        elif sec.startswith("JOINT_EN="):
            vals = [int(x) for x in sec[len("JOINT_EN=") :].split(",") if x]
            result["joint_en"] = vals
        elif sec.startswith("CART_EN_WRF="):
            vals = [int(x) for x in sec[len("CART_EN_WRF=") :].split(",") if x]
            result["cart_en_wrf"] = vals
        elif sec.startswith("CART_EN_TRF="):
            vals = [int(x) for x in sec[len("CART_EN_TRF=") :].split(",") if x]
            result["cart_en_trf"] = vals

    # Basic validation: accept if at least one of the core groups is present
    if (
        result["pose"] is None
        and result["angles"] is None
        and result["io"] is None
        and result["gripper"] is None
        and result["action_current"] is None
    ):
        return None

    return cast(StatusAggregate, result)
