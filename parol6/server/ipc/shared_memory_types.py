"""
Shared memory layouts and utilities for inter-process communication.

This module defines the memory structures used to communicate between
the main controller process and the MockSerial/IK worker subprocesses.
"""

import struct
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Tuple

import numpy as np


# ==============================================================================
# MockSerial Shared Memory Layout
# ==============================================================================


@dataclass(frozen=True)
class MockSerialRxLayout:
    """
    Layout for RX buffer (robot -> controller).

    Total size: 72 bytes
    """

    PAYLOAD_OFFSET: int = 0
    PAYLOAD_SIZE: int = 52
    VERSION_OFFSET: int = 52  # uint64 - incremented each frame
    TIMESTAMP_OFFSET: int = 60  # float64 - time.time() of frame
    TOTAL_SIZE: int = 72


@dataclass(frozen=True)
class MockSerialTxLayout:
    """
    Layout for TX buffer (controller -> robot).

    Total size: 80 bytes
    """

    # Command frame data (matches write_frame arguments)
    POSITION_OUT_OFFSET: int = 0  # int32[6] = 24 bytes
    SPEED_OUT_OFFSET: int = 24  # float64[6] = 48 bytes (was int32, need float for JOG)
    COMMAND_OUT_OFFSET: int = 72  # uint8 = 1 byte
    # Padding for alignment
    CMD_SEQ_OFFSET: int = 73  # uint64 = 8 bytes - sequence number
    TOTAL_SIZE: int = 81


MOCK_RX_SHM_SIZE = MockSerialRxLayout.TOTAL_SIZE
MOCK_TX_SHM_SIZE = MockSerialTxLayout.TOTAL_SIZE


def pack_tx_command(
    buf: memoryview,
    position_out: np.ndarray,
    speed_out: np.ndarray,
    command_out: int,
    cmd_seq: int,
) -> None:
    """Pack TX command data into shared memory buffer."""
    layout = MockSerialTxLayout
    # Position (6 x int32)
    struct.pack_into("<6i", buf, layout.POSITION_OUT_OFFSET, *position_out[:6])
    # Speed (6 x float64)
    struct.pack_into("<6d", buf, layout.SPEED_OUT_OFFSET, *speed_out[:6])
    # Command
    buf[layout.COMMAND_OUT_OFFSET] = command_out & 0xFF
    # Sequence
    struct.pack_into("<Q", buf, layout.CMD_SEQ_OFFSET, cmd_seq)


def unpack_tx_command(buf: memoryview) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Unpack TX command data from shared memory buffer."""
    layout = MockSerialTxLayout
    position_out = np.array(
        struct.unpack_from("<6i", buf, layout.POSITION_OUT_OFFSET), dtype=np.int32
    )
    speed_out = np.array(
        struct.unpack_from("<6d", buf, layout.SPEED_OUT_OFFSET), dtype=np.float64
    )
    command_out = buf[layout.COMMAND_OUT_OFFSET]
    cmd_seq = struct.unpack_from("<Q", buf, layout.CMD_SEQ_OFFSET)[0]
    return position_out, speed_out, command_out, cmd_seq


def pack_rx_frame(
    buf: memoryview,
    payload: bytes | memoryview,
    version: int,
    timestamp: float,
) -> None:
    """Pack RX frame data into shared memory buffer."""
    layout = MockSerialRxLayout
    buf[layout.PAYLOAD_OFFSET : layout.PAYLOAD_OFFSET + layout.PAYLOAD_SIZE] = payload[
        :52
    ]
    struct.pack_into("<Q", buf, layout.VERSION_OFFSET, version)
    struct.pack_into("<d", buf, layout.TIMESTAMP_OFFSET, timestamp)


def unpack_rx_header(buf: memoryview) -> Tuple[int, float]:
    """Unpack just the version and timestamp from RX buffer (for polling)."""
    layout = MockSerialRxLayout
    version = struct.unpack_from("<Q", buf, layout.VERSION_OFFSET)[0]
    timestamp = struct.unpack_from("<d", buf, layout.TIMESTAMP_OFFSET)[0]
    return version, timestamp


# ==============================================================================
# IK Worker Shared Memory Layout
# ==============================================================================


@dataclass(frozen=True)
class IKInputLayout:
    """
    Layout for IK input buffer (main process -> IK worker).

    Total size: 200 bytes
    """

    Q_RAD_OFFSET: int = 0  # float64[6] = 48 bytes (joint angles in radians)
    T_FLAT_OFFSET: int = 48  # float64[16] = 128 bytes (4x4 transform matrix)
    REQ_SEQ_OFFSET: int = 176  # uint64 = 8 bytes (request sequence number)
    FLAGS_OFFSET: int = 184  # uint64 = 8 bytes (tool_changed, etc.)
    TOTAL_SIZE: int = 200


@dataclass(frozen=True)
class IKOutputLayout:
    """
    Layout for IK output buffer (IK worker -> main process).

    Total size: 48 bytes
    """

    JOINT_EN_OFFSET: int = 0  # uint8[12] = 12 bytes
    CART_EN_WRF_OFFSET: int = 12  # uint8[12] = 12 bytes
    CART_EN_TRF_OFFSET: int = 24  # uint8[12] = 12 bytes
    RESP_SEQ_OFFSET: int = 36  # uint64 = 8 bytes
    TOTAL_SIZE: int = 48


IK_INPUT_SHM_SIZE = IKInputLayout.TOTAL_SIZE
IK_OUTPUT_SHM_SIZE = IKOutputLayout.TOTAL_SIZE


def pack_ik_request(
    buf: memoryview,
    q_rad: np.ndarray,
    T_matrix: np.ndarray,
    req_seq: int,
    flags: int = 0,
) -> None:
    """Pack IK request into input shared memory buffer."""
    layout = IKInputLayout
    # Joint angles (6 x float64)
    struct.pack_into("<6d", buf, layout.Q_RAD_OFFSET, *q_rad[:6])
    # Transform matrix flattened (16 x float64)
    T_flat = T_matrix.flatten()[:16]
    struct.pack_into("<16d", buf, layout.T_FLAT_OFFSET, *T_flat)
    # Request sequence
    struct.pack_into("<Q", buf, layout.REQ_SEQ_OFFSET, req_seq)
    # Flags
    struct.pack_into("<Q", buf, layout.FLAGS_OFFSET, flags)


def unpack_ik_request(buf: memoryview) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Unpack IK request from input shared memory buffer."""
    layout = IKInputLayout
    q_rad = np.array(
        struct.unpack_from("<6d", buf, layout.Q_RAD_OFFSET), dtype=np.float64
    )
    T_flat = np.array(
        struct.unpack_from("<16d", buf, layout.T_FLAT_OFFSET), dtype=np.float64
    )
    T_matrix = T_flat.reshape((4, 4))
    req_seq = struct.unpack_from("<Q", buf, layout.REQ_SEQ_OFFSET)[0]
    flags = struct.unpack_from("<Q", buf, layout.FLAGS_OFFSET)[0]
    return q_rad, T_matrix, req_seq, flags


def pack_ik_response(
    buf: memoryview,
    joint_en: np.ndarray,
    cart_en_wrf: np.ndarray,
    cart_en_trf: np.ndarray,
    resp_seq: int,
) -> None:
    """Pack IK response into output shared memory buffer."""
    layout = IKOutputLayout
    buf[layout.JOINT_EN_OFFSET : layout.JOINT_EN_OFFSET + 12] = joint_en[:12].tobytes()
    buf[layout.CART_EN_WRF_OFFSET : layout.CART_EN_WRF_OFFSET + 12] = cart_en_wrf[
        :12
    ].tobytes()
    buf[layout.CART_EN_TRF_OFFSET : layout.CART_EN_TRF_OFFSET + 12] = cart_en_trf[
        :12
    ].tobytes()
    struct.pack_into("<Q", buf, layout.RESP_SEQ_OFFSET, resp_seq)


def unpack_ik_response(
    buf: memoryview,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Unpack IK response from output shared memory buffer."""
    layout = IKOutputLayout
    joint_en = np.frombuffer(
        buf[layout.JOINT_EN_OFFSET : layout.JOINT_EN_OFFSET + 12], dtype=np.uint8
    ).copy()
    cart_en_wrf = np.frombuffer(
        buf[layout.CART_EN_WRF_OFFSET : layout.CART_EN_WRF_OFFSET + 12], dtype=np.uint8
    ).copy()
    cart_en_trf = np.frombuffer(
        buf[layout.CART_EN_TRF_OFFSET : layout.CART_EN_TRF_OFFSET + 12], dtype=np.uint8
    ).copy()
    resp_seq = struct.unpack_from("<Q", buf, layout.RESP_SEQ_OFFSET)[0]
    return joint_en, cart_en_wrf, cart_en_trf, resp_seq


def get_ik_resp_seq(buf: memoryview) -> int:
    """Get just the response sequence number (for polling)."""
    return struct.unpack_from("<Q", buf, IKOutputLayout.RESP_SEQ_OFFSET)[0]


# ==============================================================================
# Shared Memory Utilities
# ==============================================================================


def create_shm(name: str, size: int) -> SharedMemory:
    """Create a new shared memory segment, cleaning up any existing one first."""
    try:
        # Try to clean up any existing segment with same name
        existing = SharedMemory(name=name)
        existing.close()
        existing.unlink()
    except FileNotFoundError:
        pass
    return SharedMemory(name=name, create=True, size=size)


def attach_shm(name: str) -> SharedMemory:
    """Attach to an existing shared memory segment."""
    return SharedMemory(name=name, create=False)


def cleanup_shm(shm: SharedMemory | None) -> None:
    """Safely close and unlink a shared memory segment."""
    if shm is None:
        return
    try:
        shm.close()
    except Exception:
        pass
    try:
        shm.unlink()
    except Exception:
        pass
