"""
Inter-process communication utilities for PAROL6 server.

This package provides shared memory layouts and utilities for
communication between the main controller process and worker subprocesses.
"""

from .shared_memory_types import (
    # MockSerial layouts
    MockSerialRxLayout,
    MockSerialTxLayout,
    MOCK_RX_SHM_SIZE,
    MOCK_TX_SHM_SIZE,
    pack_tx_command,
    unpack_tx_command,
    pack_rx_frame,
    unpack_rx_header,
    # IK layouts
    IKInputLayout,
    IKOutputLayout,
    IK_INPUT_SHM_SIZE,
    IK_OUTPUT_SHM_SIZE,
    pack_ik_request,
    unpack_ik_request,
    pack_ik_response,
    unpack_ik_response,
    unpack_ik_response_into,
    # Utilities
    create_shm,
    attach_shm,
    cleanup_shm,
)

__all__ = [
    # MockSerial
    "MockSerialRxLayout",
    "MockSerialTxLayout",
    "MOCK_RX_SHM_SIZE",
    "MOCK_TX_SHM_SIZE",
    "pack_tx_command",
    "unpack_tx_command",
    "pack_rx_frame",
    "unpack_rx_header",
    # IK
    "IKInputLayout",
    "IKOutputLayout",
    "IK_INPUT_SHM_SIZE",
    "IK_OUTPUT_SHM_SIZE",
    "pack_ik_request",
    "unpack_ik_request",
    "pack_ik_response",
    "unpack_ik_response",
    "unpack_ik_response_into",
    # Utilities
    "create_shm",
    "attach_shm",
    "cleanup_shm",
]
