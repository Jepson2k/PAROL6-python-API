"""
Type definitions for PAROL6 protocol.

Defines enums and dataclasses used across the public API.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal


# Stream mode state enum
class StreamModeState(Enum):
    """Stream mode state for jog commands."""

    OFF = 0  # Stream mode disabled (default FIFO queueing)
    ON = 1  # Stream mode enabled (latest-wins for jog commands)


# Frame literals
Frame = Literal["WRF", "TRF"]

# Axis literals
Axis = Literal[
    "X+", "X-", "Y+", "Y-", "Z+", "Z-", "RX+", "RX-", "RY+", "RY-", "RZ+", "RZ-"
]

# Acknowledgment status literals
AckStatus = Literal[
    "SENT",
    "QUEUED",
    "EXECUTING",
    "COMPLETED",
    "FAILED",
    "INVALID",
    "CANCELLED",
    "TIMEOUT",
    "NO_TRACKING",
]


@dataclass(slots=True, frozen=True)
class IOStatus:
    """Digital I/O status."""

    in1: int
    in2: int
    out1: int
    out2: int
    estop: int


@dataclass(slots=True, frozen=True)
class GripperStatus:
    """Electric gripper status."""

    id: int
    position: int
    speed: int
    current: int
    status_byte: int
    object_detect: int


@dataclass(slots=True)
class TrackingStatus:
    """Command tracking status."""

    command_id: str | None
    status: AckStatus
    details: str
    completed: bool
    ack_time: datetime | None


@dataclass(slots=True)
class SendResult:
    """Standardized result for command-sending APIs."""

    command_id: str | None
    status: AckStatus
    details: str
    completed: bool
    ack_time: datetime | None


@dataclass(slots=True, frozen=True)
class PingResult:
    """Parsed PING response."""

    serial_connected: bool
    raw: str
