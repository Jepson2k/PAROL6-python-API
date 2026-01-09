"""
MockSerial process adapter.

This module provides an adapter that presents the same interface as
MockSerialTransport but delegates to a subprocess via shared memory.
"""

import atexit
import logging
import multiprocessing
import time
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event

import numpy as np

from parol6 import config as cfg
from parol6.config import LIMITS
from parol6.server.ipc import (
    MOCK_RX_SHM_SIZE,
    MOCK_TX_SHM_SIZE,
    MockSerialRxLayout,
    cleanup_shm,
    create_shm,
    pack_tx_command,
    unpack_rx_header,
)
from parol6.server.state import ControllerState
from parol6.server.transports.mock_serial_process import mock_serial_worker_main

logger = logging.getLogger(__name__)


class MockSerialProcessAdapter:
    """
    Adapter that presents the same interface as MockSerialTransport
    but delegates to a subprocess via shared memory.

    This allows the physics simulation to run in a separate process,
    bypassing Python's GIL for true parallelism.
    """

    def __init__(
        self,
        port: str | None = None,
        baudrate: int = 2000000,
        timeout: float = 0,
    ):
        """
        Initialize the mock serial process adapter.

        Args:
            port: Ignored (for interface compatibility)
            baudrate: Ignored (for interface compatibility)
            timeout: Ignored (for interface compatibility)
        """
        self.port = port or "MOCK_SERIAL_PROCESS"
        self.baudrate = baudrate
        self.timeout = timeout

        # Connection state
        self._connected = False

        # Subprocess management
        self._process: Process | None = None
        self._shutdown_event: Event | None = None

        # Shared memory
        self._rx_shm: SharedMemory | None = None
        self._tx_shm: SharedMemory | None = None
        self._rx_mv: memoryview | None = None
        self._tx_mv: memoryview | None = None

        # Frame tracking
        self._cmd_seq = 0
        self._last_frame_version = 0
        self._frame_interval = cfg.INTERVAL_S

        # Statistics
        self._frames_sent = 0
        self._frames_received = 0

        # Unique names for shared memory segments
        self._shm_suffix = f"_{id(self)}"

        logger.info("MockSerialProcessAdapter initialized")

    def connect(self, port: str | None = None) -> bool:
        """
        Spawn the subprocess and establish shared memory connection.

        Args:
            port: Optional port name (ignored)

        Returns:
            True if connection successful
        """
        if self._connected:
            return True

        if port:
            self.port = port

        try:
            # Create shared memory segments
            rx_name = f"parol6_mock_rx{self._shm_suffix}"
            tx_name = f"parol6_mock_tx{self._shm_suffix}"

            self._rx_shm = create_shm(rx_name, MOCK_RX_SHM_SIZE)
            self._tx_shm = create_shm(tx_name, MOCK_TX_SHM_SIZE)
            assert self._rx_shm.buf is not None
            assert self._tx_shm.buf is not None
            self._rx_mv = memoryview(self._rx_shm.buf)
            self._tx_mv = memoryview(self._tx_shm.buf)

            # Initialize TX buffer with idle command
            pack_tx_command(
                self._tx_mv,
                np.zeros(6, dtype=np.int32),
                np.zeros(6, dtype=np.float64),
                0,  # IDLE
                0,
            )

            # Prepare subprocess arguments
            self._shutdown_event = multiprocessing.Event()

            # Extract config values to pass to subprocess
            standby_angles = tuple(cfg.STANDBY_ANGLES_DEG)
            home_angles = tuple(cfg.HOME_ANGLES_DEG)
            joint_limits = LIMITS.joint.position.steps.astype(np.int64)
            velocity_limits = LIMITS.joint.hard.velocity_steps.copy()

            # Calculate deg_to_steps ratios per joint
            deg_to_steps_ratios = np.array(
                [cfg.deg_to_steps(1.0, i) for i in range(6)], dtype=np.float64
            )

            # Spawn subprocess
            self._process = Process(
                target=mock_serial_worker_main,
                args=(
                    rx_name,
                    tx_name,
                    self._shutdown_event,
                    standby_angles,
                    home_angles,
                    joint_limits,
                    velocity_limits,
                    deg_to_steps_ratios,
                    cfg.INTERVAL_S,
                ),
                daemon=True,
                name="MockSerialProcess",
            )
            self._process.start()

            # Wait for first frame
            if not self._wait_for_first_frame(timeout=2.0):
                logger.error("MockSerial subprocess did not produce first frame")
                self._cleanup()
                return False

            self._connected = True
            logger.info(
                f"MockSerialProcessAdapter connected, subprocess PID: {self._process.pid}"
            )

            # Register cleanup on exit
            atexit.register(self._cleanup)

            return True

        except Exception as e:
            logger.exception("Failed to start MockSerial subprocess: %s", e)
            self._cleanup()
            return False

    def _wait_for_first_frame(self, timeout: float = 2.0) -> bool:
        """Wait for the subprocess to produce its first frame."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._rx_mv:
                version, _ = unpack_rx_header(self._rx_mv)
                if version > 0:
                    self._last_frame_version = version
                    return True
            time.sleep(0.01)
        return False

    def disconnect(self) -> None:
        """Signal shutdown and wait for subprocess to exit."""
        self._cleanup()
        self._connected = False
        logger.info("MockSerialProcessAdapter disconnected")

    def _cleanup(self) -> None:
        """Clean up subprocess and shared memory."""
        # Signal shutdown
        if self._shutdown_event:
            self._shutdown_event.set()

        # Wait for process to exit
        if self._process and self._process.is_alive():
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                logger.warning(
                    "MockSerial subprocess did not exit cleanly, terminating"
                )
                self._process.terminate()
                self._process.join(timeout=1.0)

        self._process = None
        self._shutdown_event = None

        # Release memoryviews before closing shared memory to avoid BufferError
        try:
            if self._rx_mv is not None:
                self._rx_mv.release()
        except Exception:
            pass
        try:
            if self._tx_mv is not None:
                self._tx_mv.release()
        except Exception:
            pass
        self._rx_mv = None
        self._tx_mv = None

        # Clean up shared memory
        cleanup_shm(self._rx_shm)
        cleanup_shm(self._tx_shm)
        self._rx_shm = None
        self._tx_shm = None

    def is_connected(self) -> bool:
        """Check if connection is active."""
        if not self._connected:
            return False
        # Also check if subprocess is still alive
        if self._process and not self._process.is_alive():
            logger.warning("MockSerial subprocess died unexpectedly")
            self._connected = False
            return False
        return True

    def auto_reconnect(self) -> bool:
        """Attempt to reconnect if not connected."""
        if not self._connected:
            return self.connect(self.port)
        return False

    def write_frame(
        self,
        position_out: np.ndarray,
        speed_out: np.ndarray,
        command_out: int,
        affected_joint_out: np.ndarray,
        inout_out: np.ndarray,
        timeout_out: int,
        gripper_data_out: np.ndarray,
    ) -> bool:
        """
        Write command to TX shared memory.

        Args:
            position_out: Target positions
            speed_out: Speed commands
            command_out: Command code
            affected_joint_out: Affected joint flags (ignored in mock)
            inout_out: I/O commands (ignored in mock)
            timeout_out: Timeout value (ignored in mock)
            gripper_data_out: Gripper commands (ignored for now)

        Returns:
            True if written successfully
        """
        if not self._connected or self._tx_mv is None:
            return False

        self._cmd_seq += 1
        pack_tx_command(
            self._tx_mv,
            position_out,
            speed_out,
            command_out,
            self._cmd_seq,
        )
        self._frames_sent += 1
        return True

    def get_latest_frame_view(self) -> tuple[memoryview | None, int, float]:
        """
        Return latest 52-byte payload memoryview, version, timestamp.

        Returns:
            Tuple of (payload_memoryview, version, timestamp)
        """
        if not self._connected or self._rx_mv is None:
            return (None, 0, 0.0)

        version, timestamp = unpack_rx_header(self._rx_mv)

        if version > self._last_frame_version:
            self._last_frame_version = version
            self._frames_received += 1

        # Return memoryview to payload portion
        layout = MockSerialRxLayout
        payload_mv = self._rx_mv[
            layout.PAYLOAD_OFFSET : layout.PAYLOAD_OFFSET + layout.PAYLOAD_SIZE
        ]
        return (payload_mv, version, timestamp)

    def start_reader(self, shutdown_event) -> None:
        """
        No-op for process adapter - the subprocess handles reading.

        This method exists for interface compatibility with SerialTransport.
        """
        # The subprocess already runs its own loop
        pass

    def sync_from_controller_state(self, state: ControllerState) -> None:
        """
        Synchronize the mock robot state from controller state.

        Note: This is a simplified implementation that just sends the
        current position as a MOVE command to sync the subprocess.
        """
        if not self._connected or self._tx_mv is None:
            return

        # Send current position as target to sync
        self._cmd_seq += 1
        pack_tx_command(
            self._tx_mv,
            state.Position_in.copy(),
            np.zeros(6, dtype=np.float64),
            156,  # MOVE command to sync position
            self._cmd_seq,
        )
        logger.info("MockSerialProcessAdapter: state sync requested")

    def get_info(self) -> dict:
        """Get information about the mock transport."""
        return {
            "port": self.port,
            "baudrate": self.baudrate,
            "connected": self._connected,
            "timeout": self.timeout,
            "mode": "MOCK_SERIAL_PROCESS",
            "frames_sent": self._frames_sent,
            "frames_received": self._frames_received,
            "simulation_rate_hz": int(1.0 / self._frame_interval),
            "subprocess_pid": self._process.pid if self._process else None,
            "subprocess_alive": self._process.is_alive() if self._process else False,
        }


def create_mock_serial_process_adapter() -> MockSerialProcessAdapter:
    """
    Factory function to create a mock serial process adapter.

    Returns:
        Configured MockSerialProcessAdapter instance
    """
    adapter = MockSerialProcessAdapter()
    adapter.connect()
    logger.info("Mock serial process adapter created and connected")
    return adapter
