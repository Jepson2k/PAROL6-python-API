"""
IK Worker client.

This module provides a client for the IK worker subprocess, allowing
the main process to submit requests and poll for results asynchronously.
"""

import atexit
import logging
import multiprocessing
from multiprocessing import Process
from multiprocessing.synchronize import Event

import numpy as np

from parol6.server.ipc import (
    IK_INPUT_SHM_SIZE,
    IK_OUTPUT_SHM_SIZE,
    cleanup_shm,
    create_shm,
    pack_ik_request,
    unpack_ik_response_into,
)
from parol6.server.ik_worker import ik_enablement_worker_main

logger = logging.getLogger(__name__)


class IKWorkerClient:
    """
    Client for asynchronous IK enablement computation.

    Submits requests to the IK worker subprocess via shared memory
    and polls for results non-blockingly.
    """

    def __init__(self):
        """Initialize the IK worker client (lazy - subprocess starts on first request)."""
        self._input_shm = None
        self._output_shm = None
        self._input_mv: memoryview | None = None
        self._output_mv: memoryview | None = None
        self._output_arr: np.ndarray | None = None  # numpy view for numba
        self._process: Process | None = None
        self._shutdown_event: Event | None = None
        self._request_event: Event | None = None

        self._last_resp_version = 0
        self._started = False

        # Unique names for shared memory segments
        self._shm_suffix = f"_{id(self)}"

        # Pre-allocated result buffers (zero-alloc reads)
        self._joint_en = np.ones(12, dtype=np.uint8)
        self._cart_en_wrf = np.ones(12, dtype=np.uint8)
        self._cart_en_trf = np.ones(12, dtype=np.uint8)

    def start(self) -> bool:
        """
        Spawn the IK worker subprocess.

        Returns:
            True if started successfully
        """
        if self._started and self._process and self._process.is_alive():
            return True

        try:
            self._started = True
            # Create shared memory segments
            input_name = f"parol6_ik_in{self._shm_suffix}"
            output_name = f"parol6_ik_out{self._shm_suffix}"

            self._input_shm = create_shm(input_name, IK_INPUT_SHM_SIZE)
            self._output_shm = create_shm(output_name, IK_OUTPUT_SHM_SIZE)
            self._input_mv = memoryview(self._input_shm.buf)
            self._output_mv = memoryview(self._output_shm.buf)
            self._output_arr = np.frombuffer(self._output_shm.buf, dtype=np.uint8)

            # Initialize with zeros (use numpy for cross-platform compatibility)
            np.frombuffer(self._input_shm.buf, dtype=np.uint8)[:] = 0
            np.frombuffer(self._output_shm.buf, dtype=np.uint8)[:] = 0

            # Spawn subprocess
            self._shutdown_event = multiprocessing.Event()
            self._request_event = multiprocessing.Event()
            self._process = Process(
                target=ik_enablement_worker_main,
                args=(
                    input_name,
                    output_name,
                    self._shutdown_event,
                    self._request_event,
                ),
                daemon=True,
                name="IKWorkerProcess",
            )
            self._process.start()

            logger.info(f"IKWorkerClient started, subprocess PID: {self._process.pid}")

            # Register cleanup on exit
            atexit.register(self.stop)

            return True

        except Exception as e:
            logger.exception("Failed to start IK worker subprocess: %s", e)
            self._cleanup()
            return False

    def stop(self) -> None:
        """Shutdown the worker subprocess."""
        self._cleanup()
        logger.info("IKWorkerClient stopped")

    def _cleanup(self) -> None:
        """Clean up subprocess and shared memory."""
        import time

        # Signal shutdown
        if self._shutdown_event:
            self._shutdown_event.set()

        # Wait for process to exit
        if self._process is not None:
            if self._process.is_alive():
                self._process.join(timeout=2.0)
                if self._process.is_alive():
                    logger.warning(
                        "IK worker subprocess did not exit cleanly, terminating"
                    )
                    self._process.terminate()
                    self._process.join(timeout=1.0)

            # Wait for exitcode to be set (indicates process fully terminated)
            # This ensures the subprocess's finally block has completed
            deadline = time.time() + 1.0
            while self._process.exitcode is None and time.time() < deadline:
                time.sleep(0.01)

        self._process = None
        self._shutdown_event = None
        self._request_event = None

        # Release all references to shared memory buffers before closing.
        # numpy views (frombuffer) hold pointers into the mmap; if they survive
        # past shm.close() the mmap raises BufferError, which during interpreter
        # shutdown can prevent later atexit handlers from running.
        self._output_arr = None
        try:
            if self._input_mv is not None:
                self._input_mv.release()
        except BufferError:
            pass
        try:
            if self._output_mv is not None:
                self._output_mv.release()
        except BufferError:
            pass
        self._input_mv = None
        self._output_mv = None

        # Clean up shared memory
        cleanup_shm(self._input_shm)
        cleanup_shm(self._output_shm)
        self._input_shm = None
        self._output_shm = None

    def is_alive(self) -> bool:
        """Check if worker subprocess is alive."""
        return self._process is not None and self._process.is_alive()

    def submit_request(self, q_rad: np.ndarray, T_matrix: np.ndarray) -> None:
        """
        Submit a new IK enablement request (non-blocking).

        Lazily starts the worker subprocess on first request.

        Args:
            q_rad: Current joint angles in radians (6 elements)
            T_matrix: Current pose as 4x4 transformation matrix
        """
        # Lazy start on first request
        if not self._started:
            self.start()

        if self._input_mv is None or self._request_event is None:
            return

        pack_ik_request(self._input_mv, q_rad, T_matrix)
        self._request_event.set()  # Wake up worker immediately

    def get_results_if_ready(self) -> bool:
        """
        Check for and update cached results if new data available (non-blocking, zero-alloc).

        Returns:
            True if new results were copied into the cached buffers, False otherwise.
            Access results via joint_en, cart_en_wrf, cart_en_trf properties.
        """
        if self._output_arr is None:
            return False

        new_version = unpack_ik_response_into(
            self._output_arr,
            self._last_resp_version,
            self._joint_en,
            self._cart_en_wrf,
            self._cart_en_trf,
        )

        if new_version > 0:
            self._last_resp_version = new_version
            return True

        return False

    @property
    def joint_en(self) -> np.ndarray:
        """Joint enablement flags (12 elements)."""
        return self._joint_en

    @property
    def cart_en_wrf(self) -> np.ndarray:
        """Cartesian enablement flags in world reference frame (12 elements)."""
        return self._cart_en_wrf

    @property
    def cart_en_trf(self) -> np.ndarray:
        """Cartesian enablement flags in tool reference frame (12 elements)."""
        return self._cart_en_trf
