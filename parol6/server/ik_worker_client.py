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
    get_ik_resp_seq,
    pack_ik_request,
    unpack_ik_response,
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
        self._process: Process | None = None
        self._shutdown_event: Event | None = None

        self._last_req_seq = 0
        self._last_resp_seq = 0
        self._started = False

        # Unique names for shared memory segments
        self._shm_suffix = f"_{id(self)}"

        # Cached results
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

            # Initialize with zeros
            self._input_mv[:] = bytes(IK_INPUT_SHM_SIZE)
            self._output_mv[:] = bytes(IK_OUTPUT_SHM_SIZE)

            # Spawn subprocess
            self._shutdown_event = multiprocessing.Event()
            self._process = Process(
                target=ik_enablement_worker_main,
                args=(input_name, output_name, self._shutdown_event),
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
        # Signal shutdown
        if self._shutdown_event:
            self._shutdown_event.set()

        # Wait for process to exit
        if self._process and self._process.is_alive():
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                logger.warning("IK worker subprocess did not exit cleanly, terminating")
                self._process.terminate()
                self._process.join(timeout=1.0)

        self._process = None
        self._shutdown_event = None

        # Release memoryviews before closing shared memory to avoid BufferError
        try:
            if self._input_mv is not None:
                self._input_mv.release()
        except Exception:
            pass
        try:
            if self._output_mv is not None:
                self._output_mv.release()
        except Exception:
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

        if self._input_mv is None:
            return

        self._last_req_seq += 1
        pack_ik_request(self._input_mv, q_rad, T_matrix, self._last_req_seq)

    def get_results_if_ready(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Check for and return results (non-blocking).

        Returns:
            Tuple of (joint_en, cart_en_wrf, cart_en_trf) if new results available,
            None otherwise.
        """
        if self._output_mv is None:
            return None

        resp_seq = get_ik_resp_seq(self._output_mv)

        if resp_seq != self._last_resp_seq and resp_seq == self._last_req_seq:
            self._last_resp_seq = resp_seq
            joint_en, cart_en_wrf, cart_en_trf, _ = unpack_ik_response(self._output_mv)

            # Cache results
            self._joint_en = joint_en
            self._cart_en_wrf = cart_en_wrf
            self._cart_en_trf = cart_en_trf

            return (joint_en, cart_en_wrf, cart_en_trf)

        return None

    def get_cached_results(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the most recent cached results.

        Returns:
            Tuple of (joint_en, cart_en_wrf, cart_en_trf)
        """
        return (
            self._joint_en.copy(),
            self._cart_en_wrf.copy(),
            self._cart_en_trf.copy(),
        )

    def has_pending_request(self) -> bool:
        """Check if there's a pending request awaiting response."""
        return self._last_req_seq > self._last_resp_seq
