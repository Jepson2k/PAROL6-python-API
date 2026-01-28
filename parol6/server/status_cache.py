"""
Cache of the aggregate STATUS payload for binary msgpack broadcasting.

The heavy IK enablement computations are delegated to a separate subprocess
via IKWorkerClient for true CPU parallelism.
"""

import time

import numpy as np
from numba import njit  # type: ignore[import-untyped]

from parol6.config import steps_to_deg, steps_to_rad
from parol6.protocol.wire import pack_status
from parol6.server.ik_worker_client import IKWorkerClient
from parol6.server.state import ControllerState, get_fkine_flat_mm, get_fkine_matrix


@njit(cache=True)
def _update_arrays(
    pos_in: np.ndarray,
    io_in: np.ndarray,
    spd_in: np.ndarray,
    grip_in: np.ndarray,
    pos_last: np.ndarray,
    angles_deg: np.ndarray,
    q_rad_buf: np.ndarray,
    io_cached: np.ndarray,
    spd_cached: np.ndarray,
    grip_cached: np.ndarray,
) -> tuple[bool, bool, bool, bool]:
    """
    Check for changes and update cached arrays.
    Returns (pos_changed, io_changed, spd_changed, grip_changed).
    """
    pos_changed = not np.array_equal(pos_in, pos_last)
    io_changed = not np.array_equal(io_in, io_cached)
    spd_changed = not np.array_equal(spd_in, spd_cached)
    grip_changed = not np.array_equal(grip_in, grip_cached)

    if pos_changed:
        pos_last[:] = pos_in
        steps_to_deg(pos_in, angles_deg)
        steps_to_rad(pos_in, q_rad_buf)
    if io_changed:
        io_cached[:] = io_in
    if spd_changed:
        spd_cached[:] = spd_in
    if grip_changed:
        grip_cached[:] = grip_in

    return pos_changed, io_changed, spd_changed, grip_changed


class StatusCache:
    """
    Cache of the aggregate STATUS payload components and formatted ASCII.

    Fields:
      - angles_deg: 6 floats
      - speeds: 6 ints (steps/sec)
      - io: 5 ints [in1,in2,out1,out2,estop]
      - gripper: >=6 ints [id,pos,spd,cur,status,obj]
      - pose: 16 floats (flattened transform)
      - last_update_s: wall clock time of last cache update
    """

    def __init__(self) -> None:
        # Public snapshots (materialized only when they change)
        self.angles_deg: np.ndarray = np.zeros((6,), dtype=np.float64)
        self.speeds: np.ndarray = np.zeros((6,), dtype=np.int32)
        self.io: np.ndarray = np.zeros((5,), dtype=np.uint8)
        self.gripper: np.ndarray = np.zeros((6,), dtype=np.int32)
        self.pose: np.ndarray = np.zeros((16,), dtype=np.float64)

        self.last_serial_s: float = 0.0  # last time a fresh serial frame was observed
        self._last_tool_name: str = "NONE"  # Track tool changes

        # Action tracking fields
        self._action_current: str = ""
        self._action_state: str = "IDLE"

        # Binary cache
        self._binary_cache: bytes = b""
        self._binary_dirty: bool = True

        # Change-detection caches to avoid expensive recomputation when inputs unchanged
        self._last_pos_in: np.ndarray = np.zeros((6,), dtype=np.int32)
        self._last_io_buf: np.ndarray = np.zeros((5,), dtype=np.uint8)

        # Pre-allocated buffer for IK request (avoids allocation per position change)
        self._q_rad_buf: np.ndarray = np.zeros(6, dtype=np.float64)

        # IK worker client for async enablement computation
        self._ik_client = IKWorkerClient()
        self._ik_client.start()

    def __del__(self) -> None:
        """Clean up IK worker on destruction."""
        if hasattr(self, "_ik_client") and self._ik_client:
            self._ik_client.stop()

    def update_from_state(self, state: ControllerState) -> None:
        """
        Update cache from current controller state with change gating:
          - Only recompute angles/pose when Position_in changes
          - Only refresh IO/speeds/gripper when their inputs actually change
          - IK enablement is computed asynchronously in a subprocess
        """
        # Do change detection
        self._last_io_buf[:] = state.InOut_in[:5]
        pos_changed, io_changed, spd_changed, grip_changed = _update_arrays(
            state.Position_in,
            self._last_io_buf,
            state.Speed_in,
            state.Gripper_data_in,
            self._last_pos_in,
            self.angles_deg,
            self._q_rad_buf,
            self.io,
            self.speeds,
            self.gripper,
        )
        tool_changed = state.current_tool != self._last_tool_name

        if pos_changed or tool_changed:
            if tool_changed:
                self._last_tool_name = state.current_tool
            self.pose[:] = get_fkine_flat_mm(state)
            # Submit IK request asynchronously (non-critical, errors are logged in client)
            try:
                T_matrix = get_fkine_matrix(state)
                self._ik_client.submit_request(self._q_rad_buf, T_matrix)
            except (ValueError, OSError):
                # IK submission failed - client not ready or invalid input
                pass

        # Poll for async IK results (non-blocking, zero-alloc)
        ik_changed = self._ik_client.get_results_if_ready()

        action_changed = (
            self._action_current != state.action_current
            or self._action_state != state.action_state
        )
        if action_changed:
            self._action_current = state.action_current
            self._action_state = state.action_state

        # Mark binary cache dirty if anything changed
        if (
            pos_changed
            or tool_changed
            or io_changed
            or spd_changed
            or grip_changed
            or ik_changed
            or action_changed
        ):
            self._binary_dirty = True

    def to_binary(self) -> bytes:
        """Return the msgpack-encoded STATUS payload."""
        if self._binary_dirty:
            self._binary_cache = pack_status(
                self.pose,
                self.angles_deg,
                self.speeds,
                self.io,
                self.gripper,
                self._action_current,
                self._action_state,
                self._ik_client.joint_en,
                self._ik_client.cart_en_wrf,
                self._ik_client.cart_en_trf,
            )
            self._binary_dirty = False
        return self._binary_cache

    def mark_serial_observed(self) -> None:
        """Mark that a fresh serial frame was observed just now."""
        self.last_serial_s = time.monotonic()

    def age_s(self) -> float:
        """Seconds since last fresh serial observation (used to gate broadcasting)."""
        if self.last_serial_s <= 0:
            return 1e9
        return time.monotonic() - self.last_serial_s


# Module-level singleton
_status_cache: StatusCache | None = None


def get_cache() -> StatusCache:
    global _status_cache
    if _status_cache is None:
        _status_cache = StatusCache()
    return _status_cache
