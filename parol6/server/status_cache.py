"""
Thread-safe cache of the aggregate STATUS payload components and formatted ASCII.

The heavy IK enablement computations are delegated to a separate subprocess
via IKWorkerClient for true CPU parallelism.
"""

import threading
import time

import numpy as np
from numba import njit  # type: ignore[import-untyped]
from numpy.typing import ArrayLike

from parol6.config import steps_to_deg, steps_to_rad
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
    Thread-safe cache of the aggregate STATUS payload components and formatted ASCII.

    Fields:
      - angles_deg: 6 floats
      - speeds: 6 ints (steps/sec)
      - io: 5 ints [in1,in2,out1,out2,estop]
      - gripper: >=6 ints [id,pos,spd,cur,status,obj]
      - pose: 16 floats (flattened transform)
      - last_update_s: wall clock time of last cache update
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        # Public snapshots (materialized only when they change)
        self.angles_deg: np.ndarray = np.zeros((6,), dtype=np.float64)
        self.speeds: np.ndarray = np.zeros((6,), dtype=np.int32)
        self.io: np.ndarray = np.zeros((5,), dtype=np.uint8)
        self.gripper: np.ndarray = np.zeros((6,), dtype=np.int32)
        self.pose: np.ndarray = np.zeros((16,), dtype=np.float64)

        self.last_update_s: float = 0.0  # last cache build (any section)
        self.last_serial_s: float = 0.0  # last time a fresh serial frame was observed
        self._last_tool_name: str = "NONE"  # Track tool changes

        # Cached ASCII fragments to reduce allocations
        self._angles_ascii: str = "0,0,0,0,0,0"
        self._speeds_ascii: str = "0,0,0,0,0,0"
        self._io_ascii: str = "0,0,0,0,0"
        self._gripper_ascii: str = "0,0,0,0,0,0"
        self._pose_ascii: str = ",".join("0" for _ in range(16))

        # Action tracking fields
        self._action_current: str = ""
        self._action_state: str = "IDLE"

        # Enablement arrays (12 ints each)
        self.joint_en = np.ones((12,), dtype=np.uint8)
        self.cart_en_wrf = np.ones((12,), dtype=np.uint8)
        self.cart_en_trf = np.ones((12,), dtype=np.uint8)
        self._joint_en_ascii: str = ",".join(str(int(v)) for v in self.joint_en)
        self._cart_en_wrf_ascii: str = ",".join(str(int(v)) for v in self.cart_en_wrf)
        self._cart_en_trf_ascii: str = ",".join(str(int(v)) for v in self.cart_en_trf)

        self._ascii_full: str = (
            f"STATUS|POSE={self._pose_ascii}"
            f"|ANGLES={self._angles_ascii}"
            f"|SPEEDS={self._speeds_ascii}"
            f"|IO={self._io_ascii}"
            f"|GRIPPER={self._gripper_ascii}"
            f"|ACTION_CURRENT={self._action_current}"
            f"|ACTION_STATE={self._action_state}"
            f"|JOINT_EN={self._joint_en_ascii}"
            f"|CART_EN_WRF={self._cart_en_wrf_ascii}"
            f"|CART_EN_TRF={self._cart_en_trf_ascii}"
        )

        # Change-detection caches to avoid expensive recomputation when inputs unchanged
        self._last_pos_in: np.ndarray = np.zeros((6,), dtype=np.int32)
        self._last_io_buf: np.ndarray = np.zeros((5,), dtype=np.uint8)

        # Pre-allocated buffer for IK request (avoids allocation per position change)
        self._q_rad_buf: np.ndarray = np.zeros(6, dtype=np.float64)

        # IK worker client for async enablement computation (lazy-started on first request)
        self._ik_client = IKWorkerClient()

    def __del__(self) -> None:
        """Clean up IK worker on destruction."""
        if hasattr(self, "_ik_client") and self._ik_client:
            self._ik_client.stop()

    def _format_csv_from_list(self, vals: ArrayLike) -> str:
        return ",".join(str(v) for v in vals)  # type: ignore

    def update_from_state(self, state: ControllerState) -> None:
        """
        Update cache from current controller state with change gating:
          - Only recompute angles/pose when Position_in changes
          - Only refresh IO/speeds/gripper when their inputs actually change
          - IK enablement is computed asynchronously in a subprocess
        """
        now = time.monotonic()

        # Do change detection outside lock to minimize critical section
        np.copyto(self._last_io_buf, state.InOut_in[:5])
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

        # Pre-compute ASCII strings outside lock if changed
        new_angles_ascii = None
        new_pose_ascii = None
        new_joint_en_ascii = None
        new_cart_en_wrf_ascii = None
        new_cart_en_trf_ascii = None
        new_io_ascii = None
        new_speeds_ascii = None
        new_gripper_ascii = None

        if pos_changed or tool_changed:
            if tool_changed:
                self._last_tool_name = state.current_tool
            new_angles_ascii = self._format_csv_from_list(self.angles_deg)
            pose_flat_mm = get_fkine_flat_mm(state)
            np.copyto(self.pose, pose_flat_mm)
            new_pose_ascii = self._format_csv_from_list(self.pose)
            # Submit IK request asynchronously
            try:
                T_matrix = get_fkine_matrix(state)
                self._ik_client.submit_request(self._q_rad_buf, T_matrix)
            except Exception:
                pass

        # Poll for async IK results (non-blocking)
        results = self._ik_client.get_results_if_ready()
        if results is not None:
            np.copyto(self.joint_en, results[0])
            np.copyto(self.cart_en_wrf, results[1])
            np.copyto(self.cart_en_trf, results[2])
            new_joint_en_ascii = self._format_csv_from_list(self.joint_en)
            new_cart_en_wrf_ascii = self._format_csv_from_list(self.cart_en_wrf)
            new_cart_en_trf_ascii = self._format_csv_from_list(self.cart_en_trf)

        if io_changed:
            new_io_ascii = self._format_csv_from_list(self.io)

        if spd_changed:
            new_speeds_ascii = self._format_csv_from_list(self.speeds)

        if grip_changed:
            new_gripper_ascii = self._format_csv_from_list(self.gripper)

        action_changed = (
            self._action_current != state.action_current
            or self._action_state != state.action_state
        )

        changed_any = (
            new_angles_ascii is not None
            or results is not None
            or new_io_ascii is not None
            or new_speeds_ascii is not None
            or new_gripper_ascii is not None
            or action_changed
        )

        # Minimal critical section - only update cached values
        if changed_any:
            with self._lock:
                if new_angles_ascii is not None:
                    self._angles_ascii = new_angles_ascii
                    self._pose_ascii = new_pose_ascii  # type: ignore[assignment]
                if results is not None:
                    self._joint_en_ascii = new_joint_en_ascii  # type: ignore[assignment]
                    self._cart_en_wrf_ascii = new_cart_en_wrf_ascii  # type: ignore[assignment]
                    self._cart_en_trf_ascii = new_cart_en_trf_ascii  # type: ignore[assignment]
                if new_io_ascii is not None:
                    self._io_ascii = new_io_ascii
                if new_speeds_ascii is not None:
                    self._speeds_ascii = new_speeds_ascii
                if new_gripper_ascii is not None:
                    self._gripper_ascii = new_gripper_ascii
                if action_changed:
                    self._action_current = state.action_current
                    self._action_state = state.action_state

                self._ascii_full = (
                    f"STATUS|POSE={self._pose_ascii}"
                    f"|ANGLES={self._angles_ascii}"
                    f"|SPEEDS={self._speeds_ascii}"
                    f"|IO={self._io_ascii}"
                    f"|GRIPPER={self._gripper_ascii}"
                    f"|ACTION_CURRENT={self._action_current}"
                    f"|ACTION_STATE={self._action_state}"
                    f"|JOINT_EN={self._joint_en_ascii}"
                    f"|CART_EN_WRF={self._cart_en_wrf_ascii}"
                    f"|CART_EN_TRF={self._cart_en_trf_ascii}"
                )
                self.last_update_s = now

    def to_ascii(self) -> str:
        """Return the full ASCII STATUS payload."""
        with self._lock:
            return self._ascii_full

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
