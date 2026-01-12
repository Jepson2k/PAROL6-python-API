"""
Thread-safe cache of the aggregate STATUS payload components and formatted ASCII.

The heavy IK enablement computations are delegated to a separate subprocess
via IKWorkerClient for true CPU parallelism.
"""

import threading
import time

import numpy as np
from numpy.typing import ArrayLike

from parol6.config import steps_to_deg, steps_to_rad
from parol6.server.ik_worker_client import IKWorkerClient
from parol6.server.state import ControllerState, get_fkine_flat_mm, get_fkine_matrix


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
        now = time.time()
        changed_any = False

        with self._lock:
            # Check if position or tool changed
            tool_changed = state.current_tool != self._last_tool_name
            pos_changed = self._last_pos_in is None or not np.array_equal(
                state.Position_in, self._last_pos_in
            )

            if pos_changed or tool_changed:
                if pos_changed:
                    np.copyto(self._last_pos_in, state.Position_in)
                if tool_changed:
                    self._last_tool_name = state.current_tool

                # Vectorized steps->deg
                steps_to_deg(state.Position_in, self.angles_deg)
                self._angles_ascii = self._format_csv_from_list(self.angles_deg)
                changed_any = True

                # Get cached fkine (automatically updates if needed)
                pose_flat_mm = get_fkine_flat_mm(state)
                np.copyto(self.pose, pose_flat_mm)
                self._pose_ascii = self._format_csv_from_list(self.pose)
                changed_any = True

                # Submit IK request asynchronously
                try:
                    q_rad = np.zeros(6, dtype=np.float64)
                    steps_to_rad(state.Position_in, q_rad)
                    T_matrix = get_fkine_matrix(state)
                    self._ik_client.submit_request(q_rad, T_matrix)
                except Exception:
                    pass  # IK request failed, will use cached values

            # Poll for async IK results (non-blocking)
            results = self._ik_client.get_results_if_ready()
            if results is not None:
                self.joint_en[:] = results[0]
                self.cart_en_wrf[:] = results[1]
                self.cart_en_trf[:] = results[2]
                self._joint_en_ascii = self._format_csv_from_list(
                    self.joint_en.tolist()
                )
                self._cart_en_wrf_ascii = self._format_csv_from_list(
                    self.cart_en_wrf.tolist()
                )
                self._cart_en_trf_ascii = self._format_csv_from_list(
                    self.cart_en_trf.tolist()
                )
                changed_any = True

            # 2) IO (first 5)
            if not np.array_equal(self.io, state.InOut_in[:5]):
                np.copyto(self.io, state.InOut_in[:5])
                self._io_ascii = self._format_csv_from_list(self.io)
                changed_any = True

            # 3) Speeds (steps/sec from Speed_in)
            if not np.array_equal(self.speeds, state.Speed_in):
                np.copyto(self.speeds, state.Speed_in)
                self._speeds_ascii = self._format_csv_from_list(self.speeds)
                changed_any = True

            # 4) Gripper
            if not np.array_equal(self.gripper, state.Gripper_data_in):
                np.copyto(self.gripper, state.Gripper_data_in)
                self._gripper_ascii = self._format_csv_from_list(self.gripper)
                changed_any = True

            # 5) Action tracking
            if (
                self._action_current != state.action_current
                or self._action_state != state.action_state
            ):
                self._action_current = state.action_current
                self._action_state = state.action_state
                changed_any = True

            # 6) Assemble full ASCII only if any section changed
            if changed_any:
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
        self.last_serial_s = time.time()

    def age_s(self) -> float:
        """Seconds since last fresh serial observation (used to gate broadcasting)."""
        if self.last_serial_s <= 0:
            return 1e9
        return time.time() - self.last_serial_s


# Module-level singleton
_status_cache: StatusCache | None = None


def get_cache() -> StatusCache:
    global _status_cache
    if _status_cache is None:
        _status_cache = StatusCache()
    return _status_cache
