"""
IK Helper Functions and Utilities
Shared functions used by multiple command classes for inverse kinematics calculations.
"""

import logging
import time
from collections.abc import Sequence

import numpy as np
from numba import njit  # type: ignore[import-untyped]
from numpy.typing import ArrayLike, NDArray
from roboticstoolbox import DHRobot
from roboticstoolbox.robot.IK import IKResultBuffer

import parol6.PAROL6_ROBOT as PAROL6_ROBOT
from parol6.config import IK_SAFETY_MARGINS_RAD

logger = logging.getLogger(__name__)

# Rate limiting for IK warnings (avoid log spam at 250Hz)
_ik_warn_state: dict = {
    "last_warn_time": 0.0,
    "interval": 1.0,  # Log at most once per second
}


def _rate_limited_warning(msg: str) -> None:
    """Log a warning with rate limiting to avoid spam."""
    now = time.monotonic()
    if now - _ik_warn_state["last_warn_time"] > _ik_warn_state["interval"]:
        logger.warning(msg)
        _ik_warn_state["last_warn_time"] = now


# Cache for robot.qlim and robot.ets() to avoid expensive recomputation
# Invalidated when robot instance changes (e.g., tool change)
_ik_cache: dict = {
    "robot_id": None,
    "qlim": None,
    "ets": None,
    "buffered_min": None,
    "buffered_max": None,
    "result_buf": None,  # Pre-allocated IKResultBuffer for zero-allocation IK
}


def _get_cached_ik_data(robot: "DHRobot") -> tuple:
    """
    Get cached qlim, ets, buffered limits, and result buffer for the robot.
    Cache is invalidated when robot instance changes.

    Returns (qlim, ets, buffered_min, buffered_max, result_buf)
    """
    robot_id = id(robot)
    if _ik_cache["robot_id"] != robot_id:
        qlim = robot.qlim
        n_joints = qlim.shape[1]
        _ik_cache["robot_id"] = robot_id
        _ik_cache["qlim"] = qlim
        _ik_cache["ets"] = robot.ets()
        _ik_cache["buffered_min"] = qlim[0, :] + IK_SAFETY_MARGINS_RAD[:, 0]
        _ik_cache["buffered_max"] = qlim[1, :] - IK_SAFETY_MARGINS_RAD[:, 1]
        _ik_cache["result_buf"] = SolveIKResultBuffer(n_joints)

    return (
        _ik_cache["qlim"],
        _ik_cache["ets"],
        _ik_cache["buffered_min"],
        _ik_cache["buffered_max"],
        _ik_cache["result_buf"],
    )


# This dictionary maps descriptive axis names to movement vectors
# Format: ([x, y, z], [rx, ry, rz])
AXIS_MAP = {
    "X+": ([1, 0, 0], [0, 0, 0]),
    "X-": ([-1, 0, 0], [0, 0, 0]),
    "Y+": ([0, 1, 0], [0, 0, 0]),
    "Y-": ([0, -1, 0], [0, 0, 0]),
    "Z+": ([0, 0, 1], [0, 0, 0]),
    "Z-": ([0, 0, -1], [0, 0, 0]),
    "RX+": ([0, 0, 0], [1, 0, 0]),
    "RX-": ([0, 0, 0], [-1, 0, 0]),
    "RY+": ([0, 0, 0], [0, 1, 0]),
    "RY-": ([0, 0, 0], [0, -1, 0]),
    "RZ+": ([0, 0, 0], [0, 0, 1]),
    "RZ-": ([0, 0, 0], [0, 0, -1]),
}


@njit(cache=True)
def unwrap_angles(
    q_solution: NDArray[np.float64],
    q_current: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Vectorized unwrap: bring solution angles near current by adding/subtracting 2*pi.
    This minimizes joint motion between consecutive configurations.
    """
    qs = np.asarray(q_solution, dtype=np.float64)
    qc = np.asarray(q_current, dtype=np.float64)
    diff = qs - qc
    q_unwrapped = qs.copy()
    q_unwrapped[diff > np.pi] -= 2 * np.pi
    q_unwrapped[diff < -np.pi] += 2 * np.pi
    return q_unwrapped


@njit(cache=True)
def _ik_safety_check(
    q: NDArray[np.float64],
    cq: NDArray[np.float64],
    buffered_min: NDArray[np.float64],
    buffered_max: NDArray[np.float64],
    qlim_min: NDArray[np.float64],
    qlim_max: NDArray[np.float64],
) -> tuple[bool, bool, int]:
    """
    JIT-compiled IK safety validation.
    Returns (ok, is_recovery_violation, violation_idx).
    """
    in_danger_old = (cq < buffered_min) | (cq > buffered_max)
    dist_old = np.minimum(np.abs(cq - qlim_min), np.abs(cq - qlim_max))
    dist_new = np.minimum(np.abs(q - qlim_min), np.abs(q - qlim_max))
    recovery_violations = in_danger_old & (dist_new < dist_old)
    in_danger_new = (q < buffered_min) | (q > buffered_max)
    safety_violations = (~in_danger_old) & in_danger_new

    if np.any(recovery_violations):
        return False, True, int(np.argmax(recovery_violations))
    if np.any(safety_violations):
        return False, False, int(np.argmax(safety_violations))
    return True, False, -1


class SolveIKResultBuffer(IKResultBuffer):
    """IKResultBuffer extended with violations field."""

    def __init__(self, n: int = 6) -> None:
        super().__init__(n)
        self.violations: str | None = None


def solve_ik(
    robot: DHRobot,
    target_pose: NDArray[np.float64],
    current_q: Sequence[float] | NDArray[np.float64],
    quiet_logging: bool = False,
) -> SolveIKResultBuffer:
    """
    IK solver with per-joint safety margins.

    Per-joint safety margins (IK_SAFETY_MARGINS_RAD) are always enforced.
    Joints like J3 (elbow) have larger margins because backwards bend creates
    trap configurations that are hard to recover from.

    Parameters
    ----------
    robot : DHRobot
        Robot model
    target_pose : NDArray[np.float64]
        Target pose as 4x4 SE3 transformation matrix
    current_q : array_like
        Current joint configuration in radians
    quiet_logging : bool, optional
        If True, suppress warning logs (default: False)

    Returns
    -------
    SolveIKResultBuffer
        success - True if solution found
        q - Joint configuration in radians
        iterations - Number of iterations used
        residual - Final error value
        violations - Error message if failed, None if successful
    """
    cq: NDArray[np.float64] = np.asarray(current_q, dtype=np.float64)

    # Get cached robot data (qlim, ets, buffered limits, result buffer)
    qlim, ets, buffered_min, buffered_max, result = _get_cached_ik_data(robot)

    # ik_LM accepts numpy 4x4 matrices
    target_matrix = np.asarray(target_pose, dtype=np.float64)

    ets.ik_LM(
        target_matrix,
        q0=cq,
        tol=1e-12,
        joint_limits=True,
        k=0.0,
        method="sugihara",
        ilimit=10,
        slimit=10,
        result=result,  # Zero-allocation: C writes to pre-allocated buffer
    )  # tol=1e-12 required for sub-mm jogs

    result.violations = None

    if result.success:
        # JIT-compiled safety validation
        ok, is_recovery, idx = _ik_safety_check(
            result.q, cq, buffered_min, buffered_max, qlim[0, :], qlim[1, :]
        )
        if not ok:
            result._scalars[0] = 0  # Set success=False via underlying storage
            if is_recovery:
                result.violations = (
                    f"J{idx + 1} moving further into danger zone (recovery blocked)"
                )
            else:
                result.violations = (
                    f"J{idx + 1} would leave safe zone (buffer violated)"
                )
            if not quiet_logging:
                _rate_limited_warning(result.violations)

    if result.success:
        # Valid solution - apply unwrapping to minimize joint motion
        q_unwrapped = unwrap_angles(result.q, current_q)

        # Verify unwrapped solution still within actual limits
        within_limits = check_limits(
            current_q, q_unwrapped, allow_recovery=True, log=not quiet_logging
        )

        if within_limits:
            result.q[:] = q_unwrapped
        # else: keep original result.q which already passed library's limit check
    else:
        result.violations = "IK failed to solve."

    return result


# -----------------------------
# Fast, vectorized limit checking with edge-triggered logging
# -----------------------------
# Pre-allocated buffers for check_limits (avoid per-call allocation)
_cl_viol = np.zeros(6, dtype=np.bool_)
_cl_below = np.zeros(6, dtype=np.bool_)
_cl_above = np.zeros(6, dtype=np.bool_)
_cl_cur_viol = np.zeros(6, dtype=np.bool_)
_cl_t_below = np.zeros(6, dtype=np.bool_)
_cl_t_above = np.zeros(6, dtype=np.bool_)
_last_violation_mask = np.zeros(6, dtype=np.bool_)
_last_any_violation = False


@njit(cache=True)
def _check_limits_core(
    q_arr: NDArray[np.float64],
    t_arr: NDArray[np.float64],
    mn: NDArray[np.float64],
    mx: NDArray[np.float64],
    allow_recovery: bool,
    has_target: bool,
    viol_out: NDArray[np.bool_],
    below_out: NDArray[np.bool_],
    above_out: NDArray[np.bool_],
    cur_viol_out: NDArray[np.bool_],
    t_below_out: NDArray[np.bool_],
    t_above_out: NDArray[np.bool_],
) -> bool:
    """JIT-compiled core of check_limits. Writes masks to output buffers."""
    for i in range(6):
        below_out[i] = q_arr[i] < mn[i]
        above_out[i] = q_arr[i] > mx[i]
        cur_viol_out[i] = below_out[i] or above_out[i]

    if not has_target:
        all_ok = True
        for i in range(6):
            viol_out[i] = cur_viol_out[i]
            if viol_out[i]:
                all_ok = False
        return all_ok

    for i in range(6):
        t_below_out[i] = t_arr[i] < mn[i]
        t_above_out[i] = t_arr[i] > mx[i]

    all_ok = True
    for i in range(6):
        t_viol = t_below_out[i] or t_above_out[i]
        if allow_recovery:
            rec_ok = (above_out[i] and t_arr[i] <= q_arr[i]) or (
                below_out[i] and t_arr[i] >= q_arr[i]
            )
        else:
            rec_ok = False
        ok = (not cur_viol_out[i] and not t_viol) or (cur_viol_out[i] and rec_ok)
        viol_out[i] = not ok
        if not ok:
            all_ok = False

    return all_ok


def check_limits(
    q: ArrayLike,
    target_q: ArrayLike | None = None,
    allow_recovery: bool = True,
    *,
    log: bool = True,
) -> bool:
    """
    Vectorized limits check in radians.
    - q: current joint angles in radians (array-like)
    - target_q: optional target joint angles in radians (array-like)
    - allow_recovery: allow movement that heads back toward valid range if currently violating
    - log: emit edge-triggered warning/info logs on violation state changes

    Returns True if move is allowed (within limits or valid recovery), False otherwise.
    """
    global _last_any_violation

    q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
    mn = PAROL6_ROBOT._joint_limits_radian[:, 0]
    mx = PAROL6_ROBOT._joint_limits_radian[:, 1]
    has_target = target_q is not None
    t_arr = (
        np.asarray(target_q, dtype=np.float64).reshape(-1)
        if has_target
        else np.zeros(6, dtype=np.float64)
    )

    all_ok = _check_limits_core(
        q_arr,
        t_arr,
        mn,
        mx,
        allow_recovery,
        has_target,
        _cl_viol,
        _cl_below,
        _cl_above,
        _cl_cur_viol,
        _cl_t_below,
        _cl_t_above,
    )

    if log:
        any_viol = not all_ok

        # Edge-triggered violation logs
        if any_viol and (
            np.any(_cl_viol != _last_violation_mask) or not _last_any_violation
        ):
            idxs = np.where(_cl_viol)[0]
            tokens = []
            for i in idxs:
                if _cl_cur_viol[i]:
                    tokens.append(
                        f"J{i + 1}:" + ("cur<min" if _cl_below[i] else "cur>max")
                    )
                elif has_target and _cl_t_below[i]:
                    tokens.append(f"J{i + 1}:target<min")
                elif has_target and _cl_t_above[i]:
                    tokens.append(f"J{i + 1}:target>max")
                else:
                    tokens.append(f"J{i + 1}:violation")
            logger.warning("LIMIT VIOLATION: %s", " ".join(tokens))
        elif (not any_viol) and _last_any_violation:
            logger.info("Limits back in range")

        _last_violation_mask[:] = _cl_viol
        _last_any_violation = any_viol

    return all_ok
