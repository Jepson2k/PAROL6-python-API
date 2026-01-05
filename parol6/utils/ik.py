"""
IK Helper Functions and Utilities
Shared functions used by multiple command classes for inverse kinematics calculations.
"""

import logging
import time
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import sophuspy as sp
from numpy.typing import ArrayLike, NDArray
from roboticstoolbox import DHRobot

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
}


def _get_cached_ik_data(robot: "DHRobot") -> tuple:
    """
    Get cached qlim, ets, and buffered limits for the robot.
    Cache is invalidated when robot instance changes.

    Returns (qlim, ets, buffered_min, buffered_max)
    """
    robot_id = id(robot)
    if _ik_cache["robot_id"] != robot_id:
        qlim = robot.qlim
        _ik_cache["robot_id"] = robot_id
        _ik_cache["qlim"] = qlim
        _ik_cache["ets"] = robot.ets()
        _ik_cache["buffered_min"] = qlim[0, :] + IK_SAFETY_MARGINS_RAD[:, 0]
        _ik_cache["buffered_max"] = qlim[1, :] - IK_SAFETY_MARGINS_RAD[:, 1]

    return (
        _ik_cache["qlim"],
        _ik_cache["ets"],
        _ik_cache["buffered_min"],
        _ik_cache["buffered_max"],
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


def unwrap_angles(
    q_solution: Sequence[float] | NDArray[np.float64],
    q_current: Sequence[float] | NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Vectorized unwrap: bring solution angles near current by adding/subtracting 2*pi.
    This minimizes joint motion between consecutive configurations.
    """
    qs = np.asarray(q_solution, dtype=float)
    qc = np.asarray(q_current, dtype=float)
    diff = qs - qc
    q_unwrapped = qs.copy()
    q_unwrapped[diff > np.pi] -= 2 * np.pi
    q_unwrapped[diff < -np.pi] += 2 * np.pi
    return q_unwrapped


class IKResult(NamedTuple):
    success: bool
    q: NDArray[np.float64] | None
    iterations: int
    residual: float
    violations: str | None


def solve_ik(
    robot: DHRobot,
    target_pose: sp.SE3,
    current_q: Sequence[float] | NDArray[np.float64],
    quiet_logging: bool = False,
) -> IKResult:
    """
    IK solver with per-joint safety margins.

    Per-joint safety margins (IK_SAFETY_MARGINS_RAD) are always enforced.
    Joints like J3 (elbow) have larger margins because backwards bend creates
    trap configurations that are hard to recover from.

    Parameters
    ----------
    robot : DHRobot
        Robot model
    target_pose : sp.SE3
        Target pose to reach (sophuspy SE3)
    current_q : array_like
        Current joint configuration in radians
    quiet_logging : bool, optional
        If True, suppress warning logs (default: False)

    Returns
    -------
    IKResult
        success - True if solution found
        q - Joint configuration in radians (or None if failed)
        iterations - Number of iterations used
        residual - Final error value
        violations - Error message if failed, None if successful
    """
    cq: NDArray[np.float64] = np.asarray(current_q, dtype=np.float64)

    # Get cached robot data (qlim, ets, buffered limits)
    qlim, ets, buffered_min, buffered_max = _get_cached_ik_data(robot)

    # ik_LM accepts numpy 4x4 matrices - extract from sophuspy SE3
    target_matrix = target_pose.matrix()

    result = ets.ik_LM(
        target_matrix, q0=cq, tol=1e-10, joint_limits=True, k=0.0, method="sugihara", ilimit=10, slimit=33
    )  # Small tol needed so it moves at slow speeds

    q = result[0]
    success = result[1] > 0
    iterations = result[2]
    remaining = result[3]

    violations = None

    if success:
        # Vectorized safety validation with per-joint direction-aware margins (using cached values)

        # Check which joints were in danger zone (beyond buffer)
        in_danger_old = (cq < buffered_min) | (cq > buffered_max)

        # Compute distance from nearest limit for all joints
        dist_old_lower = np.abs(cq - qlim[0, :])
        dist_old_upper = np.abs(cq - qlim[1, :])
        dist_old = np.minimum(dist_old_lower, dist_old_upper)

        dist_new_lower = np.abs(q - qlim[0, :])
        dist_new_upper = np.abs(q - qlim[1, :])
        dist_new = np.minimum(dist_new_lower, dist_new_upper)

        # Check for recovery violations (was in danger, moved closer to limit)
        recovery_violations = in_danger_old & (dist_new < dist_old)

        # Check for safety violations (was safe, left buffer zone)
        in_danger_new = (q < buffered_min) | (q > buffered_max)
        safety_violations = (~in_danger_old) & in_danger_new

        # Report first violation found
        if np.any(recovery_violations):
            idx = np.argmax(recovery_violations)
            success = False
            violations = (
                f"J{idx + 1} moving further into danger zone (recovery blocked)"
            )
            if not quiet_logging:
                _rate_limited_warning(violations)
        elif np.any(safety_violations):
            idx = np.argmax(safety_violations)
            success = False
            violations = f"J{idx + 1} would leave safe zone (buffer violated)"
            if not quiet_logging:
                _rate_limited_warning(violations)

    if success:
        # Valid solution - apply unwrapping to minimize joint motion
        q_unwrapped = unwrap_angles(q, current_q)

        # Verify unwrapped solution still within actual limits
        within_limits = check_limits(
            current_q, q_unwrapped, allow_recovery=True, log=not quiet_logging
        )

        if within_limits:
            q = q_unwrapped
        # else: use original result.q which already passed library's limit check
    else:
        violations = "IK failed to solve."

    return IKResult(
        success=success,
        q=q if success else None,
        iterations=iterations,
        residual=remaining,
        violations=violations,
    )


# -----------------------------
# Fast, vectorized limit checking with edge-triggered logging
# -----------------------------
_last_violation_mask = np.zeros(6, dtype=bool)
_last_any_violation = False


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
    global _last_violation_mask, _last_any_violation

    q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
    mn = PAROL6_ROBOT._joint_limits_radian[:, 0]
    mx = PAROL6_ROBOT._joint_limits_radian[:, 1]

    below = q_arr < mn
    above = q_arr > mx
    cur_viol = below | above

    if target_q is None:
        ok_mask = ~cur_viol
        t_below = t_above = None
    else:
        t = np.asarray(target_q, dtype=np.float64).reshape(-1)
        t_below = t < mn
        t_above = t > mx
        t_viol = t_below | t_above
        if allow_recovery:
            rec_ok = (above & (t <= q_arr)) | (below & (t >= q_arr))
        else:
            rec_ok = np.zeros(6, dtype=bool)
        ok_mask = (~cur_viol & ~t_viol) | (cur_viol & rec_ok)

    all_ok = bool(np.all(ok_mask))

    if log:
        viol = ~ok_mask
        any_viol = bool(np.any(viol))

        # Edge-triggered violation logs
        if any_viol and (
            np.any(viol != _last_violation_mask) or not _last_any_violation
        ):
            idxs = np.where(viol)[0]
            tokens = []
            for i in idxs:
                if cur_viol[i]:
                    tokens.append(f"J{i + 1}:" + ("cur<min" if below[i] else "cur>max"))
                else:
                    # target violates
                    if t_below is not None and t_below[i]:
                        tokens.append(f"J{i + 1}:target<min")
                    elif t_above is not None and t_above[i]:
                        tokens.append(f"J{i + 1}:target>max")
                    else:
                        tokens.append(f"J{i + 1}:violation")
            logger.warning("LIMIT VIOLATION: %s", " ".join(tokens))
        elif (not any_viol) and _last_any_violation:
            logger.info("Limits back in range")

        _last_violation_mask[:] = viol
        _last_any_violation = any_viol

    return all_ok


def check_limits_mask(
    q: ArrayLike, target_q: ArrayLike | None = None, allow_recovery: bool = True
) -> NDArray[np.bool_]:
    """Return per-joint boolean mask (True if OK for that joint)."""
    q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
    mn = PAROL6_ROBOT._joint_limits_radian[:, 0]
    mx = PAROL6_ROBOT._joint_limits_radian[:, 1]
    below = q_arr < mn
    above = q_arr > mx
    cur_viol = below | above

    if target_q is None:
        return ~cur_viol
    t = np.asarray(target_q, dtype=np.float64).reshape(-1)
    t_below = t < mn
    t_above = t > mx
    t_viol = t_below | t_above
    if allow_recovery:
        rec_ok = (above & (t <= q_arr)) | (below & (t >= q_arr))
    else:
        rec_ok = np.zeros(6, dtype=bool)
    ok_mask = (~cur_viol & ~t_viol) | (cur_viol & rec_ok)
    return ok_mask
