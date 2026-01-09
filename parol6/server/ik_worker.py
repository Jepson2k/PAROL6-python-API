"""
IK Worker subprocess.

This module runs the computationally expensive IK enablement calculations
in a separate process, communicating with the main process via shared memory.
"""

import logging
import math
import time
from multiprocessing.synchronize import Event

import numpy as np
import sophuspy as sp

from parol6.server.ipc import (
    IKInputLayout,
    attach_shm,
    pack_ik_response,
    unpack_ik_request,
)

logger = logging.getLogger(__name__)


def ik_enablement_worker_main(
    input_shm_name: str,
    output_shm_name: str,
    shutdown_event: Event,
) -> None:
    """
    Main entry point for IK enablement worker subprocess.

    This worker monitors the input shared memory for new requests,
    computes joint and cartesian enablement, and writes results to
    the output shared memory.

    Args:
        input_shm_name: Name of input shared memory segment
        output_shm_name: Name of output shared memory segment
        shutdown_event: Event to signal shutdown
    """
    # Attach to shared memory
    input_shm = attach_shm(input_shm_name)
    output_shm = attach_shm(output_shm_name)
    assert input_shm.buf is not None
    assert output_shm.buf is not None
    input_mv = memoryview(input_shm.buf)
    output_mv = memoryview(output_shm.buf)

    # Initialize robot model in this process
    import parol6.PAROL6_ROBOT as PAROL6_ROBOT
    from parol6.utils.ik import AXIS_MAP, solve_ik
    from parol6.utils.se3_utils import (
        se3_from_matrix,
        se3_from_trans,
        se3_rx,
        se3_ry,
        se3_rz,
    )

    robot = PAROL6_ROBOT.robot
    assert robot is not None
    qlim = robot.qlim

    last_req_seq = 0

    logger.info("IK worker subprocess started")

    try:
        while not shutdown_event.is_set():
            # Read request sequence number (cheap check)
            req_seq = np.frombuffer(
                input_mv[
                    IKInputLayout.REQ_SEQ_OFFSET : IKInputLayout.REQ_SEQ_OFFSET + 8
                ],
                dtype=np.uint64,
            )[0]

            if req_seq != last_req_seq and req_seq > 0:
                # New request - read inputs
                q_rad, T_matrix, _, flags = unpack_ik_request(input_mv)

                # Reconstruct SE3 from matrix
                T = se3_from_matrix(T_matrix)

                # Compute enablement
                joint_en = _compute_joint_enable(q_rad, qlim)
                cart_en_wrf = _compute_cart_enable(
                    T,
                    "WRF",
                    q_rad,
                    robot,
                    solve_ik,
                    AXIS_MAP,
                    se3_from_trans,
                    se3_rx,
                    se3_ry,
                    se3_rz,
                )
                cart_en_trf = _compute_cart_enable(
                    T,
                    "TRF",
                    q_rad,
                    robot,
                    solve_ik,
                    AXIS_MAP,
                    se3_from_trans,
                    se3_rx,
                    se3_ry,
                    se3_rz,
                )

                # Write results
                pack_ik_response(output_mv, joint_en, cart_en_wrf, cart_en_trf, req_seq)

                last_req_seq = req_seq
            else:
                # No new request, sleep briefly
                time.sleep(0.001)

    except Exception as e:
        logger.exception("IK worker subprocess error: %s", e)
    finally:
        # Release memoryviews before closing shared memory to avoid BufferError
        try:
            input_mv.release()
        except Exception:
            pass
        try:
            output_mv.release()
        except Exception:
            pass
        input_shm.close()
        output_shm.close()
        logger.info("IK worker subprocess exiting")


def _compute_joint_enable(
    q_rad: np.ndarray,
    qlim: np.ndarray,
    delta_rad: float = math.radians(0.2),
) -> np.ndarray:
    """
    Compute per-joint +/- enable bits based on joint limits and a small delta.

    Returns 12-element array: [J1+, J1-, J2+, J2-, ..., J6+, J6-]
    """
    if qlim is None:
        return np.ones(12, dtype=np.uint8)

    allow_plus = (q_rad + delta_rad) <= qlim[1, :]
    allow_minus = (q_rad - delta_rad) >= qlim[0, :]

    bits = np.zeros(12, dtype=np.uint8)
    for i in range(6):
        bits[i * 2] = 1 if allow_plus[i] else 0
        bits[i * 2 + 1] = 1 if allow_minus[i] else 0

    return bits


def _compute_cart_enable(
    T: sp.SE3,
    frame: str,
    q_rad: np.ndarray,
    robot,
    solve_ik,
    axis_map: dict,
    se3_from_trans,
    se3_rx,
    se3_ry,
    se3_rz,
    delta_mm: float = 0.5,
    delta_deg: float = 0.5,
) -> np.ndarray:
    """
    Compute per-axis +/- enable bits for the given frame (WRF/TRF) via small-step IK.

    Returns 12-element array for the 12 axes in AXIS_MAP order.
    """
    bits = []
    t_step_m = delta_mm / 1000.0
    r_step_rad = math.radians(delta_deg)

    for axis, (v_lin, v_rot) in axis_map.items():
        # Compose delta SE3 for this axis
        dT = sp.SE3()

        # Translation
        dx = v_lin[0] * t_step_m
        dy = v_lin[1] * t_step_m
        dz = v_lin[2] * t_step_m
        if abs(dx) > 0 or abs(dy) > 0 or abs(dz) > 0:
            dT = dT * se3_from_trans(dx, dy, dz)

        # Rotation
        rx = v_rot[0] * r_step_rad
        ry = v_rot[1] * r_step_rad
        rz = v_rot[2] * r_step_rad
        if abs(rx) > 0:
            dT = dT * se3_rx(rx)
        if abs(ry) > 0:
            dT = dT * se3_ry(ry)
        if abs(rz) > 0:
            dT = dT * se3_rz(rz)

        # Apply in specified frame
        if frame == "WRF":
            T_target = dT * T
        else:  # TRF
            T_target = T * dT

        try:
            ik = solve_ik(robot, T_target, q_rad, quiet_logging=True)
            bits.append(1 if ik.success else 0)
        except Exception:
            bits.append(0)

    return np.array(bits, dtype=np.uint8)
