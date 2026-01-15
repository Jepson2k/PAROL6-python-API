"""
JIT warmup utilities.

Call warmup_jit() on startup to pre-compile all numba functions before the control loop.
With cache=True, this is fast (~100ms) if cache exists, slower (~3-10s) on first run.
"""

import logging
import time

import numpy as np

from parol6.config import (
    deg_to_steps,
    deg_to_steps_scalar,
    rad_to_steps,
    rad_to_steps_scalar,
    speed_deg_to_steps,
    speed_deg_to_steps_scalar,
    speed_rad_to_steps,
    speed_rad_to_steps_scalar,
    speed_steps_to_deg,
    speed_steps_to_deg_scalar,
    speed_steps_to_rad,
    speed_steps_to_rad_scalar,
    steps_to_deg,
    steps_to_deg_scalar,
    steps_to_rad,
    steps_to_rad_scalar,
)
from parol6.motion.streaming_executors import _transform_vel_wrf_to_body
from parol6.protocol.wire import (
    _pack_bitfield,
    _pack_positions,
    _unpack_bitfield,
    _unpack_positions,
    fuse_2_bytes,
    fuse_3_bytes,
    split_to_3_bytes,
)
from parol6.server.status_cache import _update_arrays
from parol6.server.ik_worker import (
    _compute_joint_enable,
    _compute_target_poses,
    _AXIS_DIRS,
)
from parol6.utils.ik import _check_limits_core, _ik_safety_check, unwrap_angles
from parol6.utils.se3_numba import (
    se3_identity,
    se3_from_trans,
    se3_rx,
    se3_ry,
    se3_rz,
    se3_mul,
    se3_copy,
)

logger = logging.getLogger(__name__)


def warmup_jit() -> float:
    """
    Pre-compile all numba JIT functions by calling them with dummy data.

    Returns the time taken in seconds.
    """
    logging.info("Warming JIT...")
    start = time.perf_counter()

    dummy_6f = np.zeros(6, dtype=np.float64)
    dummy_6i = np.zeros(6, dtype=np.int32)
    out_6f = np.zeros(6, dtype=np.float64)
    out_6i = np.zeros(6, dtype=np.int32)

    # parol6/config.py
    deg_to_steps(dummy_6f, out_6i)
    deg_to_steps_scalar(0.0, 0)
    steps_to_deg(dummy_6i, out_6f)
    steps_to_deg_scalar(0, 0)
    rad_to_steps(dummy_6f, out_6i)
    rad_to_steps_scalar(0.0, 0)
    steps_to_rad(dummy_6i, out_6f)
    steps_to_rad_scalar(0, 0)
    speed_steps_to_deg(dummy_6i, out_6f)
    speed_steps_to_deg_scalar(0.0, 0)
    speed_deg_to_steps(dummy_6f, out_6i)
    speed_deg_to_steps_scalar(0.0, 0)
    speed_steps_to_rad(dummy_6i, out_6f)
    speed_steps_to_rad_scalar(0.0, 0)
    speed_rad_to_steps(dummy_6f, out_6i)
    speed_rad_to_steps_scalar(0.0, 0)

    # parol6/utils/ik.py
    unwrap_angles(dummy_6f, dummy_6f)
    _ik_safety_check(dummy_6f, dummy_6f, dummy_6f, dummy_6f, dummy_6f, dummy_6f)
    viol = np.zeros(6, dtype=np.bool_)
    _check_limits_core(
        dummy_6f,
        dummy_6f,
        dummy_6f,
        dummy_6f,
        True,
        False,
        viol,
        viol,
        viol,
        viol,
        viol,
        viol,
    )

    # parol6/protocol/wire.py
    dummy_pos = np.zeros(6, dtype=np.int32)
    dummy_bits = np.zeros(8, dtype=np.uint8)
    dummy_out = np.zeros(20, dtype=np.uint8)
    _pack_positions(dummy_out, dummy_pos, 0)
    _unpack_positions(dummy_out, dummy_pos)
    _pack_bitfield(dummy_bits)
    _unpack_bitfield(0, dummy_bits)
    split_to_3_bytes(0)
    fuse_3_bytes(0, 0, 0)
    fuse_2_bytes(0, 0)

    # parol6/motion/streaming_executors.py
    dummy_3x3 = np.eye(3, dtype=np.float64)
    dummy_6f_out = np.zeros(6, dtype=np.float64)
    _transform_vel_wrf_to_body(dummy_3x3, dummy_6f, dummy_6f_out)

    # parol6/server/status_cache.py
    dummy_5u8 = np.zeros(5, dtype=np.uint8)
    _update_arrays(
        dummy_6i,
        dummy_5u8,
        dummy_6i,
        dummy_6i,
        dummy_6i,
        dummy_6f,
        dummy_6f,
        dummy_5u8,
        dummy_6i,
        dummy_6i,
    )

    # parol6/utils/se3_numba.py
    dummy_4x4 = np.zeros((4, 4), dtype=np.float64)
    dummy_4x4_b = np.zeros((4, 4), dtype=np.float64)
    dummy_4x4_out = np.zeros((4, 4), dtype=np.float64)
    se3_identity(dummy_4x4)
    se3_from_trans(0.0, 0.0, 0.0, dummy_4x4)
    se3_rx(0.0, dummy_4x4)
    se3_ry(0.0, dummy_4x4)
    se3_rz(0.0, dummy_4x4)
    se3_mul(dummy_4x4, dummy_4x4_b, dummy_4x4_out)
    se3_copy(dummy_4x4, dummy_4x4_b)

    # parol6/server/ik_worker.py
    dummy_qlim = np.zeros((2, 6), dtype=np.float64)
    dummy_12u8 = np.zeros(12, dtype=np.uint8)
    _compute_joint_enable(dummy_6f, dummy_qlim, dummy_12u8)
    dummy_targets = np.zeros((12, 4, 4), dtype=np.float64)
    _compute_target_poses(dummy_4x4, 0.001, 0.01, True, _AXIS_DIRS, dummy_targets)

    elapsed = time.perf_counter() - start
    logger.info(f"\tJIT warmup completed in {elapsed * 1000:.1f}ms")
    return elapsed
