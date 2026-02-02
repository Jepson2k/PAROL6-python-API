"""
JIT warmup utilities.

Call warmup_jit() on startup to pre-compile all numba functions before the control loop.
With cache=True, this is fast (~100ms) if cache exists, slower (~3-10s) on first run.
"""

import logging
import time

import numpy as np

from parol6.commands.cartesian_commands import (
    _apply_velocity_delta_trf_jit,
    _apply_velocity_delta_wrf_jit,
)
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
from parol6.motion.streaming_executors import (
    _pose_to_tangent_jit,
    _tangent_to_pose_jit,
)
from parol6.protocol.wire import (
    _pack_bitfield,
    _pack_positions,
    _unpack_bitfield,
    _unpack_positions,
    fuse_2_bytes,
    fuse_3_bytes,
    pack_tx_frame_into,
    split_to_3_bytes,
    unpack_rx_frame_into,
)
from parol6.server.ik_worker import (
    _AXIS_DIRS,
    _compute_joint_enable,
    _compute_target_poses,
)
from parol6.server.ipc import unpack_ik_response_into
from parol6.server.loop_timer import (
    _compute_event_rate,
    _compute_loop_stats,
    _compute_phase_stats,
    _quickselect,
    _quickselect_partition,
)
from parol6.server.status_cache import _update_arrays
from parol6.utils.se3_utils import arrays_equal_6
from parol6.server.transport_manager import _arrays_changed
from parol6.server.transports.mock_serial_transport import (
    _encode_payload_jit,
    _simulate_motion_jit,
    _write_frame_jit,
)
from parol6.utils.ik import _check_limits_core, _ik_safety_check
from parol6.utils.se3_utils import (
    _compute_V_inv_matrix,
    _compute_V_matrix,
    se3_angdist,
    se3_copy,
    se3_exp,
    se3_exp_ws,
    se3_from_rpy,
    se3_from_trans,
    se3_identity,
    se3_interp,
    se3_interp_ws,
    se3_inverse,
    se3_log,
    se3_log_ws,
    se3_mul,
    se3_rpy,
    se3_rx,
    se3_ry,
    se3_rz,
    so3_exp,
    so3_from_rpy,
    so3_log,
    so3_rpy,
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

    dummy_3x3 = np.eye(3, dtype=np.float64)

    # parol6/utils/se3_utils.py - arrays_equal_6
    arrays_equal_6(dummy_6i, dummy_6i)
    arrays_equal_6(dummy_6f, dummy_6f)

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

    # parol6/utils/se3_utils.py
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

    # parol6/server/ipc - IK response unpacking
    dummy_ik_buf = np.zeros(44, dtype=np.uint8)
    dummy_joint_en = np.zeros(12, dtype=np.uint8)
    dummy_cart_wrf = np.zeros(12, dtype=np.uint8)
    dummy_cart_trf = np.zeros(12, dtype=np.uint8)
    unpack_ik_response_into(
        dummy_ik_buf, 0, dummy_joint_en, dummy_cart_wrf, dummy_cart_trf
    )

    # parol6/protocol/wire.py - frame packing/unpacking
    dummy_tx_frame = memoryview(bytearray(64))
    dummy_grip_3f = np.zeros(3, dtype=np.float64)
    dummy_gripper_data = np.zeros(6, dtype=np.int32)
    pack_tx_frame_into(
        dummy_tx_frame,  # out
        dummy_6i,  # position_out
        dummy_6i,  # speed_out
        0,  # command_code
        dummy_6i,  # affected_joint_out
        dummy_6i,  # inout_out
        0,  # timeout_out
        dummy_gripper_data,  # gripper_data_out
    )
    dummy_rx_frame = memoryview(bytearray(64))
    dummy_io_5u8 = np.zeros(5, dtype=np.uint8)
    dummy_8u8_homed = np.zeros(8, dtype=np.uint8)
    dummy_8u8_io = np.zeros(8, dtype=np.uint8)
    dummy_8u8_temp = np.zeros(8, dtype=np.uint8)
    dummy_8u8_poserr = np.zeros(8, dtype=np.uint8)
    dummy_timing_out = np.zeros(2, dtype=np.int32)
    dummy_grip_out = np.zeros(5, dtype=np.int32)
    unpack_rx_frame_into(
        dummy_rx_frame,  # data
        dummy_6i,  # pos_out
        dummy_6i,  # spd_out
        dummy_8u8_homed,  # homed_out
        dummy_8u8_io,  # io_out
        dummy_8u8_temp,  # temp_out
        dummy_8u8_poserr,  # poserr_out
        dummy_timing_out,  # timing_out
        dummy_grip_out,  # grip_out
    )

    # parol6/server/transport_manager.py
    _arrays_changed(
        dummy_6f,
        dummy_6f,
        dummy_6f,
        dummy_6f,
        dummy_6f,
        dummy_6f,
        dummy_6f,
        dummy_6f,
        dummy_grip_3f,
        dummy_grip_3f,
    )

    # parol6/server/loop_timer.py - stats computation
    dummy_1000f = np.zeros(1000, dtype=np.float64)
    dummy_1000f_scratch = np.zeros(1000, dtype=np.float64)
    # Fill with some data for realistic warmup
    dummy_1000f[:100] = np.linspace(0.004, 0.006, 100)
    _quickselect_partition(dummy_1000f_scratch[:10].copy(), 0, 9)
    _quickselect(dummy_1000f_scratch[:100].copy(), 50)
    _compute_phase_stats(dummy_1000f, dummy_1000f_scratch, 100)
    _compute_loop_stats(dummy_1000f, dummy_1000f_scratch, 100)
    _compute_event_rate(dummy_1000f, 100, 1.0, 1.0)

    # parol6/server/transports/mock_serial_transport.py
    dummy_pos_f = np.zeros(6, dtype=np.float64)
    dummy_8u8 = np.zeros(8, dtype=np.uint8)
    dummy_gripper_6i = np.zeros(6, dtype=np.int32)
    _simulate_motion_jit(
        dummy_pos_f,  # position_f
        dummy_6i,  # position_in
        dummy_6i,  # speed_in
        dummy_6i,  # speed_out
        dummy_6i,  # position_out
        dummy_8u8,  # homed_in
        dummy_8u8,  # io_in
        dummy_6f.copy(),  # prev_pos_f
        dummy_6f.copy(),  # vmax_f
        dummy_6f.copy(),  # jmin_f
        dummy_6f.copy(),  # jmax_f
        dummy_6f.copy(),  # home_angles_deg
        0,  # command_out
        0.004,  # dt
        0,  # homing_countdown
    )
    _write_frame_jit(
        dummy_6i,  # state_position_out
        dummy_6i,  # state_speed_out
        dummy_gripper_6i,  # state_gripper_data_in
        dummy_6i,  # position_out
        dummy_6i,  # speed_out
        dummy_gripper_6i,  # gripper_data_out
    )
    dummy_payload = memoryview(bytearray(64))
    dummy_timing = np.zeros(1, dtype=np.int32)
    dummy_gripper_in = np.zeros(5, dtype=np.int32)
    _encode_payload_jit(
        dummy_payload,  # out
        dummy_6i,  # position_in
        dummy_6i,  # speed_in
        dummy_8u8,  # homed_in
        dummy_io_5u8,  # io_in
        dummy_8u8,  # temp_err_in
        dummy_8u8,  # pos_err_in
        dummy_timing,  # timing_in
        dummy_gripper_in,  # gripper_in
    )

    # parol6/utils/se3_utils.py - additional functions
    dummy_3x3_out = np.zeros((3, 3), dtype=np.float64)
    dummy_3f = np.zeros(3, dtype=np.float64)
    dummy_twist = np.zeros(6, dtype=np.float64)

    so3_from_rpy(0.0, 0.0, 0.0, dummy_3x3_out)
    se3_from_rpy(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dummy_4x4)
    so3_rpy(dummy_3x3, dummy_3f)
    se3_rpy(dummy_4x4, dummy_3f)
    se3_inverse(dummy_4x4, dummy_4x4_out)
    so3_log(dummy_3x3, dummy_3f)
    so3_exp(dummy_3f, dummy_3x3_out)
    _compute_V_matrix(dummy_3f, dummy_3x3_out)
    _compute_V_inv_matrix(dummy_3f, dummy_3x3_out)
    se3_log(dummy_4x4, dummy_twist)
    se3_exp(dummy_twist, dummy_4x4_out)
    se3_interp(dummy_4x4, dummy_4x4_b, 0.5, dummy_4x4_out)
    se3_angdist(dummy_4x4, dummy_4x4_b)
    # Workspace arrays for _ws functions
    omega_ws = np.zeros(3, dtype=np.float64)
    R_ws = np.zeros((3, 3), dtype=np.float64)
    V_ws = np.zeros((3, 3), dtype=np.float64)
    V_inv_ws = np.zeros((3, 3), dtype=np.float64)
    T1_inv_ws = np.zeros((4, 4), dtype=np.float64)
    delta_ws = np.zeros((4, 4), dtype=np.float64)
    twist_ws = np.zeros(6, dtype=np.float64)
    delta_scaled_ws = np.zeros((4, 4), dtype=np.float64)
    se3_log_ws(dummy_4x4, dummy_twist, omega_ws, R_ws, V_inv_ws)
    se3_exp_ws(dummy_twist, dummy_4x4_out, omega_ws, R_ws, V_ws)
    se3_interp_ws(
        dummy_4x4,
        dummy_4x4_b,
        0.5,
        dummy_4x4_out,
        T1_inv_ws,
        delta_ws,
        twist_ws,
        delta_scaled_ws,
        omega_ws,
        R_ws,
        V_ws,
    )

    # parol6/motion/streaming_executors.py - additional functions
    ref_inv = np.zeros((4, 4), dtype=np.float64)
    delta_4x4 = np.zeros((4, 4), dtype=np.float64)
    _pose_to_tangent_jit(
        dummy_4x4,
        dummy_4x4_b,
        ref_inv,
        delta_4x4,
        dummy_twist,
        omega_ws,
        R_ws,
        V_inv_ws,
    )
    _tangent_to_pose_jit(
        dummy_4x4, dummy_twist, delta_4x4, dummy_4x4_out, omega_ws, R_ws, V_ws
    )

    # parol6/commands/cartesian_commands.py
    vel_lin = np.zeros(3, dtype=np.float64)
    vel_ang = np.zeros(3, dtype=np.float64)
    world_twist = np.zeros(6, dtype=np.float64)
    body_twist = np.zeros(6, dtype=np.float64)
    _apply_velocity_delta_wrf_jit(
        dummy_3x3,  # R
        dummy_6f,  # smoothed_vel
        0.004,  # dt
        dummy_4x4,  # current_pose
        vel_lin,  # vel_lin
        vel_ang,  # vel_ang
        world_twist,  # world_twist
        delta_4x4,  # delta
        dummy_4x4_out,  # out
        omega_ws,  # omega_ws
        R_ws,  # R_ws
        V_ws,  # V_ws
    )
    _apply_velocity_delta_trf_jit(
        dummy_6f,  # smoothed_vel
        0.004,  # dt
        dummy_4x4,  # current_pose
        body_twist,  # body_twist
        delta_4x4,  # delta
        dummy_4x4_out,  # out
        omega_ws,  # omega_ws
        R_ws,  # R_ws
        V_ws,  # V_ws
    )

    elapsed = time.perf_counter() - start
    logger.info(f"\tJIT warmup completed in {elapsed * 1000:.1f}ms")
    return elapsed
