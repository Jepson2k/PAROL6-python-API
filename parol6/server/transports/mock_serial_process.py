"""
MockSerial subprocess worker.

This module runs the robot physics simulation in a separate process,
communicating with the main controller via shared memory.
"""

import logging
import signal
import time
from dataclasses import dataclass, field
from multiprocessing.synchronize import Event

import numpy as np

from parol6.server.ipc import (
    attach_shm,
    pack_rx_frame,
    unpack_tx_command,
)

logger = logging.getLogger(__name__)


@dataclass
class MockRobotState:
    """Internal state of the simulated robot."""

    # Joint positions (in steps)
    position_in: np.ndarray = field(
        default_factory=lambda: np.zeros((6,), dtype=np.int32)
    )
    # Floating accumulator for high-fidelity integration (steps, float)
    position_f: np.ndarray = field(
        default_factory=lambda: np.zeros((6,), dtype=np.float64)
    )
    # Joint speeds (in steps/sec)
    speed_in: np.ndarray = field(default_factory=lambda: np.zeros((6,), dtype=np.int32))
    # Homed status per joint
    homed_in: np.ndarray = field(default_factory=lambda: np.zeros((8,), dtype=np.uint8))
    # I/O states
    io_in: np.ndarray = field(default_factory=lambda: np.zeros((8,), dtype=np.uint8))
    # Error states
    temperature_error_in: np.ndarray = field(
        default_factory=lambda: np.zeros((8,), dtype=np.uint8)
    )
    position_error_in: np.ndarray = field(
        default_factory=lambda: np.zeros((8,), dtype=np.uint8)
    )
    # Gripper state
    gripper_data_in: np.ndarray = field(
        default_factory=lambda: np.zeros((6,), dtype=np.int32)
    )
    # Timing data
    timing_data_in: np.ndarray = field(
        default_factory=lambda: np.zeros((1,), dtype=np.int32)
    )

    # Simulation parameters
    update_rate: float = 0.004  # 250 Hz
    last_update: float = field(default_factory=time.perf_counter)
    homing_countdown: int = 0

    # Command state from controller
    command_out: int = 0  # IDLE
    position_out: np.ndarray = field(
        default_factory=lambda: np.zeros((6,), dtype=np.int32)
    )
    speed_out: np.ndarray = field(
        default_factory=lambda: np.zeros((6,), dtype=np.float64)
    )


def _split_to_3_bytes(val: int) -> tuple:
    """Split a 24-bit signed integer into 3 bytes."""
    if val < 0:
        val = val + (1 << 24)
    return (val >> 16) & 0xFF, (val >> 8) & 0xFF, val & 0xFF


def mock_serial_worker_main(
    rx_shm_name: str,
    tx_shm_name: str,
    shutdown_event: Event,
    standby_angles_deg: tuple | list,
    home_angles_deg: tuple | list,
    joint_limits_steps: np.ndarray,
    velocity_limits_steps: np.ndarray,
    deg_to_steps_ratios: np.ndarray,
    interval_s: float = 0.004,
) -> None:
    """
    Main entry point for MockSerial subprocess.

    Args:
        rx_shm_name: Name of RX shared memory segment
        tx_shm_name: Name of TX shared memory segment
        shutdown_event: Event to signal shutdown
        standby_angles_deg: Initial standby angles in degrees
        home_angles_deg: Home position angles in degrees
        joint_limits_steps: Joint limits in steps [6, 2] (min, max)
        velocity_limits_steps: Max velocity per joint in steps/sec
        deg_to_steps_ratios: Conversion ratios for deg to steps
        interval_s: Control loop interval (default 4ms for 250 Hz)
    """
    import sys
    import traceback

    # Ignore SIGINT in worker - main process handles shutdown via shutdown_event
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Configure logging for subprocess (spawn creates fresh process without logging)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    sub_logger = logging.getLogger("parol6.mock_serial_subprocess")
    sub_logger.info(
        "MockSerial subprocess starting (rx=%s, tx=%s)", rx_shm_name, tx_shm_name
    )

    # Initialize variables for finally block
    rx_shm = None
    tx_shm = None
    rx_mv = None
    tx_mv = None
    frame_mv = None

    try:
        # Attach to shared memory
        sub_logger.debug("Attaching to RX shared memory: %s", rx_shm_name)
        rx_shm = attach_shm(rx_shm_name)
        sub_logger.debug("Attaching to TX shared memory: %s", tx_shm_name)
        tx_shm = attach_shm(tx_shm_name)

        if rx_shm.buf is None:
            raise RuntimeError(f"RX shared memory buffer is None for {rx_shm_name}")
        if tx_shm.buf is None:
            raise RuntimeError(f"TX shared memory buffer is None for {tx_shm_name}")

        rx_mv = memoryview(rx_shm.buf)
        tx_mv = memoryview(tx_shm.buf)
        sub_logger.debug("Shared memory attached successfully")

        # Initialize robot state
        state = MockRobotState(update_rate=interval_s)

        # Set initial positions to standby position
        for i in range(6):
            deg = float(standby_angles_deg[i])
            steps = int(deg * deg_to_steps_ratios[i])
            state.position_in[i] = steps
        state.position_f = state.position_in.astype(np.float64)

        # Ensure E-stop is not pressed
        state.io_in[4] = 1

        # Precompute motion simulation constants
        vmax_f = velocity_limits_steps.astype(np.float64)
        vmax_i32 = velocity_limits_steps.copy()
        jmin_f = joint_limits_steps[:, 0].astype(np.float64)
        jmax_f = joint_limits_steps[:, 1].astype(np.float64)

        # Scratch buffers for motion simulation
        prev_pos_f = np.zeros((6,), dtype=np.float64)
        v_cmd_f = np.zeros((6,), dtype=np.float64)
        new_pos_f = np.zeros((6,), dtype=np.float64)
        realized_v = np.zeros((6,), dtype=np.int32)

        # Frame buffer
        frame_buf = bytearray(52)
        frame_mv = memoryview(frame_buf)
        frame_version = 0

        # Timing
        last_cmd_seq = 0
        next_deadline = time.perf_counter()
        state.last_update = next_deadline

        # Command codes
        CMD_IDLE = 0
        CMD_HOME = 100
        CMD_JOG = 123
        CMD_MOVE = 156

        sub_logger.info("MockSerial subprocess initialized, entering main loop")

        # Write initial frame immediately before entering loop
        _encode_payload_into(frame_mv, state)
        frame_version += 1
        pack_rx_frame(rx_mv, frame_mv, frame_version, time.time())
        sub_logger.info(
            "MockSerial subprocess wrote initial frame (version=%d)", frame_version
        )

        loop_count = 0
        while not shutdown_event.is_set():
            loop_count += 1
            now = time.perf_counter()

            if now >= next_deadline:
                # Read TX commands from shared memory
                position_out, speed_out, command_out, cmd_seq = unpack_tx_command(tx_mv)

                if cmd_seq != last_cmd_seq:
                    state.command_out = command_out
                    np.copyto(state.position_out, position_out)
                    np.copyto(state.speed_out, speed_out)
                    last_cmd_seq = cmd_seq

                # Advance simulation
                dt = now - state.last_update
                if dt > 0:
                    # Handle homing countdown
                    if state.homing_countdown > 0:
                        state.homing_countdown -= 1
                        if state.homing_countdown == 0:
                            # Homing complete
                            state.homed_in.fill(1)
                            for i in range(6):
                                steps = int(
                                    float(home_angles_deg[i]) * deg_to_steps_ratios[i]
                                )
                                state.position_in[i] = steps
                                state.position_f[i] = float(steps)
                                state.speed_in[i] = 0
                            state.command_out = CMD_IDLE

                    # Ensure E-stop stays released
                    state.io_in[4] = 1

                    # Simulate motion based on command type
                    if state.command_out == CMD_HOME:
                        if state.homing_countdown == 0:
                            for i in range(6):
                                state.homed_in[i] = 0
                            state.homing_countdown = max(1, int(0.2 / interval_s + 0.5))
                        state.speed_in.fill(0)

                    elif state.command_out == CMD_JOG:
                        # Speed control mode
                        np.copyto(prev_pos_f, state.position_f)
                        np.copyto(v_cmd_f, state.speed_out)
                        np.clip(v_cmd_f, -vmax_f, vmax_f, out=v_cmd_f)
                        np.multiply(v_cmd_f, dt, out=new_pos_f)
                        np.add(state.position_f, new_pos_f, out=new_pos_f)
                        np.clip(new_pos_f, jmin_f, jmax_f, out=state.position_f)

                        if dt > 0:
                            np.subtract(state.position_f, prev_pos_f, out=new_pos_f)
                            new_pos_f /= dt
                            np.rint(new_pos_f, out=new_pos_f)
                            realized_v[:] = new_pos_f
                            np.clip(realized_v, -vmax_i32, vmax_i32, out=state.speed_in)
                        else:
                            state.speed_in.fill(0)

                    elif state.command_out == CMD_MOVE:
                        # Position control mode
                        prev_pos = state.position_f.copy()
                        for i in range(6):
                            target = float(state.position_out[i])
                            current_f = float(state.position_f[i])
                            err_f = target - current_f

                            max_step_f = float(velocity_limits_steps[i]) * float(dt)
                            if max_step_f < 1.0:
                                max_step_f = 1.0

                            move = float(err_f)
                            if move > max_step_f:
                                move = max_step_f
                            elif move < -max_step_f:
                                move = -max_step_f

                            new_pos = current_f + move
                            jmin, jmax = joint_limits_steps[i]
                            if new_pos < float(jmin):
                                new_pos = float(jmin)
                            elif new_pos > float(jmax):
                                new_pos = float(jmax)

                            state.position_f[i] = new_pos

                        if dt > 0:
                            realized = np.rint(
                                (state.position_f - prev_pos) / dt
                            ).astype(np.int32)
                        else:
                            realized = np.zeros(6, dtype=np.int32)
                        np.clip(realized, -vmax_i32, vmax_i32, out=state.speed_in)

                    else:
                        # Idle or unknown - hold position
                        state.speed_in.fill(0)

                    # Sync integer telemetry from high-fidelity accumulator
                    state.position_in[:] = np.rint(state.position_f).astype(np.int32)
                    state.last_update = now

                # Encode frame
                _encode_payload_into(frame_mv, state)
                frame_version += 1

                # Write to RX shared memory
                pack_rx_frame(rx_mv, frame_mv, frame_version, time.time())

                # Advance deadline
                next_deadline += interval_s
                if next_deadline < now - interval_s:
                    next_deadline = now + interval_s
            else:
                # Sleep until next deadline
                sleep_time = min(next_deadline - now, 0.002)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        sub_logger.info(
            "MockSerial subprocess loop exited after %d iterations (shutdown_event.is_set=%s)",
            loop_count,
            shutdown_event.is_set(),
        )
    except Exception as e:
        sub_logger.exception("MockSerial subprocess error: %s", e)
        # Also print to stderr directly in case logging isn't working
        print(f"MockSerial subprocess FATAL ERROR: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        # Release memoryviews before closing shared memory to avoid BufferError
        for obj, method in [
            (rx_mv, "release"),
            (tx_mv, "release"),
            (frame_mv, "release"),
            (rx_shm, "close"),
            (tx_shm, "close"),
        ]:
            if obj is not None:
                try:
                    getattr(obj, method)()
                except Exception:
                    pass
        sub_logger.info("MockSerial subprocess exiting")


def _encode_payload_into(out_mv: memoryview, state: MockRobotState) -> None:
    """
    Build a 52-byte payload per firmware layout from the simulated state.
    """
    # Positions (6 * 3 bytes)
    off = 0
    for i in range(6):
        b0, b1, b2 = _split_to_3_bytes(int(state.position_in[i]))
        out_mv[off] = b0
        out_mv[off + 1] = b1
        out_mv[off + 2] = b2
        off += 3

    # Speeds (6 * 3 bytes)
    off = 18
    for i in range(6):
        b0, b1, b2 = _split_to_3_bytes(int(state.speed_in[i]))
        out_mv[off] = b0
        out_mv[off + 1] = b1
        out_mv[off + 2] = b2
        off += 3

    # Bitfields
    out_mv[36] = np.packbits(state.homed_in[:8].astype(np.uint8))[0]
    out_mv[37] = np.packbits(state.io_in[:8].astype(np.uint8))[0]
    out_mv[38] = np.packbits(state.temperature_error_in[:8].astype(np.uint8))[0]
    out_mv[39] = np.packbits(state.position_error_in[:8].astype(np.uint8))[0]

    # Timing (two bytes)
    t = int(state.timing_data_in[0]) if state.timing_data_in.any() else 0
    out_mv[40] = (t >> 8) & 0xFF
    out_mv[41] = t & 0xFF

    # Reserved
    out_mv[42] = 0
    out_mv[43] = 0

    # Gripper
    gd = state.gripper_data_in
    dev_id = int(gd[0]) if gd.any() else 0
    pos = int(gd[1]) & 0xFFFF
    spd = int(gd[2]) & 0xFFFF
    cur = int(gd[3]) & 0xFFFF
    status = int(gd[4]) & 0xFF if gd.any() else 0

    out_mv[44] = dev_id & 0xFF
    out_mv[45] = (pos >> 8) & 0xFF
    out_mv[46] = pos & 0xFF
    out_mv[47] = (spd >> 8) & 0xFF
    out_mv[48] = spd & 0xFF
    out_mv[49] = (cur >> 8) & 0xFF
    out_mv[50] = cur & 0xFF
    out_mv[51] = status & 0xFF
