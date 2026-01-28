"""
Basic Robot Commands
Contains fundamental movement commands: Home, Jog, and MultiJog
"""

import logging
from enum import Enum, auto
from math import ceil

import numpy as np

from parol6.config import (
    INTERVAL_S,
    JOG_MIN_STEPS,
    LIMITS,
    rad_to_steps,
    speed_steps_to_rad_scalar,
    steps_to_rad,
)
from parol6.protocol.wire import (
    CmdType,
    HomeCmd,
    JogCmd,
    MultiJogCmd,
    SetToolCmd,
)
from parol6.protocol.wire import CommandCode
from parol6.server.command_registry import register_command
from parol6.server.state import ControllerState

from .base import (
    ExecutionStatus,
    MotionCommand,
)

logger = logging.getLogger(__name__)


class HomeState(Enum):
    """State machine states for the homing sequence."""

    START = auto()
    WAITING_FOR_UNHOMED = auto()
    WAITING_FOR_HOMED = auto()


@register_command(CmdType.HOME)
class HomeCommand(MotionCommand):
    """
    A non-blocking command that tells the robot to perform its internal homing sequence.
    This version uses a state machine to allow re-homing even if the robot is already homed.
    """

    PARAMS_TYPE = HomeCmd

    __slots__ = (
        "state",
        "start_cmd_counter",
        "timeout_counter",
    )

    def __init__(self):
        super().__init__()
        self.state = HomeState.START
        self.start_cmd_counter = 10
        self.timeout_counter = 2000

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Manages the homing command and monitors for completion using a state machine."""
        if self.state == HomeState.START:
            logger.debug(
                f"  -> Sending home signal (100)... Countdown: {self.start_cmd_counter}"
            )
            state.Command_out = CommandCode.HOME
            self.start_cmd_counter -= 1
            if self.start_cmd_counter <= 0:
                self.state = HomeState.WAITING_FOR_UNHOMED
            return ExecutionStatus.executing("Homing: start")

        if self.state == HomeState.WAITING_FOR_UNHOMED:
            state.Command_out = CommandCode.IDLE
            if np.any(state.Homed_in[:6] == 0):
                logger.info("  -> Homing sequence initiated by robot.")
                self.state = HomeState.WAITING_FOR_HOMED
            self.timeout_counter -= 1
            if self.timeout_counter <= 0:
                raise TimeoutError(
                    "Timeout waiting for robot to start homing sequence."
                )
            return ExecutionStatus.executing("Homing: waiting for unhomed")

        if self.state == HomeState.WAITING_FOR_HOMED:
            state.Command_out = CommandCode.IDLE
            if np.all(state.Homed_in[:6] == 1):
                self.log_info("Homing sequence complete. All joints reported home.")
                self.is_finished = True
                self.stop_and_idle(state)
                return ExecutionStatus.completed("Homing complete")

            return ExecutionStatus.executing("Homing: waiting for homed")

        return ExecutionStatus.executing("Homing")


@register_command(CmdType.JOG)
class JogCommand(MotionCommand):
    """
    A non-blocking command to jog a joint for a specific duration.
    """

    PARAMS_TYPE = JogCmd
    streamable = True

    __slots__ = (
        "command_step",
        "direction",
        "joint_index",
        "speed_out",
        "_jog_initialized",
        "_speeds_array",
        "_jog_vel_buf",
    )

    def __init__(self):
        super().__init__()
        self.command_step = 0
        self.direction = 1
        self.joint_index = 0
        self.speed_out = 0
        self._jog_initialized = False
        self._speeds_array = np.zeros(6, dtype=np.float64)
        self._jog_vel_buf = [0.0] * 6

    def do_setup(self, state):
        """Pre-computes speeds using live data."""
        assert self.p is not None

        self.direction = 1 if 0 <= self.p.joint <= 5 else -1
        self.joint_index = self.p.joint if self.direction == 1 else self.p.joint - 6

        speed_steps_per_sec = int(
            self.linmap_pct(
                abs(self.p.speed_pct),
                JOG_MIN_STEPS,
                int(LIMITS.joint.jog.velocity_steps[self.joint_index]),
            )
        )

        self.speed_out = speed_steps_per_sec * self.direction
        self.start_timer(self.p.duration)

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """This is the EXECUTION phase. It runs on every loop cycle."""
        if state.stream_mode and state.streaming_executor is not None:
            return self._execute_streaming(state)

        return self._execute_velocity_jog(state)

    def _execute_streaming(self, state: "ControllerState") -> ExecutionStatus:
        """Execute using StreamingExecutor for smooth jogging with velocity control."""
        assert state.streaming_executor is not None
        assert self.joint_index is not None

        se = state.streaming_executor

        # Sync position on first tick
        if not self._jog_initialized:
            steps_to_rad(state.Position_in, self._q_rad_buf)
            current_q_rad = list(self._q_rad_buf)
            se.sync_position(current_q_rad)
            self._jog_initialized = True

        # Check stop conditions
        stop_reason = self._check_stop_conditions(state)

        if stop_reason:
            # Decelerate to stop using velocity mode (reuse buffer)
            for i in range(6):
                self._jog_vel_buf[i] = 0.0
            se.set_jog_velocity(self._jog_vel_buf)
            pos_rad, vel, finished = se.tick()
            rad_to_steps(np.asarray(pos_rad), self._steps_buf)
            self.set_move_position(state, self._steps_buf)

            # Check if actually stopped (velocity near zero)
            if finished or all(abs(v) < 0.001 for v in vel):
                if stop_reason.startswith("Limit"):
                    logger.warning(stop_reason)
                else:
                    self.log_trace(stop_reason)
                self.is_finished = True
                return ExecutionStatus.completed(stop_reason)
            return ExecutionStatus.executing("Jogging (stopping)")

        # Set jog velocity for this joint (reuse buffer)
        speed_rad = float(
            speed_steps_to_rad_scalar(abs(self.speed_out), self.joint_index)
        )
        for i in range(6):
            self._jog_vel_buf[i] = 0.0
        self._jog_vel_buf[self.joint_index] = speed_rad * self.direction

        se.set_jog_velocity(self._jog_vel_buf)
        pos_rad, _vel, _finished = se.tick()

        rad_to_steps(np.asarray(pos_rad), self._steps_buf)
        self.set_move_position(state, self._steps_buf)

        self.command_step += 1
        return ExecutionStatus.executing("Jogging")

    def _execute_velocity_jog(self, state: "ControllerState") -> ExecutionStatus:
        """Execute using standard velocity-based jogging."""
        assert self.joint_index is not None

        stop_reason = self._check_stop_conditions(state)

        if stop_reason:
            if stop_reason.startswith("Limit"):
                logger.warning(stop_reason)
            else:
                self.log_trace(stop_reason)

            self.is_finished = True
            self.stop_and_idle(state)
            return ExecutionStatus.completed(stop_reason)

        state.Speed_out.fill(0)
        state.Speed_out[self.joint_index] = self.speed_out
        state.Command_out = CommandCode.JOG
        self.command_step += 1
        return ExecutionStatus.executing("Jogging")

    def _check_stop_conditions(self, state: "ControllerState") -> str | None:
        """Check if jog should stop. Returns stop reason or None."""
        assert self.joint_index is not None

        if self.timer_expired():
            return "Timed jog finished."

        self._speeds_array.fill(0)
        self._speeds_array[self.joint_index] = self.speed_out
        limit_mask = self.limit_hit_mask(state.Position_in, self._speeds_array)
        if limit_mask[self.joint_index]:
            return f"Limit reached on joint {self.joint_index + 1}."

        return None


@register_command(CmdType.MULTIJOG)
class MultiJogCommand(MotionCommand):
    """
    A non-blocking command to jog multiple joints simultaneously for a specific duration.
    """

    PARAMS_TYPE = MultiJogCmd
    streamable = True

    __slots__ = (
        "command_step",
        "command_len",
        "speeds_out",
        "_lims_steps",
    )

    def __init__(self):
        super().__init__()
        self.command_step = 0
        self.command_len = 0
        self.speeds_out = np.zeros(6, dtype=np.int32)
        self._lims_steps = LIMITS.joint.position.steps

    def do_setup(self, state):
        """Pre-computes the speeds for each joint."""
        assert self.p is not None

        self.command_len = ceil(self.p.duration / INTERVAL_S)
        joints_arr = np.asarray(self.p.joints, dtype=int)
        speeds_pct = np.asarray(self.p.speeds, dtype=float)

        # Map to base joint index (0-5) and direction (+/-)
        direction = np.where((joints_arr >= 0) & (joints_arr <= 5), 1, -1)
        joint_index = np.where(direction == 1, joints_arr, joints_arr - 6)

        # Validate indices
        invalid_mask = (joint_index < 0) | (joint_index >= 6)
        if np.any(invalid_mask):
            bad = joint_index[invalid_mask]
            raise ValueError(f"Invalid joint indices {bad.tolist()}")

        pct = np.clip(np.abs(speeds_pct) / 100.0, 0.0, 1.0)
        for i, idx in enumerate(joint_index):
            jog_max = float(LIMITS.joint.jog.velocity_steps[idx])
            self.speeds_out[idx] = (
                int(self.linmap_pct(pct[i] * 100.0, JOG_MIN_STEPS, jog_max))
                * direction[i]
            )

        self.start_timer(self.p.duration)

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """This is the EXECUTION phase. It runs on every loop cycle."""
        if self.timer_expired() or self.command_step >= self.command_len:
            self.is_finished = True
            self.stop_and_idle(state)
            return ExecutionStatus.completed("MultiJog")

        limit_mask = self.limit_hit_mask(state.Position_in, self.speeds_out)
        if np.any(limit_mask):
            i = np.argmax(limit_mask)
            logger.warning(f"Limit reached on joint {i + 1}. Stopping jog.")
            self.is_finished = True
            self.stop_and_idle(state)
            return ExecutionStatus.completed(f"Limit reached on J{i + 1}")

        np.copyto(state.Speed_out, self.speeds_out, casting="no")
        state.Command_out = CommandCode.JOG
        self.command_step += 1
        return ExecutionStatus.executing("MultiJog")


@register_command(CmdType.SET_TOOL)
class SetToolCommand(MotionCommand):
    """
    Set the current end-effector tool configuration.
    """

    PARAMS_TYPE = SetToolCmd

    __slots__ = ()

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Set the tool in state and update robot kinematics."""
        assert self.p is not None
        tool_name = self.p.tool_name.strip().upper()

        # Update server state - property setter handles tool application and cache invalidation
        state.current_tool = tool_name

        self.log_info(f"Tool set to: {tool_name}")
        self.is_finished = True
        return ExecutionStatus.completed(f"Tool set: {tool_name}")
