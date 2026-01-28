"""
Gripper Control Commands
Contains commands for electric and pneumatic gripper control
"""

import logging
from enum import Enum

from parol6.commands.base import Debouncer, ExecutionStatus, MotionCommand
from parol6.protocol.wire import CmdType, ElectricGripperCmd, PneumaticGripperCmd
from parol6.server.command_registry import register_command
from parol6.server.state import ControllerState

logger = logging.getLogger(__name__)


class GripperState(Enum):
    START = "START"
    SEND_CALIBRATE = "SEND_CALIBRATE"
    WAITING_CALIBRATION = "WAITING_CALIBRATION"
    WAIT_FOR_POSITION = "WAIT_FOR_POSITION"


@register_command(CmdType.PNEUMATICGRIPPER)
class PneumaticGripperCommand(MotionCommand):
    """Control pneumatic gripper (open/close)."""

    PARAMS_TYPE = PneumaticGripperCmd

    __slots__ = (
        "state",
        "timeout_counter",
        "_state_to_set",
        "_port_index",
    )

    def __init__(self):
        super().__init__()
        self.state = GripperState.START
        self.timeout_counter = 1000
        self._state_to_set: int = 0
        self._port_index: int = 0

    def do_setup(self, state: "ControllerState") -> None:
        """Compute port index and state to set from params."""
        assert self.p is not None

        # open=True means set to 1, open=False means set to 0
        self._state_to_set = 1 if self.p.open else 0
        # port 1 -> index 2, port 2 -> index 3
        self._port_index = 2 if self.p.port == 1 else 3

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """Execute pneumatic gripper command."""
        self.timeout_counter -= 1
        if self.timeout_counter <= 0:
            raise TimeoutError(f"Gripper command timed out in state {self.state}.")

        state.InOut_out[self._port_index] = self._state_to_set
        logger.info("  -> Pneumatic gripper command sent.")
        self.is_finished = True
        return ExecutionStatus.completed("Pneumatic gripper toggled")


@register_command(CmdType.ELECTRICGRIPPER)
class ElectricGripperCommand(MotionCommand):
    """Control electric gripper (move/calibrate)."""

    PARAMS_TYPE = ElectricGripperCmd

    __slots__ = (
        "state",
        "timeout_counter",
        "object_debouncer",
        "wait_counter",
    )

    def __init__(self):
        super().__init__()
        self.state = GripperState.START
        self.timeout_counter = 1000
        self.object_debouncer = Debouncer(5)
        self.wait_counter = 0

    def do_setup(self, state: "ControllerState") -> None:
        """Initialize wait counter for calibration mode."""
        assert self.p is not None
        if self.p.calibrate:
            self.wait_counter = 200

    def execute_step(self, state: "ControllerState") -> ExecutionStatus:
        """State-based execution for electric gripper."""
        assert self.p is not None

        self.timeout_counter -= 1
        if self.timeout_counter <= 0:
            raise TimeoutError(f"Gripper command timed out in state {self.state}.")

        if self.state == GripperState.START:
            if self.p.calibrate:
                self.state = GripperState.SEND_CALIBRATE
            else:
                self.state = GripperState.WAIT_FOR_POSITION

        if self.state == GripperState.SEND_CALIBRATE:
            logger.debug("  -> Sending one-shot calibrate command...")
            state.Gripper_data_out[4] = 1
            self.state = GripperState.WAITING_CALIBRATION
            return ExecutionStatus.executing("Calibrating")

        if self.state == GripperState.WAITING_CALIBRATION:
            self.wait_counter -= 1
            if self.wait_counter <= 0:
                logger.info("  -> Calibration delay finished.")
                state.Gripper_data_out[4] = 0
                self.is_finished = True
                return ExecutionStatus.completed("Calibration complete")
            return ExecutionStatus.executing("Calibrating")

        if self.state == GripperState.WAIT_FOR_POSITION:
            state.Gripper_data_out[0] = self.p.position
            state.Gripper_data_out[1] = self.p.speed
            state.Gripper_data_out[2] = self.p.current
            state.Gripper_data_out[4] = 0

            bits = [1, 1, int(not state.InOut_in[4]), 1, 0, 0, 0, 0]
            val = 0
            for b in bits:
                val = (val << 1) | int(b)
            state.Gripper_data_out[3] = val

            object_detection = (
                state.Gripper_data_in[5] if len(state.Gripper_data_in) > 5 else 0
            )
            logger.debug(
                f" -> Gripper moving to {self.p.position} (current: {state.Gripper_data_in[1]}), object detected: {object_detection}"
            )

            object_detected = self.object_debouncer.tick(object_detection != 0)

            current_position = state.Gripper_data_in[1]
            if abs(current_position - self.p.position) <= 5:
                logger.info("  -> Gripper move complete.")
                self.is_finished = True
                bits = [1, 0, int(not state.InOut_in[4]), 1, 0, 0, 0, 0]
                val = 0
                for b in bits:
                    val = (val << 1) | int(b)
                state.Gripper_data_out[3] = val
                return ExecutionStatus.completed("Gripper move complete")

            if object_detected:
                if (object_detection == 1) and (self.p.position > current_position):
                    logger.info(
                        "  -> Gripper move holding position due to object detection when closing."
                    )
                    self.is_finished = True
                    return ExecutionStatus.completed(
                        "Object detected while closing - hold"
                    )

                if (object_detection == 2) and (self.p.position < current_position):
                    logger.info(
                        "  -> Gripper move holding position due to object detection when opening."
                    )
                    self.is_finished = True
                    return ExecutionStatus.completed(
                        "Object detected while opening - hold"
                    )

            return ExecutionStatus.executing("Moving gripper")

        return ExecutionStatus.failed("Unknown gripper state")
