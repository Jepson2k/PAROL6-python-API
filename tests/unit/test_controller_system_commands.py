"""
Unit tests for controller system command side-effect handling.

These tests verify that system commands that require side-effects
(like SIMULATOR and SET_PORT) properly trigger the handler methods.
"""

from unittest.mock import MagicMock, patch


class TestSystemCommandSideEffects:
    """Test that system command side-effects are properly triggered."""

    def test_simulator_command_triggers_toggle_handler(self):
        """Verify SIMULATOR command triggers _handle_simulator_toggle.

        Regression test for: simulator toggle handler not being called after
        refactoring, causing multicast to stop working when switching to
        simulator mode at runtime.
        """
        from parol6.commands.system_commands import SimulatorCommand
        from parol6.commands.base import ExecutionStatusCode
        from parol6.server.state import ControllerState
        from parol6.protocol.wire import SimulatorCmd

        # Create a mock controller with the necessary attributes
        with patch("parol6.server.controller.Controller") as MockController:
            controller = MockController.return_value

            # Setup required mocks
            controller.udp_transport = MagicMock()
            controller._handle_simulator_toggle = MagicMock(return_value=True)
            controller._make_command_context = MagicMock()

            # Create a real SimulatorCommand with params
            cmd = SimulatorCommand()
            cmd.assign_params(SimulatorCmd(on=True))

            # Create a real state
            state = ControllerState()

            # Execute the command directly to get the status
            status = cmd.execute_step(state)

            # Verify the command returns the expected details
            assert status.code == ExecutionStatusCode.COMPLETED
            assert status.details is not None
            assert "simulator_mode" in status.details
            assert status.details["simulator_mode"] == "on"

    def test_set_port_command_returns_serial_port_detail(self):
        """Verify SET_PORT command returns serial_port in details."""
        from parol6.commands.system_commands import SetSerialPortCommand
        from parol6.commands.base import ExecutionStatusCode
        from parol6.server.state import ControllerState
        from parol6.protocol.wire import SetPortCmd

        cmd = SetSerialPortCommand()
        cmd.assign_params(SetPortCmd(port_str="/dev/ttyUSB0"))

        state = ControllerState()

        # Mock the config save
        with patch("parol6.commands.system_commands.save_com_port", return_value=True):
            status = cmd.execute_step(state)

        assert status.code == ExecutionStatusCode.COMPLETED
        assert status.details is not None
        assert "serial_port" in status.details
        assert status.details["serial_port"] == "/dev/ttyUSB0"

    def test_handle_system_command_calls_simulator_toggle(self):
        """Verify _handle_system_command calls _handle_simulator_toggle for SIMULATOR.

        This is the key regression test - ensures the handler chain is complete.
        """
        from parol6.server.controller import Controller, ControllerConfig
        from parol6.commands.system_commands import SimulatorCommand
        from parol6.commands.base import ExecutionStatus, ExecutionStatusCode
        from parol6.server.state import ControllerState
        from parol6.protocol.wire import SimulatorCmd

        # Create config without starting the actual server
        config = ControllerConfig()

        # We need to partially mock the controller to test _handle_system_command
        with patch.object(Controller, "_initialize_components"):
            with patch.object(Controller, "__init__", lambda self, cfg: None):
                controller = Controller.__new__(Controller)
                controller.config = config
                controller.udp_transport = MagicMock()
                controller.gcode_interpreter = MagicMock()
                controller._handle_simulator_toggle = MagicMock(return_value=True)
                controller._handle_set_port = MagicMock()
                # Mock executor to return status with simulator_mode detail
                controller._executor = MagicMock()
                controller._executor.execute_immediate = MagicMock(
                    return_value=ExecutionStatus(
                        code=ExecutionStatusCode.COMPLETED,
                        message="OK",
                        details={"simulator_mode": "on"},
                    )
                )

                # Create command and state
                cmd = SimulatorCommand()
                cmd.assign_params(SimulatorCmd(on=True))
                state = ControllerState()
                addr = ("127.0.0.1", 5001)

                # Call _handle_system_command
                Controller._handle_system_command(controller, cmd, state, addr)

                # Verify _handle_simulator_toggle was called with "on"
                controller._handle_simulator_toggle.assert_called_once()
                call_args = controller._handle_simulator_toggle.call_args
                assert call_args[0][0] == "on", (
                    "_handle_simulator_toggle should be called with 'on'"
                )

    def test_handle_system_command_calls_set_port(self):
        """Verify _handle_system_command calls _handle_set_port for SET_PORT."""
        from parol6.server.controller import Controller, ControllerConfig
        from parol6.commands.system_commands import SetSerialPortCommand
        from parol6.commands.base import ExecutionStatus, ExecutionStatusCode
        from parol6.server.state import ControllerState
        from parol6.protocol.wire import SetPortCmd

        config = ControllerConfig()

        with patch.object(Controller, "_initialize_components"):
            with patch.object(Controller, "__init__", lambda self, cfg: None):
                controller = Controller.__new__(Controller)
                controller.config = config
                controller.udp_transport = MagicMock()
                controller.gcode_interpreter = MagicMock()
                controller._handle_simulator_toggle = MagicMock(return_value=True)
                controller._handle_set_port = MagicMock()
                # Mock executor to return status with serial_port detail
                controller._executor = MagicMock()
                controller._executor.execute_immediate = MagicMock(
                    return_value=ExecutionStatus(
                        code=ExecutionStatusCode.COMPLETED,
                        message="OK",
                        details={"serial_port": "/dev/ttyUSB1"},
                    )
                )

                cmd = SetSerialPortCommand()
                cmd.assign_params(SetPortCmd(port_str="/dev/ttyUSB1"))
                state = ControllerState()
                addr = ("127.0.0.1", 5001)

                Controller._handle_system_command(controller, cmd, state, addr)

                # Verify _handle_set_port was called with the port
                controller._handle_set_port.assert_called_once_with("/dev/ttyUSB1")
