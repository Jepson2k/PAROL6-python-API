"""Tests for RESET command."""

import pytest
from parol6.commands.utility_commands import ResetCommand
from parol6.server.state import ControllerState
import numpy as np


class TestResetCommandParsing:
    """Test ResetCommand.do_match parsing."""

    def test_parse_no_params(self):
        """RESET takes no parameters."""
        cmd = ResetCommand()
        ok, err = cmd.do_match(["RESET"])
        assert ok is True
        assert err is None
        assert cmd.is_valid is True

    def test_parse_with_params_fails(self):
        """RESET should reject extra parameters."""
        cmd = ResetCommand()
        ok, err = cmd.do_match(["RESET", "extra"])
        assert ok is False
        assert "no parameters" in err


class TestResetCommandExecution:
    """Test ResetCommand.tick resets state correctly."""

    def test_reset_clears_positions(self):
        """Reset should zero out position buffers."""
        state = ControllerState()
        state.Position_in = np.array(
            [1000, 2000, 3000, 4000, 5000, 6000], dtype=np.int32
        )
        state.Speed_in = np.array([10, 20, 30, 40, 50, 60], dtype=np.int32)

        cmd = ResetCommand()
        cmd.do_match(["RESET"])
        cmd.tick(state)  # Reset executes in tick, not setup

        assert np.all(state.Position_in == 0)
        assert np.all(state.Speed_in == 0)

    def test_reset_clears_errors(self):
        """Reset should clear error states."""
        state = ControllerState()
        state.e_stop_active = True
        state.soft_error = True
        state.disabled_reason = "some error"

        cmd = ResetCommand()
        cmd.do_match(["RESET"])
        cmd.tick(state)  # Reset executes in tick, not setup

        assert state.e_stop_active is False
        assert state.soft_error is False
        assert state.disabled_reason == ""

    def test_reset_clears_tool(self):
        """Reset should reset tool to NONE."""
        state = ControllerState()
        state._current_tool = "GRIPPER"

        cmd = ResetCommand()
        cmd.do_match(["RESET"])
        cmd.tick(state)  # Reset executes in tick, not setup

        assert state._current_tool == "NONE"

    def test_reset_clears_queues(self):
        """Reset should clear command queues."""
        state = ControllerState()
        state.command_queue.append("cmd1")
        state.command_queue.append("cmd2")
        state.incoming_command_buffer.append(("msg", ("addr", 123)))

        cmd = ResetCommand()
        cmd.do_match(["RESET"])
        cmd.tick(state)  # Reset executes in tick, not setup

        assert len(state.command_queue) == 0
        assert len(state.incoming_command_buffer) == 0

    def test_reset_preserves_connection_state(self):
        """Reset should NOT reset connection-related state."""
        state = ControllerState()
        state.ip = "192.168.1.100"
        state.port = 9999
        state.start_time = 12345.0
        state.ser = "mock_serial"

        cmd = ResetCommand()
        cmd.do_match(["RESET"])
        cmd.tick(state)  # Reset executes in tick, not setup

        assert state.ip == "192.168.1.100"
        assert state.port == 9999
        assert state.start_time == 12345.0
        assert state.ser == "mock_serial"

    def test_reset_finishes_immediately(self):
        """Reset command should complete in single tick."""
        state = ControllerState()
        cmd = ResetCommand()
        cmd.do_match(["RESET"])
        cmd.tick(state)  # Reset completes in single tick

        assert cmd.is_finished is True


@pytest.mark.integration
class TestResetIntegration:
    """Integration tests for RESET command via client."""

    def test_reset_command_succeeds(self, client, server_proc):
        """Test reset command executes successfully via client."""
        result = client.reset()
        assert result is True

    def test_reset_multiple_times(self, client, server_proc):
        """Test reset can be called multiple times."""
        for _ in range(3):
            result = client.reset()
            assert result is True
