"""
Unit tests for action-related query commands.

Tests GET_CURRENT_ACTION and GET_QUEUE query commands without requiring a running server.
Uses stub UDP transport and minimal state objects to test command logic in isolation.
"""

from types import SimpleNamespace

from parol6.commands.base import CommandContext, ExecutionStatusCode
from parol6.commands.query_commands import GetCurrentActionCommand, GetQueueCommand
from parol6.protocol.wire import MsgType, QueryType, decode


class StubUDPTransport:
    """Stub UDP transport that captures sent responses."""

    def __init__(self):
        self.sent: list[tuple[bytes, tuple[str, int]]] = []

    def send(self, data: bytes, addr: tuple[str, int]) -> bool:
        """Capture sent binary responses for verification."""
        self.sent.append((data, addr))
        return True

    def get_last_response(self) -> tuple | list | None:
        """Decode and return the last sent response."""
        if not self.sent:
            return None
        data, _ = self.sent[-1]
        return decode(data)


def test_get_current_action_command_init():
    """Test that GET_CURRENT_ACTION command initializes correctly."""
    cmd = GetCurrentActionCommand()

    # Command should initialize with valid state
    assert cmd.is_valid
    assert not cmd.is_finished
    assert cmd.PARAMS_TYPE is not None


def test_get_current_action_replies_json():
    """Test that GET_CURRENT_ACTION returns correct binary response."""
    # Setup
    udp = StubUDPTransport()
    ctx = CommandContext(
        udp_transport=udp, addr=("127.0.0.1", 5001), gcode_interpreter=None, dt=0.01
    )

    # Create minimal state with action tracking fields
    state = SimpleNamespace(
        action_current="MovePoseCommand",
        action_state="EXECUTING",
        action_next="HomeCommand",
    )

    # Execute command
    cmd = GetCurrentActionCommand()
    cmd.bind(ctx)
    cmd.setup(state)
    status = cmd.tick(state)

    # Verify response sent
    assert len(udp.sent) == 1
    msg = udp.get_last_response()

    # Verify binary array message format: [RESPONSE, query_type, value]
    assert msg[0] == MsgType.RESPONSE
    assert msg[1] == QueryType.CURRENT_ACTION
    # payload is [current, state, next]
    current, state, next_ = msg[2]

    assert current == "MovePoseCommand"
    assert state == "EXECUTING"
    assert next_ == "HomeCommand"

    # Verify command completed
    assert status.code == ExecutionStatusCode.COMPLETED
    assert cmd.is_finished


def test_get_current_action_with_idle_state():
    """Test GET_CURRENT_ACTION when robot is idle."""
    udp = StubUDPTransport()
    ctx = CommandContext(
        udp_transport=udp, addr=("127.0.0.1", 5001), gcode_interpreter=None, dt=0.01
    )

    # Idle state - no current action
    state = SimpleNamespace(action_current="", action_state="IDLE", action_next="")

    cmd = GetCurrentActionCommand()
    cmd.bind(ctx)
    cmd.setup(state)
    cmd.tick(state)

    # Verify response
    assert len(udp.sent) == 1
    msg = udp.get_last_response()
    # payload is [current, state, next]
    current, state, next_ = msg[2]

    assert current == ""
    assert state == "IDLE"
    assert next_ == ""


def test_get_queue_command_init():
    """Test that GET_QUEUE command initializes correctly."""
    cmd = GetQueueCommand()

    # Command should initialize with valid state
    assert cmd.is_valid
    assert not cmd.is_finished
    assert cmd.PARAMS_TYPE is not None


def test_get_queue_replies_json():
    """Test that GET_QUEUE returns correct binary response."""
    # Setup
    udp = StubUDPTransport()
    ctx = CommandContext(
        udp_transport=udp, addr=("127.0.0.1", 5001), gcode_interpreter=None, dt=0.01
    )

    # Create state with queued commands
    state = SimpleNamespace(
        queue_nonstreamable=["MovePoseCommand", "HomeCommand", "MoveJointCommand"]
    )

    # Execute command
    cmd = GetQueueCommand()
    cmd.bind(ctx)
    cmd.setup(state)
    status = cmd.tick(state)

    # Verify response sent
    assert len(udp.sent) == 1
    msg = udp.get_last_response()

    # Verify binary array message format: [RESPONSE, query_type, value]
    assert msg[0] == MsgType.RESPONSE
    assert msg[1] == QueryType.QUEUE
    # payload is just the queue list
    payload = msg[2]

    assert payload == [
        "MovePoseCommand",
        "HomeCommand",
        "MoveJointCommand",
    ]

    # Verify command completed
    assert status.code == ExecutionStatusCode.COMPLETED
    assert cmd.is_finished


def test_get_queue_with_empty_queue():
    """Test GET_QUEUE when queue is empty."""
    udp = StubUDPTransport()
    ctx = CommandContext(
        udp_transport=udp, addr=("127.0.0.1", 5001), gcode_interpreter=None, dt=0.01
    )

    # Empty queue
    state = SimpleNamespace(queue_nonstreamable=[])

    cmd = GetQueueCommand()
    cmd.bind(ctx)
    cmd.setup(state)
    cmd.tick(state)

    # Verify response
    assert len(udp.sent) == 1
    msg = udp.get_last_response()
    # payload is just the queue list
    payload = msg[2]

    assert payload == []


def test_get_queue_excludes_streamable():
    """Test that queue only contains non-streamable commands (by design)."""
    # This test verifies the API contract - the queue_nonstreamable field
    # should already have streamable commands filtered out by the controller

    udp = StubUDPTransport()
    ctx = CommandContext(
        udp_transport=udp, addr=("127.0.0.1", 5001), gcode_interpreter=None, dt=0.01
    )

    # State should only contain non-streamable commands
    # (streamable commands like JogJointCommand are filtered by controller)
    state = SimpleNamespace(queue_nonstreamable=["MovePoseCommand", "HomeCommand"])

    cmd = GetQueueCommand()
    cmd.bind(ctx)
    cmd.setup(state)
    cmd.tick(state)

    msg = udp.get_last_response()
    # payload is just the queue list
    payload = msg[2]

    # Verify only non-streamable commands in response
    assert "MovePoseCommand" in payload
    assert "HomeCommand" in payload
    assert len(payload) == 2
