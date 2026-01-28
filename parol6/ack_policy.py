import os
from collections.abc import Callable

from parol6.protocol.wire import CmdType

# System command types (always require ACK)
SYSTEM_CMD_TYPES: set[CmdType] = {
    CmdType.STOP,
    CmdType.ENABLE,
    CmdType.DISABLE,
    CmdType.SET_PORT,
    CmdType.STREAM,
    CmdType.SIMULATOR,
    CmdType.SET_PROFILE,
    CmdType.RESET,
}

# Query command types (use request/response, not ACK)
QUERY_CMD_TYPES: set[CmdType] = {
    CmdType.GET_POSE,
    CmdType.GET_ANGLES,
    CmdType.GET_IO,
    CmdType.GET_GRIPPER,
    CmdType.GET_SPEEDS,
    CmdType.GET_STATUS,
    CmdType.GET_GCODE_STATUS,
    CmdType.GET_LOOP_STATS,
    CmdType.GET_CURRENT_ACTION,
    CmdType.GET_QUEUE,
    CmdType.GET_TOOL,
    CmdType.GET_PROFILE,
    CmdType.PING,
}


class AckPolicy:
    """
    Centralized heuristic for deciding if a command requires an acknowledgment.

    Rules:
    - If force_ack is set, it overrides everything.
    - System commands always require ack.
    - Query commands use request/response, not ACKs.
    - Motion and other commands: ACKs only when forced.
    """

    def __init__(
        self,
        get_stream_mode: Callable[[], bool],
        force_ack: bool | None = None,
    ) -> None:
        self._get_stream_mode = get_stream_mode
        self._force_ack = force_ack

    @staticmethod
    def from_env(get_stream_mode: Callable[[], bool]) -> "AckPolicy":
        raw = os.getenv("PAROL6_FORCE_ACK", "").strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            force = True
        elif raw in {"0", "false", "no", "off"}:
            force = False
        else:
            force = None
        return AckPolicy(get_stream_mode=get_stream_mode, force_ack=force)

    def requires_ack(self, cmd_type: CmdType) -> bool:
        """Check if a command type requires an ACK response."""
        # Forced override (e.g., diagnostics)
        if self._force_ack is not None:
            return bool(self._force_ack)

        # System commands always require ACKs
        if cmd_type in SYSTEM_CMD_TYPES:
            return True

        # Query commands use request/response, not ACKs
        if cmd_type in QUERY_CMD_TYPES:
            return False

        # Motion and other commands: ACKs only when forced
        return False
