"""Tests for MoveCartCommand parsing via msgspec structs.

Commands now use msgspec Structs for parameter validation:
Format: MoveCartCmd(pose, duration, speed_pct, accel_pct)
"""

import msgspec
import pytest

from parol6.commands.cartesian_commands import MoveCartCommand
from parol6.protocol.wire import MoveCartCmd


class TestMoveCartCommandParsing:
    """Test MoveCartCmd struct parsing and validation."""

    def test_parse_with_speed(self):
        """Parse with explicit speed."""
        # Create params struct
        params = MoveCartCmd(
            pose=[100.0, 200.0, 300.0, 0.0, 0.0, 0.0],
            speed_pct=50.0,
            accel_pct=75.0,
        )

        cmd = MoveCartCommand()
        cmd.assign_params(params)

        assert cmd.p.pose == [100.0, 200.0, 300.0, 0.0, 0.0, 0.0]
        assert cmd.p.duration == 0.0  # default
        assert cmd.p.speed_pct == 50.0
        assert cmd.p.accel_pct == 75.0

    def test_parse_accel_default(self):
        """Default acceleration should be 100."""
        params = MoveCartCmd(
            pose=[100.0, 200.0, 300.0, 0.0, 0.0, 0.0],
            speed_pct=50.0,
        )

        cmd = MoveCartCommand()
        cmd.assign_params(params)

        assert cmd.p.accel_pct == 100.0  # default

    def test_parse_with_duration(self):
        """Parse with duration instead of speed."""
        params = MoveCartCmd(
            pose=[100.0, 200.0, 300.0, 0.0, 0.0, 0.0],
            duration=2.5,
            accel_pct=80.0,
        )

        cmd = MoveCartCommand()
        cmd.assign_params(params)

        assert cmd.p.duration == 2.5
        assert cmd.p.speed_pct == 0.0  # default
        assert cmd.p.accel_pct == 80.0

    def test_parse_full_accel_range(self):
        """Test acceleration values at boundaries."""
        # Min accel (must be > 0)
        params1 = MoveCartCmd(
            pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            speed_pct=50.0,
            accel_pct=0.1,
        )
        cmd1 = MoveCartCommand()
        cmd1.assign_params(params1)
        assert cmd1.p.accel_pct == 0.1

        # Max accel
        params2 = MoveCartCmd(
            pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            speed_pct=50.0,
            accel_pct=100.0,
        )
        cmd2 = MoveCartCommand()
        cmd2.assign_params(params2)
        assert cmd2.p.accel_pct == 100.0

    def test_validation_requires_duration_or_speed(self):
        """Must have either duration > 0 or speed_pct > 0."""
        with pytest.raises((ValueError, msgspec.ValidationError)):
            # Both zero (default) should fail __post_init__
            MoveCartCmd(pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_validation_rejects_both_duration_and_speed(self):
        """Cannot have both duration > 0 and speed_pct > 0."""
        with pytest.raises((ValueError, msgspec.ValidationError)):
            MoveCartCmd(
                pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                duration=2.0,
                speed_pct=50.0,
            )

    def test_validation_pose_length(self):
        """Pose must have exactly 6 elements when decoded from wire format."""
        from parol6.protocol.wire import CmdType, decode_command

        # Validation happens during decode from msgpack (wire format)
        # Create a raw command with wrong pose length
        import msgspec

        encoder = msgspec.msgpack.Encoder()
        # Wire format: [tag, pose, duration, speed_pct, accel_pct]
        raw = encoder.encode([int(CmdType.MOVECART), [0.0, 0.0, 0.0], 0.0, 50.0, 100.0])

        with pytest.raises(msgspec.ValidationError):
            decode_command(raw)

    def test_command_init(self):
        """Test that MoveCartCommand initializes correctly."""
        cmd = MoveCartCommand()

        assert cmd.p is None  # No params until assigned
        assert cmd.is_valid
        assert not cmd.is_finished
