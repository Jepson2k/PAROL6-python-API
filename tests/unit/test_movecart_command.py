"""Tests for MoveCartCommand parsing, including acceleration parameter."""

from parol6.commands.cartesian_commands import MoveCartCommand
from parol6.config import DEFAULT_ACCEL_PERCENT


class TestMoveCartCommandParsing:
    """Test MoveCartCommand.do_match parsing."""

    def test_parse_10_params_with_accel(self):
        """10-parameter format with explicit acceleration."""
        cmd = MoveCartCommand()
        # Format: MOVECART|x|y|z|rx|ry|rz|duration|speed|accel
        parts = ["MOVECART", "100", "200", "300", "0", "0", "0", "NONE", "50", "75"]
        ok, err = cmd.do_match(parts)

        assert ok is True
        assert err is None
        assert cmd.pose == [100.0, 200.0, 300.0, 0.0, 0.0, 0.0]
        assert cmd.duration is None
        assert cmd.velocity_percent == 50.0
        assert cmd.accel_percent == 75.0

    def test_parse_10_params_accel_none(self):
        """10-parameter format with NONE acceleration should use default."""
        cmd = MoveCartCommand()
        parts = ["MOVECART", "100", "200", "300", "0", "0", "0", "NONE", "50", "NONE"]
        ok, err = cmd.do_match(parts)

        assert ok is True
        assert err is None
        assert cmd.accel_percent == DEFAULT_ACCEL_PERCENT

    def test_parse_with_duration(self):
        """Parse with duration instead of speed."""
        cmd = MoveCartCommand()
        parts = ["MOVECART", "100", "200", "300", "0", "0", "0", "2.5", "NONE", "80"]
        ok, err = cmd.do_match(parts)

        assert ok is True
        assert err is None
        assert cmd.duration == 2.5
        assert cmd.velocity_percent is None
        assert cmd.accel_percent == 80.0

    def test_parse_full_accel_range(self):
        """Test acceleration values at boundaries."""
        cmd = MoveCartCommand()

        # Min accel
        parts = ["MOVECART", "0", "0", "0", "0", "0", "0", "NONE", "50", "1"]
        ok, _ = cmd.do_match(parts)
        assert ok is True
        assert cmd.accel_percent == 1.0

        # Max accel
        cmd2 = MoveCartCommand()
        parts = ["MOVECART", "0", "0", "0", "0", "0", "0", "NONE", "50", "100"]
        ok, _ = cmd2.do_match(parts)
        assert ok is True
        assert cmd2.accel_percent == 100.0

    def test_parse_too_few_params_fails(self):
        """Fewer than 10 parameters should fail."""
        cmd = MoveCartCommand()
        parts = ["MOVECART", "100", "200", "300", "0", "0", "0", "NONE", "50"]  # Only 9
        ok, err = cmd.do_match(parts)

        assert ok is False
        assert err is not None
        assert "9 parameters" in err or "parameters" in err.lower()

    def test_parse_too_many_params_fails(self):
        """More than 10 parameters should fail."""
        cmd = MoveCartCommand()
        parts = [
            "MOVECART",
            "100",
            "200",
            "300",
            "0",
            "0",
            "0",
            "NONE",
            "50",
            "75",
            "EXTRA",
        ]
        ok, err = cmd.do_match(parts)

        assert ok is False
        assert err is not None

    def test_init_defaults(self):
        """Test that MoveCartCommand initializes with proper defaults."""
        cmd = MoveCartCommand()

        assert cmd.accel_percent == DEFAULT_ACCEL_PERCENT
        assert cmd.velocity_percent is None
        assert cmd.duration is None
        assert cmd.pose is None  # pose is None until do_match parses a command
