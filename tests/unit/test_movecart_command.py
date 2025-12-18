"""Tests for MoveCartCommand parsing, including acceleration parameter."""

import pytest
from parol6.commands.cartesian_commands import MoveCartCommand
from parol6.config import DEFAULT_ACCEL_PERCENT


class TestMoveCartCommandParsing:
    """Test MoveCartCommand.do_match parsing."""

    def test_parse_legacy_9_params_no_accel(self):
        """Legacy 9-parameter format (without accel) should use default accel."""
        cmd = MoveCartCommand()
        # Format: MOVECART|x|y|z|rx|ry|rz|duration|speed
        parts = ["MOVECART", "100", "200", "300", "0", "0", "0", "NONE", "50"]
        ok, err = cmd.do_match(parts)

        assert ok is True
        assert err is None
        assert cmd.pose == [100.0, 200.0, 300.0, 0.0, 0.0, 0.0]
        assert cmd.duration is None
        assert cmd.velocity_percent == 50.0
        assert cmd.accel_percent == DEFAULT_ACCEL_PERCENT

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
        """Fewer than 9 parameters should fail."""
        cmd = MoveCartCommand()
        parts = ["MOVECART", "100", "200", "300", "0", "0", "0", "NONE"]  # Only 8
        ok, err = cmd.do_match(parts)

        assert ok is False
        assert err is not None
        assert "8-9" in err or "parameters" in err.lower()

    def test_parse_too_many_params_fails(self):
        """More than 10 parameters should fail."""
        cmd = MoveCartCommand()
        parts = ["MOVECART", "100", "200", "300", "0", "0", "0", "NONE", "50", "75", "EXTRA"]
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


class TestAccelAffectsDuration:
    """Test that acceleration parameter affects motion duration."""

    def test_trapezoidal_duration_edge_cases(self):
        """Helper handles zero/invalid inputs correctly."""
        td = MoveCartCommand._trapezoidal_duration
        assert td(0, 1, 1) == 0.0  # Zero distance
        assert td(1, 0, 1) == 0.0  # Zero velocity
        assert td(1, 1, 0) == 0.0  # Zero accel

    def test_trapezoidal_duration_formulas(self):
        """Helper uses correct trapezoid/triangle formulas."""
        import math

        td = MoveCartCommand._trapezoidal_duration
        # Long move: t = d/v + v/a = 10 + 1 = 11s
        assert abs(td(1.0, 0.1, 0.1) - 11.0) < 0.001
        # Short move: t = 2*sqrt(d/a) ≈ 1.414s
        assert abs(td(0.05, 0.1, 0.1) - 2 * math.sqrt(0.5)) < 0.001

    def test_higher_accel_gives_shorter_duration(self):
        """Higher acceleration percentage results in shorter motion duration."""
        td = MoveCartCommand._trapezoidal_duration
        low_accel_dur = td(0.1, 0.1, 0.05)  # Low accel
        high_accel_dur = td(0.1, 0.1, 0.3)  # High accel
        assert high_accel_dur < low_accel_dur
        assert low_accel_dur / high_accel_dur > 1.5  # Meaningful difference
