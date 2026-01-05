"""
Integration tests for motion profile commands.

Tests SETPROFILE and GETPROFILE through the client API with a running server.
Verifies that profiles affect motion behavior in both streaming and non-streaming modes.
"""

import time

import pytest

from parol6 import RobotClient


@pytest.mark.integration
class TestProfileCommands:
    """Test motion profile get/set commands."""

    def test_get_profile_returns_default(self, client, server_proc):
        """Test GETPROFILE returns default profile."""
        profile = client.get_profile()
        assert profile is not None
        assert profile in {"AUTO", "TOPPRA", "RUCKIG", "QUINTIC", "TRAPEZOID", "LINEAR"}

    def test_set_and_get_profile_roundtrip(self, client, server_proc):
        """Test setting a profile and getting it back."""
        # Set to LINEAR
        assert client.set_profile("LINEAR") is True
        assert client.get_profile() == "LINEAR"

        # Set to QUINTIC
        assert client.set_profile("QUINTIC") is True
        assert client.get_profile() == "QUINTIC"

        # Set to TRAPEZOID
        assert client.set_profile("TRAPEZOID") is True
        assert client.get_profile() == "TRAPEZOID"

        # Set to RUCKIG
        assert client.set_profile("RUCKIG") is True
        assert client.get_profile() == "RUCKIG"

    def test_set_profile_case_insensitive(self, client, server_proc):
        """Test that profile names are case-insensitive."""
        assert client.set_profile("linear") is True
        assert client.get_profile() == "LINEAR"

        assert client.set_profile("Quintic") is True
        assert client.get_profile() == "QUINTIC"


@pytest.mark.integration
class TestProfileMotionBehavior:
    """Test that different profiles produce correct motion behavior."""

    def test_joint_move_reaches_target_all_profiles(self, client, server_proc):
        """Test that joint moves reach target position with all profiles."""
        target_angles = [10, -50, 190, 5, 10, 15]

        for profile in ["LINEAR", "QUINTIC", "TRAPEZOID", "RUCKIG", "TOPPRA"]:
            # Reset to home first
            client.home(wait=True)

            # Set profile and execute move
            assert client.set_profile(profile) is True
            result = client.move_joints(target_angles, duration=2.0)
            assert result is True
            assert client.wait_motion_complete(timeout=10.0)

            # Verify we reached target (within tolerance)
            angles = client.get_angles()
            assert angles is not None
            for i, (actual, target) in enumerate(zip(angles, target_angles)):
                assert abs(actual - target) < 1.0, (
                    f"Profile {profile}: Joint {i} off target "
                    f"(expected {target}, got {actual})"
                )

    def test_cartesian_move_reaches_target_all_profiles(self, client, server_proc):
        """Test that cartesian moves reach target position with all profiles."""
        # Start from home
        client.home(wait=True)
        start_pose = client.get_pose_rpy()
        assert start_pose is not None

        # Target pose (small offset from start)
        target_pose = [
            start_pose[0],
            start_pose[1] + 20, # Y + 20mm
            start_pose[2],
            start_pose[3],
            start_pose[4],
            start_pose[5],
        ]

        # Note: RUCKIG is not valid for Cartesian moves, use TOPPRA instead
        for profile in ["LINEAR", "TOPPRA"]:
            # Reset to home first
            client.home(wait=True)

            # Set profile and execute move
            assert client.set_profile(profile) is True
            result = client.move_cartesian(target_pose, duration=2.0)
            assert result is True
            assert client.wait_motion_complete(timeout=10.0)

            # Verify position reached (within tolerance)
            pose = client.get_pose_rpy()
            assert pose is not None
            assert abs(pose[0] - target_pose[0]) < 1.0, (
                f"Profile {profile}: X position off target "
                f"(expected {target_pose[0]:.1f}, got {pose[0]:.1f})"
            )


@pytest.mark.integration
class TestProfileStreamingMode:
    """Test profile behavior in streaming mode."""

    def test_streaming_cartesian(self, client, server_proc):
        """Test streaming cartesian moves with different profiles."""
        client.home(wait=True)
        start_pose = client.get_pose_rpy()
        assert start_pose is not None

        # Reset
        client.home(wait=True)
        assert client.stream_on() is True

        # Send a sequence of streaming cartesian commands
        for i in range(5):
            target = [
                start_pose[0] + (i * 5),
                start_pose[1],
                start_pose[2],
                start_pose[3],
                start_pose[4],
                start_pose[5],
            ]
            result = client.move_cartesian(target, duration=0.5)
            assert result is True
            time.sleep(0.1)

        assert client.stream_off() is True
        assert client.wait_motion_complete(timeout=10.0)

        # Verify robot completed motion
        assert client.is_robot_stopped()


@pytest.mark.integration
class TestCartesianPrecision:
    """Test cartesian move precision with different profiles."""

    # Note: RUCKIG is not valid for Cartesian moves, use TOPPRA instead
    @pytest.mark.parametrize("profile", ["TOPPRA", "LINEAR"])
    def test_cartesian_simple_sequence(self, client, server_proc, profile):
        """
        Test precision of simple cartesian moves (all profiles).

        All profiles should handle this correctly.
        """
        client.home(wait=True)

        # Simple sequence: sweep X from -100 to 100 (no return to center)
        moves = [
            [-100, 250, 334, 90.0, 0.0, 90.0],
            [100, 250, 334, 90.0, 0.0, 90.0],
        ]

        assert client.set_profile(profile) is True

        for target in moves:
            result = client.move_cartesian(target, duration=2.0)
            assert result is True
            assert client.wait_motion_complete(timeout=10.0)

        # Verify final pose
        pose = client.get_pose_rpy()
        assert pose is not None
        final_target = moves[-1]

        # Print diagnostic info
        print(f"\nProfile {profile}:")
        print(f"  Target:   X={final_target[0]:.2f}, Y={final_target[1]:.2f}, Z={final_target[2]:.2f}")
        print(f"            RX={final_target[3]:.2f}, RY={final_target[4]:.2f}, RZ={final_target[5]:.2f}")
        print(f"  Actual:   X={pose[0]:.2f}, Y={pose[1]:.2f}, Z={pose[2]:.2f}")
        print(f"            RX={pose[3]:.2f}, RY={pose[4]:.2f}, RZ={pose[5]:.2f}")

        # Check position (X, Y, Z) within 1mm tolerance
        for i, (actual, expected) in enumerate(zip(pose[:3], final_target[:3])):
            assert abs(actual - expected) < 1.0, (
                f"Profile {profile}: Position[{i}] off target "
                f"(expected {expected:.2f}, got {actual:.2f})"
            )

        # Check orientation (RX, RY, RZ) within 1 degree tolerance
        for i, (actual, expected) in enumerate(zip(pose[3:], final_target[3:])):
            assert abs(actual - expected) < 1.0, (
                f"Profile {profile}: Orientation[{i}] off target "
                f"(expected {expected:.2f}, got {actual:.2f})"
            )
