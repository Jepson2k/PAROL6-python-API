"""
Integration tests for motion profile commands.

Tests SETJOINTPROFILE/GETJOINTPROFILE and SETCARTPROFILE/GETCARTPROFILE through
the client API with a running server.
"""

import time

import pytest

from parol6 import RobotClient


@pytest.mark.integration
class TestJointProfileCommands:
    """Test joint motion profile get/set commands."""

    def test_get_joint_profile_returns_default(self, client, server_proc):
        """Test GETJOINTPROFILE returns default profile (TOPPRA) after reset."""
        # Reset to restore server defaults (fixture sets to LINEAR for speed)
        client.reset()
        profile = client.get_joint_profile()
        assert profile is not None
        assert profile == "TOPPRA"

    def test_set_and_get_joint_profile_roundtrip(self, client, server_proc):
        """Test setting a joint profile and getting it back."""
        # Test all valid joint profiles
        for profile in ["LINEAR", "QUINTIC", "TRAPEZOID", "SCURVE", "RUCKIG", "TOPPRA"]:
            assert client.set_joint_profile(profile) is True
            assert client.get_joint_profile() == profile

    def test_set_joint_profile_case_insensitive(self, client, server_proc):
        """Test that joint profile names are case-insensitive."""
        assert client.set_joint_profile("linear") is True
        assert client.get_joint_profile() == "LINEAR"

        assert client.set_joint_profile("Quintic") is True
        assert client.get_joint_profile() == "QUINTIC"


@pytest.mark.integration
class TestCartesianProfileCommands:
    """Test Cartesian motion profile get/set commands."""

    def test_get_cartesian_profile_returns_default(self, client, server_proc):
        """Test GETCARTPROFILE returns default profile (TOPPRA) after reset."""
        # Reset to restore server defaults (fixture sets to LINEAR for speed)
        client.reset()
        profile = client.get_cartesian_profile()
        assert profile is not None
        assert profile == "TOPPRA"

    def test_set_and_get_cartesian_profile_roundtrip(self, client, server_proc):
        """Test setting a Cartesian profile and getting it back."""
        # Only TOPPRA and LINEAR are valid for Cartesian moves
        for profile in ["LINEAR", "TOPPRA"]:
            assert client.set_cartesian_profile(profile) is True
            assert client.get_cartesian_profile() == profile

    def test_set_cartesian_profile_rejects_invalid(self, client, server_proc):
        """Test that SETCARTPROFILE rejects profiles that can't follow Cartesian paths."""
        # These profiles cannot be used for Cartesian moves
        invalid_profiles = ["RUCKIG", "QUINTIC", "TRAPEZOID", "SCURVE"]

        for profile in invalid_profiles:
            # Should return False (command rejected)
            result = client.set_cartesian_profile(profile)
            assert result is False, f"Expected {profile} to be rejected for Cartesian"

            # Profile should remain unchanged (TOPPRA from previous test or default)
            current = client.get_cartesian_profile()
            assert current in ("TOPPRA", "LINEAR"), f"Profile changed to {current} unexpectedly"

    def test_set_cartesian_profile_case_insensitive(self, client, server_proc):
        """Test that Cartesian profile names are case-insensitive."""
        assert client.set_cartesian_profile("linear") is True
        assert client.get_cartesian_profile() == "LINEAR"

        assert client.set_cartesian_profile("Toppra") is True
        assert client.get_cartesian_profile() == "TOPPRA"


@pytest.mark.integration
class TestProfileMotionBehavior:
    """Test that different profiles produce correct motion behavior."""

    def test_joint_move_reaches_target_all_profiles(self, client, server_proc):
        """Test that joint moves reach target position with all profiles."""
        target_angles = [10, -50, 190, 5, 10, 15]

        for profile in ["LINEAR", "QUINTIC", "TRAPEZOID", "SCURVE", "RUCKIG", "TOPPRA"]:
            # Reset to home first
            client.home(wait=True)

            # Set joint profile and execute move
            assert client.set_joint_profile(profile) is True
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
        """Test that Cartesian moves reach target position with all valid profiles."""
        # Start from home
        client.home(wait=True)
        start_pose = client.get_pose_rpy()
        assert start_pose is not None

        # Target pose (small offset from start)
        target_pose = [
            start_pose[0],
            start_pose[1] + 20,  # Y + 20mm
            start_pose[2],
            start_pose[3],
            start_pose[4],
            start_pose[5],
        ]

        # Only LINEAR and TOPPRA are valid for Cartesian moves
        for profile in ["LINEAR", "TOPPRA"]:
            # Reset to home first
            client.home(wait=True)

            # Set Cartesian profile and execute move
            assert client.set_cartesian_profile(profile) is True
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
        """Test streaming Cartesian moves with different profiles."""
        client.home(wait=True)
        start_pose = client.get_pose_rpy()
        assert start_pose is not None

        # Reset
        client.home(wait=True)
        assert client.stream_on() is True

        # Send a sequence of streaming Cartesian commands
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
    """Test Cartesian move precision with different profiles."""

    @pytest.mark.parametrize("profile", ["TOPPRA", "LINEAR"])
    def test_cartesian_simple_sequence(self, client, server_proc, profile):
        """
        Test precision of simple Cartesian moves (all valid profiles).

        All valid Cartesian profiles (TOPPRA, LINEAR) should handle this correctly.
        """
        client.home(wait=True)

        # Simple sequence: sweep X from -100 to 100 (no return to center)
        moves = [
            [-100, 250, 334, 90.0, 0.0, 90.0],
            [100, 250, 334, 90.0, 0.0, 90.0],
        ]

        assert client.set_cartesian_profile(profile) is True

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


@pytest.mark.integration
class TestIndependentProfiles:
    """Test that joint and Cartesian profiles are independent."""

    def test_profiles_are_independent(self, client, server_proc):
        """Test that setting one profile doesn't affect the other."""
        # Set joint to RUCKIG, Cartesian to LINEAR
        assert client.set_joint_profile("RUCKIG") is True
        assert client.set_cartesian_profile("LINEAR") is True

        # Verify both are set correctly
        assert client.get_joint_profile() == "RUCKIG"
        assert client.get_cartesian_profile() == "LINEAR"

        # Change joint profile, Cartesian should stay the same
        assert client.set_joint_profile("QUINTIC") is True
        assert client.get_joint_profile() == "QUINTIC"
        assert client.get_cartesian_profile() == "LINEAR"

        # Change Cartesian profile, joint should stay the same
        assert client.set_cartesian_profile("TOPPRA") is True
        assert client.get_joint_profile() == "QUINTIC"
        assert client.get_cartesian_profile() == "TOPPRA"
