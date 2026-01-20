"""
Integration test for streaming Cartesian move accuracy.

Tests the CartesianStreamingExecutor path used for TCP dragging.
Catches bugs where reference pose gets corrupted (e.g., aliasing with FK cache).
"""

import numpy as np
import pytest


def angle_diff(a: float, b: float) -> float:
    """Compute smallest angle difference considering wrapping."""
    diff = (a - b + 180) % 360 - 180
    return abs(diff)


def assert_pose_accuracy(
    final_pose: list[float],
    target: list[float],
    pos_tol_mm: float = 1.0,
    ori_tol_deg: float = 1.0,
    context: str = "",
) -> None:
    """Assert that final pose matches target within tolerances."""
    # Position check
    pos_error = np.linalg.norm(np.array(final_pose[:3]) - np.array(target[:3]))
    assert pos_error < pos_tol_mm, (
        f"{context}Position error {pos_error:.3f}mm exceeds {pos_tol_mm}mm tolerance. "
        f"Target: {target[:3]}, Final: {final_pose[:3]}"
    )

    # Orientation check
    for i, axis in enumerate(["RX", "RY", "RZ"]):
        ori_error = angle_diff(final_pose[3 + i], target[3 + i])
        assert ori_error < ori_tol_deg, (
            f"{context}{axis} error {ori_error:.3f}° exceeds {ori_tol_deg}° tolerance. "
            f"Target: {target[3 + i]:.1f}°, Final: {final_pose[3 + i]:.1f}°"
        )


@pytest.mark.integration
class TestStreamingCartesianAccuracy:
    """Test that streaming cartesian moves reach correct targets."""

    def test_streaming_movecart_reaches_target(self, client, server_proc):
        """Streaming cartesian move should arrive at the requested target.

        Tests the CartesianStreamingExecutor path activated by stream_on().
        """
        assert client.enable() is True
        assert client.home() is True
        assert client.wait_motion_complete(timeout=15.0)

        # Get starting pose
        start_pose = client.get_pose_rpy()
        print(f"\nStart pose: {start_pose}")

        # Enable streaming mode (activates CartesianStreamingExecutor)
        assert client.stream_on() is True

        try:
            # Target: offset from start (like beginning of a TCP drag)
            target = list(start_pose)
            target[0] += 30.0  # +15mm in X

            print(f"Target pose: {target}")

            # Send streaming cartesian move
            result = client.move_cartesian(target, speed=100)
            assert result is True

            # Wait for motion to settle
            assert client.wait_motion_complete(timeout=10.0)

            # Verify final pose
            final_pose = client.get_pose_rpy()
            print(f"Final pose:  {final_pose}")

            assert_pose_accuracy(final_pose, target)
            print("✓ Streaming movecart reached target accurately")

        finally:
            # Always disable streaming mode
            client.stream_off()

    def test_streaming_movecart_sequential_targets(self, client, server_proc):
        """Sequential streaming moves should each reach their target.

        Simulates TCP dragging behavior where multiple MOVECART commands
        are sent in sequence while streaming mode is active.
        """
        assert client.enable() is True
        assert client.home() is True
        assert client.wait_motion_complete(timeout=15.0)

        start_pose = client.get_pose_rpy()
        print(f"\nStart pose: {start_pose}")

        # Enable streaming mode
        assert client.stream_on() is True

        try:
            # Simulate a drag path: series of small incremental moves
            # This pattern catches bugs where reference pose gets corrupted
            # between moves (like the FK cache aliasing bug)
            offsets = [
                (30.0, 0.0, 0.0),  # +30mm X
                (30.0, 30.0, 0.0),  # +30mm X, +30mm Y
                (30.0, 30.0, -30.0),  # +30mm X, +30mm Y, -30mm Z
                (-30.0, -30.0, -30.0),  # back toward start
                (0.0, 0.0, 0.0),  # back to start
            ]

            for i, (dx, dy, dz) in enumerate(offsets):
                target = list(start_pose)
                target[0] += dx
                target[1] += dy
                target[2] += dz

                print(f"\n--- Move {i + 1}/{len(offsets)} ---")
                print(f"Target: {target[:3]}")

                result = client.move_cartesian(target, speed=100)
                assert result is True

                # Wait for this move to complete before next
                assert client.wait_motion_complete(timeout=10.0, settle_window=2.0)

                final_pose = client.get_pose_rpy()
                start_pose = final_pose
                print(f"Final:  {final_pose[:3]}")

                assert_pose_accuracy(final_pose, target, context=f"Move {i + 1}: ")
                print(f"✓ Move {i + 1} accurate")

            print("\n✓ All sequential streaming moves reached targets accurately")

        finally:
            client.stream_off()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
