#!/usr/bin/env python3
"""
Measure TCP path precision during Cartesian motion.

Runs the same moves as programs/precision.py and samples actual TCP position
to measure deviation from the ideal straight-line path.

Starts its own controller in simulator mode - no external server needed.
"""

import asyncio
import logging
import time
import numpy as np
from parol6 import AsyncRobotClient
from parol6.client.manager import managed_server

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("toppra").setLevel(logging.INFO)


# Same moves as programs/precision.py
MOVES = [
    ([-0.000, 200, 425, 90.009, -0.000, 90.000], 100, 100),
    ([200, 250, 300, 90, -0.000, 90.000], 100, 100),
    ([-115, 250, 300, 90, -0, 90], 50, 50),
    ([-0.000, 350.000, 300, 90, -0.000, 90.000], 100, 100),
    ([0.000, 250, 80, -0, -0.000, 90.000], 100, 100),
    ([0.000, 200, 80, -0, -0.000, 90.000], 100, 100),
    ([0.000, 300, 80, -0, -0.000, 90.000], 100, 100),
    ([0.000, 250, 80, -0, -0.000, 90.000], 100, 100),
    ([0.000, 250, 80, -50, -0.000, 90.000], 100, 100),
    ([0.000, 250, 80, 50, -0.000, 90.000], 100, 100),
    ([0.000, 250, 80, -0, -0.000, 90.000], 100, 100),
    ([0.000, 250, 80, -0, 50.000, 90.000], 100, 100),
    ([0.000, 250, 80, -0, -50.000, 90.000], 100, 100),
    ([0.000, 250, 80, -0, -0.000, 90.000], 100, 100),
]


def compute_line_deviation(start_pos, end_pos, sample_pos):
    """
    Compute perpendicular distance from sample_pos to line from start_pos to end_pos.
    All positions are [x, y, z] in mm.
    """
    start = np.array(start_pos[:3])
    end = np.array(end_pos[:3])
    sample = np.array(sample_pos[:3])

    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len < 0.001:
        # Start and end are same point
        return np.linalg.norm(sample - start)

    line_unit = line_vec / line_len
    point_vec = sample - start

    # Project point onto line
    projection_len = np.dot(point_vec, line_unit)
    projection = start + projection_len * line_unit

    # Distance from point to projection
    return np.linalg.norm(sample - projection)


async def measure_move(client, start_pose, target_pose, speed, accel, move_num):
    """Execute a move and measure path deviation."""
    samples = []
    sample_times = []

    print(
        f"\n--- Move {move_num}: [{target_pose[0]:.1f}, {target_pose[1]:.1f}, {target_pose[2]:.1f}] speed={speed} ---"
    )

    # Start the move
    start_time = time.time()
    await client.move_cartesian(target_pose, speed=speed, accel=accel)

    # Sample position during motion
    while True:
        pose = await client.get_pose_rpy()
        if pose:
            t = time.time() - start_time
            samples.append(pose)
            sample_times.append(t)

        # Check if motion complete
        dist_to_target = (
            np.sqrt(
                (pose[0] - target_pose[0]) ** 2
                + (pose[1] - target_pose[1]) ** 2
                + (pose[2] - target_pose[2]) ** 2
            )
            if pose
            else 999
        )

        if dist_to_target < 1.0:
            # Capture a few more samples
            for _ in range(3):
                await asyncio.sleep(0.01)
                pose = await client.get_pose_rpy()
                if pose:
                    samples.append(pose)
                    sample_times.append(time.time() - start_time)
            break

        if time.time() - start_time > 10.0:
            print("  Motion timeout!")
            break

        await asyncio.sleep(0.015)  # ~66Hz sampling

    # Wait for motion to complete
    await client.wait_motion_complete(timeout=5.0)

    duration = time.time() - start_time

    if len(samples) < 2:
        print(f"  Not enough samples ({len(samples)})")
        return None

    # Compute path deviations
    deviations = [compute_line_deviation(start_pose, target_pose, s) for s in samples]

    # Compute velocities
    positions = np.array([[s[0], s[1], s[2]] for s in samples])
    times = np.array(sample_times)

    if len(positions) > 1:
        dt = np.diff(times)
        dp = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        velocities = dp / np.where(dt > 0.001, dt, 0.001)
    else:
        velocities = [0]

    # Report
    max_dev = max(deviations)
    mean_dev = np.mean(deviations)
    max_vel = max(velocities)
    mean_vel = np.mean(velocities)

    print(f"  Duration: {duration:.2f}s, Samples: {len(samples)}")
    print(f"  Path deviation: max={max_dev:.2f}mm, mean={mean_dev:.2f}mm")
    print(f"  Velocity: max={max_vel:.1f}mm/s, mean={mean_vel:.1f}mm/s")

    return {
        "move_num": move_num,
        "duration": duration,
        "samples": len(samples),
        "max_deviation": max_dev,
        "mean_deviation": mean_dev,
        "max_velocity": max_vel,
        "mean_velocity": mean_vel,
    }


async def run_precision_test(host: str, port: int):
    """Run all moves and measure path deviation."""
    client = AsyncRobotClient(host=host, port=port)

    try:
        print("=== TCP Path Precision Test ===", flush=True)
        print(f"Running {len(MOVES)} moves from programs/precision.py\n", flush=True)

        # Get status
        status = await client.get_status()
        print(f"Robot status: {status}")

        # Set Cartesian motion profile for path following
        await client.set_cartesian_profile("TOPPRA")
        print("Cartesian motion profile set to TOPPRA")

        # Get initial pose
        current_pose = await client.get_pose_rpy()
        if not current_pose:
            print("ERROR: Could not get current pose")
            return

        print(
            f"Start pose: X={current_pose[0]:.1f}, Y={current_pose[1]:.1f}, Z={current_pose[2]:.1f}"
        )

        results = []

        for i, (target, speed, accel) in enumerate(MOVES):
            start_pose = await client.get_pose_rpy()
            if not start_pose:
                print(f"ERROR: Could not get pose before move {i + 1}")
                continue

            result = await measure_move(client, start_pose, target, speed, accel, i + 1)
            if result:
                results.append(result)

            # Brief pause between moves
            await asyncio.sleep(0.1)

        # Summary
        print("\n" + "=" * 50)
        print("=== SUMMARY ===")
        print("=" * 50)

        if results:
            all_max_dev = [r["max_deviation"] for r in results]
            all_mean_dev = [r["mean_deviation"] for r in results]
            all_max_vel = [r["max_velocity"] for r in results]
            all_durations = [r["duration"] for r in results]

            print("\nPath Deviation (mm):")
            print(
                f"  Worst max deviation: {max(all_max_dev):.2f}mm (move {all_max_dev.index(max(all_max_dev)) + 1})"
            )
            print(f"  Average max deviation: {np.mean(all_max_dev):.2f}mm")
            print(f"  Average mean deviation: {np.mean(all_mean_dev):.2f}mm")

            print("\nVelocity (mm/s):")
            print(f"  Peak velocity: {max(all_max_vel):.1f}mm/s")
            print(f"  Average peak: {np.mean(all_max_vel):.1f}mm/s")

            print("\nTiming:")
            print(f"  Total time: {sum(all_durations):.1f}s")
            print(f"  Average move time: {np.mean(all_durations):.2f}s")
        else:
            print("No valid results collected")

    finally:
        await client.close()


def main():
    """Entry point with proper server management."""
    host = "127.0.0.1"
    port = 5001

    print("Starting controller in simulator mode...", flush=True)

    with managed_server(
        host=host,
        port=port,
        extra_env={
            "PAROL6_FAKE_SERIAL": "1",
            "PAROL6_NOAUTOHOME": "1",
            "PAROL6_LOG_LEVEL": "DEBUG",
        },
    ):
        print("Controller ready.\n", flush=True)
        asyncio.run(run_precision_test(host, port))

    print("\nController stopped.", flush=True)


if __name__ == "__main__":
    main()
