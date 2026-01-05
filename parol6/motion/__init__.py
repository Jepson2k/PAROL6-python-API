"""
Motion pipeline for trajectory generation and execution.

This module provides a unified pipeline for all motion commands:
- Cartesian commands (MoveCart, MoveCartRelTrf)
- Joint commands (MovePose, MoveJoint)
- Smooth shapes (Circle, Arc, Helix, Spline)

All commands produce a JointPath that gets converted to a Trajectory
via time-optimal path parameterization (TOPP-RA).

Streaming executors provide real-time jerk-limited motion for jogging.
"""

from parol6.motion.streaming_executors import (
    CartesianStreamingExecutor,
    RuckigExecutorBase,
    StreamingExecutor,
)
from parol6.motion.trajectory import (
    JointPath,
    ProfileType,
    Trajectory,
    TrajectoryBuilder,
)

__all__ = [
    "JointPath",
    "Trajectory",
    "TrajectoryBuilder",
    "ProfileType",
    "StreamingExecutor",
    "CartesianStreamingExecutor",
    "RuckigExecutorBase",
]
