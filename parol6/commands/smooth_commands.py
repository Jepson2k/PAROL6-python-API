"""
Smooth Geometry Commands

Commands for generating smooth geometric paths: circles, arcs, and splines.
These use the unified motion pipeline with TOPP-RA for time-optimal path parameterization.
"""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from parol6.commands.base import TrajectoryMoveCommandBase
from parol6.config import INTERVAL_S, LIMITS, steps_to_rad
from parol6.motion import CircularMotion, JointPath, SplineMotion, TrajectoryBuilder
from parol6.protocol.wire import (
    CmdType,
    SmoothArcCenterCmd,
    SmoothArcParamCmd,
    SmoothCircleCmd,
    SmoothSplineCmd,
)
from parol6.server.command_registry import register_command
from parol6.server.state import get_fkine_se3
from parol6.utils.errors import IKError
from parol6.utils.se3_utils import se3_from_rpy, se3_from_trans, se3_rpy

if TYPE_CHECKING:
    from parol6.server.state import ControllerState

logger = logging.getLogger(__name__)


# =============================================================================
# TRF/WRF Transformation Utilities
# =============================================================================

# Plane normal vectors in Tool Reference Frame
_PLANE_NORMALS_TRF: dict[str, NDArray] = {
    "XY": np.array([0.0, 0.0, 1.0]),  # Tool's Z-axis
    "XZ": np.array([0.0, 1.0, 0.0]),  # Tool's Y-axis
    "YZ": np.array([1.0, 0.0, 0.0]),  # Tool's X-axis
}

# Pre-allocated workspace buffers for TRF/WRF transformations (command setup phase)
_pose_trf_buf: np.ndarray = np.zeros((4, 4), dtype=np.float64)
_pose_wrf_buf: np.ndarray = np.zeros((4, 4), dtype=np.float64)
_rpy_rad_buf: np.ndarray = np.zeros(3, dtype=np.float64)
_result_6_buf: np.ndarray = np.zeros(6, dtype=np.float64)


def _pose6_trf_to_wrf(
    pose6_mm_deg: Sequence[float], tool_pose: np.ndarray
) -> list[float]:
    """Convert 6D pose [x,y,z,rx,ry,rz] from TRF to WRF (mm, degrees)."""
    se3_from_rpy(
        pose6_mm_deg[0] / 1000.0,
        pose6_mm_deg[1] / 1000.0,
        pose6_mm_deg[2] / 1000.0,
        np.radians(pose6_mm_deg[3]),
        np.radians(pose6_mm_deg[4]),
        np.radians(pose6_mm_deg[5]),
        _pose_trf_buf,
    )
    np.matmul(tool_pose, _pose_trf_buf, out=_pose_wrf_buf)
    se3_rpy(_pose_wrf_buf, _rpy_rad_buf)
    # Build result in pre-allocated buffer
    _result_6_buf[:3] = _pose_wrf_buf[:3, 3] * 1000.0
    _result_6_buf[3:] = np.degrees(_rpy_rad_buf)
    return _result_6_buf.tolist()


def _transform_center_trf_to_wrf(
    params: dict[str, Any], tool_pose: np.ndarray, transformed: dict[str, Any]
) -> None:
    """Transform 'center' parameter from TRF (mm) to WRF (mm)."""
    se3_from_trans(
        params["center"][0] / 1000.0,
        params["center"][1] / 1000.0,
        params["center"][2] / 1000.0,
        _pose_trf_buf,
    )
    np.matmul(tool_pose, _pose_trf_buf, out=_pose_wrf_buf)
    transformed["center"] = (_pose_wrf_buf[:3, 3] * 1000.0).tolist()


def _transform_command_params_to_wrf(
    command_type: str, params: dict[str, Any], frame: str
) -> dict[str, Any]:
    """Transform command parameters from TRF to WRF. No-op for WRF."""
    if frame == "WRF":
        return params

    tool_pose = get_fkine_se3()
    transformed = params.copy()

    if command_type == "SMOOTH_CIRCLE":
        if "center" in params:
            _transform_center_trf_to_wrf(params, tool_pose, transformed)
        if "plane" in params:
            normal_trf = _PLANE_NORMALS_TRF[params["plane"]]
            normal_wrf = tool_pose[:3, :3] @ normal_trf
            transformed["normal_vector"] = normal_wrf.tolist()

    elif command_type == "SMOOTH_ARC_CENTER":
        if "center" in params:
            _transform_center_trf_to_wrf(params, tool_pose, transformed)
        if "end_pose" in params:
            transformed["end_pose"] = _pose6_trf_to_wrf(params["end_pose"], tool_pose)
        if "plane" in params:
            normal_trf = _PLANE_NORMALS_TRF[params["plane"]]
            normal_wrf = tool_pose[:3, :3] @ normal_trf
            transformed["normal_vector"] = normal_wrf.tolist()

    elif command_type == "SMOOTH_ARC_PARAM":
        if "end_pose" in params:
            transformed["end_pose"] = _pose6_trf_to_wrf(params["end_pose"], tool_pose)
        if "plane" not in params:
            params["plane"] = "XY"
        normal_trf = _PLANE_NORMALS_TRF[params.get("plane", "XY")]
        normal_wrf = tool_pose[:3, :3] @ normal_trf
        transformed["normal_vector"] = normal_wrf.tolist()

    elif command_type == "SMOOTH_SPLINE":
        if "waypoints" in params:
            transformed["waypoints"] = [
                _pose6_trf_to_wrf(wp, tool_pose) for wp in params["waypoints"]
            ]

    return transformed


# =============================================================================
# Smooth Motion Command Base
# =============================================================================


class BaseSmoothMotionCommand(TrajectoryMoveCommandBase):
    """Base class for smooth geometry commands (circle, arc, helix, spline).

    Subclasses implement generate_main_trajectory() to create Cartesian geometry.
    This base class handles IK conversion and trajectory building.
    """

    __slots__ = (
        "description",
        "normal_vector",
    )

    def __init__(self, description: str = "smooth geometry") -> None:
        super().__init__()
        self.description = description
        self.normal_vector: list[float] | None = None

    def _transform_params(
        self, command_type: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Transform params from TRF to WRF if needed. No-op for WRF frame."""
        assert self.p is not None
        return _transform_command_params_to_wrf(command_type, params, self.p.frame)

    def get_current_pose(self, state: "ControllerState") -> np.ndarray:
        """Get current TCP pose as [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]."""
        current_se3 = get_fkine_se3(state)
        current_xyz = current_se3[:3, 3] * 1000  # m -> mm
        rpy_rad = np.zeros(3, dtype=np.float64)
        se3_rpy(current_se3, rpy_rad)
        return np.concatenate([current_xyz, np.degrees(rpy_rad)])

    def do_setup(self, state: "ControllerState") -> None:
        """Pre-compute trajectory from current position."""
        assert self.p is not None

        self.log_debug("  -> Preparing %s...", self.description)

        current_pose = self.get_current_pose(state)
        self.log_info(
            "  -> Generating %s from position: %s",
            self.description,
            [round(p, 1) for p in current_pose[:3]],
        )

        cartesian_trajectory = self.generate_main_trajectory(current_pose)
        if cartesian_trajectory is None or len(cartesian_trajectory) == 0:
            self.fail("Trajectory generation returned empty result")
            return

        steps_to_rad(state.Position_in, self._q_rad_buf)

        try:
            joint_path = JointPath.from_poses(
                cartesian_trajectory, self._q_rad_buf, quiet_logging=True
            )
        except IKError as e:
            self.log_error("  -> ERROR: IK failed during trajectory generation: %s", e)
            self.fail(str(e))
            return

        # Get duration and velocity/accel percent from params
        duration = (
            self.p.duration
            if self.p.duration is not None and self.p.duration > 0.0
            else None
        )
        vel_pct = self.p.speed_pct if self.p.speed_pct is not None else 100.0
        acc_pct = self.p.accel_pct

        cart_vel_max = LIMITS.cart.hard.velocity.linear * (vel_pct / 100.0)
        cart_acc_max = LIMITS.cart.hard.acceleration.linear * (acc_pct / 100.0)

        builder = TrajectoryBuilder(
            joint_path=joint_path,
            profile=state.motion_profile,
            velocity_percent=vel_pct if duration is None else None,
            accel_percent=acc_pct,
            duration=duration,
            dt=INTERVAL_S,
            cart_vel_limit=cart_vel_max,
            cart_acc_limit=cart_acc_max,
        )

        trajectory = builder.build()
        self.trajectory_steps = trajectory.steps

        self.log_info(
            "  -> Trajectory prepared: %d steps, %.2fs duration",
            len(self.trajectory_steps),
            trajectory.duration,
        )

    def generate_main_trajectory(self, effective_start_pose):
        """Override this in subclasses to generate the specific motion trajectory."""
        raise NotImplementedError("Subclasses must implement generate_main_trajectory")


@register_command(CmdType.SMOOTH_CIRCLE)
class SmoothCircleCommand(BaseSmoothMotionCommand):
    """Execute smooth circular motion."""

    PARAMS_TYPE = SmoothCircleCmd

    __slots__ = ("_center",)

    def __init__(self) -> None:
        super().__init__(description="circle")
        self._center: list[float] | None = None

    def do_setup(self, state: "ControllerState") -> None:
        """Transform parameters if in TRF, then prepare trajectory."""
        assert self.p is not None

        self.description = f"circle (r={self.p.radius}mm)"

        transformed = self._transform_params(
            "SMOOTH_CIRCLE", {"center": list(self.p.center), "plane": self.p.plane}
        )
        self._center = transformed["center"]
        self.normal_vector = transformed.get("normal_vector")
        return super().do_setup(state)

    def generate_main_trajectory(self, effective_start_pose):
        """Generate circle geometry starting from actual position."""
        assert self.p is not None

        motion_gen = CircularMotion()

        # Determine normal vector
        if self.normal_vector is not None:
            normal = np.array(self.normal_vector)
        else:
            plane_normals = {"XY": [0, 0, 1], "XZ": [0, 1, 0], "YZ": [1, 0, 0]}
            normal = np.array(plane_normals.get(self.p.plane, [0, 0, 1]))

        # Handle center_mode
        if self.p.center_mode == "TOOL":
            actual_center = np.array(effective_start_pose[:3])
        elif self.p.center_mode == "RELATIVE":
            center_np = (
                np.asarray(self._center, dtype=float)
                if self._center is not None
                else np.zeros(3)
            )
            actual_center = np.array(effective_start_pose[:3]) + center_np
        else:
            actual_center = (
                np.array(self._center) if self._center is not None else np.zeros(3)
            )

        # Use duration for geometry sampling, default to 4s for circle if not specified
        geom_duration = (
            self.p.duration
            if self.p.duration is not None and self.p.duration > 0.0
            else 4.0
        )
        trajectory = motion_gen.generate_circle(
            center=actual_center,
            radius=self.p.radius,
            normal=normal,
            duration=geom_duration,
            start_point=effective_start_pose,
        )

        if self.p.clockwise:
            trajectory = trajectory[::-1]

        # Update orientations to match start pose
        for i in range(len(trajectory)):
            trajectory[i][3:] = effective_start_pose[3:]

        return trajectory


@register_command(CmdType.SMOOTH_ARC_CENTER)
class SmoothArcCenterCommand(BaseSmoothMotionCommand):
    """Execute smooth arc motion defined by center point."""

    PARAMS_TYPE = SmoothArcCenterCmd

    __slots__ = ("_end_pose", "_center")

    def __init__(self) -> None:
        super().__init__(description="arc (center)")
        self._end_pose: list[float] | None = None
        self._center: list[float] | None = None

    def do_setup(self, state: "ControllerState") -> None:
        """Transform parameters if in TRF."""
        assert self.p is not None

        transformed = self._transform_params(
            "SMOOTH_ARC_CENTER",
            {"end_pose": list(self.p.end_pose), "center": list(self.p.center)},
        )
        self._end_pose = transformed["end_pose"]
        self._center = transformed["center"]
        self.normal_vector = transformed.get("normal_vector")
        return super().do_setup(state)

    def generate_main_trajectory(self, effective_start_pose):
        """Generate arc geometry from actual start to end."""
        assert self.p is not None
        assert self._end_pose is not None
        assert self._center is not None

        motion_gen = CircularMotion()

        # Use duration for geometry sampling, default to 5s if not specified
        geom_duration = (
            self.p.duration
            if self.p.duration is not None and self.p.duration > 0.0
            else 5.0
        )
        return motion_gen.generate_arc(
            start_pose=effective_start_pose,
            end_pose=self._end_pose,
            center=self._center,
            normal=self.normal_vector,
            clockwise=self.p.clockwise,
            duration=geom_duration,
        )


@register_command(CmdType.SMOOTH_ARC_PARAM)
class SmoothArcParamCommand(BaseSmoothMotionCommand):
    """Execute smooth arc motion defined by radius and angle."""

    PARAMS_TYPE = SmoothArcParamCmd

    __slots__ = ("_end_pose",)

    def __init__(self) -> None:
        super().__init__(description="arc (param)")
        self._end_pose: list[float] | None = None

    def do_setup(self, state: "ControllerState") -> None:
        """Transform parameters if in TRF."""
        assert self.p is not None

        self.description = f"arc (r={self.p.radius}mm)"

        transformed = self._transform_params(
            "SMOOTH_ARC_PARAM", {"end_pose": list(self.p.end_pose), "plane": "XY"}
        )
        self._end_pose = transformed["end_pose"]
        self.normal_vector = transformed.get("normal_vector")
        return super().do_setup(state)

    def generate_main_trajectory(self, effective_start_pose):
        """Generate arc based on radius and angle from actual start."""
        assert self.p is not None
        assert self._end_pose is not None

        # Use duration for geometry sampling, default to 5s if not specified
        geom_duration = (
            self.p.duration
            if self.p.duration is not None and self.p.duration > 0.0
            else 5.0
        )
        return CircularMotion().generate_arc_from_endpoints(
            start_pose=effective_start_pose,
            end_pose=self._end_pose,
            radius=self.p.radius,
            normal=self.normal_vector,
            clockwise=self.p.clockwise,
            duration=geom_duration,
        )


@register_command(CmdType.SMOOTH_SPLINE)
class SmoothSplineCommand(BaseSmoothMotionCommand):
    """Execute smooth spline motion through waypoints."""

    PARAMS_TYPE = SmoothSplineCmd

    __slots__ = ("_waypoints",)

    def __init__(self) -> None:
        super().__init__(description="spline")
        self._waypoints: list[list[float]] | None = None

    def do_setup(self, state: "ControllerState") -> None:
        """Transform parameters if in TRF."""
        assert self.p is not None

        self.description = f"spline ({len(self.p.waypoints)} points, {self.p.frame})"

        transformed = self._transform_params(
            "SMOOTH_SPLINE", {"waypoints": [list(wp) for wp in self.p.waypoints]}
        )
        self._waypoints = transformed["waypoints"]
        return super().do_setup(state)

    def generate_main_trajectory(self, effective_start_pose):
        """Generate spline starting from actual position."""
        assert self.p is not None
        assert self._waypoints is not None

        wps = self._waypoints
        motion_gen = SplineMotion()

        # Always start from the effective start pose
        first_wp_error = np.linalg.norm(
            np.array(wps[0][:3]) - np.array(effective_start_pose[:3])
        )

        if first_wp_error > 5.0:
            # First waypoint is far, prepend the start position
            modified_waypoints = [effective_start_pose] + wps
            logger.info(
                f"    Added start position as first waypoint (distance: {first_wp_error:.1f}mm)"
            )
        else:
            # Replace first waypoint with actual start to ensure continuity
            modified_waypoints = [effective_start_pose] + wps[1:]
            logger.info("    Replaced first waypoint with actual start position")

        # Use duration for geometry sampling, None lets SplineMotion estimate from path length
        duration = (
            self.p.duration
            if self.p.duration is not None and self.p.duration > 0.0
            else None
        )
        trajectory = motion_gen.generate_spline(
            waypoints=modified_waypoints,
            duration=duration,
        )

        logger.debug(f"    Generated spline with {len(trajectory)} points")

        return trajectory
