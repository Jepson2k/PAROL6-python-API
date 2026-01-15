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

from parol6.commands.base import TrajectoryMoveCommandBase, parse_motion_params
from parol6.config import INTERVAL_S, LIMITS, steps_to_rad
from parol6.motion import CircularMotion, JointPath, SplineMotion, TrajectoryBuilder
from parol6.server.command_registry import register_command
from parol6.server.state import get_fkine_se3
from parol6.utils.errors import IKError
from parol6.utils.se3_utils import se3_from_rpy, se3_from_trans, se3_rpy

if TYPE_CHECKING:
    import sophuspy as sp
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


def _pose6_trf_to_wrf(
    pose6_mm_deg: Sequence[float], tool_pose: "sp.SE3"
) -> list[float]:
    """Convert 6D pose [x,y,z,rx,ry,rz] from TRF to WRF (mm, degrees)."""
    pose_trf = se3_from_rpy(
        pose6_mm_deg[0] / 1000.0,
        pose6_mm_deg[1] / 1000.0,
        pose6_mm_deg[2] / 1000.0,
        pose6_mm_deg[3],
        pose6_mm_deg[4],
        pose6_mm_deg[5],
        degrees=True,
    )
    pose_wrf = tool_pose * pose_trf
    return np.concatenate(
        [pose_wrf.translation() * 1000.0, se3_rpy(pose_wrf, degrees=True)]
    ).tolist()


def _transform_center_trf_to_wrf(
    params: dict[str, Any], tool_pose: "sp.SE3", transformed: dict[str, Any]
) -> None:
    """Transform 'center' parameter from TRF (mm) to WRF (mm)."""
    center_trf = se3_from_trans(
        params["center"][0] / 1000.0,
        params["center"][1] / 1000.0,
        params["center"][2] / 1000.0,
    )
    center_wrf = tool_pose * center_trf
    transformed["center"] = (center_wrf.translation() * 1000.0).tolist()


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
            normal_wrf = tool_pose.rotationMatrix() @ normal_trf
            transformed["normal_vector"] = normal_wrf.tolist()

    elif command_type == "SMOOTH_ARC_CENTER":
        if "center" in params:
            _transform_center_trf_to_wrf(params, tool_pose, transformed)
        if "end_pose" in params:
            transformed["end_pose"] = _pose6_trf_to_wrf(params["end_pose"], tool_pose)
        if "plane" in params:
            normal_trf = _PLANE_NORMALS_TRF[params["plane"]]
            normal_wrf = tool_pose.rotationMatrix() @ normal_trf
            transformed["normal_vector"] = normal_wrf.tolist()

    elif command_type == "SMOOTH_ARC_PARAM":
        if "end_pose" in params:
            transformed["end_pose"] = _pose6_trf_to_wrf(params["end_pose"], tool_pose)
        if "plane" not in params:
            params["plane"] = "XY"
        normal_trf = _PLANE_NORMALS_TRF[params.get("plane", "XY")]
        normal_wrf = tool_pose.rotationMatrix() @ normal_trf
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
        "frame",
        "normal_vector",
        "duration",
        "velocity_percent",
        "accel_percent",
    )

    VALID_FRAMES = frozenset(("WRF", "TRF"))
    CLOCKWISE_VALUES = frozenset(("CW", "CLOCKWISE", "TRUE"))

    def __init__(self, description: str = "smooth geometry") -> None:
        super().__init__()
        self.description = description
        self.frame: str = "WRF"
        self.normal_vector: list[float] | None = None
        self.duration: float | None = None
        self.velocity_percent: float | None = None
        self.accel_percent: float = 100.0

    def _transform_params(
        self, command_type: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Transform params from TRF to WRF if needed. No-op for WRF frame."""
        return _transform_command_params_to_wrf(command_type, params, self.frame)

    @staticmethod
    def _parse_frame(frame_str: str) -> tuple[str | None, str | None]:
        """Parse and validate frame. Returns (frame, error)."""
        frame = frame_str.upper()
        if frame not in BaseSmoothMotionCommand.VALID_FRAMES:
            return None, f"Invalid frame: {frame_str}"
        return frame, None

    @staticmethod
    def _is_clockwise(value: str) -> bool:
        """Check if value indicates clockwise direction."""
        return value.upper() in BaseSmoothMotionCommand.CLOCKWISE_VALUES

    def _parse_motion_params(self, parts: list[str], start_idx: int) -> None:
        """Parse duration|velocity_percent|accel_percent from parts[start_idx:]."""
        params = parse_motion_params(parts, start_idx)
        self.duration = params.duration
        self.velocity_percent = params.velocity_percent
        self.accel_percent = params.accel_percent

    def get_current_pose(self, state: "ControllerState") -> np.ndarray:
        """Get current TCP pose as [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]."""
        current_se3 = get_fkine_se3(state)
        current_xyz = current_se3.translation() * 1000  # m -> mm
        current_rpy = se3_rpy(current_se3, degrees=True)
        return np.concatenate([current_xyz, current_rpy])

    def do_setup(self, state: "ControllerState") -> None:
        """Pre-compute trajectory from current position."""
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

        # Scale limits by velocity/accel percent (default 100% if not specified)
        vel_pct = self.velocity_percent if self.velocity_percent is not None else 100.0
        acc_pct = self.accel_percent if self.accel_percent is not None else 100.0
        cart_vel_max = LIMITS.cart.hard.velocity.linear * (vel_pct / 100.0)
        cart_acc_max = LIMITS.cart.hard.acceleration.linear * (acc_pct / 100.0)

        builder = TrajectoryBuilder(
            joint_path=joint_path,
            profile=state.motion_profile,
            velocity_percent=vel_pct,
            accel_percent=acc_pct,
            duration=self.duration,
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


@register_command("SMOOTH_CIRCLE")
class SmoothCircleCommand(BaseSmoothMotionCommand):
    """Execute smooth circular motion."""

    __slots__ = (
        "center",
        "radius",
        "plane",
        "clockwise",
        "center_mode",
    )

    def __init__(self) -> None:
        super().__init__(description="circle")
        self.center: NDArray[np.floating] | None = None
        self.radius: float = 100.0
        self.plane: str = "XY"
        self.clockwise: bool = False
        self.center_mode: str = "ABSOLUTE"

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
        """Parse SMOOTH_CIRCLE: center|radius|plane|frame|duration|velocity|accel|[cw]|[center_mode]"""
        if parts[0].upper() != "SMOOTH_CIRCLE":
            return False, None
        if len(parts) < 5:
            return False, "SMOOTH_CIRCLE requires at least 4 parameters"

        try:
            center_list = list(map(float, parts[1].split(",")))
            if len(center_list) != 3:
                return False, "Center must have 3 coordinates"
            self.center = np.asarray(center_list, dtype=np.float64)
            self.radius = float(parts[2])

            self.plane = parts[3].upper()
            if self.plane not in ("XY", "XZ", "YZ"):
                return False, f"Invalid plane: {self.plane}"

            frame, err = self._parse_frame(parts[4])
            if err:
                return False, err
            assert frame is not None
            self.frame = frame

            # Parse duration|velocity|accel (all optional)
            self._parse_motion_params(parts, 5)

            # Check for optional cw and center_mode after motion params
            idx = 8  # After duration|velocity|accel
            if idx < len(parts) and self._is_clockwise(parts[idx]):
                self.clockwise = True
                idx += 1
            if idx < len(parts) and parts[idx].upper() in (
                "ABSOLUTE",
                "TOOL",
                "RELATIVE",
            ):
                self.center_mode = parts[idx].upper()

            self.description = f"circle (r={self.radius}mm)"
            return True, None
        except (ValueError, IndexError) as e:
            return False, f"Invalid SMOOTH_CIRCLE parameters: {e}"

    def do_setup(self, state: "ControllerState") -> None:
        """Transform parameters if in TRF, then prepare trajectory."""
        transformed = self._transform_params(
            "SMOOTH_CIRCLE", {"center": self.center, "plane": self.plane}
        )
        self.center = transformed["center"]
        self.normal_vector = transformed.get("normal_vector")
        return super().do_setup(state)

    def generate_main_trajectory(self, effective_start_pose):
        """Generate circle geometry starting from actual position."""
        motion_gen = CircularMotion()

        # Determine normal vector
        if self.normal_vector is not None:
            normal = np.array(self.normal_vector)
        else:
            plane_normals = {"XY": [0, 0, 1], "XZ": [0, 1, 0], "YZ": [1, 0, 0]}
            normal = np.array(plane_normals.get(self.plane, [0, 0, 1]))

        # Handle center_mode
        if self.center_mode == "TOOL":
            actual_center = np.array(effective_start_pose[:3])
        elif self.center_mode == "RELATIVE":
            center_np = (
                np.asarray(self.center, dtype=float)
                if self.center is not None
                else np.zeros(3)
            )
            actual_center = np.array(effective_start_pose[:3]) + center_np
        else:
            actual_center = (
                np.array(self.center) if self.center is not None else np.zeros(3)
            )

        # Use duration for geometry sampling, default to 4s for circle if not specified
        geom_duration = self.duration if self.duration is not None else 4.0
        trajectory = motion_gen.generate_circle(
            center=actual_center,
            radius=self.radius,
            normal=normal,
            duration=geom_duration,
            start_point=effective_start_pose,
        )

        if self.clockwise:
            trajectory = trajectory[::-1]

        # Update orientations to match start pose
        for i in range(len(trajectory)):
            trajectory[i][3:] = effective_start_pose[3:]

        return trajectory


@register_command("SMOOTH_ARC_CENTER")
class SmoothArcCenterCommand(BaseSmoothMotionCommand):
    """Execute smooth arc motion defined by center point."""

    __slots__ = (
        "end_pose",
        "center",
        "clockwise",
    )

    def __init__(self) -> None:
        super().__init__(description="arc")
        self.end_pose: list[float] | None = None
        self.center: list[float] | None = None
        self.clockwise: bool = False

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
        """Parse SMOOTH_ARC_CENTER: end_pose|center|frame|duration|velocity|accel|[cw]"""
        if parts[0].upper() != "SMOOTH_ARC_CENTER":
            return False, None
        if len(parts) < 4:
            return False, "SMOOTH_ARC_CENTER requires at least 3 parameters"

        try:
            self.end_pose = list(map(float, parts[1].split(",")))
            if len(self.end_pose) != 6:
                return False, "End pose must have 6 values (x,y,z,rx,ry,rz)"

            self.center = list(map(float, parts[2].split(",")))
            if len(self.center) != 3:
                return False, "Center must have 3 coordinates"

            frame, err = self._parse_frame(parts[3])
            if err:
                return False, err
            assert frame is not None
            self.frame = frame

            # Parse duration|velocity|accel (all optional)
            self._parse_motion_params(parts, 4)

            # Check for optional cw after motion params
            idx = 7  # After duration|velocity|accel
            if idx < len(parts) and self._is_clockwise(parts[idx]):
                self.clockwise = True

            self.description = "arc (center)"
            return True, None
        except (ValueError, IndexError) as e:
            return False, f"Invalid SMOOTH_ARC_CENTER parameters: {e}"

    def do_setup(self, state: "ControllerState") -> None:
        """Transform parameters if in TRF."""
        transformed = self._transform_params(
            "SMOOTH_ARC_CENTER", {"end_pose": self.end_pose, "center": self.center}
        )
        self.end_pose = transformed["end_pose"]
        self.center = transformed["center"]
        self.normal_vector = transformed.get("normal_vector")
        return super().do_setup(state)

    def generate_main_trajectory(self, effective_start_pose):
        """Generate arc geometry from actual start to end."""
        motion_gen = CircularMotion()

        assert self.end_pose is not None
        assert self.center is not None

        # Use duration for geometry sampling, default to 5s if not specified
        geom_duration = self.duration if self.duration is not None else 5.0
        return motion_gen.generate_arc(
            start_pose=effective_start_pose,
            end_pose=self.end_pose,
            center=self.center,
            normal=self.normal_vector,
            clockwise=self.clockwise,
            duration=geom_duration,
        )


@register_command("SMOOTH_ARC_PARAM")
class SmoothArcParamCommand(BaseSmoothMotionCommand):
    """Execute smooth arc motion defined by radius and angle."""

    __slots__ = (
        "end_pose",
        "radius",
        "arc_angle",
        "clockwise",
    )

    def __init__(self) -> None:
        super().__init__(description="arc (param)")
        self.end_pose: list[float] | None = None
        self.radius: float = 100.0
        self.arc_angle: float = 90.0
        self.clockwise: bool = False

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
        """Parse SMOOTH_ARC_PARAM: end_pose|radius|arc_angle|frame|duration|velocity|accel|[cw]"""
        if parts[0].upper() != "SMOOTH_ARC_PARAM":
            return False, None
        if len(parts) < 5:
            return False, "SMOOTH_ARC_PARAM requires at least 4 parameters"

        try:
            self.end_pose = list(map(float, parts[1].split(",")))
            if len(self.end_pose) != 6:
                return False, "End pose must have 6 values (x,y,z,rx,ry,rz)"

            self.radius = float(parts[2])
            self.arc_angle = float(parts[3])

            frame, err = self._parse_frame(parts[4])
            if err:
                return False, err
            assert frame is not None
            self.frame = frame

            # Parse duration|velocity|accel (all optional)
            self._parse_motion_params(parts, 5)

            # Check for optional cw after motion params
            idx = 8  # After duration|velocity|accel
            if idx < len(parts) and self._is_clockwise(parts[idx]):
                self.clockwise = True

            self.description = f"arc (r={self.radius}mm)"
            return True, None
        except (ValueError, IndexError) as e:
            return False, f"Invalid SMOOTH_ARC_PARAM parameters: {e}"

    def do_setup(self, state: "ControllerState") -> None:
        """Transform parameters if in TRF."""
        transformed = self._transform_params(
            "SMOOTH_ARC_PARAM", {"end_pose": self.end_pose, "plane": "XY"}
        )
        self.end_pose = transformed["end_pose"]
        self.normal_vector = transformed.get("normal_vector")
        return super().do_setup(state)

    def generate_main_trajectory(self, effective_start_pose):
        """Generate arc based on radius and angle from actual start."""
        assert self.end_pose is not None
        # Use duration for geometry sampling, default to 5s if not specified
        geom_duration = self.duration if self.duration is not None else 5.0
        return CircularMotion().generate_arc_from_endpoints(
            start_pose=effective_start_pose,
            end_pose=self.end_pose,
            radius=self.radius,
            normal=self.normal_vector,
            clockwise=self.clockwise,
            duration=geom_duration,
        )


@register_command("SMOOTH_SPLINE")
class SmoothSplineCommand(BaseSmoothMotionCommand):
    """Execute smooth spline motion through waypoints."""

    __slots__ = ("waypoints",)

    def __init__(self) -> None:
        super().__init__(description="spline")
        self.waypoints: list[list[float]] | None = None

    def do_match(self, parts: list[str]) -> tuple[bool, str | None]:
        """Parse SMOOTH_SPLINE: waypoints|frame|duration|velocity|accel (two formats supported)"""
        if parts[0].upper() != "SMOOTH_SPLINE":
            return False, None
        if len(parts) < 3:
            return False, "SMOOTH_SPLINE requires at least 2 parameters"

        try:
            # Alt format: SMOOTH_SPLINE|<count>|<frame>|duration|velocity|accel|<flattened waypoints...>
            if len(parts) >= 4 and parts[1].isdigit():
                return self._parse_flattened_format(parts)
            return self._parse_semicolon_format(parts)
        except (ValueError, IndexError) as e:
            return False, f"Invalid SMOOTH_SPLINE parameters: {e}"

    def _parse_flattened_format(self, parts: list[str]) -> tuple[bool, str | None]:
        """Parse: count|frame|duration|velocity|accel|flattened_waypoints..."""
        num = int(parts[1])
        frame, err = self._parse_frame(parts[2])
        if err:
            return False, err
        assert frame is not None
        self.frame = frame

        # Parse duration|velocity|accel
        self._parse_motion_params(parts, 3)

        # Waypoints start after motion params
        idx = 6
        needed = num * 6
        if len(parts) - idx < needed:
            return False, "Insufficient waypoint values"
        vals = list(map(float, parts[idx : idx + needed]))
        self.waypoints = [vals[i : i + 6] for i in range(0, needed, 6)]

        self.description = f"spline ({len(self.waypoints)} points, {self.frame})"
        return True, None

    def _parse_semicolon_format(self, parts: list[str]) -> tuple[bool, str | None]:
        """Parse: wp1;wp2;...|frame|duration|velocity|accel"""
        waypoint_strs = parts[1].split(";")
        self.waypoints = []
        for wp_str in waypoint_strs:
            wp = list(map(float, wp_str.split(",")))
            if len(wp) != 6:
                return False, "Each waypoint must have 6 values (x,y,z,rx,ry,rz)"
            self.waypoints.append(wp)

        if len(self.waypoints) < 2:
            return False, "SMOOTH_SPLINE requires at least 2 waypoints"

        frame, err = self._parse_frame(parts[2])
        if err:
            return False, err
        assert frame is not None
        self.frame = frame

        # Parse duration|velocity|accel
        self._parse_motion_params(parts, 3)

        self.description = f"spline ({len(self.waypoints)} points, {self.frame})"
        return True, None

    def _calc_path_length(self) -> float:
        """Estimate path length from waypoints."""
        if not self.waypoints:
            return 0.0
        length = 0.0
        for i in range(1, len(self.waypoints)):
            length += float(
                np.linalg.norm(
                    np.array(self.waypoints[i][:3])
                    - np.array(self.waypoints[i - 1][:3])
                )
            )
        return length

    def do_setup(self, state: "ControllerState") -> None:
        """Transform parameters if in TRF."""
        transformed = self._transform_params(
            "SMOOTH_SPLINE", {"waypoints": self.waypoints}
        )
        self.waypoints = transformed["waypoints"]
        return super().do_setup(state)

    def generate_main_trajectory(self, effective_start_pose):
        """Generate spline starting from actual position."""
        assert self.waypoints is not None
        wps = self.waypoints
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
        trajectory = motion_gen.generate_spline(
            waypoints=modified_waypoints,
            duration=self.duration,
        )

        logger.debug(f"    Generated spline with {len(trajectory)} points")

        return trajectory
