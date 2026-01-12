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
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from parol6.commands.base import TrajectoryMoveCommandBase, parse_motion_params
from parol6.config import CONTROL_RATE_HZ, INTERVAL_S, LIMITS, steps_to_rad
from parol6.motion import JointPath, TrajectoryBuilder
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
# Shape Generators
# =============================================================================


class _ShapeGenerator:
    """Base class for geometry generation (circles, arcs, splines)."""

    def __init__(self, control_rate: float | None = None):
        self.control_rate = (
            control_rate if control_rate is not None else CONTROL_RATE_HZ
        )

    def _get_perpendicular_vector(self, v: np.ndarray) -> np.ndarray:
        """Find a vector perpendicular to the given vector."""
        if abs(v[0]) < 0.9:
            cross = np.cross(v, [1, 0, 0])
        else:
            cross = np.cross(v, [0, 1, 0])
        return cross / np.linalg.norm(cross)


class CircularMotion(_ShapeGenerator):
    """Generate circular and arc trajectories in 3D space."""

    def generate_arc(
        self,
        start_pose: Sequence[float],
        end_pose: Sequence[float],
        center: Sequence[float] | NDArray,
        normal: Sequence[float] | NDArray | None = None,
        clockwise: bool = False,
        duration: float = 2.0,
    ) -> np.ndarray:
        """Generate a 3D circular arc trajectory (uniformly sampled geometry)."""
        return self._generate_arc_geometry(
            start_pose, end_pose, center, normal, clockwise, duration
        )

    def generate_arc_from_endpoints(
        self,
        start_pose: Sequence[float],
        end_pose: Sequence[float],
        radius: float,
        normal: Sequence[float] | NDArray | None = None,
        clockwise: bool = False,
        duration: float = 2.0,
    ) -> np.ndarray:
        """Generate arc by calculating center from endpoints and radius."""
        start_xyz = np.array(start_pose[:3])
        end_xyz = np.array(end_pose[:3])
        chord_vec = end_xyz - start_xyz
        chord_length = float(np.linalg.norm(chord_vec))

        # Adjust radius if points are too far apart
        if chord_length > 2 * radius:
            logger.warning(
                "Points too far apart (%.1fmm) for radius %.1fmm, adjusting",
                chord_length,
                radius,
            )
            radius = chord_length / 2 + 1

        # Calculate arc center
        chord_mid = (start_xyz + end_xyz) / 2
        h = float(np.sqrt(max(0.0, radius**2 - (chord_length / 2) ** 2)))

        if normal is not None:
            # 3D arc with specified normal
            normal_np = np.array(normal, dtype=float)
            normal_np = normal_np / np.linalg.norm(normal_np)
            if chord_length > 0:
                chord_dir = chord_vec / chord_length
                perp = np.cross(normal_np, chord_dir)
                if np.linalg.norm(perp) > 0.001:
                    perp = perp / np.linalg.norm(perp)
                else:
                    perp = np.array([1, 0, 0])
            else:
                perp = np.array([1, 0, 0])
            center = chord_mid + ((-h if clockwise else h) * perp)
        else:
            # XY plane arc
            normal_np = np.array([0, 0, 1])
            if chord_length > 0:
                perp_2d = np.array(
                    [-(end_xyz[1] - start_xyz[1]), end_xyz[0] - start_xyz[0]]
                )
                perp_2d = perp_2d / np.linalg.norm(perp_2d)
                center_2d = chord_mid[:2] + ((-h if clockwise else h) * perp_2d)
                center = np.array(
                    [center_2d[0], center_2d[1], (start_xyz[2] + end_xyz[2]) / 2]
                )
            else:
                center = start_xyz.copy()

        return self.generate_arc(
            start_pose,
            end_pose,
            center,
            normal=normal_np if normal is not None else None,
            clockwise=clockwise,
            duration=duration,
        )

    def generate_circle(
        self,
        center: Sequence[float] | NDArray,
        radius: float,
        normal: Sequence[float] | NDArray = (0, 0, 1),
        duration: float = 4.0,
        start_point: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Generate a complete circle trajectory (uniformly sampled geometry)."""
        return self._generate_circle_geometry(
            center, radius, normal, duration, start_point
        )

    def _generate_arc_geometry(
        self,
        start_pose: Sequence[float],
        end_pose: Sequence[float],
        center: Sequence[float] | NDArray,
        normal: Sequence[float] | NDArray | None,
        clockwise: bool,
        duration: float,
    ) -> np.ndarray:
        """Generate uniformly-spaced arc geometry."""
        start_pos = np.array(start_pose[:3])
        end_pos = np.array(end_pose[:3])
        center_pt = np.array(center)

        r1 = start_pos - center_pt
        r2 = end_pos - center_pt

        if normal is None:
            normal = np.cross(r1, r2)
            if np.linalg.norm(normal) < 1e-6:
                normal = np.array([0, 0, 1])
        normal_np = np.array(normal, dtype=float)
        normal_np = normal_np / np.linalg.norm(normal_np)

        r1_norm = r1 / np.linalg.norm(r1)
        r2_norm = r2 / np.linalg.norm(r2)
        cos_angle = np.clip(np.dot(r1_norm, r2_norm), -1, 1)
        arc_angle = np.arccos(cos_angle)

        cross = np.cross(r1_norm, r2_norm)
        if np.dot(cross, normal_np) < 0:
            arc_angle = 2 * np.pi - arc_angle
        if clockwise:
            arc_angle = -arc_angle

        num_points = max(2, int(duration * self.control_rate))

        # Vectorized arc generation using scipy Rotation
        t_values = np.linspace(0, 1, num_points) if num_points > 1 else np.array([1.0])
        angles = t_values * arc_angle

        # Batch rotation using rotvec (axis-angle)
        rotvecs = np.outer(angles, normal_np)  # (num_points, 3)
        rotations = Rotation.from_rotvec(rotvecs)
        positions = center_pt + rotations.apply(r1)  # (num_points, 3)

        # Batch orientation interpolation (slerp)
        start_orient = np.array(start_pose[3:])
        end_orient = np.array(end_pose[3:])
        r_start = Rotation.from_euler("xyz", start_orient, degrees=True)
        r_end = Rotation.from_euler("xyz", end_orient, degrees=True)
        key_rots = Rotation.from_quat(np.stack([r_start.as_quat(), r_end.as_quat()]))
        slerp = Slerp(np.array([0.0, 1.0]), key_rots)
        orientations = slerp(t_values).as_euler("xyz", degrees=True)  # (num_points, 3)

        # Combine positions and orientations
        trajectory = np.concatenate([positions, orientations], axis=1)

        return trajectory

    def _generate_circle_geometry(
        self,
        center: Sequence[float] | NDArray,
        radius: float,
        normal: Sequence[float] | NDArray,
        duration: float,
        start_point: Sequence[float] | None,
    ) -> np.ndarray:
        """Generate uniformly-spaced circle geometry."""
        normal_np = np.array(normal, dtype=float)
        normal_np = normal_np / np.linalg.norm(normal_np)
        u = self._get_perpendicular_vector(normal_np)
        v = np.cross(normal_np, u)
        center_np = np.array(center, dtype=float)

        start_angle = 0.0
        if start_point is not None:
            start_pos = np.array(start_point[:3])
            to_start = start_pos - center_np
            to_start_plane = to_start - np.dot(to_start, normal_np) * normal_np
            dist_in_plane = np.linalg.norm(to_start_plane)

            if dist_in_plane > 0.001:
                to_start_normalized = to_start_plane / dist_in_plane
                u_comp = np.dot(to_start_normalized, u)
                v_comp = np.dot(to_start_normalized, v)
                start_angle = np.arctan2(v_comp, u_comp)

        num_points = max(2, int(duration * self.control_rate))

        # Vectorized circle generation
        if num_points > 1:
            angles = start_angle + np.linspace(0, 2 * np.pi, num_points)
        else:
            angles = np.array([start_angle])
        cos_a = np.cos(angles).reshape(-1, 1)
        sin_a = np.sin(angles).reshape(-1, 1)
        positions = center_np + radius * (cos_a * u + sin_a * v)

        # Add zero orientations
        trajectory = np.zeros((num_points, 6), dtype=np.float64)
        trajectory[:, :3] = positions

        return trajectory

    def _rotation_matrix_from_axis_angle(
        self, axis: np.ndarray, angle: float
    ) -> np.ndarray:
        """Generate rotation matrix using Rodrigues' formula."""
        axis = axis / np.linalg.norm(axis)
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

    def _slerp_orientation(
        self, start_orient: NDArray, end_orient: NDArray, t: float
    ) -> np.ndarray:
        """Spherical linear interpolation for orientation."""
        r1 = Rotation.from_euler("xyz", start_orient, degrees=True)
        r2 = Rotation.from_euler("xyz", end_orient, degrees=True)
        key_rots = Rotation.from_quat(np.stack([r1.as_quat(), r2.as_quat()]))
        slerp = Slerp(np.array([0.0, 1.0]), key_rots)
        interp_rot = slerp(np.array([t]))
        return interp_rot.as_euler("xyz", degrees=True)[0]


class SplineMotion(_ShapeGenerator):
    """Generate smooth spline trajectories through waypoints."""

    def generate_spline(
        self,
        waypoints: Sequence[Sequence[float]],
        timestamps: Sequence[float] | None = None,
        duration: float | None = None,
        velocity_start: Sequence[float] | None = None,
        velocity_end: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Generate spline trajectory (uniformly sampled geometry)."""
        return self._generate_spline_geometry(
            waypoints, timestamps, duration, velocity_start, velocity_end
        )

    def _generate_spline_geometry(
        self,
        waypoints: Sequence[Sequence[float]],
        timestamps: Sequence[float] | None,
        duration: float | None,
        velocity_start: Sequence[float] | None,
        velocity_end: Sequence[float] | None,
    ) -> np.ndarray:
        """Generate uniformly-spaced spline geometry using cubic interpolation."""
        waypoints_arr = np.asarray(waypoints, dtype=float)
        num_waypoints = len(waypoints_arr)

        if num_waypoints < 2:
            return waypoints_arr

        if timestamps is None:
            total_dist = 0.0
            for i in range(1, num_waypoints):
                dist = np.linalg.norm(waypoints_arr[i, :3] - waypoints_arr[i - 1, :3])
                total_dist += float(dist)

            if duration is not None:
                total_time = duration
            else:
                total_time = max(0.1, total_dist / 50.0)

            timestamps_arr = np.linspace(0, total_time, num_waypoints)
        else:
            timestamps_arr = np.asarray(timestamps, dtype=float)
            if duration is not None:
                scale = duration / timestamps_arr[-1] if timestamps_arr[-1] > 0 else 1.0
                timestamps_arr = timestamps_arr * scale

        if len(timestamps_arr) != len(waypoints_arr):
            raise ValueError(
                f"Timestamps length ({len(timestamps_arr)}) must match "
                f"waypoints length ({len(waypoints_arr)})"
            )

        pos_splines = []
        for i in range(3):
            bc: Any
            if velocity_start is not None and velocity_end is not None:
                bc = ((1, float(velocity_start[i])), (1, float(velocity_end[i])))
            else:
                bc = "not-a-knot"
            spline = CubicSpline(timestamps_arr, waypoints_arr[:, i], bc_type=bc)
            pos_splines.append(spline)

        # Batch convert euler angles to rotations (vectorized)
        euler_angles = waypoints_arr[:, 3:]
        key_rots = Rotation.from_euler("xyz", euler_angles, degrees=True)
        slerp = Slerp(timestamps_arr, key_rots)

        total_time = float(timestamps_arr[-1])
        num_points = max(2, int(total_time * self.control_rate))
        t_eval = np.linspace(0, total_time, num_points)

        trajectory: list[np.ndarray] = []
        for t in t_eval:
            pos = [float(spline(float(t))) for spline in pos_splines]
            rot = slerp(np.array([float(t)]))
            orient = rot.as_euler("xyz", degrees=True)[0]
            trajectory.append(np.concatenate([pos, orient]))

        return np.array(trajectory)


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
