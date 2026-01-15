"""
Geometry generation for smooth motion paths.

This module provides pure geometry generators for circles, arcs, and splines.
These are used by both the controller (smooth_commands.py) and the GUI
(DryRunRobotClient) for path preview and visualization.

All generators are stateless - they produce Cartesian path geometry without
depending on controller state or executing any motion.
"""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

if TYPE_CHECKING:
    from roboticstoolbox import DHRobot

from parol6.config import CONTROL_RATE_HZ

logger = logging.getLogger(__name__)

# Default control rate for geometry sampling
DEFAULT_CONTROL_RATE = CONTROL_RATE_HZ


class _ShapeGenerator:
    """Base class for geometry generation (circles, arcs, splines)."""

    def __init__(self, control_rate: float | None = None):
        self.control_rate = (
            control_rate if control_rate is not None else DEFAULT_CONTROL_RATE
        )

    def _get_perpendicular_vector(self, v: np.ndarray) -> np.ndarray:
        """Find a vector perpendicular to the given vector."""
        if abs(v[0]) < 0.9:
            cross = np.cross(v, [1, 0, 0])
        else:
            cross = np.cross(v, [0, 1, 0])
        return cross / np.linalg.norm(cross)


class CircularMotion(_ShapeGenerator):
    """Generate circular and arc trajectories in 3D space.

    All methods return (N, 6) arrays of [x, y, z, rx, ry, rz] poses.
    Position units match input units (typically mm).
    Orientation is in degrees.
    """

    def generate_arc(
        self,
        start_pose: Sequence[float],
        end_pose: Sequence[float],
        center: Sequence[float] | NDArray,
        normal: Sequence[float] | NDArray | None = None,
        clockwise: bool = False,
        duration: float = 2.0,
    ) -> np.ndarray:
        """Generate a 3D circular arc trajectory (uniformly sampled geometry).

        Args:
            start_pose: Start pose [x, y, z, rx, ry, rz]
            end_pose: End pose [x, y, z, rx, ry, rz]
            center: Arc center point [x, y, z]
            normal: Normal vector defining arc plane (auto-computed if None)
            clockwise: If True, arc goes clockwise when viewed from normal
            duration: Affects number of sample points (duration * control_rate)

        Returns:
            (N, 6) array of poses along the arc
        """
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
        """Generate arc by calculating center from endpoints and radius.

        Args:
            start_pose: Start pose [x, y, z, rx, ry, rz]
            end_pose: End pose [x, y, z, rx, ry, rz]
            radius: Arc radius (same units as position)
            normal: Normal vector defining arc plane (XY plane if None)
            clockwise: If True, arc goes clockwise when viewed from normal
            duration: Affects number of sample points

        Returns:
            (N, 6) array of poses along the arc
        """
        start_xyz = np.array(start_pose[:3])
        end_xyz = np.array(end_pose[:3])
        chord_vec = end_xyz - start_xyz
        chord_length = float(np.linalg.norm(chord_vec))

        # Adjust radius if points are too far apart
        if chord_length > 2 * radius:
            logger.warning(
                "Points too far apart (%.1f) for radius %.1f, adjusting",
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
        """Generate a complete circle trajectory (uniformly sampled geometry).

        Args:
            center: Circle center [x, y, z]
            radius: Circle radius (same units as center)
            normal: Normal vector defining circle plane
            duration: Affects number of sample points
            start_point: If provided, circle starts at nearest point to this

        Returns:
            (N, 6) array of poses around the circle
        """
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
    """Generate smooth spline trajectories through waypoints.

    Uses cubic spline interpolation for position and SLERP for orientation.
    """

    def generate_spline(
        self,
        waypoints: Sequence[Sequence[float]],
        timestamps: Sequence[float] | None = None,
        duration: float | None = None,
        velocity_start: Sequence[float] | None = None,
        velocity_end: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Generate spline trajectory (uniformly sampled geometry).

        Args:
            waypoints: List of [x, y, z, rx, ry, rz] waypoints
            timestamps: Optional timestamps for each waypoint
            duration: Total duration (overrides timestamps scaling)
            velocity_start: Start velocity for position [vx, vy, vz]
            velocity_end: End velocity for position [vx, vy, vz]

        Returns:
            (N, 6) array of poses along the spline
        """
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


def joint_path_to_tcp_poses(
    joint_positions: NDArray[np.float64],
    robot: "DHRobot | None" = None,
) -> NDArray[np.float64]:
    """Convert joint-space path to TCP poses using forward kinematics.

    This is useful for visualizing the actual TCP trajectory that results
    from joint-space interpolation (which traces an arc, not a straight line).

    Args:
        joint_positions: (N, 6) array of joint angles in radians
        robot: roboticstoolbox robot model (uses PAROL6_ROBOT.robot if None)

    Returns:
        (N, 6) array of [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg] poses
    """
    from scipy.spatial.transform import Rotation

    if robot is None:
        import parol6.PAROL6_ROBOT as PAROL6_ROBOT

        robot = PAROL6_ROBOT.robot

    if robot is None:
        raise ValueError("Robot model not available")

    n_points = len(joint_positions)
    tcp_poses = np.empty((n_points, 6), dtype=np.float64)

    for i, q in enumerate(joint_positions):
        # Forward kinematics returns SE3 (spatialmath)
        T = robot.fkine(q)
        # Extract position (meters -> mm)
        tcp_poses[i, :3] = T.t * 1000.0  # m -> mm
        # Extract orientation - T.R is the rotation matrix
        rpy = Rotation.from_matrix(T.R).as_euler("xyz", degrees=True)
        tcp_poses[i, 3:] = rpy

    return tcp_poses


# Plane normal vectors (useful for TRF/WRF transformations)
PLANE_NORMALS: dict[str, NDArray] = {
    "XY": np.array([0.0, 0.0, 1.0]),
    "XZ": np.array([0.0, 1.0, 0.0]),
    "YZ": np.array([1.0, 0.0, 0.0]),
}
