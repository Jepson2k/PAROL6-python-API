# Clean, hierarchical, vectorized, and typed robot configuration and helpers
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import roboticstoolbox as rtb
from numpy.typing import NDArray
from roboticstoolbox import ET, Link
from roboticstoolbox.tools.urdf import URDF

from parol6.tools import get_tool_transform

logger = logging.getLogger(__name__)

# -----------------------------
# Typing aliases
# -----------------------------
Vec6f = NDArray[np.float64]
Vec6i = NDArray[np.int32]
Limits2f = NDArray[np.float64]  # shape (6,2)

# -----------------------------
# Kinematics and conversion constants
# -----------------------------
Joint_num = 6
Microstep = 32
steps_per_revolution = 200

# Conversion constants
degree_per_step_constant: float = 360.0 / (Microstep * steps_per_revolution)
radian_per_step_constant: float = (2.0 * np.pi) / (Microstep * steps_per_revolution)
radian_per_sec_2_deg_per_sec_const: float = 360.0 / (2.0 * np.pi)
deg_per_sec_2_radian_per_sec_const: float = (2.0 * np.pi) / 360.0

# -----------------------------
# Joint limits
# -----------------------------
# Limits (deg) you get after homing and moving to extremes
_joint_limits_degree: Limits2f = np.array(
    [
        [-123.046875, 123.046875],
        [-145.0088, -3.375],
        [107.866, 287.8675],
        [-105.46975, 105.46975],
        [-90.0, 90.0],
        [0.0, 360.0],
    ],
    dtype=np.float64,
)

_joint_limits_radian: Limits2f = np.deg2rad(_joint_limits_degree).astype(np.float64)


# URDF-based robot model (frames/limits aligned with controller)
def _load_urdf() -> URDF:
    """Load and cache the URDF object for robot reconstruction."""
    base_path = Path(__file__).resolve().parent / "urdf_model"
    urdf_path = base_path / "urdf" / "PAROL6.urdf"
    urdf_string = urdf_path.read_text(encoding="utf-8")
    return URDF.loadstr(urdf_string, str(urdf_path), base_path=base_path)


# Cache the URDF object (parsed once, reused for robot reconstruction)
_cached_urdf = _load_urdf()

# Current robot instance (rebuilt when tool changes)
robot = None


def apply_tool(tool_name: str) -> None:
    """
    Rebuild the robot with the specified tool as an additional link.
    This ensures the tool transform is properly integrated into the kinematic chain
    and affects forward kinematics calculations.

    Parameters
    ----------
    tool_name : str
        Name of the tool from tools.TOOL_CONFIGS
    """
    global robot

    # Get tool transform
    T_tool = get_tool_transform(tool_name)

    # Get the base elinks from cached URDF
    base_links = list(_cached_urdf.elinks)

    # Create a tool link if there's a non-identity transform
    if tool_name != "NONE" and not np.allclose(T_tool, np.eye(4)):
        # Create an ELink for the tool
        # The tool is a fixed transform from the last joint
        # ET.SE3 accepts a 4x4 numpy array directly
        tool_link = Link(
            ET.SE3(T_tool),
            name=f"tool_{tool_name}",
            parent=base_links[-1],  # Attach to the last link
        )

        # Add tool link to the chain
        all_links = base_links + [tool_link]
        logger.info(f"Applied tool '{tool_name}' to robot model as link")
    else:
        all_links = base_links
        logger.info(f"Applied tool '{tool_name}' (no additional link needed)")

    # Create robot with the complete link chain
    robot = rtb.Robot(
        all_links,
        name=_cached_urdf.name,
    )


# Initialize with no tool
apply_tool("NONE")

# -----------------------------
# Additional raw parameter arrays
# -----------------------------
# Reduction ratio per joint
_joint_ratio: NDArray[np.float64] = np.array(
    [6.4, 20.0, 20.0 * (38.0 / 42.0), 4.0, 4.0, 10.0], dtype=np.float64
)

# Joint speeds (steps/s)
_joint_max_speed: Vec6i = np.array(
    [9750, 27000, 30000, 30000, 33000, 33000], dtype=np.int32
)
_joint_min_speed: Vec6i = np.array([100, 100, 100, 100, 100, 100], dtype=np.int32)

# Jog speeds (steps/s) - 80% of max for safety margin during jogging
_joint_max_jog_speed: Vec6i = (_joint_max_speed * 0.5).astype(np.int32)
_joint_min_jog_speed: Vec6i = np.array([100, 100, 100, 100, 100, 100], dtype=np.int32)

# Joint accelerations (steps/s^2) per joint
# Derived: a_max = v_max * 3 (reach max speed in ~0.33s)
_joint_max_acc: Vec6i = (_joint_max_speed * 3).astype(np.int32)

# Maximum jerk limits (steps/s^3) per joint
# Derived: j_max = a_max * 10 (reach max accel in ~0.1s)
_joint_max_jerk: Vec6i = (_joint_max_acc * 10).astype(np.int32)

# Compute joint angular velocities/accelerations in rad/s
_joint_speed_rad = (
    _joint_max_speed.astype(float) * radian_per_step_constant / _joint_ratio
)
_joint_acc_rad = _joint_max_acc.astype(float) * radian_per_step_constant / _joint_ratio
_joint_jerk_rad = (
    _joint_max_jerk.astype(float) * radian_per_step_constant / _joint_ratio
)


def _compute_tcp_velocity_at_config(
    q: NDArray, direction: int, v_max_joint: NDArray
) -> float | None:
    """
    Compute max TCP velocity in a direction while maintaining orientation.

    Uses Jacobian pseudoinverse to find joint velocities that achieve pure
    linear TCP motion (no rotation). This models real Cartesian motion where
    wrist joints must compensate to maintain tool orientation.

    Args:
        q: Joint configuration in radians (6,)
        direction: 0=X, 1=Y, 2=Z
        v_max_joint: Joint velocity limits in rad/s (6,)

    Returns:
        Max TCP velocity in m/s, or None if near singularity
    """
    try:
        assert robot is not None
        J = robot.jacob0(q)
        if np.linalg.cond(J) > 1e6:
            return None  # Near singularity

        # Desired TCP velocity: 1 m/s in direction, zero angular velocity
        desired = np.zeros(6)
        desired[direction] = 1.0

        # Pseudoinverse gives minimum-norm joint velocities
        J_pinv = np.linalg.pinv(J)
        q_dot = J_pinv @ desired

        # Verify orientation is maintained (angular velocity near zero)
        omega = J[3:, :] @ q_dot
        if np.linalg.norm(omega) > 0.01:
            return None  # Can't maintain orientation

        # Find limiting joint and scale factor
        q_dot_abs = np.abs(q_dot) + 1e-10
        scale_factors = v_max_joint / q_dot_abs
        max_scale = np.min(scale_factors)

        return max_scale  # m/s
    except Exception:
        return None


def _compute_jacobian_velocity_bound() -> tuple[float, float]:
    """
    Compute Cartesian velocity bound using Jacobian pseudoinverse sampling.

    Samples the workspace and computes achievable TCP velocity while maintaining
    orientation (zero angular velocity). This is more accurate than column-norm
    methods because it accounts for joint coupling required for Cartesian motion.

    Method:
        1. Sample random configurations within joint limits
        2. For each config, compute max TCP velocity in X, Y, Z directions
           using J_pinv @ [v, 0, 0, 0, 0, 0] to find required joint velocities
        3. Scale by limiting joint to find max achievable velocity
        4. Return median across workspace (conservative but realistic)

    Returns:
        (v_linear_max, ω_angular_max) in (m/s, rad/s)
    """
    np.random.seed(42)  # Reproducible results
    n_samples = 500

    velocities = []

    for _ in range(n_samples):
        # Random config within joint limits
        q = np.array(
            [
                np.random.uniform(
                    _joint_limits_radian[j, 0], _joint_limits_radian[j, 1]
                )
                for j in range(6)
            ]
        )

        # Test X, Y, Z directions
        for direction in range(3):
            v = _compute_tcp_velocity_at_config(q, direction, _joint_speed_rad)
            if v is not None and v > 0.001:  # Filter near-singular configs
                velocities.append(v)

    if not velocities:
        # Fallback to conservative estimate
        return 0.1, 1.0

    # Use median for conservative but realistic estimate
    median_vel = float(np.median(velocities))

    # Angular velocity: estimate from wrist joint speeds
    # (less critical, use simple estimate)
    angular_vel = float(np.mean(_joint_speed_rad[3:6]))

    return median_vel, angular_vel


def _compute_jacobian_accel_bound() -> tuple[float, float]:
    """
    Compute Cartesian acceleration bound using same approach as velocity.

    Returns:
        (a_linear_max, a_angular_max) in (m/s², rad/s²)
    """
    np.random.seed(43)  # Different seed for variety
    n_samples = 200

    accelerations = []

    for _ in range(n_samples):
        q = np.array(
            [
                np.random.uniform(
                    _joint_limits_radian[j, 0], _joint_limits_radian[j, 1]
                )
                for j in range(6)
            ]
        )

        for direction in range(3):
            a = _compute_tcp_velocity_at_config(q, direction, _joint_acc_rad)
            if a is not None and a > 0.001:
                accelerations.append(a)

    if not accelerations:
        linear_acc = 1.0  # Fallback
    else:
        linear_acc = float(np.median(accelerations))

    # Angular acceleration: estimate from wrist joint accelerations
    angular_acc = float(np.mean(_joint_acc_rad[3:6]))

    return linear_acc, angular_acc


def _compute_jacobian_jerk_bound() -> tuple[float, float]:
    """
    Compute Cartesian jerk bound using same approach as velocity/acceleration.

    Returns:
        (j_linear_max, j_angular_max) in (m/s³, rad/s³)
    """
    np.random.seed(44)  # Different seed
    n_samples = 200

    jerks = []

    for _ in range(n_samples):
        q = np.array(
            [
                np.random.uniform(
                    _joint_limits_radian[j, 0], _joint_limits_radian[j, 1]
                )
                for j in range(6)
            ]
        )

        for direction in range(3):
            j = _compute_tcp_velocity_at_config(q, direction, _joint_jerk_rad)
            if j is not None and j > 0.001:
                jerks.append(j)

    if not jerks:
        linear_jerk = 10.0  # Fallback
    else:
        linear_jerk = float(np.median(jerks))

    # Angular jerk: estimate from wrist joint jerks
    angular_jerk = float(np.mean(_joint_jerk_rad[3:6]))

    return linear_jerk, angular_jerk


# Cartesian limits derived from Jacobian analysis
_cart_linear_velocity_max: float = 0.0  # Set after robot init
_cart_angular_velocity_max: float = 0.0
_cart_linear_acc_max: float = 0.0
_cart_angular_acc_max: float = 0.0
_cart_linear_jerk_max: float = 0.0
_cart_angular_jerk_max: float = 0.0

# Min values as fraction of max
_cart_linear_velocity_min: float = 0.0
_cart_angular_velocity_min: float = 0.0
_cart_linear_acc_min: float = 0.0
_cart_angular_acc_min: float = 0.0
_cart_linear_jerk_min: float = 0.0
_cart_angular_jerk_min: float = 0.0

# Jog limits (80% of max for safety margin)
_cart_linear_velocity_max_JOG: float = 0.0
_cart_linear_velocity_min_JOG: float = 0.0


def _init_cartesian_limits() -> None:
    """Initialize Cartesian limits after robot model is loaded.

    Linear velocity/acceleration/jerk are stored in mm/s, mm/s², mm/s³.
    Angular velocity/acceleration/jerk are stored in deg/s, deg/s², deg/s³.
    """
    global _cart_linear_velocity_max, _cart_angular_velocity_max
    global _cart_linear_velocity_min, _cart_angular_velocity_min
    global _cart_linear_acc_max, _cart_linear_acc_min
    global _cart_angular_acc_max, _cart_angular_acc_min
    global _cart_linear_jerk_max, _cart_linear_jerk_min
    global _cart_angular_jerk_max, _cart_angular_jerk_min
    global _cart_linear_velocity_max_JOG, _cart_linear_velocity_min_JOG

    linear_vel_m_s, angular_vel_rad_s = _compute_jacobian_velocity_bound()
    linear_acc_m_s2, angular_acc_rad_s2 = _compute_jacobian_accel_bound()
    linear_jerk_m_s3, angular_jerk_rad_s3 = _compute_jacobian_jerk_bound()

    # Convert linear units from m/s to mm/s (and similar for accel/jerk)
    _cart_linear_velocity_max = linear_vel_m_s * 1000.0
    _cart_linear_acc_max = linear_acc_m_s2 * 1000.0
    _cart_linear_jerk_max = linear_jerk_m_s3 * 1000.0

    # Convert angular units from rad/s to deg/s (and similar for accel/jerk)
    _cart_angular_velocity_max = np.degrees(angular_vel_rad_s)
    _cart_angular_acc_max = np.degrees(angular_acc_rad_s2)
    _cart_angular_jerk_max = np.degrees(angular_jerk_rad_s3)

    # Min values as 1% of max
    _cart_linear_velocity_min = _cart_linear_velocity_max * 0.01
    _cart_angular_velocity_min = _cart_angular_velocity_max * 0.01
    _cart_linear_acc_min = _cart_linear_acc_max * 0.01
    _cart_angular_acc_min = _cart_angular_acc_max * 0.01
    _cart_linear_jerk_min = _cart_linear_jerk_max * 0.01
    _cart_angular_jerk_min = _cart_angular_jerk_max * 0.01

    # Jog limits (80% of max for additional safety during jogging)
    _cart_linear_velocity_max_JOG = _cart_linear_velocity_max * 0.8
    _cart_linear_velocity_min_JOG = _cart_linear_velocity_min


def log_derived_limits() -> None:
    """Log the derived Cartesian limits. Call at controller startup."""
    logger.info("=== Derived Kinematic Limits ===")
    logger.info("Joint velocity (rad/s): %s", np.round(_joint_speed_rad, 3))
    logger.info("Joint accel (rad/s²): %s", np.round(_joint_acc_rad, 2))
    logger.info("Joint jerk (rad/s³): %s", np.round(_joint_jerk_rad, 1))
    logger.info(
        "Cartesian linear velocity: %.1f mm/s (jog: %.1f mm/s)",
        _cart_linear_velocity_max,
        _cart_linear_velocity_max_JOG,
    )
    logger.info("Cartesian angular velocity: %.2f deg/s", _cart_angular_velocity_max)
    logger.info(
        "Cartesian linear accel: %.1f mm/s², angular: %.2f deg/s²",
        _cart_linear_acc_max,
        _cart_angular_acc_max,
    )
    logger.info(
        "Cartesian linear jerk: %.1f mm/s³, angular: %.2f deg/s³",
        _cart_linear_jerk_max,
        _cart_angular_jerk_max,
    )
    logger.info("================================")


# Standby positions
_standby_deg: Vec6f = np.array([90.0, -90.0, 180.0, 0.0, 0.0, 180.0], dtype=np.float64)

# Initialize Cartesian limits (depends on robot model and standby positions)
_init_cartesian_limits()


# -----------------------------
# Typed hierarchical API
# -----------------------------
@dataclass(frozen=True)
class Joint:
    """Minimal joint configuration - all values in native units (deg for position, steps/s for speed)."""

    limits_deg: Limits2f  # Position limits in degrees [6, 2]
    speed_max: Vec6i  # Max speed in steps/s
    speed_min: Vec6i  # Min speed in steps/s
    jog_speed_max: Vec6i  # Max jog speed in steps/s
    jog_speed_min: Vec6i  # Min jog speed in steps/s
    acc_max: Vec6i  # Max acceleration in steps/s²
    jerk_max: Vec6i  # Max jerk in steps/s³
    ratio: Vec6f  # Gear ratio per joint
    standby_deg: Vec6f  # Standby position in degrees


@dataclass(frozen=True)
class RangeF:
    min: float
    max: float


@dataclass(frozen=True)
class CartVel:
    linear: RangeF
    jog: RangeF
    angular: RangeF


@dataclass(frozen=True)
class CartAcc:
    linear: RangeF
    angular: RangeF


@dataclass(frozen=True)
class CartJerk:
    linear: RangeF
    angular: RangeF


@dataclass(frozen=True)
class Cart:
    vel: CartVel
    acc: CartAcc
    jerk: CartJerk


@dataclass(frozen=True)
class Conv:
    degree_per_step: float
    radian_per_step: float
    rad_sec_to_deg_sec: float
    deg_sec_to_rad_sec: float


joint: Final[Joint] = Joint(
    limits_deg=_joint_limits_degree,
    speed_max=_joint_max_speed,
    speed_min=_joint_min_speed,
    jog_speed_max=_joint_max_jog_speed,
    jog_speed_min=_joint_min_jog_speed,
    acc_max=_joint_max_acc,
    jerk_max=_joint_max_jerk,
    ratio=_joint_ratio,
    standby_deg=_standby_deg,
)

cart: Final[Cart] = Cart(
    vel=CartVel(
        linear=RangeF(min=_cart_linear_velocity_min, max=_cart_linear_velocity_max),
        jog=RangeF(
            min=_cart_linear_velocity_min_JOG, max=_cart_linear_velocity_max_JOG
        ),
        angular=RangeF(min=_cart_angular_velocity_min, max=_cart_angular_velocity_max),
    ),
    acc=CartAcc(
        linear=RangeF(min=_cart_linear_acc_min, max=_cart_linear_acc_max),
        angular=RangeF(min=_cart_angular_acc_min, max=_cart_angular_acc_max),
    ),
    jerk=CartJerk(
        linear=RangeF(min=_cart_linear_jerk_min, max=_cart_linear_jerk_max),
        angular=RangeF(min=_cart_angular_jerk_min, max=_cart_angular_jerk_max),
    ),
)

conv: Final[Conv] = Conv(
    degree_per_step=degree_per_step_constant,
    radian_per_step=radian_per_step_constant,
    rad_sec_to_deg_sec=radian_per_sec_2_deg_per_sec_const,
    deg_sec_to_rad_sec=deg_per_sec_2_radian_per_sec_const,
)


# -----------------------------
# CAN helpers and bitfield utils (used by transports/gripper)
# -----------------------------
def extract_from_can_id(can_id: int) -> tuple[int, int, int]:
    id2 = (can_id >> 7) & 0xF
    can_command = (can_id >> 1) & 0x3F
    error_bit = can_id & 0x1
    return id2, can_command, error_bit


def combine_2_can_id(id2: int, can_command: int, error_bit: int) -> int:
    can_id = 0
    can_id |= (id2 & 0xF) << 7
    can_id |= (can_command & 0x3F) << 1
    can_id |= error_bit & 0x1
    return can_id


def fuse_bitfield_2_bytearray(var_in: list[int] | tuple[int, ...]) -> bytes:
    number = 0
    for b in var_in:
        number = (2 * number) + int(b)
    return bytes([number])


def split_2_bitfield(var_in: int) -> list[int]:
    return [(var_in >> i) & 1 for i in range(7, -1, -1)]


if __name__ == "__main__":
    # Simple sanity prints
    from parol6.config import steps_to_rad

    j_step_rad = steps_to_rad(np.array([1, 1, 1, 1, 1, 1], dtype=np.int32))
    print("Smallest step (deg):", np.rad2deg(j_step_rad))
    print("Standby deg:", joint.standby_deg)
    print("Standby rad:", np.deg2rad(joint.standby_deg))
