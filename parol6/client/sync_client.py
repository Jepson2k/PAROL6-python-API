"""
Synchronous facade for AsyncRobotClient.

- In sync code: use RobotClient and call methods directly.
- In async code (event loop running): use AsyncRobotClient and `await` the methods.
"""

import asyncio
import atexit
import threading
from collections.abc import Callable, Coroutine
from typing import Any, Literal, TypeVar

from ..protocol.types import Axis, Frame, PingResult, StatusAggregate
from .async_client import AsyncRobotClient

T = TypeVar("T")


# Persistent background event loop for sync wrapper
_SYNC_LOOP: asyncio.AbstractEventLoop | None = None
_SYNC_THREAD: threading.Thread | None = None
_SYNC_LOOP_READY = threading.Event()


def _loop_worker(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    _SYNC_LOOP_READY.set()
    loop.run_forever()


def _stop_sync_loop() -> None:
    global _SYNC_LOOP, _SYNC_THREAD
    if _SYNC_LOOP is not None:
        try:
            _SYNC_LOOP.call_soon_threadsafe(_SYNC_LOOP.stop)
        except Exception:
            pass
        _SYNC_LOOP = None
        _SYNC_THREAD = None


def _ensure_sync_loop() -> None:
    """Start a persistent background event loop if not started yet."""
    global _SYNC_LOOP, _SYNC_THREAD
    if _SYNC_LOOP is None:
        _SYNC_LOOP = asyncio.new_event_loop()
        _SYNC_THREAD = threading.Thread(
            target=_loop_worker,
            args=(_SYNC_LOOP,),
            name="parol6-sync-loop",
            daemon=True,
        )
        _SYNC_THREAD.start()
        _SYNC_LOOP_READY.wait(timeout=1.0)
        atexit.register(_stop_sync_loop)


def _run(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine to completion using a persistent background event loop.
    If a loop is already running in this thread, raise to avoid deadlocks.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in this thread -> submit to persistent loop
        _ensure_sync_loop()
        assert _SYNC_LOOP is not None
        fut = asyncio.run_coroutine_threadsafe(coro, _SYNC_LOOP)
        return fut.result()
    # A loop is running in this thread; blocking would be unsafe.
    raise RuntimeError(
        "RobotClient was used while an event loop is running.\n"
        "Use AsyncRobotClient and `await` the method instead."
    )


class RobotClient:
    """
    Synchronous wrapper around AsyncRobotClient.
    All methods return concrete results (never coroutines).

    Can be used as a context manager to ensure proper cleanup:

        with RobotClient() as client:
            client.enable()
            ...
    """

    # ---------- lifecycle ----------

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5001,
        timeout: float = 2.0,
        retries: int = 1,
    ) -> None:
        self._inner = AsyncRobotClient(
            host=host, port=port, timeout=timeout, retries=retries
        )

    def close(self) -> None:
        """Close underlying AsyncRobotClient and release resources."""
        _run(self._inner.close())

    def __enter__(self) -> "RobotClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def async_client(self) -> AsyncRobotClient:
        """Access the underlying async client if you need it."""
        return self._inner

    # Expose common configuration attributes
    @property
    def host(self) -> str:
        return self._inner.host

    @property
    def port(self) -> int:
        return self._inner.port

    # ---------- motion / control ----------

    def home(self, wait: bool = False, **wait_kwargs) -> bool:
        """Home the robot to its home position.

        Args:
            wait: If True, block until motion completes.
            **wait_kwargs: Arguments passed to wait_motion_complete().

        Returns:
            True if the command was acknowledged successfully.
        """
        return _run(self._inner.home(wait=wait, **wait_kwargs))

    def enable(self) -> bool:
        """Enable the robot controller, allowing motion commands.

        Returns:
            True if the command was acknowledged successfully.
        """
        return _run(self._inner.enable())

    def disable(self) -> bool:
        """Disable the robot controller, stopping all motion.

        Returns:
            True if the command was acknowledged successfully.
        """
        return _run(self._inner.disable())

    def stop(self) -> bool:
        """Alias for disable() - stops motion and disables controller."""
        return self.disable()

    def start(self) -> bool:
        """Alias for enable() - enables controller."""
        return self.enable()

    def stream_on(self) -> bool:
        """Enable streaming mode for high-frequency motion commands.

        Returns:
            True if the command was acknowledged successfully.
        """
        return _run(self._inner.stream_on())

    def stream_off(self) -> bool:
        """Disable streaming mode.

        Returns:
            True if the command was acknowledged successfully.
        """
        return _run(self._inner.stream_off())

    def simulator_on(self) -> bool:
        """Enable simulator mode (no physical robot hardware required).

        Returns:
            True if the command was acknowledged successfully.
        """
        return _run(self._inner.simulator_on())

    def simulator_off(self) -> bool:
        """Disable simulator mode, switching to real hardware.

        Returns:
            True if the command was acknowledged successfully.
        """
        return _run(self._inner.simulator_off())

    def set_serial_port(self, port_str: str) -> bool:
        """Set the serial port for robot hardware communication.

        Args:
            port_str: Serial port path (e.g., '/dev/ttyUSB0' or 'COM3').

        Returns:
            True if the command was acknowledged successfully.
        """
        return _run(self._inner.set_serial_port(port_str))

    def reset(self) -> bool:
        """Reset controller state to initial values."""
        return _run(self._inner.reset())

    # ---------- status / queries ----------
    def ping(self) -> PingResult | None:
        """Ping the controller to check connectivity.

        Returns:
            PingResult with serial_connected status, or None on timeout.
        """
        return _run(self._inner.ping())

    def get_angles(self) -> list[float] | None:
        """Get current joint angles in degrees.

        Returns:
            List of 6 joint angles [J1-J6] in degrees, or None on timeout.
        """
        return _run(self._inner.get_angles())

    def get_io(self) -> list[int] | None:
        """Get digital I/O status.

        Returns:
            List of 5 integers [in1, in2, out1, out2, estop], or None on timeout.
        """
        return _run(self._inner.get_io())

    def get_gripper_status(self) -> list[int] | None:
        """Get electric gripper status.

        Returns:
            List of integers [id, pos, speed, current, status, obj_detected],
            or None on timeout.
        """
        return _run(self._inner.get_gripper_status())

    def get_speeds(self) -> list[float] | None:
        """Get current joint speeds in steps per second.

        Returns:
            List of 6 joint speeds [J1-J6] in steps/sec, or None on timeout.
        """
        return _run(self._inner.get_speeds())

    def get_pose(self) -> list[float] | None:
        """Get current robot pose as a 4x4 transformation matrix.

        Returns:
            16-element flattened transformation matrix (row-major) with
            translation in mm, or None on timeout.
        """
        return _run(self._inner.get_pose())

    def get_gripper(self) -> list[int] | None:
        """Alias for get_gripper_status()."""
        return _run(self._inner.get_gripper())

    def get_status(self) -> dict | None:
        """Get aggregate robot status.

        Returns:
            Dict with keys: pose, angles, io, gripper, or None on timeout.
        """
        return _run(self._inner.get_status())

    def get_loop_stats(self) -> dict | None:
        """Get control loop runtime statistics.

        Returns:
            Dict with loop timing metrics, or None on timeout.
        """
        return _run(self._inner.get_loop_stats())

    def get_tool(self) -> dict | None:
        """
        Get the current tool configuration and available tools.

        Returns:
            Dict with keys: 'tool' (current tool name), 'available' (list of available tools)
        """
        return _run(self._inner.get_tool())

    def set_tool(self, tool_name: str) -> bool:
        """
        Set the current end-effector tool configuration.

        Args:
            tool_name: Name of the tool ('NONE', 'PNEUMATIC', 'ELECTRIC')

        Returns:
            True if successful
        """
        return _run(self._inner.set_tool(tool_name))

    def get_current_action(self) -> dict | None:
        """
        Get the current executing action/command and its state.

        Returns:
            Dict with keys: 'current' (current action name), 'state' (action state),
                           'next' (next action if any)
        """
        return _run(self._inner.get_current_action())

    def get_queue(self) -> dict | None:
        """
        Get the list of queued non-streamable commands.

        Returns:
            Dict with keys: 'non_streamable' (list of queued commands), 'size' (queue size)
        """
        return _run(self._inner.get_queue())

    # ---------- helper methods ----------

    def get_pose_rpy(self) -> list[float] | None:
        """Get robot pose as [x, y, z, rx, ry, rz] in mm and degrees.

        Returns:
            List of 6 floats [x, y, z, rx, ry, rz], or None on error.
        """
        return _run(self._inner.get_pose_rpy())

    def get_pose_xyz(self) -> list[float] | None:
        """Get robot position as [x, y, z] in mm.

        Returns:
            List of 3 floats [x, y, z], or None on error.
        """
        return _run(self._inner.get_pose_xyz())

    def is_estop_pressed(self) -> bool:
        """Check if E-stop is pressed.

        Returns:
            True if E-stop is pressed, False otherwise.
        """
        return _run(self._inner.is_estop_pressed())

    def is_robot_stopped(self, threshold_speed: float = 2.0) -> bool:
        """Check if robot has stopped moving.

        Args:
            threshold_speed: Speed threshold in steps/sec.

        Returns:
            True if all joints below threshold.
        """
        return _run(self._inner.is_robot_stopped(threshold_speed))

    def wait_motion_complete(
        self,
        timeout: float = 90.0,
        settle_window: float = 1.0,
        speed_threshold: float = 2.0,
        angle_threshold: float = 0.5,
    ) -> bool:
        """Wait for robot to stop moving.

        Args:
            timeout: Maximum time to wait in seconds.
            settle_window: How long robot must be stable.
            speed_threshold: Max joint speed to be considered stopped.
            angle_threshold: Max angle change to be considered stopped.

        Returns:
            True if robot stopped, False if timeout.
        """
        return _run(
            self._inner.wait_motion_complete(
                timeout=timeout,
                settle_window=settle_window,
                speed_threshold=speed_threshold,
                angle_threshold=angle_threshold,
            )
        )

    # ---------- responsive waits / raw send ----------

    def wait_for_server_ready(
        self, timeout: float = 5.0, interval: float = 0.05
    ) -> bool:
        """Poll ping() until server responds or timeout."""
        return _run(
            self._inner.wait_for_server_ready(timeout=timeout, interval=interval)
        )

    def wait_for_status(
        self, predicate: Callable[[StatusAggregate], bool], timeout: float = 5.0
    ) -> bool:
        """
        Wait until a multicast status satisfies predicate(status) within timeout.
        Note: predicate is executed in the client's event loop thread.
        """
        return _run(self._inner.wait_for_status(predicate, timeout=timeout))

    def send_raw(
        self, message: str, await_reply: bool = False, timeout: float = 2.0
    ) -> bool | str | None:
        """
        Send a raw UDP message; optionally await a single reply and return its text.
        Returns True on fire-and-forget send, str on reply, or None on timeout/error when awaiting.
        """
        return _run(
            self._inner.send_raw(message, await_reply=await_reply, timeout=timeout)
        )

    # ---------- extended controls / motion ----------

    def move_joints(
        self,
        joint_angles: list[float],
        duration: float | None = None,
        speed_percentage: int | None = None,
        accel_percentage: int | None = None,
        wait: bool = False,
        **wait_kwargs,
    ) -> bool:
        """Move to specified joint angles.

        Args:
            joint_angles: Target joint angles in degrees [J1-J6].
            duration: Time to complete motion in seconds.
            speed_percentage: Speed as percentage (1-100).
            accel_percentage: Acceleration as percentage (1-100).
            wait: If True, block until motion completes.
            **wait_kwargs: Arguments passed to wait_motion_complete().

        Returns:
            True if command sent successfully.
        """
        return _run(
            self._inner.move_joints(
                joint_angles,
                duration,
                speed_percentage,
                accel_percentage,
                wait=wait,
                **wait_kwargs,
            )
        )

    def move_pose(
        self,
        pose: list[float],
        duration: float | None = None,
        speed_percentage: int | None = None,
        accel_percentage: int | None = None,
        wait: bool = False,
        **wait_kwargs,
    ) -> bool:
        """Move to specified pose using joint-space interpolation.

        Args:
            pose: Target pose [x, y, z, rx, ry, rz] in mm and degrees.
            duration: Time to complete motion in seconds.
            speed_percentage: Speed as percentage (1-100).
            accel_percentage: Acceleration as percentage (1-100).
            wait: If True, block until motion completes.
            **wait_kwargs: Arguments passed to wait_motion_complete().

        Returns:
            True if command sent successfully.
        """
        return _run(
            self._inner.move_pose(
                pose,
                duration,
                speed_percentage,
                accel_percentage,
                wait=wait,
                **wait_kwargs,
            )
        )

    def move_cartesian(
        self,
        pose: list[float],
        duration: float | None = None,
        speed_percentage: float | None = None,
        accel_percentage: int | None = None,
        wait: bool = False,
        **wait_kwargs,
    ) -> bool:
        """Move to specified pose using Cartesian-space interpolation.

        Args:
            pose: Target pose [x, y, z, rx, ry, rz] in mm and degrees.
            duration: Time to complete motion in seconds.
            speed_percentage: Speed as percentage (1-100).
            accel_percentage: Acceleration as percentage (1-100).
            wait: If True, block until motion completes.
            **wait_kwargs: Arguments passed to wait_motion_complete().

        Returns:
            True if command sent successfully.
        """
        return _run(
            self._inner.move_cartesian(
                pose,
                duration,
                speed_percentage,
                accel_percentage,
                wait=wait,
                **wait_kwargs,
            )
        )

    def move_cartesian_rel_trf(
        self,
        deltas: list[float],
        duration: float | None = None,
        speed_percentage: float | None = None,
        accel_percentage: int | None = None,
        profile: str | None = None,
        tracking: str | None = None,
        wait: bool = False,
        **wait_kwargs,
    ) -> bool:
        """Move relative to current pose in Tool Reference Frame.

        Args:
            deltas: Relative movement [dx, dy, dz, rx, ry, rz] in mm and degrees.
            duration: Time to complete motion in seconds.
            speed_percentage: Speed as percentage (1-100).
            accel_percentage: Acceleration as percentage (1-100).
            profile: Motion profile type.
            tracking: Tracking mode.
            wait: If True, block until motion completes.
            **wait_kwargs: Arguments passed to wait_motion_complete().

        Returns:
            True if command sent successfully.
        """
        return _run(
            self._inner.move_cartesian_rel_trf(
                deltas,
                duration,
                speed_percentage,
                accel_percentage,
                profile,
                tracking,
                wait=wait,
                **wait_kwargs,
            )
        )

    def jog_joint(
        self,
        joint_index: int,
        speed_percentage: int,
        duration: float | None = None,
        distance_deg: float | None = None,
    ) -> bool:
        """Jog a single joint at a specified speed.

        Args:
            joint_index: Joint to jog (0-5 positive, 6-11 negative direction).
            speed_percentage: Speed as percentage (1-100).
            duration: Time to jog in seconds.
            distance_deg: Distance to jog in degrees.

        Returns:
            True if command sent successfully.
        """
        return _run(
            self._inner.jog_joint(
                joint_index,
                speed_percentage,
                duration,
                distance_deg,
            )
        )

    def jog_cartesian(
        self,
        frame: Frame,
        axis: Axis,
        speed_percentage: int,
        duration: float,
    ) -> bool:
        """Jog the robot in Cartesian space along a specified axis.

        Args:
            frame: Reference frame ('TRF' for Tool, 'WRF' for World).
            axis: Axis and direction to jog (e.g., 'X+', 'Y-', 'RZ+').
            speed_percentage: Speed as percentage (1-100).
            duration: Time to jog in seconds.

        Returns:
            True if command sent successfully.
        """
        return _run(self._inner.jog_cartesian(frame, axis, speed_percentage, duration))

    def jog_multiple(
        self,
        joints: list[int],
        speeds: list[float],
        duration: float,
    ) -> bool:
        """Jog multiple joints simultaneously.

        Args:
            joints: List of joint indices to jog (0-5).
            speeds: List of speeds for each joint (can be negative).
            duration: Time to jog in seconds.

        Returns:
            True if command sent successfully.
        """
        return _run(self._inner.jog_multiple(joints, speeds, duration))

    def set_io(self, index: int, value: int) -> bool:
        """Set digital I/O bit (0..7) to 0 or 1."""
        return _run(self._inner.set_io(index, value))

    def delay(self, seconds: float) -> bool:
        """Insert a non-blocking delay in the motion queue."""
        return _run(self._inner.delay(seconds))

    # ---------- IO / gripper ----------

    def control_pneumatic_gripper(
        self,
        action: str,
        port: int,
        wait: bool = False,
        **wait_kwargs,
    ) -> bool:
        """Control pneumatic gripper via digital outputs.

        Args:
            action: 'open' or 'close'.
            port: Port number (1 or 2).
            wait: If True, block until motion completes.
            **wait_kwargs: Arguments passed to wait_motion_complete().

        Returns:
            True if command sent successfully.
        """
        return _run(self._inner.control_pneumatic_gripper(action, port, wait=wait, **wait_kwargs))

    def control_electric_gripper(
        self,
        action: str,
        position: int | None = 255,
        speed: int | None = 150,
        current: int | None = 500,
        wait: bool = False,
        **wait_kwargs,
    ) -> bool:
        """Control electric gripper.

        Args:
            action: 'move' or 'calibrate'.
            position: Target position (0-255).
            speed: Movement speed (0-255).
            current: Current limit in mA (100-1000).
            wait: If True, block until motion completes.
            **wait_kwargs: Arguments passed to wait_motion_complete().

        Returns:
            True if command sent successfully.
        """
        return _run(
            self._inner.control_electric_gripper(action, position, speed, current, wait=wait, **wait_kwargs)
        )

    # ---------- GCODE ----------

    def execute_gcode(
        self,
        gcode_line: str,
    ) -> bool:
        """Execute a single G-code line.

        Args:
            gcode_line: G-code command to execute (e.g., 'G0 X100').

        Returns:
            True if command sent successfully.
        """
        return _run(self._inner.execute_gcode(gcode_line))

    def execute_gcode_program(
        self,
        program_lines: list[str],
    ) -> bool:
        """Execute a G-code program from a list of lines.

        Args:
            program_lines: List of G-code lines to execute.

        Returns:
            True if command sent successfully.
        """
        return _run(self._inner.execute_gcode_program(program_lines))

    def load_gcode_file(
        self,
        filepath: str,
    ) -> bool:
        """Load and execute a G-code program from a file.

        Args:
            filepath: Path to the G-code file.

        Returns:
            True if command sent successfully.
        """
        return _run(self._inner.load_gcode_file(filepath))

    def get_gcode_status(self) -> dict | None:
        """Get the current status of the G-code interpreter.

        Returns:
            Dict with interpreter state, or None on timeout.
        """
        return _run(self._inner.get_gcode_status())

    def pause_gcode_program(self) -> bool:
        """Pause the currently running G-code program.

        Returns:
            True if command sent successfully.
        """
        return _run(self._inner.pause_gcode_program())

    def resume_gcode_program(self) -> bool:
        """Resume a paused G-code program.

        Returns:
            True if command sent successfully.
        """
        return _run(self._inner.resume_gcode_program())

    def stop_gcode_program(self) -> bool:
        """Stop the currently running G-code program.

        Returns:
            True if command sent successfully.
        """
        return _run(self._inner.stop_gcode_program())

    # ---------- smooth motion ----------

    def smooth_circle(
        self,
        center: list[float],
        radius: float,
        plane: Literal["XY", "XZ", "YZ"] = "XY",
        frame: Literal["WRF", "TRF"] = "WRF",
        center_mode: Literal["ABSOLUTE", "TOOL", "RELATIVE"] = "ABSOLUTE",
        entry_mode: Literal["AUTO", "TANGENT", "DIRECT", "NONE"] = "NONE",
        start_pose: list[float] | None = None,
        duration: float | None = None,
        speed_percentage: float | None = None,
        clockwise: bool = False,
        trajectory_type: Literal["cubic", "quintic", "s_curve"] = "cubic",
        jerk_limit: float | None = None,
        wait: bool = False,
        **wait_kwargs,
    ) -> bool:
        """Execute a smooth circular motion.

        Args:
            center: Circle center [x, y, z] in mm.
            radius: Circle radius in mm.
            plane: Plane of the circle ('XY', 'XZ', 'YZ').
            frame: Reference frame ('WRF' or 'TRF').
            center_mode: How to interpret center point.
            entry_mode: How to approach circle if not on perimeter.
            start_pose: Optional start pose [x, y, z, rx, ry, rz].
            duration: Time to complete motion in seconds.
            speed_percentage: Speed as percentage (1-100).
            clockwise: Direction of motion.
            trajectory_type: Trajectory type ('cubic', 'quintic', 's_curve').
            jerk_limit: Optional jerk limit for s_curve.
            wait: If True, block until motion completes.
            **wait_kwargs: Arguments passed to wait_motion_complete().

        Returns:
            True if command sent successfully.
        """
        return _run(
            self._inner.smooth_circle(
                center=center,
                radius=radius,
                plane=plane,
                frame=frame,
                center_mode=center_mode,
                entry_mode=entry_mode,
                start_pose=start_pose,
                duration=duration,
                speed_percentage=speed_percentage,
                clockwise=clockwise,
                trajectory_type=trajectory_type,
                jerk_limit=jerk_limit,
                wait=wait,
                **wait_kwargs,
            )
        )

    def smooth_arc_center(
        self,
        end_pose: list[float],
        center: list[float],
        frame: Literal["WRF", "TRF"] = "WRF",
        start_pose: list[float] | None = None,
        duration: float | None = None,
        speed_percentage: float | None = None,
        clockwise: bool = False,
        trajectory_type: Literal["cubic", "quintic", "s_curve"] = "cubic",
        jerk_limit: float | None = None,
        wait: bool = False,
        **wait_kwargs,
    ) -> bool:
        """Execute a smooth arc motion defined by center point.

        Args:
            end_pose: End pose [x, y, z, rx, ry, rz] in mm and degrees.
            center: Arc center [x, y, z] in mm.
            frame: Reference frame ('WRF' or 'TRF').
            start_pose: Optional start pose.
            duration: Time to complete motion in seconds.
            speed_percentage: Speed as percentage (1-100).
            clockwise: Direction of motion.
            trajectory_type: Trajectory type.
            jerk_limit: Optional jerk limit for s_curve.
            wait: If True, block until motion completes.
            **wait_kwargs: Arguments passed to wait_motion_complete().

        Returns:
            True if command sent successfully.
        """
        return _run(
            self._inner.smooth_arc_center(
                end_pose=end_pose,
                center=center,
                frame=frame,
                start_pose=start_pose,
                duration=duration,
                speed_percentage=speed_percentage,
                clockwise=clockwise,
                trajectory_type=trajectory_type,
                jerk_limit=jerk_limit,
                wait=wait,
                **wait_kwargs,
            )
        )

    def smooth_arc_param(
        self,
        end_pose: list[float],
        radius: float,
        arc_angle: float,
        frame: Literal["WRF", "TRF"] = "WRF",
        start_pose: list[float] | None = None,
        duration: float | None = None,
        speed_percentage: float | None = None,
        trajectory_type: Literal["cubic", "quintic", "s_curve"] = "cubic",
        jerk_limit: float | None = None,
        clockwise: bool = False,
        wait: bool = False,
        **wait_kwargs,
    ) -> bool:
        """Execute a smooth arc motion defined parametrically.

        Args:
            end_pose: End pose [x, y, z, rx, ry, rz] in mm and degrees.
            radius: Arc radius in mm.
            arc_angle: Arc angle in degrees.
            frame: Reference frame ('WRF' or 'TRF').
            start_pose: Optional start pose.
            duration: Time to complete motion in seconds.
            speed_percentage: Speed as percentage (1-100).
            trajectory_type: Trajectory type.
            jerk_limit: Optional jerk limit for s_curve.
            clockwise: Direction of motion.
            wait: If True, block until motion completes.
            **wait_kwargs: Arguments passed to wait_motion_complete().

        Returns:
            True if command sent successfully.
        """
        return _run(
            self._inner.smooth_arc_param(
                end_pose=end_pose,
                radius=radius,
                arc_angle=arc_angle,
                frame=frame,
                start_pose=start_pose,
                duration=duration,
                speed_percentage=speed_percentage,
                trajectory_type=trajectory_type,
                jerk_limit=jerk_limit,
                clockwise=clockwise,
                wait=wait,
                **wait_kwargs,
            )
        )

    def smooth_spline(
        self,
        waypoints: list[list[float]],
        frame: Literal["WRF", "TRF"] = "WRF",
        start_pose: list[float] | None = None,
        duration: float | None = None,
        speed_percentage: float | None = None,
        trajectory_type: Literal["cubic", "quintic", "s_curve"] = "cubic",
        jerk_limit: float | None = None,
        wait: bool = False,
        **wait_kwargs,
    ) -> bool:
        """Execute a smooth spline motion through waypoints.

        Args:
            waypoints: List of poses [x, y, z, rx, ry, rz] in mm and degrees.
            frame: Reference frame ('WRF' or 'TRF').
            start_pose: Optional start pose.
            duration: Total time for motion in seconds.
            speed_percentage: Speed as percentage (1-100).
            trajectory_type: Trajectory type.
            jerk_limit: Optional jerk limit for s_curve.
            wait: If True, block until motion completes.
            **wait_kwargs: Arguments passed to wait_motion_complete().

        Returns:
            True if command sent successfully.
        """
        return _run(
            self._inner.smooth_spline(
                waypoints=waypoints,
                frame=frame,
                start_pose=start_pose,
                duration=duration,
                speed_percentage=speed_percentage,
                trajectory_type=trajectory_type,
                jerk_limit=jerk_limit,
                wait=wait,
                **wait_kwargs,
            )
        )

    def smooth_helix(
        self,
        center: list[float],
        radius: float,
        pitch: float,
        height: float,
        frame: Literal["WRF", "TRF"] = "WRF",
        trajectory_type: Literal["cubic", "quintic", "s_curve"] = "cubic",
        jerk_limit: float | None = None,
        start_pose: list[float] | None = None,
        duration: float | None = None,
        speed_percentage: float | None = None,
        clockwise: bool = False,
        wait: bool = False,
        **wait_kwargs,
    ) -> bool:
        """Execute a smooth helical motion.

        Args:
            center: Helix center [x, y, z] in mm.
            radius: Helix radius in mm.
            pitch: Vertical distance per revolution in mm.
            height: Total height of helix in mm.
            frame: Reference frame ('WRF' or 'TRF').
            trajectory_type: Trajectory type.
            jerk_limit: Optional jerk limit for s_curve.
            start_pose: Optional start pose.
            duration: Time to complete motion in seconds.
            speed_percentage: Speed as percentage (1-100).
            clockwise: Direction of motion.
            wait: If True, block until motion completes.
            **wait_kwargs: Arguments passed to wait_motion_complete().

        Returns:
            True if command sent successfully.
        """
        return _run(
            self._inner.smooth_helix(
                center=center,
                radius=radius,
                pitch=pitch,
                height=height,
                frame=frame,
                trajectory_type=trajectory_type,
                jerk_limit=jerk_limit,
                start_pose=start_pose,
                duration=duration,
                speed_percentage=speed_percentage,
                clockwise=clockwise,
                wait=wait,
                **wait_kwargs,
            )
        )

    def smooth_blend(
        self,
        segments: list[dict],
        blend_time: float = 0.5,
        frame: Literal["WRF", "TRF"] = "WRF",
        start_pose: list[float] | None = None,
        duration: float | None = None,
        speed_percentage: float | None = None,
        wait: bool = False,
        **wait_kwargs,
    ) -> bool:
        """Execute a blended motion through multiple segments.

        Args:
            segments: List of segment dictionaries.
            blend_time: Time to blend between segments in seconds.
            frame: Reference frame ('WRF' or 'TRF').
            start_pose: Optional start pose.
            duration: Total time for motion in seconds.
            speed_percentage: Speed as percentage (1-100).
            wait: If True, block until motion completes.
            **wait_kwargs: Arguments passed to wait_motion_complete().

        Returns:
            True if command sent successfully.
        """
        return _run(
            self._inner.smooth_blend(
                segments=segments,
                blend_time=blend_time,
                frame=frame,
                start_pose=start_pose,
                duration=duration,
                speed_percentage=speed_percentage,
                wait=wait,
                **wait_kwargs,
            )
        )

    def smooth_waypoints(
        self,
        waypoints: list[list[float]],
        blend_radii: Literal["AUTO"] | list[float] = "AUTO",
        blend_mode: Literal["parabolic", "circular", "none"] = "parabolic",
        via_modes: list[str] | None = None,
        max_velocity: float = 100.0,
        max_acceleration: float = 500.0,
        frame: Literal["WRF", "TRF"] = "WRF",
        trajectory_type: Literal["cubic", "quintic", "s_curve"] = "quintic",
        duration: float | None = None,
        wait: bool = False,
        **wait_kwargs,
    ) -> bool:
        """Execute a waypoint trajectory with blending.

        Args:
            waypoints: List of poses [x, y, z, rx, ry, rz] in mm and degrees.
            blend_radii: Blend radii for intermediate waypoints ('AUTO' or list).
            blend_mode: Blending mode ('parabolic', 'circular', 'none').
            via_modes: List of 'via' or 'stop' for each waypoint.
            max_velocity: Maximum velocity.
            max_acceleration: Maximum acceleration.
            frame: Reference frame ('WRF' or 'TRF').
            trajectory_type: Trajectory type.
            duration: Total time for motion in seconds.
            wait: If True, block until motion completes.
            **wait_kwargs: Arguments passed to wait_motion_complete().

        Returns:
            True if command sent successfully.
        """
        return _run(
            self._inner.smooth_waypoints(
                waypoints=waypoints,
                blend_radii=blend_radii,
                blend_mode=blend_mode,
                via_modes=via_modes,
                max_velocity=max_velocity,
                max_acceleration=max_acceleration,
                frame=frame,
                trajectory_type=trajectory_type,
                duration=duration,
                wait=wait,
                **wait_kwargs,
            )
        )

    # ---------- work coordinate helpers ----------

    def set_work_coordinate_offset(
        self,
        coordinate_system: str,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
    ) -> bool:
        """Set work coordinate system offsets (G54-G59).

        Args:
            coordinate_system: Work coordinate system ('G54' through 'G59').
            x: X axis offset in mm (None to keep current).
            y: Y axis offset in mm (None to keep current).
            z: Z axis offset in mm (None to keep current).

        Returns:
            True if command sent successfully.
        """
        return _run(
            self._inner.set_work_coordinate_offset(
                coordinate_system=coordinate_system,
                x=x,
                y=y,
                z=z,
            )
        )

    def zero_work_coordinates(
        self,
        coordinate_system: str = "G54",
    ) -> bool:
        """Set the current position as zero in the specified work coordinate system.

        Args:
            coordinate_system: Work coordinate system ('G54' through 'G59').

        Returns:
            True if command sent successfully.
        """
        return _run(
            self._inner.zero_work_coordinates(
                coordinate_system=coordinate_system,
            )
        )
