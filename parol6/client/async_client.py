"""
Async UDP client for PAROL6 robot control.
"""

import asyncio
import contextlib
import logging
import random
import socket
import struct
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Literal, cast

import msgspec
import numpy as np

from .. import config as cfg
from ..ack_policy import QUERY_CMD_TYPES, SYSTEM_CMD_TYPES, AckPolicy
from ..protocol.wire import (
    STRUCT_TO_CMDTYPE,
    decode_status_bin_into,
    CartJogCmd,
    CurrentActionResultStruct,
    DelayCmd,
    DisableCmd,
    ElectricGripperCmd,
    EnableCmd,
    ErrorMsg,
    GcodeCmd,
    GcodePauseCmd,
    GcodeProgramCmd,
    GcodeResumeCmd,
    GcodeStopCmd,
    GcodeStatusResultStruct,
    GetAnglesCmd,
    GetCurrentActionCmd,
    GetGcodeStatusCmd,
    GetGripperCmd,
    GetIOCmd,
    GetLoopStatsCmd,
    GetPoseCmd,
    GetProfileCmd,
    GetQueueCmd,
    GetSpeedsCmd,
    GetStatusCmd,
    GetToolCmd,
    HomeCmd,
    JogCmd,
    LoopStatsResultStruct,
    MoveCartCmd,
    MoveCartRelTrfCmd,
    MoveJointCmd,
    MovePoseCmd,
    MultiJogCmd,
    OkMsg,
    PingCmd,
    PneumaticGripperCmd,
    ResetCmd,
    ResetLoopStatsCmd,
    ResponseMsg,
    SetIOCmd,
    SetPortCmd,
    SetProfileCmd,
    SetToolCmd,
    SimulatorCmd,
    SmoothArcCenterCmd,
    SmoothArcParamCmd,
    SmoothCircleCmd,
    SmoothSplineCmd,
    StatusBuffer,
    StatusResultStruct,
    StreamCmd,
    ToolResultStruct,
    encode_command,
    parse_message,
)
from ..protocol.types import (
    Axis,
    Frame,
    PingResult,
)
from ..utils.se3_utils import so3_rpy

logger = logging.getLogger(__name__)


class _UDPClientProtocol(asyncio.DatagramProtocol):
    def __init__(self, rx_queue: asyncio.Queue[tuple[bytes, tuple[str, int]]]):
        self.rx_queue = rx_queue
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = cast(asyncio.DatagramTransport, transport)

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        try:
            self.rx_queue.put_nowait((data, addr))
        except asyncio.QueueFull:
            pass  # Drop packet when queue is full (expected under load)

    def error_received(self, exc: Exception) -> None:
        pass

    def connection_lost(self, exc: Exception | None) -> None:
        pass


def _create_multicast_socket(group: str, port: int, iface_ip: str) -> socket.socket:
    """Create and configure a multicast socket with loopback-first semantics."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

    # Allow multiple listeners on same port
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except AttributeError:
        pass
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)

    # Bind to port
    try:
        sock.bind(("", port))
    except OSError:
        sock.bind((iface_ip, port))

    # Helper to detect primary NIC IP
    def _detect_primary_ip() -> str:
        tmp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            tmp.connect(("1.1.1.1", 80))
            return tmp.getsockname()[0]
        except Exception:
            return "127.0.0.1"
        finally:
            with contextlib.suppress(Exception):
                tmp.close()

    # Join multicast group with fallbacks
    try:
        mreq = struct.pack("=4s4s", socket.inet_aton(group), socket.inet_aton(iface_ip))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    except Exception:
        try:
            primary_ip = _detect_primary_ip()
            mreq = struct.pack(
                "=4s4s", socket.inet_aton(group), socket.inet_aton(primary_ip)
            )
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        except Exception:
            mreq_any = struct.pack("=4sl", socket.inet_aton(group), socket.INADDR_ANY)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq_any)

    sock.setblocking(False)
    return sock


def _create_unicast_socket(port: int, host: str) -> socket.socket:
    """Create and configure a plain UDP socket for unicast reception."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except AttributeError:
        pass
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    try:
        sock.bind((host, port))
    except OSError:
        sock.bind(("", port))
    sock.setblocking(False)
    return sock


if TYPE_CHECKING:
    from typing import Protocol

    class _StatusNotifier(Protocol):
        _shared_status: StatusBuffer
        _status_generation: int
        _status_event: asyncio.Event
        _closed: bool


class _StatusProtocol(asyncio.DatagramProtocol):
    """Protocol handler for status datagrams - decodes directly into shared buffer."""

    def __init__(self, client: "_StatusNotifier"):
        self._client = client
        self._transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self._transport = cast(asyncio.DatagramTransport, transport)

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        if self._client._closed:
            return
        # Zero-allocation decode directly into shared buffer
        if decode_status_bin_into(data, self._client._shared_status):
            self._client._status_generation += 1
            # Event.set() is synchronous and wakes all waiters
            self._client._status_event.set()

    def error_received(self, exc: Exception) -> None:
        pass

    def connection_lost(self, exc: Exception | None) -> None:
        pass


class AsyncRobotClient:
    """
    Async UDP client for the PAROL6 headless controller.

    Motion/control commands: fire-and-forget via UDP
    Query commands: request/response with timeout and simple retry
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5001,
        timeout: float = 1.0,
        retries: int = 1,
    ) -> None:
        # Endpoint configuration (host/port immutable after endpoint creation)
        self._host = host
        self._port = port
        self.timeout = timeout
        self.retries = retries

        # Persistent asyncio datagram endpoint
        self._transport: asyncio.DatagramTransport | None = None
        self._protocol: _UDPClientProtocol | None = None
        self._rx_queue: asyncio.Queue[tuple[bytes, tuple[str, int]]] = asyncio.Queue(
            maxsize=256
        )
        self._ep_lock = asyncio.Lock()

        # Serialize request/response
        self._req_lock = asyncio.Lock()

        # Stream flag for UI convenience
        self._stream_mode = False
        # ACK policy
        self._ack_policy = AckPolicy.from_env(lambda: self._stream_mode)

        # Shared status state (single buffer, event-based notification)
        self._status_transport: asyncio.DatagramTransport | None = None
        self._status_sock: socket.socket | None = None
        self._shared_status: StatusBuffer = StatusBuffer()
        self._status_generation: int = 0
        self._status_event: asyncio.Event = asyncio.Event()

        # Lifecycle flag
        self._closed: bool = False

    # --------------- Endpoint configuration properties ---------------

    @property
    def host(self) -> str:
        return self._host

    @host.setter
    def host(self, value: str) -> None:
        if self._transport is not None:
            raise RuntimeError(
                "AsyncRobotClient.host is read-only after endpoint creation"
            )
        self._host = value

    @property
    def port(self) -> int:
        return self._port

    @port.setter
    def port(self, value: int) -> None:
        if self._transport is not None:
            raise RuntimeError(
                "AsyncRobotClient.port is read-only after endpoint creation"
            )
        self._port = value

    # --------------- Internal helpers ---------------

    async def _ensure_endpoint(self) -> None:
        """Lazily create a persistent asyncio UDP datagram endpoint."""
        if self._closed:
            raise RuntimeError("AsyncRobotClient is closed")
        if self._transport is not None:
            return
        async with self._ep_lock:
            if self._closed:
                raise RuntimeError("AsyncRobotClient is closed")
            if self._transport is not None:
                return
            loop = asyncio.get_running_loop()
            self._rx_queue = asyncio.Queue(maxsize=256)
            transport, protocol = await loop.create_datagram_endpoint(
                lambda: _UDPClientProtocol(self._rx_queue),
                remote_addr=(self.host, self.port),
            )
            self._transport = transport
            self._protocol = protocol
            logger.info(
                f"AsyncRobotClient UDP endpoint: remote={self.host}:{self.port}, timeout={self.timeout}, retries={self.retries}"
            )

            # Start multicast listener
            await self._start_multicast_listener()

    async def _start_multicast_listener(self) -> None:
        """Start listening for multicast/unicast status broadcasts.

        Creates a UDP socket and protocol that decodes status datagrams directly
        into the shared buffer, notifying consumers via condition variable.
        """
        if self._status_transport is not None:
            return

        logger.info(
            f"Status subscriber config: transport={cfg.STATUS_TRANSPORT} group={cfg.MCAST_GROUP} port={cfg.MCAST_PORT} iface={cfg.MCAST_IF}"
        )
        # Quick readiness check (no blind wait): bounded by client's own timeout
        try:
            await self.wait_for_server_ready(
                timeout=min(1.0, float(self.timeout or 0.3)), interval=0.5
            )
        except Exception:
            pass

        # Create the socket based on configured transport
        if cfg.STATUS_TRANSPORT == "UNICAST":
            self._status_sock = _create_unicast_socket(
                cfg.MCAST_PORT, cfg.STATUS_UNICAST_HOST
            )
        else:
            self._status_sock = _create_multicast_socket(
                cfg.MCAST_GROUP, cfg.MCAST_PORT, cfg.MCAST_IF
            )

        # Create the datagram endpoint with the status protocol
        loop = asyncio.get_running_loop()
        self._status_transport, _ = await loop.create_datagram_endpoint(
            lambda: _StatusProtocol(self),  # type: ignore[arg-type]
            sock=self._status_sock,
        )

    # --------------- Lifecycle / context management ---------------

    async def close(self) -> None:
        """Release UDP transport and background tasks.

        Safe to call multiple times.
        """
        if self._closed:
            return
        logging.debug("Closing Client...")
        self._closed = True

        # Wake all status_stream consumers
        self._status_event.set()

        # Close status transport - yield first to let pending I/O complete
        if self._status_transport is not None:
            with contextlib.suppress(Exception):
                await asyncio.sleep(0)
                self._status_transport.close()
            self._status_transport = None
        if self._status_sock is not None:
            with contextlib.suppress(Exception):
                self._status_sock.close()
            self._status_sock = None

        # Close UDP command transport
        if self._transport is not None:
            with contextlib.suppress(Exception):
                await asyncio.sleep(0)
                self._transport.close()
            self._transport = None
            self._protocol = None

        # Best-effort drain for RX queue to free memory
        with contextlib.suppress(Exception):
            while not self._rx_queue.empty():
                self._rx_queue.get_nowait()

    async def __aenter__(self) -> "AsyncRobotClient":
        if self._closed:
            raise RuntimeError("AsyncRobotClient is closed")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def status_stream(self) -> AsyncIterator[StatusBuffer]:
        """Async generator that yields status updates from multicast broadcasts.

        Usage:
            async for status in client.status_stream():
                print(f"Angles: {status.angles}")

        This generator terminates automatically when :meth:`close` is
        called on the client, so callers do not need to manually cancel
        their consumer tasks.

        Each yielded StatusBuffer is a copy - safe to store or process async.
        For zero-copy hot paths, use :meth:`status_stream_shared` instead.

        Slow consumers automatically skip to the latest state (desired for real-time).
        """
        async for status in self.status_stream_shared():
            yield status.copy()

    async def status_stream_shared(self) -> AsyncIterator[StatusBuffer]:
        """Zero-copy async generator that yields the shared status buffer.

        Usage:
            async for status in client.status_stream_shared():
                # Process immediately - don't store references
                print(f"Angles: {status.angles}")

        WARNING: The same buffer instance is yielded on every iteration.
        Do not store references to the yielded object - data will be
        overwritten on the next iteration. For safe storage, use
        :meth:`status_stream` or call status.copy().

        This generator terminates automatically when :meth:`close` is
        called on the client.

        Slow consumers automatically skip to the latest state (desired for real-time).
        """
        await self._ensure_endpoint()
        last_gen = 0

        while not self._closed:
            # Clear before waiting - only affects future waits, not current waiters
            self._status_event.clear()

            # Check if we already have new data (arrived between yield and clear)
            if self._status_generation != last_gen:
                last_gen = self._status_generation
                yield self._shared_status
                continue

            # Wait for next update
            await self._status_event.wait()

            if self._closed:
                break
            if self._status_generation != last_gen:
                last_gen = self._status_generation
                yield self._shared_status

    async def _send(self, cmd: msgspec.Struct) -> bool:
        """
        Send a binary command based on AckPolicy:
        - System commands: wait for server OK/ERROR, return True on OK, False on ERROR
        - Motion commands: wait iff policy requires ACK; otherwise fire-and-forget (return True on send)
        - Query commands: should use _request path; if invoked here, just fire-and-forget

        Args:
            cmd: Typed command struct
        """
        await self._ensure_endpoint()
        assert self._transport is not None

        # Get command type from struct's tag
        cmd_type = STRUCT_TO_CMDTYPE.get(type(cmd))
        if cmd_type is None:
            return False

        # Encode the struct
        data = encode_command(cmd)

        # System commands: wait for OK/ERROR
        if cmd_type in SYSTEM_CMD_TYPES:
            try:
                return await self._request_ok_raw(data, self.timeout)
            except RuntimeError:
                # Server rejected command
                return False

        # Motion and other non-query commands
        if cmd_type not in QUERY_CMD_TYPES:
            if self._ack_policy.requires_ack(cmd_type):
                try:
                    return await self._request_ok_raw(data, self.timeout)
                except RuntimeError:
                    return False
            # Fire-and-forget
            self._transport.sendto(data)
            return True

        # Queries: fire-and-forget here (query methods use _request())
        self._transport.sendto(data)
        return True

    async def _request(self, cmd: msgspec.Struct) -> ResponseMsg | None:
        """Send a request and wait for a response.

        Args:
            cmd: Typed command struct

        Returns:
            ResponseMsg with query_type and value, or None on timeout/error.
        """
        await self._ensure_endpoint()
        assert self._transport is not None
        data = encode_command(cmd)
        for attempt in range(self.retries + 1):
            try:
                async with self._req_lock:
                    self._transport.sendto(data)
                    resp_data, _ = await asyncio.wait_for(
                        self._rx_queue.get(), timeout=self.timeout
                    )
                    try:
                        raw = msgspec.msgpack.decode(resp_data)
                        parsed = parse_message(raw)
                        if isinstance(parsed, ResponseMsg):
                            return parsed
                        return None
                    except Exception:
                        return None
            except (asyncio.TimeoutError, TimeoutError):
                if attempt < self.retries:
                    backoff = min(0.5, 0.05 * (2**attempt)) + random.uniform(0, 0.05)
                    await asyncio.sleep(backoff)
                    continue
            except Exception:
                break
        return None

    async def _request_ok_raw(self, data: bytes, timeout: float) -> bool:
        """
        Send pre-encoded binary command and wait for 'OK' or 'ERROR' reply.

        Args:
            data: Pre-encoded msgpack bytes
            timeout: Timeout in seconds.

        Returns True on OK; raises RuntimeError on ERROR, TimeoutError on timeout.
        """
        await self._ensure_endpoint()
        assert self._transport is not None

        end_time = time.monotonic() + timeout
        async with self._req_lock:
            self._transport.sendto(data)
            while time.monotonic() < end_time:
                try:
                    resp_data, _addr = await asyncio.wait_for(
                        self._rx_queue.get(),
                        timeout=max(0.0, end_time - time.monotonic()),
                    )
                    try:
                        resp = msgspec.msgpack.decode(resp_data)
                        match parse_message(resp):
                            case OkMsg():
                                return True
                            case ErrorMsg(message):
                                raise RuntimeError(f"ERROR|{message}")
                    except msgspec.DecodeError:
                        pass  # Ignore non-msgpack datagrams
                except (asyncio.TimeoutError, TimeoutError):
                    break
        raise TimeoutError("Timeout waiting for OK")

    # --------------- Motion / Control ---------------

    async def home(self, wait: bool = False, **wait_kwargs) -> bool:
        """Home the robot to its home position.

        Args:
            wait: If True, block until motion completes
            **wait_kwargs: Arguments passed to wait_motion_complete() (timeout, settle_window, etc.)
        """
        result = await self._send(HomeCmd())
        if wait and result:
            await self.wait_motion_complete(**wait_kwargs)
        return result

    async def enable(self) -> bool:
        """Enable the robot controller, allowing motion commands.

        Returns:
            True if the command was acknowledged successfully.
        """
        return await self._send(EnableCmd())

    async def disable(self) -> bool:
        """Disable the robot controller, stopping all motion.

        Returns:
            True if the command was acknowledged successfully.
        """
        return await self._send(DisableCmd())

    async def stop(self) -> bool:
        """Alias for disable() - stops motion and disables controller."""
        return await self.disable()

    async def start(self) -> bool:
        """Alias for enable() - enables controller."""
        return await self.enable()

    async def stream_on(self) -> bool:
        """Enable streaming mode for high-frequency motion commands.

        In streaming mode, motion commands are sent without waiting for
        acknowledgment, allowing higher command rates for smooth motion.

        Returns:
            True if the command was acknowledged successfully.
        """
        self._stream_mode = True
        return await self._send(StreamCmd(on=True))

    async def stream_off(self) -> bool:
        """Disable streaming mode, returning to acknowledged motion commands.

        Returns:
            True if the command was acknowledged successfully.
        """
        self._stream_mode = False
        return await self._send(StreamCmd(on=False))

    async def simulator_on(self) -> bool:
        """Enable simulator mode (no physical robot hardware required).

        The controller will use simulated robot dynamics instead of
        communicating with real hardware over serial.

        Returns:
            True if the command was acknowledged successfully.
        """
        return await self._send(SimulatorCmd(on=True))

    async def simulator_off(self) -> bool:
        """Disable simulator mode, switching to real hardware communication.

        Returns:
            True if the command was acknowledged successfully.
        """
        return await self._send(SimulatorCmd(on=False))

    async def set_serial_port(self, port_str: str) -> bool:
        """Set the serial port for robot hardware communication.

        Args:
            port_str: Serial port path (e.g., '/dev/ttyUSB0' or 'COM3').

        Returns:
            True if the command was acknowledged successfully.

        Raises:
            ValueError: If port_str is empty.
        """
        if not port_str:
            raise ValueError("No port provided")
        return await self._send(SetPortCmd(port_str=port_str))

    async def reset(self) -> bool:
        """Reset controller state to initial values.

        Instantly resets positions to home, clears queues, resets tool/errors.
        Preserves serial connection. Useful for fast test isolation.
        """
        return await self._send(ResetCmd())

    # --------------- Status / Queries ---------------
    async def ping(self) -> PingResult | None:
        """Return parsed ping result with serial_connected status."""
        resp = await self._request(PingCmd())
        if resp is None:
            return None
        serial = int(resp.value) if resp.value is not None else 0
        return PingResult(serial_connected=bool(serial), raw=str(resp))

    async def get_angles(self) -> list[float] | None:
        """Get current joint angles in degrees [J1, J2, J3, J4, J5, J6]."""
        resp = await self._request(GetAnglesCmd())
        return cast(list[float], resp.value) if resp else None

    async def get_io(self) -> list[int] | None:
        """Get digital I/O status [in1, in2, out1, out2, estop]."""
        resp = await self._request(GetIOCmd())
        return cast(list[int], resp.value) if resp else None

    async def get_gripper_status(self) -> list[int] | None:
        """Get electric gripper status [id, pos, speed, current, status, obj_detected]."""
        resp = await self._request(GetGripperCmd())
        return cast(list[int], resp.value) if resp else None

    async def get_speeds(self) -> list[float] | None:
        """Get current joint speeds in steps/sec [J1, J2, J3, J4, J5, J6]."""
        resp = await self._request(GetSpeedsCmd())
        return cast(list[float], resp.value) if resp else None

    async def get_pose(
        self, frame: Literal["WRF", "TRF"] = "WRF"
    ) -> list[float] | None:
        """Get 16-element transformation matrix (flattened) with translation in mm."""
        resp = await self._request(GetPoseCmd(frame=frame))
        return cast(list[float], resp.value) if resp else None

    async def get_gripper(self) -> list[int] | None:
        """Alias for get_gripper_status."""
        return await self.get_gripper_status()

    async def get_status(self) -> StatusResultStruct | None:
        """Get aggregate status (pose, angles, speeds, io, gripper)."""
        resp = await self._request(GetStatusCmd())
        return StatusResultStruct(*resp.value) if resp else None

    async def get_loop_stats(self) -> LoopStatsResultStruct | None:
        """Fetch control-loop runtime metrics."""
        resp = await self._request(GetLoopStatsCmd())
        return LoopStatsResultStruct(*resp.value) if resp else None

    async def reset_loop_stats(self) -> bool:
        """Reset control-loop min/max metrics and overrun count."""
        return await self._send(ResetLoopStatsCmd())

    async def set_tool(self, tool_name: str) -> bool:
        """
        Set the current end-effector tool configuration.

        Args:
            tool_name: Name of the tool ('NONE', 'PNEUMATIC', 'ELECTRIC')

        Returns:
            True if successful
        """
        return await self._send(SetToolCmd(tool_name=tool_name.upper()))

    async def set_profile(self, profile: str) -> bool:
        """
        Set the motion profile for all moves.

        Args:
            profile: Motion profile type ('TOPPRA', 'RUCKIG', 'QUINTIC', 'TRAPEZOID', 'LINEAR')
                Note: RUCKIG is point-to-point only; Cartesian moves will use TOPPRA.

        Returns:
            True if successful
        """
        return await self._send(SetProfileCmd(profile=profile.upper()))

    async def get_profile(self) -> str | None:
        """Get the current motion profile."""
        resp = await self._request(GetProfileCmd())
        return str(resp.value).upper() if resp and resp.value else None

    async def get_tool(self) -> ToolResultStruct | None:
        """Get the current tool and available tools."""
        resp = await self._request(GetToolCmd())
        if resp and isinstance(resp.value, (list, tuple)) and len(resp.value) >= 2:
            return ToolResultStruct(*resp.value)
        return None

    async def get_current_action(self) -> CurrentActionResultStruct | None:
        """Get the current executing action (current, state, next)."""
        resp = await self._request(GetCurrentActionCmd())
        if resp and isinstance(resp.value, (list, tuple)) and len(resp.value) >= 3:
            return CurrentActionResultStruct(*resp.value)
        return None

    async def get_queue(self) -> list | None:
        """Get the list of queued non-streamable commands."""
        resp = await self._request(GetQueueCmd())
        return cast(list, resp.value) if resp else None

    # --------------- Helper methods ---------------

    async def get_pose_rpy(self) -> list[float] | None:
        """
        Get robot pose as [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg] using RPY order='xyz'.
        """
        pose_matrix = await self.get_pose()
        if not pose_matrix or len(pose_matrix) != 16:
            return None

        try:
            x, y, z = pose_matrix[3], pose_matrix[7], pose_matrix[11]
            # Rotation matrix rows (row-major layout)
            R = np.array(
                [
                    [pose_matrix[0], pose_matrix[1], pose_matrix[2]],
                    [pose_matrix[4], pose_matrix[5], pose_matrix[6]],
                    [pose_matrix[8], pose_matrix[9], pose_matrix[10]],
                ]
            )
            # Use xyz convention (rx, ry, rz) - Roll-Pitch-Yaw
            rpy_rad = np.zeros(3, dtype=np.float64)
            so3_rpy(R, rpy_rad)
            rpy_deg = np.degrees(rpy_rad)
            return [x, y, z, rpy_deg[0], rpy_deg[1], rpy_deg[2]]
        except (ValueError, IndexError, ImportError):
            return None

    async def get_pose_xyz(self) -> list[float] | None:
        """Get robot position as [x, y, z] in mm."""
        pose_rpy = await self.get_pose_rpy()
        return pose_rpy[:3] if pose_rpy else None

    async def is_estop_pressed(self) -> bool:
        """Check if E-stop is pressed. Returns True if pressed."""
        io_status = await self.get_io()
        if io_status and len(io_status) >= 5:
            return io_status[4] == 0  # E-stop at index 4, 0 means pressed
        return False

    async def is_robot_stopped(self, threshold_speed: float = 2.0) -> bool:
        """
        Check if robot has stopped moving.

        Args:
            threshold_speed: Speed threshold in steps/sec

        Returns:
            True if all joints below threshold
        """
        speeds = await self.get_speeds()
        if not speeds:
            return False
        return max(abs(s) for s in speeds) < threshold_speed

    async def wait_motion_complete(
        self,
        timeout: float = 10.0,
        settle_window: float = 0.25,
        speed_threshold: float = 2.0,
        angle_threshold: float = 0.5,
        motion_start_timeout: float = 1.0,
    ) -> bool:
        """
        Wait for robot to stop moving using multicast status broadcasts.

        This method first waits for motion to START (speeds above threshold),
        then waits for motion to COMPLETE (speeds below threshold for settle_window).
        This avoids a race condition where the method returns immediately if
        called before motion has begun.

        Args:
            timeout: Maximum time to wait in seconds
            settle_window: How long robot must be stable to be considered stopped
            speed_threshold: Max joint speed to be considered stopped (steps/sec)
            angle_threshold: Max angle change to be considered stopped (degrees)
            motion_start_timeout: Max time to wait for motion to start (seconds)

        Returns:
            True if robot stopped, False if timeout
        """
        await self._ensure_endpoint()

        last_angles: np.ndarray | None = None
        settle_start: float | None = None
        motion_started = False
        start_time = time.monotonic()

        try:
            async with asyncio.timeout(timeout):
                async for status in self.status_stream_shared():
                    speeds = status.speeds
                    angles = status.angles

                    max_speed = float(np.abs(speeds).max())

                    max_angle_change = 0.0
                    if last_angles is not None:
                        max_angle_change = float(np.abs(angles - last_angles).max())
                        np.copyto(last_angles, angles)
                    else:
                        last_angles = angles.copy()

                    now = time.monotonic()

                    # Phase 1: Wait for motion to start
                    if not motion_started:
                        if (
                            max_speed >= speed_threshold
                            or max_angle_change >= angle_threshold
                        ):
                            motion_started = True
                            settle_start = None
                        elif now - start_time > motion_start_timeout:
                            motion_started = True

                    # Phase 2: Wait for motion to complete
                    if motion_started:
                        if (
                            max_speed < speed_threshold
                            and max_angle_change < angle_threshold
                        ):
                            if settle_start is None:
                                settle_start = now
                            elif now - settle_start > settle_window:
                                return True
                        else:
                            settle_start = None
        except TimeoutError:
            return False

        return False

    # --------------- Additional waits and utilities ---------------

    async def wait_for_server_ready(
        self, timeout: float = 5.0, interval: float = 0.05
    ) -> bool:
        """Poll ping() until server responds or timeout.

        Args:
            timeout: Maximum time to wait for server to respond
            interval: Polling interval between ping attempts

        Returns:
            True if server responded to PING, False on timeout
        """
        end_time = time.monotonic() + timeout
        while time.monotonic() < end_time:
            result = await self.ping()
            if result:
                return True
            await asyncio.sleep(interval)
        return False

    async def wait_for_status(
        self, predicate: Callable[[StatusBuffer], bool], timeout: float = 5.0
    ) -> bool:
        """Wait until a multicast status satisfies predicate(status) within timeout."""
        await self._ensure_endpoint()
        last_gen = 0
        end_time = time.monotonic() + timeout

        while time.monotonic() < end_time and not self._closed:
            self._status_event.clear()

            # Check if we already have new data
            if self._status_generation != last_gen:
                last_gen = self._status_generation
                try:
                    if predicate(self._shared_status):
                        return True
                except Exception:
                    pass
                continue

            # Wait for next update with timeout
            remaining = max(0.0, end_time - time.monotonic())
            if remaining <= 0:
                return False
            try:
                await asyncio.wait_for(
                    self._status_event.wait(),
                    timeout=min(remaining, 0.5),
                )
            except asyncio.TimeoutError:
                continue

            if self._closed:
                return False
            if self._status_generation != last_gen:
                last_gen = self._status_generation
                try:
                    if predicate(self._shared_status):
                        return True
                except Exception:
                    pass
        return False

    # --------------- Motion encoders ---------------

    async def move_joints(
        self,
        joint_angles: list[float],
        duration: float | None = None,
        speed: int | None = None,
        accel: int | None = None,
        wait: bool = False,
        **wait_kwargs,
    ) -> bool:
        """Move to specified joint angles.

        Args:
            joint_angles: Target joint angles in degrees
            duration: Time to complete motion in seconds
            speed: Speed as percentage (1-100)
            accel: Acceleration as percentage (1-100)
            wait: If True, block until motion completes
            **wait_kwargs: Arguments passed to wait_motion_complete()
        """
        if duration is None and speed is None:
            raise RuntimeError("You must provide either a duration or a speed.")
        cmd = MoveJointCmd(
            angles=joint_angles,
            duration=duration if duration else 0.0,
            speed_pct=float(speed) if speed else 0.0,
            accel_pct=float(accel) if accel else 100.0,
        )
        result = await self._send(cmd)
        if wait and result:
            await self.wait_motion_complete(**wait_kwargs)
        return result

    async def move_pose(
        self,
        pose: list[float],
        duration: float | None = None,
        speed: int | None = None,
        accel: int | None = None,
        wait: bool = False,
        **wait_kwargs,
    ) -> bool:
        """Move to specified pose using joint-space interpolation.

        Args:
            pose: Target pose [x, y, z, rx, ry, rz] in mm and degrees
            duration: Time to complete motion in seconds
            speed: Speed as percentage (1-100)
            accel: Acceleration as percentage (1-100)
            wait: If True, block until motion completes
            **wait_kwargs: Arguments passed to wait_motion_complete()
        """
        if duration is None and speed is None:
            raise RuntimeError("You must provide either a duration or a speed.")
        cmd = MovePoseCmd(
            pose=pose,
            duration=duration if duration else 0.0,
            speed_pct=float(speed) if speed else 0.0,
            accel_pct=float(accel) if accel else 100.0,
        )
        result = await self._send(cmd)
        if wait and result:
            await self.wait_motion_complete(**wait_kwargs)
        return result

    async def move_cartesian(
        self,
        pose: list[float],
        duration: float | None = None,
        speed: float | None = None,
        accel: float | None = None,
        wait: bool = False,
        **wait_kwargs,
    ) -> bool:
        """Move to specified pose using Cartesian-space interpolation.

        Args:
            pose: Target pose [x, y, z, rx, ry, rz] in mm and degrees
            duration: Time to complete motion in seconds
            speed: Speed as percentage (1-100)
            accel: Acceleration as percentage (1-100)
            wait: If True, block until motion completes
            **wait_kwargs: Arguments passed to wait_motion_complete()
        """
        if duration is None and speed is None:
            raise RuntimeError("You must provide either a duration or a speed.")
        cmd = MoveCartCmd(
            pose=pose,
            duration=duration if duration else 0.0,
            speed_pct=float(speed) if speed else 0.0,
            accel_pct=float(accel) if accel else 100.0,
        )
        result = await self._send(cmd)
        if wait and result:
            await self.wait_motion_complete(**wait_kwargs)
        return result

    async def move_cartesian_rel_trf(
        self,
        deltas: list[float],  # [dx, dy, dz, rx, ry, rz]
        duration: float | None = None,
        speed: float | None = None,
        accel: int | None = None,
        wait: bool = False,
        **wait_kwargs,
    ) -> bool:
        """Send a MOVECARTRELTRF (relative straight-line in TRF) command.

        Args:
            deltas: Relative movement [dx, dy, dz, rx, ry, rz] in mm and degrees
            duration: Time to complete motion in seconds
            speed: Speed as percentage (1-100)
            accel: Acceleration as percentage (1-100)
            wait: If True, block until motion completes
            **wait_kwargs: Arguments passed to wait_motion_complete()
        """
        if duration is None and speed is None:
            raise RuntimeError("Error: You must provide either a duration or a speed.")
        cmd = MoveCartRelTrfCmd(
            deltas=deltas,
            duration=duration if duration else 0.0,
            speed_pct=float(speed) if speed else 0.0,
            accel_pct=float(accel) if accel else 100.0,
        )
        result = await self._send(cmd)
        if wait and result:
            await self.wait_motion_complete(**wait_kwargs)
        return result

    async def jog_joint(
        self,
        joint_index: int,
        speed: int,
        duration: float,
    ) -> bool:
        """Jog a single joint at a specified speed for a duration.

        Args:
            joint_index: Joint to jog (0-5 for positive direction,
                6-11 for negative/reverse direction).
            speed: Speed as percentage (1-100).
            duration: Time to jog in seconds.

        Returns:
            True if the command was sent successfully.
        """
        cmd = JogCmd(
            joint=joint_index,
            speed_pct=float(speed),
            duration=duration,
        )
        return await self._send(cmd)

    async def jog_cartesian(
        self,
        frame: Frame,
        axis: Axis,
        speed: int,
        duration: float,
    ) -> bool:
        """Jog the robot in Cartesian space along a specified axis.

        Args:
            frame: Reference frame ('TRF' for Tool, 'WRF' for World).
            axis: Axis and direction to jog (e.g., 'X+', 'X-', 'Y+', 'RZ-').
            speed: Speed as percentage (1-100).
            duration: Time to jog in seconds.

        Returns:
            True if the command was sent successfully.
        """
        cmd = CartJogCmd(
            frame=frame,
            axis=axis,
            speed_pct=float(speed),
            duration=duration,
        )
        return await self._send(cmd)

    async def jog_multiple(
        self,
        joints: list[int],
        speeds: list[float],
        duration: float,
    ) -> bool:
        """Jog multiple joints simultaneously.

        Args:
            joints: List of joint indices to jog (0-5).
            speeds: List of speeds for each joint (percentage, can be negative).
            duration: Time to jog in seconds.

        Returns:
            True if the command was sent successfully.

        Raises:
            ValueError: If joints and speeds lists have different lengths.
        """
        if len(joints) != len(speeds):
            raise ValueError(
                "Error: The number of joints must match the number of speeds."
            )
        cmd = MultiJogCmd(joints=joints, speeds=speeds, duration=duration)
        return await self._send(cmd)

    async def set_io(self, index: int, value: int) -> bool:
        """Set a digital I/O output bit.

        Args:
            index: Output index (0-7).
            value: Output value (0 or 1).

        Returns:
            True if the command was sent successfully.

        Raises:
            ValueError: If index is not 0-7 or value is not 0 or 1.
        """
        if index < 0 or index > 7:
            raise ValueError("I/O index must be 0..7")
        if value not in (0, 1):
            raise ValueError("I/O value must be 0 or 1")
        cmd = SetIOCmd(port_index=index, value=value)
        return await self._send(cmd)

    async def delay(self, seconds: float) -> bool:
        """Insert a non-blocking delay in the motion queue.

        The delay is queued with other motion commands and executes
        in sequence without blocking the client.

        Args:
            seconds: Delay duration in seconds (must be positive).

        Returns:
            True if the command was sent successfully.

        Raises:
            ValueError: If seconds is not positive.
        """
        if seconds <= 0:
            raise ValueError("Delay must be positive")
        cmd = DelayCmd(seconds=seconds)
        return await self._send(cmd)

    # --------------- IO / Gripper ---------------

    async def control_pneumatic_gripper(
        self, action: str, port: int, wait: bool = False, **wait_kwargs
    ) -> bool:
        """Control pneumatic gripper via digital outputs.

        Args:
            action: 'open' or 'close'
            port: 1 or 2
            wait: If True, block until motion completes
            **wait_kwargs: Arguments passed to wait_motion_complete()
        """
        action = action.lower()
        if action not in ("open", "close"):
            raise ValueError("Invalid pneumatic action")
        if port not in (1, 2):
            raise ValueError("Invalid pneumatic port")
        cmd = PneumaticGripperCmd(open=(action == "open"), port=port)
        result = await self._send(cmd)
        if wait and result:
            await self.wait_motion_complete(**wait_kwargs)
        return result

    async def control_electric_gripper(
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
            action: 'move' or 'calibrate'
            position: 0..255
            speed: 1..255
            current: 100..1000 (mA)
            wait: If True, block until motion completes
            **wait_kwargs: Arguments passed to wait_motion_complete()
        """
        action = action.lower()
        if action not in ("move", "calibrate"):
            raise ValueError("Invalid electric gripper action")
        pos = 0 if position is None else int(position)
        spd = 1 if speed is None or speed <= 0 else int(speed)
        cur = 100 if current is None else int(current)
        cmd = ElectricGripperCmd(
            calibrate=(action == "calibrate"),
            position=pos,
            speed=spd,
            current=cur,
        )
        result = await self._send(cmd)
        if wait and result:
            await self.wait_motion_complete(**wait_kwargs)
        return result

    # --------------- GCODE ---------------

    async def execute_gcode(self, gcode_line: str) -> bool:
        """Execute a single G-code line.

        Args:
            gcode_line: G-code command to execute (e.g., 'G0 X100 Y50').

        Returns:
            True if the command was sent successfully.
        """
        cmd = GcodeCmd(line=gcode_line)
        return await self._send(cmd)

    async def execute_gcode_program(self, program_lines: list[str]) -> bool:
        """Execute a G-code program from a list of lines.

        Args:
            program_lines: List of G-code lines to execute sequentially.

        Returns:
            True if the command was sent successfully.
        """
        cmd = GcodeProgramCmd(lines=program_lines)
        return await self._send(cmd)

    async def load_gcode_file(self, filepath: str) -> bool:
        """Load and execute a G-code program from a file.

        Args:
            filepath: Path to the G-code file on the controller.

        Returns:
            True if the command was sent successfully.
        """
        # Read file and send as program
        try:
            with open(filepath) as f:
                lines = [line.strip() for line in f if line.strip()]
            return await self.execute_gcode_program(lines)
        except OSError:
            return False

    async def get_gcode_status(self) -> GcodeStatusResultStruct | None:
        """Get the current status of the G-code interpreter."""
        resp = await self._request(GetGcodeStatusCmd())
        if resp and isinstance(resp.value, (list, tuple)) and len(resp.value) >= 5:
            return GcodeStatusResultStruct(*resp.value)
        return None

    async def pause_gcode_program(self) -> bool:
        """Pause the currently running GCODE program."""
        return await self._send(GcodePauseCmd())

    async def resume_gcode_program(self) -> bool:
        """Resume a paused GCODE program."""
        return await self._send(GcodeResumeCmd())

    async def stop_gcode_program(self) -> bool:
        """Stop the currently running GCODE program."""
        return await self._send(GcodeStopCmd())

    # --------------- Smooth motion ---------------
    async def smooth_circle(
        self,
        center: list[float],
        radius: float,
        plane: Literal["XY", "XZ", "YZ"] = "XY",
        frame: Literal["WRF", "TRF"] = "WRF",
        center_mode: Literal["ABSOLUTE", "TOOL", "RELATIVE"] = "ABSOLUTE",
        duration: float | None = None,
        velocity_percent: float | None = None,
        accel_percent: float | None = None,
        clockwise: bool = False,
        wait: bool = False,
        **wait_kwargs,
    ) -> bool:
        """Execute a smooth circular motion.

        Args:
            center: Center point [x, y, z] in mm
            radius: Circle radius in mm
            plane: Motion plane ("XY", "XZ", or "YZ")
            frame: Reference frame ("WRF" or "TRF")
            center_mode: How center is interpreted ("ABSOLUTE", "TOOL", "RELATIVE")
            duration: Motion duration in seconds
            velocity_percent: Speed as percentage (1-100)
            accel_percent: Acceleration as percentage (1-100)
            clockwise: True for clockwise motion
            wait: If True, block until motion completes
        """
        cmd = SmoothCircleCmd(
            center=center,
            radius=radius,
            plane=plane,
            frame=frame,
            center_mode=center_mode,
            duration=duration,
            speed_pct=velocity_percent,
            accel_pct=accel_percent if accel_percent else 100.0,
            clockwise=clockwise,
        )
        result = await self._send(cmd)
        if wait and result:
            await self.wait_motion_complete(**wait_kwargs)
        return result

    async def smooth_arc_center(
        self,
        end_pose: list[float],
        center: list[float],
        frame: Literal["WRF", "TRF"] = "WRF",
        duration: float | None = None,
        velocity_percent: float | None = None,
        accel_percent: float | None = None,
        clockwise: bool = False,
        wait: bool = False,
        **wait_kwargs,
    ) -> bool:
        """Execute a smooth arc motion defined by center point.

        Args:
            end_pose: Target pose [x, y, z, rx, ry, rz] in mm and degrees
            center: Arc center [x, y, z] in mm
            frame: Reference frame ("WRF" or "TRF")
            duration: Motion duration in seconds
            velocity_percent: Speed as percentage (1-100)
            accel_percent: Acceleration as percentage (1-100)
            clockwise: True for clockwise motion
            wait: If True, block until motion completes
        """
        cmd = SmoothArcCenterCmd(
            end_pose=end_pose,
            center=center,
            frame=frame,
            duration=duration,
            speed_pct=velocity_percent,
            accel_pct=accel_percent if accel_percent else 100.0,
            clockwise=clockwise,
        )
        result = await self._send(cmd)
        if wait and result:
            await self.wait_motion_complete(**wait_kwargs)
        return result

    async def smooth_arc_param(
        self,
        end_pose: list[float],
        radius: float,
        arc_angle: float,
        frame: Literal["WRF", "TRF"] = "WRF",
        duration: float | None = None,
        velocity_percent: float | None = None,
        accel_percent: float | None = None,
        clockwise: bool = False,
        wait: bool = False,
        **wait_kwargs,
    ) -> bool:
        """Execute a smooth arc motion defined parametrically.

        Args:
            end_pose: Target pose [x, y, z, rx, ry, rz] in mm and degrees
            radius: Arc radius in mm
            arc_angle: Arc angle in degrees
            frame: Reference frame ("WRF" or "TRF")
            duration: Motion duration in seconds
            velocity_percent: Speed as percentage (1-100)
            accel_percent: Acceleration as percentage (1-100)
            clockwise: True for clockwise motion
            wait: If True, block until motion completes
        """
        cmd = SmoothArcParamCmd(
            end_pose=end_pose,
            radius=radius,
            arc_angle=arc_angle,
            frame=frame,
            duration=duration,
            speed_pct=velocity_percent,
            accel_pct=accel_percent if accel_percent else 100.0,
            clockwise=clockwise,
        )
        result = await self._send(cmd)
        if wait and result:
            await self.wait_motion_complete(**wait_kwargs)
        return result

    async def smooth_spline(
        self,
        waypoints: list[list[float]],
        frame: Literal["WRF", "TRF"] = "WRF",
        duration: float | None = None,
        velocity_percent: float | None = None,
        accel_percent: float | None = None,
        wait: bool = False,
        **wait_kwargs,
    ) -> bool:
        """Execute a smooth spline motion through waypoints.

        Args:
            waypoints: List of poses [[x, y, z, rx, ry, rz], ...] in mm and degrees
            frame: Reference frame ("WRF" or "TRF")
            duration: Motion duration in seconds
            velocity_percent: Speed as percentage (1-100)
            accel_percent: Acceleration as percentage (1-100)
            wait: If True, block until motion completes
        """
        cmd = SmoothSplineCmd(
            waypoints=waypoints,
            frame=frame,
            duration=duration,
            speed_pct=velocity_percent,
            accel_pct=accel_percent if accel_percent else 100.0,
        )
        result = await self._send(cmd)
        if wait and result:
            await self.wait_motion_complete(**wait_kwargs)
        return result

    # --------------- Work coordinate helpers ---------------

    async def set_work_coordinate_offset(
        self,
        coordinate_system: str,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
    ) -> bool:
        """
        Set work coordinate system offsets (G54-G59).

        Args:
            coordinate_system: Work coordinate system to set ('G54' through 'G59')
            x: X axis offset in mm (None to keep current)
            y: Y axis offset in mm (None to keep current)
            z: Z axis offset in mm (None to keep current)

        Returns:
            Success message, command ID, or dict with status details

        Example:
            # Set G54 origin to current position
            await client.set_work_coordinate_offset('G54', x=0, y=0, z=0)

            # Offset G55 by 100mm in X
            await client.set_work_coordinate_offset('G55', x=100)
        """
        valid_systems = ["G54", "G55", "G56", "G57", "G58", "G59"]
        if coordinate_system not in valid_systems:
            raise RuntimeError(
                f"Invalid coordinate system: {coordinate_system}. Must be one of {valid_systems}"
            )

        coord_num = int(coordinate_system[1:]) - 53  # G54=1, G55=2, etc.
        offset_params = []
        if x is not None:
            offset_params.append(f"X{x}")
        if y is not None:
            offset_params.append(f"Y{y}")
        if z is not None:
            offset_params.append(f"Z{z}")

        # Always select CS first, then apply offset if any
        ok = await self.execute_gcode(coordinate_system)
        if not ok:
            return False
        if offset_params:
            offset_cmd = f"G10 L2 P{coord_num} {' '.join(offset_params)}"
            return await self.execute_gcode(offset_cmd)
        return True

    async def zero_work_coordinates(
        self,
        coordinate_system: str = "G54",
    ) -> bool:
        """
        Set the current position as zero in the specified work coordinate system.

        Args:
            coordinate_system: Work coordinate system to zero ('G54' through 'G59')

        Returns:
            Success message, command ID, or dict with status details

        Example:
            # Set current position as origin in G54
            await client.zero_work_coordinates('G54')
        """
        # This sets the current position as 0,0,0 in the work coordinate system
        return await self.set_work_coordinate_offset(coordinate_system, x=0, y=0, z=0)
