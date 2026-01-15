"""
Main controller for PAROL6 robot server.
"""

import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from parol6.ack_policy import AckPolicy
from parol6.commands.base import (
    CommandContext,
    ExecutionStatusCode,
    MotionCommand,
    QueryCommand,
    SystemCommand,
)
from parol6.gcode import GcodeInterpreter
from parol6.server.command_executor import CommandExecutor
from parol6.protocol.wire import CommandCode, unpack_rx_frame_into
from parol6.server.command_registry import create_command_from_parts, discover_commands
from parol6.server.state import ControllerState, StateManager
from parol6.server.status_broadcast import StatusBroadcaster
from parol6.server.async_logging import AsyncLogHandler
from parol6.server.loop_timer import LoopTimer
from parol6.server.status_cache import get_cache
from parol6.server.transport_manager import TransportManager
from parol6.server.transports.udp_transport import UDPTransport
from parol6.config import (
    TRACE,
    INTERVAL_S,
    MCAST_GROUP,
    MCAST_PORT,
    MCAST_IF,
    MCAST_TTL,
    STATUS_RATE_HZ,
    STATUS_STALE_S,
)

logger = logging.getLogger("parol6.server.controller")


@dataclass
class ControllerConfig:
    """Configuration for the controller."""

    udp_host: str = "0.0.0.0"
    udp_port: int = 5001
    serial_port: str | None = None
    serial_baudrate: int = 3000000
    loop_interval: float = INTERVAL_S
    estop_recovery_delay: float = 1.0
    auto_home: bool = False


class Controller:
    """
    Main controller that orchestrates all components of the PAROL6 server.

    This replaces the monolithic controller.py with a modular design:
    - State management via StateManager singleton
    - Transport abstraction for UDP and Serial
    - Command execution via CommandExecutor
    - Automatic command discovery and registration
    """

    def __init__(self, config: ControllerConfig):
        """
        Initialize the controller with configuration.

        Args:
            config: Configuration object for the controller
        """
        self.config = config
        self.running = False
        self.shutdown_event = threading.Event()
        self._initialized = False

        # Core components
        self.state_manager = StateManager()
        self.udp_transport: UDPTransport | None = None

        # ACK management
        self.ack_socket: socket.socket | None = None
        try:
            self.ack_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except Exception as e:
            logger.error(f"Failed to create ACK socket: {e}")

        # E-stop recovery
        self.estop_active: bool | None = (
            None  # None = unknown, True = active, False = released
        )

        # TX keepalive timeout
        self._tx_keepalive_s = float(os.getenv("PAROL6_TX_KEEPALIVE_S", "0.2"))

        # Thread for command processing
        self.command_thread: threading.Thread | None = None

        # GCODE interpreter
        self.gcode_interpreter = GcodeInterpreter()

        # Status services (updater + multicast broadcaster)
        self._status_updater: Any | None = None
        self._status_broadcaster: Any | None = None

        # Helper classes
        self._timer = LoopTimer(self.config.loop_interval)
        self._async_log = AsyncLogHandler()
        self._transport_mgr = TransportManager(
            shutdown_event=self.shutdown_event,
            serial_port=self.config.serial_port,
            serial_baudrate=self.config.serial_baudrate,
        )
        self._executor = CommandExecutor(
            state_manager=self.state_manager,
            gcode_interpreter=self.gcode_interpreter,
            udp_transport_getter=lambda: self.udp_transport,
            send_ack=self._send_ack,
            sync_mock_from_state=self._transport_mgr.sync_mock_from_state,
            dt=self.config.loop_interval,
        )

        # Periodic logging state
        self._prev_print_time = 0.0

        # Initialize components on construction
        self._initialize_components()

    def _send_ack(
        self, cmd_id: str, status: str, details: str, addr: tuple[str, int]
    ) -> None:
        """
        Send an acknowledgment message.

        Args:
            cmd_id: Command ID to acknowledge
            status: Status (QUEUED, EXECUTING, COMPLETED, FAILED, CANCELLED)
            details: Optional details message
            addr: Address tuple (host, port) to send to
        """
        if not cmd_id or not self.ack_socket:
            return

        # Debug/Trace log all outgoing ACKs
        logger.log(
            TRACE,
            "ack_send id=%s status=%s details=%s addr=%s",
            cmd_id,
            status,
            details,
            addr,
        )

        message = f"ACK|{cmd_id}|{status}|{details}".encode("ascii")

        try:
            self.ack_socket.sendto(message, addr)
        except Exception as e:
            logger.error(f"Failed to send ACK to {addr[0]}:{addr[1]} - {e}")

    def _initialize_components(self) -> None:
        """
        Initialize all components during construction.

        Raises:
            RuntimeError: If critical components fail to initialize
        """
        try:
            # Discover and register all commands
            discover_commands()

            # Initialize UDP transport
            logger.info(
                f"Starting UDP server on {self.config.udp_host}:{self.config.udp_port}"
            )
            self.udp_transport = UDPTransport(
                self.config.udp_host, self.config.udp_port
            )
            if not self.udp_transport.create_socket():
                raise RuntimeError("Failed to create UDP socket")

            # Initialize robot state
            self.state_manager.reset_state()
            state = self.state_manager.get_state()

            # Initialize serial transport via TransportManager
            self._transport_mgr.initialize(
                {
                    "Position_out": state.Position_out,
                    "Speed_out": state.Speed_out,
                    "Affected_joint_out": state.Affected_joint_out,
                    "InOut_out": state.InOut_out,
                    "Gripper_data_out": state.Gripper_data_out,
                }
            )

            # Optionally queue auto-home per policy (default OFF)
            if self.config.auto_home:
                try:
                    home_cmd, _ = create_command_from_parts(["HOME"])
                    if home_cmd:
                        # Queue without address/id for auto-home
                        self._executor.queue_command(("127.0.0.1", 0), home_cmd, None)
                        logger.info("Auto-home queued")
                except Exception as e:
                    logger.warning(f"Failed to queue auto-home: {e}")

            # Start status updater and broadcaster (ASCII multicast at configured rate)
            try:
                logger.info(
                    f"StatusBroadcaster config: group={MCAST_GROUP} port={MCAST_PORT} ttl={MCAST_TTL} iface={MCAST_IF} rate_hz={STATUS_RATE_HZ} stale_s={STATUS_STALE_S}"
                )
                broadcaster = StatusBroadcaster(
                    state_mgr=self.state_manager,
                    group=MCAST_GROUP,
                    port=MCAST_PORT,
                    ttl=MCAST_TTL,
                    iface_ip=MCAST_IF,
                    rate_hz=STATUS_RATE_HZ,
                    stale_s=STATUS_STALE_S,
                )

                broadcaster.start()
                logger.info("Status updater and broadcaster started")
            except Exception as e:
                logger.warning(f"Failed to start status services: {e}")

            self._initialized = True
            logger.info("Controller initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize controller: {e}")
            self._initialized = False
            raise RuntimeError(f"Controller initialization failed: {e}")

    def is_initialized(self) -> bool:
        """Check if controller is properly initialized."""
        return self._initialized

    def start(self):
        """Start the main control loop."""
        if self.running:
            logger.warning("Controller already running")
            return

        self.running = True

        # Start async logging to move I/O off the control loop thread
        self._async_log.start()

        # Start command processing thread
        self.command_thread = threading.Thread(target=self._command_processing_loop)
        self.command_thread.start()

        # Start main control loop
        logger.info("Starting main control loop")
        self._main_control_loop()

    def stop(self):
        """Stop the controller and clean up resources."""
        logger.info("Stopping controller...")
        self.running = False
        self.shutdown_event.set()

        # Wait for threads to finish
        if self.command_thread and self.command_thread.is_alive():
            self.command_thread.join(timeout=2.0)

        # Stop status services
        try:
            if self._status_broadcaster:
                self._status_broadcaster.stop()
            if self._status_updater:
                self._status_updater.stop()
        except Exception:
            pass

        # Clean up transports
        if self.udp_transport:
            self.udp_transport.close_socket()

        self._transport_mgr.disconnect()

        # Stop async logging (flushes queued messages)
        self._async_log.stop()

        logger.info("Controller stopped")

    def _read_from_firmware(self, state: ControllerState) -> None:
        """Phase 1: Read latest frame from serial transport and handle auto-reconnect."""
        if self._transport_mgr.is_connected():
            try:
                mv, ver, ts = self._transport_mgr.get_latest_frame()
                if mv is not None and ver != self._transport_mgr._last_version:
                    ok = unpack_rx_frame_into(
                        mv,
                        pos_out=state.Position_in,
                        spd_out=state.Speed_in,
                        homed_out=state.Homed_in,
                        io_out=state.InOut_in,
                        temp_out=state.Temperature_error_in,
                        poserr_out=state.Position_error_in,
                        timing_out=state.Timing_data_in,
                        grip_out=state.Gripper_data_in,
                    )
                    if ok:
                        get_cache().mark_serial_observed()
                        if not self._transport_mgr.first_frame_received:
                            self._transport_mgr.first_frame_received = True
                            logger.info("First frame received from robot")
                        self._transport_mgr._last_version = ver
            except Exception as e:
                logger.warning(f"Error decoding latest serial frame: {e}")

        # Serial auto-reconnect when a port is known
        self._transport_mgr.auto_reconnect()

    def _handle_estop(self, state: ControllerState) -> None:
        """Phase 2: Handle E-stop activation and recovery."""
        if not (
            self._transport_mgr.is_connected()
            and self._transport_mgr.first_frame_received
        ):
            return

        if state.InOut_in[4] == 0:  # E-stop pressed
            if not self.estop_active:
                logger.warning("E-STOP activated")
                self.estop_active = True
                if self._executor.active_command:
                    self._executor.cancel_active_command("E-Stop activated")
                self._executor.clear_queue("E-Stop activated")
                state.Command_out = CommandCode.DISABLE
                state.Speed_out.fill(0)
        elif state.InOut_in[4] == 1:  # E-stop released
            if self.estop_active:
                logger.info("E-STOP released - automatic recovery")
                self.estop_active = False
                state.enabled = True
                state.disabled_reason = ""
                state.Command_out = CommandCode.IDLE
                state.Speed_out.fill(0)

    def _execute_commands(self, state: ControllerState) -> None:
        """Phase 3: Execute active command or fetch from GCODE."""
        if self._executor.active_command or self._executor.command_queue:
            self._executor.execute_active_command()
        elif self.gcode_interpreter.is_running:
            self._executor.fetch_gcode_commands()
        else:
            state.Command_out = CommandCode.IDLE
            state.Speed_out.fill(0)
            np.copyto(state.Position_out, state.Position_in, casting="no")

    def _write_to_firmware(self, state: ControllerState) -> None:
        """Phase 4: Write state to serial transport if changed."""
        ok = self._transport_mgr.write_frame(
            state.Position_out,
            state.Speed_out,
            state.Command_out.value,
            state.Affected_joint_out,
            state.InOut_out,
            state.Timeout_out,
            state.Gripper_data_out,
            keepalive_s=self._tx_keepalive_s,
        )
        if ok:
            # Auto-reset one-shot gripper modes after successful send
            if state.Gripper_data_out[4] in (1, 2):
                state.Gripper_data_out[4] = 0

    def _sync_timer_metrics(self, state: ControllerState) -> None:
        """Copy timing metrics from LoopTimer to controller state."""
        state.loop_count = self._timer.metrics.loop_count
        state.overrun_count = self._timer.metrics.overrun_count
        state.last_period_s = self._timer.metrics.last_period_s
        state.ema_period_s = self._timer.metrics.ema_period_s

    def _log_periodic_status(self, state: ControllerState) -> None:
        """Log performance metrics every 5 seconds."""
        now = time.perf_counter()
        if now - self._prev_print_time <= 5:
            return

        self._prev_print_time = now
        tick = self._timer.interval

        # Warn if average period degraded >10% vs target
        if state.ema_period_s > tick * 1.10:
            logger.warning(
                f"Control loop avg period degraded by +{((state.ema_period_s / tick) - 1.0) * 100.0:.0f}% "
                f"(avg={state.ema_period_s:.4f}s target={tick:.4f}s)"
            )

        # Calculate command frequency
        cmd_hz = 0.0
        if state.ema_command_period_s > 0.0:
            cmd_hz = 1.0 / state.ema_command_period_s

        # Calculate short-term command rate from recent timestamps
        short_term_cmd_hz = 0.0
        if len(state.command_timestamps) >= 2:
            time_span = state.command_timestamps[-1] - state.command_timestamps[0]
            if time_span > 0:
                short_term_cmd_hz = (len(state.command_timestamps) - 1) / time_span

        logger.debug(
            "loop=%.2fHz cmd=%.2fHz s=%.2f/%d q=%d ov=%d",
            (1.0 / state.ema_period_s) if state.ema_period_s > 0.0 else 0.0,
            cmd_hz,
            short_term_cmd_hz,
            max(0, len(state.command_timestamps) - 1),
            state.command_count,
            state.overrun_count,
        )

    def _main_control_loop(self):
        """Main control loop with phase-based structure and precise timing."""
        self._timer.start()
        self._prev_print_time = time.perf_counter()

        while self.running:
            try:
                state = self.state_manager.get_state()

                self._read_from_firmware(state)
                self._handle_estop(state)

                if not self.estop_active:
                    self._execute_commands(state)

                self._write_to_firmware(state)

                self._sync_timer_metrics(state)
                self._log_periodic_status(state)
                self._timer.wait_for_next_tick()

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.stop()
                break
            except Exception as e:
                logger.error(f"Error in main control loop: {e}", exc_info=True)

    def _make_command_context(self, addr: tuple[str, int]) -> CommandContext:
        """Create a CommandContext for command execution."""
        return CommandContext(
            udp_transport=self.udp_transport,
            addr=addr,
            gcode_interpreter=self.gcode_interpreter,
            dt=self.config.loop_interval,
        )

    def _try_stream_fast_path(
        self,
        cmd_parts: list[str],
        cmd_name: str,
        addr: tuple[str, int],
        state: ControllerState,
        policy: AckPolicy,
        cmd_str: str,
    ) -> bool:
        """Attempt stream fast-path for active streamable command.

        Returns True if command was handled via fast-path (caller should continue to next).
        """
        active_cmd = self._executor.active_command
        if not (state.stream_mode and active_cmd):
            return False

        logger.log(
            TRACE,
            "stream_fast_path_considered active=%s incoming=%s",
            type(active_cmd.command).__name__,
            cmd_name,
        )

        active_inst = active_cmd.command
        if not (isinstance(active_inst, MotionCommand) and active_inst.streamable):
            return False

        active_name = active_inst._registered_name
        if active_name != cmd_name:
            return False

        can_handle, match_err = active_inst.match(cmd_parts)
        if can_handle:
            try:
                active_inst.bind(self._make_command_context(addr))
                active_inst.setup(state)
                logger.log(TRACE, "stream_fast_path_applied name=%s", active_name)
                return True
            except Exception as e:
                logger.error("Stream fast-path setup failed for %s: %s", active_name, e)
        elif match_err:
            if self.udp_transport and policy.requires_ack(cmd_str):
                self.udp_transport.send_response(f"ERROR|{match_err}", addr)
            logger.log(
                TRACE,
                "Stream fast-path match failed for %s: %s",
                active_name,
                match_err,
            )

        return False

    def _handle_set_port(self, port: str) -> None:
        """Handle SET_PORT command side-effect."""
        self.config.serial_port = port
        self._transport_mgr.switch_to_port(port)

    def _handle_simulator_toggle(
        self, mode: str, state: ControllerState, addr: tuple[str, int]
    ) -> bool:
        """Handle SIMULATOR command side-effect.

        Returns False if an error response was sent and caller should skip OK response.
        """
        enable = mode in ("on", "1", "true", "yes")

        # Pre-switch safety
        try:
            state.Command_out = CommandCode.IDLE
            state.Speed_out.fill(0)
            self._executor.cancel_active_command("Simulator mode toggle")
            self._executor.clear_queue("Simulator mode toggle")
        except Exception as e:
            logger.debug("Simulator toggle pre-switch safety failed: %s", e)

        success, error = self._transport_mgr.switch_simulator_mode(
            enable, sync_state=state
        )
        if not success and error:
            if self.udp_transport:
                self.udp_transport.send_response(f"ERROR|{error}", addr)
            return False

        return True

    def _handle_system_command(
        self, command: SystemCommand, state: ControllerState, addr: tuple[str, int]
    ) -> bool:
        """Execute system command and handle side-effects.

        Returns False if caller should skip sending OK response (error already sent).
        """
        command.bind(self._make_command_context(addr))
        logger.log(TRACE, "syscmd_execute_start name=%s", type(command).__name__)

        command.setup(state)
        status = command.tick(state)

        logger.log(
            TRACE,
            "syscmd_execute_%s name=%s msg=%s",
            "ok" if status.code == ExecutionStatusCode.COMPLETED else "err",
            type(command).__name__,
            status.message,
        )

        # Handle side-effects
        try:
            if status.details and isinstance(status.details, dict):
                if "serial_port" in status.details:
                    port = status.details.get("serial_port")
                    if port:
                        self._handle_set_port(port)

                if "simulator_mode" in status.details:
                    mode = str(status.details.get("simulator_mode", "")).lower()
                    if not self._handle_simulator_toggle(mode, state, addr):
                        return False
        except Exception as e:
            logger.debug(f"System command side-effect handling failed: {e}")

        # Send response
        if self.udp_transport:
            if status.code == ExecutionStatusCode.COMPLETED:
                self.udp_transport.send_response("OK", addr)
            else:
                msg = status.message or "Failed"
                self.udp_transport.send_response(f"ERROR|{msg}", addr)

        return True

    def _handle_query_command(
        self, command: QueryCommand, state: ControllerState, addr: tuple[str, int]
    ) -> None:
        """Execute query command immediately."""
        command.bind(self._make_command_context(addr))
        command.setup(state)
        command.tick(state)

    def _prepare_stream_mode(self, command: MotionCommand) -> None:
        """Prepare for stream mode by clearing stale commands."""
        if self.udp_transport:
            drained = self.udp_transport.drain_buffer()
            if drained > 0:
                logger.log(TRACE, "udp_buffer_drained count=%d", drained)

        # Cancel active streamable command
        ac = self._executor.active_command
        if (
            ac
            and isinstance(ac.command, MotionCommand)
            and getattr(ac.command, "streamable", False)
        ):
            self._executor.active_command = None

        # Clear queued streamable commands
        removed = 0
        for queued_cmd in list(self._executor.command_queue):
            if isinstance(queued_cmd.command, MotionCommand) and getattr(
                queued_cmd.command, "streamable", False
            ):
                self._executor.command_queue.remove(queued_cmd)
                removed += 1
        if removed:
            logger.log(TRACE, "queued_streamables_removed count=%d", removed)

    def _update_command_metrics(self, state: ControllerState) -> None:
        """Update command reception frequency metrics."""
        now = time.perf_counter()
        if state.last_command_time > 0:
            period = now - state.last_command_time
            state.last_command_period_s = period
            if state.ema_command_period_s <= 0.0:
                state.ema_command_period_s = period
            else:
                state.ema_command_period_s = (
                    0.1 * period + 0.9 * state.ema_command_period_s
                )

        state.last_command_time = now
        state.command_count += 1
        state.command_timestamps.append(now)

    def _command_processing_loop(self):
        """Separate thread for processing incoming commands from UDP."""
        while self.running and self.udp_transport:
            try:
                tup = self.udp_transport.receive_one()
                if tup is None:
                    continue

                cmd_str, addr = tup
                try:
                    cmd_name = (
                        cmd_str.split("|", 1)[0].upper()
                        if isinstance(cmd_str, str)
                        else "UNKNOWN"
                    )
                except Exception:
                    cmd_name = "UNKNOWN"

                logger.log(
                    TRACE,
                    "cmd_received name=%s from=%s cmd_str=%s",
                    cmd_name,
                    addr,
                    cmd_str,
                )

                state = self.state_manager.get_state()
                self._update_command_metrics(state)

                cmd_parts = cmd_str.split("|")
                cmd_name = cmd_parts[0].upper()
                policy = AckPolicy.from_env(lambda: state.stream_mode)

                # Try stream fast-path
                if self._try_stream_fast_path(
                    cmd_parts, cmd_name, addr, state, policy, cmd_str
                ):
                    continue

                # Create command instance
                command, error = create_command_from_parts(cmd_parts)
                if not command:
                    if error:
                        logger.warning(
                            f"Command validation failed: {cmd_str} - {error}"
                        )
                        if self.udp_transport:
                            self.udp_transport.send_response(f"ERROR|{error}", addr)
                    else:
                        logger.warning(f"Unknown command: {cmd_str}")
                        if self.udp_transport:
                            self.udp_transport.send_response(
                                "ERROR|Unknown command", addr
                            )
                    continue

                # Handle by command type
                if isinstance(command, SystemCommand):
                    self._handle_system_command(command, state, addr)
                    continue

                if isinstance(command, MotionCommand) and not state.enabled:
                    if self.udp_transport and policy.requires_ack(cmd_str):
                        reason = state.disabled_reason or "Controller disabled"
                        self.udp_transport.send_response(f"ERROR|{reason}", addr)
                    logger.warning(
                        f"Motion command rejected - controller disabled: {cmd_name}"
                    )
                    continue

                if isinstance(command, QueryCommand):
                    self._handle_query_command(command, state, addr)
                    continue

                # Handle streamable motion commands in stream mode
                if (
                    state.stream_mode
                    and isinstance(command, MotionCommand)
                    and command.streamable
                ):
                    self._prepare_stream_mode(command)

                # Queue the command
                status = self._executor.queue_command(addr, command, None)
                logger.log(
                    TRACE, "Command %s queued with status: %s", cmd_name, status.code
                )

                # ACK for motion commands
                if isinstance(command, MotionCommand) and self.udp_transport:
                    if policy.requires_ack(cmd_str):
                        if status.code == ExecutionStatusCode.QUEUED:
                            self.udp_transport.send_response("OK", addr)
                        else:
                            msg = status.message or "Queue error"
                            self.udp_transport.send_response(f"ERROR|{msg}", addr)

            except Exception as e:
                logger.error(f"Error in command processing: {e}", exc_info=True)
