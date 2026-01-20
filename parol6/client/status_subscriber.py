import asyncio
import contextlib
import logging
import socket
import struct
import time
from collections.abc import AsyncIterator

from parol6 import config as cfg
from parol6.protocol.wire import (
    StatusBuffer,
    decode_status_into,
)
from parol6.server.loop_timer import LoopMetrics, format_hz_summary

logger = logging.getLogger(__name__)


class UDPProtocol(asyncio.DatagramProtocol):
    """Protocol handler for UDP datagrams (multicast or unicast)."""

    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.transport = None

        # RX rate tracking with LoopMetrics
        self._metrics = LoopMetrics()
        self._metrics.configure(1.0 / cfg.STATUS_RATE_HZ, int(cfg.STATUS_RATE_HZ))

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        now = time.monotonic()
        self._metrics.tick(now)

        # Rate-limited debug log every 3s
        if self._metrics.should_log(now, 3.0):
            logger.debug("rx: %s", format_hz_summary(self._metrics))

        try:
            self.queue.put_nowait((data, addr))
        except asyncio.QueueFull:
            # Drop oldest packet if queue is full
            try:
                self.queue.get_nowait()
                self.queue.put_nowait((data, addr))
            except Exception:
                pass

    def error_received(self, exc):
        logger.error(f"Error received: {exc}")

    def connection_lost(self, exc):
        logger.info(f"Connection lost: {exc}")


def _create_multicast_socket(group: str, port: int, iface_ip: str) -> socket.socket:
    """Create and configure a multicast socket with loopback-first semantics and robust joins."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

    # Allow multiple listeners on same port; prefer SO_REUSEPORT where available
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except Exception:
        # Not available or not permitted on this platform; continue
        pass
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    # Bind to port (try wildcard first, then iface_ip)
    try:
        sock.bind(("", port))
    except OSError:
        sock.bind((iface_ip, port))

    # Helper to determine active NIC IP (no packets sent)
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

    # Join multicast group on specified interface (loopback preferred), with fallbacks
    try:
        mreq = struct.pack("=4s4s", socket.inet_aton(group), socket.inet_aton(iface_ip))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    except Exception:
        # Retry using primary NIC IP
        try:
            primary_ip = _detect_primary_ip()
            mreq = struct.pack(
                "=4s4s", socket.inet_aton(group), socket.inet_aton(primary_ip)
            )
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        except Exception:
            # Final fallback: INADDR_ANY variant
            mreq_any = struct.pack("=4sl", socket.inet_aton(group), socket.INADDR_ANY)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq_any)

    # Non-blocking for asyncio
    sock.setblocking(False)
    return sock


def _create_unicast_socket(port: int, host: str) -> socket.socket:
    """Create and configure a plain UDP socket for unicast reception.

    Binds to the provided host (default 127.0.0.1) and port with large RCVBUF.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except Exception:
        pass
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    try:
        sock.bind((host, port))
    except OSError:
        # Fallback to wildcard
        sock.bind(("", port))
    sock.setblocking(False)
    return sock


async def subscribe_status_into(
    buf: StatusBuffer,
    group: str | None = None,
    port: int | None = None,
    iface_ip: str | None = None,
) -> AsyncIterator[StatusBuffer]:
    """Zero-allocation status subscription - fills caller-provided buffer.

    This is the preferred API for high-frequency status consumers that want
    to avoid GC pressure. The caller provides their own StatusBuffer and
    this generator fills it in place on each iteration.

    WARNING: The same buffer instance is yielded on every iteration.
    Caller must process data before the next iteration overwrites it.

    Args:
        buf: Caller-owned StatusBuffer to fill with each status update
        group: Multicast group (default from config)
        port: UDP port (default from config)
        iface_ip: Interface IP for multicast (default from config)

    Yields:
        The same StatusBuffer instance, filled with new data each iteration

    Example:
        buf = StatusBuffer()
        async for _ in subscribe_status_into(buf):
            process(buf.angles)  # Must process before next iteration
    """
    group = group or cfg.MCAST_GROUP
    port = port or cfg.MCAST_PORT
    iface_ip = iface_ip or cfg.MCAST_IF

    logger.info(
        f"subscribe_status_into starting: transport={cfg.STATUS_TRANSPORT} group={group}, port={port}, iface_ip={iface_ip}"
    )

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[bytes, tuple[str, int]]] = asyncio.Queue(maxsize=100)

    # Create the socket based on configured transport
    if cfg.STATUS_TRANSPORT == "UNICAST":
        sock = _create_unicast_socket(port, cfg.STATUS_UNICAST_HOST)
    else:
        # Multicast socket bound to ("", port) will also receive unicast datagrams to that port
        sock = _create_multicast_socket(group, port, iface_ip)

    # Create the datagram endpoint with our protocol
    transport = None
    try:
        transport, _ = await loop.create_datagram_endpoint(
            lambda: UDPProtocol(queue), sock=sock
        )

        while True:
            try:
                # Wait for data with timeout
                data, addr = await asyncio.wait_for(queue.get(), timeout=2.0)
                text = data.decode("ascii", errors="ignore")

                # Zero-allocation path: fill caller's buffer
                if decode_status_into(text, buf):
                    yield buf

            except (asyncio.TimeoutError, TimeoutError):
                logger.warning(
                    f"No status received for 2s on {('unicast' if cfg.STATUS_TRANSPORT == 'UNICAST' else 'multicast')} {group}:{port} (iface={iface_ip})"
                )
                continue

    except (asyncio.CancelledError, GeneratorExit):
        # Normal shutdown - don't log
        pass
    except Exception as e:
        # Log unexpected errors, but not "Event loop is closed" during shutdown
        if "Event loop is closed" not in str(e):
            logger.error(f"Error in subscribe_status_into: {e}")
    finally:
        try:
            if transport:
                transport.close()
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass


async def subscribe_status(
    group: str | None = None,
    port: int | None = None,
    iface_ip: str | None = None,
) -> AsyncIterator[StatusBuffer]:
    """Async generator yielding status updates with owned data.

    Each yielded StatusBuffer is a fresh copy - safe to store or process
    asynchronously. For zero-allocation hot paths, use subscribe_status_into().

    Args:
        group: Multicast group (default from config)
        port: UDP port (default from config)
        iface_ip: Interface IP for multicast (default from config)

    Yields:
        StatusBuffer with copied array data (safe to store)
    """
    buf = StatusBuffer()
    async for _ in subscribe_status_into(buf, group, port, iface_ip):
        yield buf.copy()
