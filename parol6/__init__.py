"""
PAROL6 Python Package

A unified library for controlling PAROL6 robot arms with async-first UDP client,
optional sync wrapper, and server management capabilities.

Key components:
- AsyncRobotClient: Async UDP client for robot operations
- RobotClient: Sync wrapper with automatic event loop handling
- ServerManager: Manages headless controller process lifecycle
- manage_server: Convenience function to start a controller process
- is_server_running: Helper to probe for an existing controller
"""
import multiprocessing
import sys

# Force multiprocessing to use 'spawn' instead of 'fork' on all platforms
# This prevents fork-safety issues with C extensions (numpy, scipy, robotics-toolbox)
# that initialize thread pools at import time. Must be set before any imports that
# use multiprocessing or before the first Process() is created.
if sys.platform.startswith("linux") or sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set (e.g., by test framework or parent process)

from . import PAROL6_ROBOT
from ._version import __version__
from .client.async_client import AsyncRobotClient
from .client.manager import ServerManager, is_server_running, manage_server
from .client.sync_client import RobotClient

__all__ = [
    "__version__",
    "AsyncRobotClient",
    "RobotClient",
    "ServerManager",
    "manage_server",
    "is_server_running",
    "PAROL6_ROBOT",
]
