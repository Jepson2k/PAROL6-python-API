"""
CLI entry point for parol6-server command.

This module provides the command-line interface for starting the PAROL6 headless controller.
"""
import multiprocessing
import sys

# Force multiprocessing to use 'spawn' instead of 'fork' on Unix platforms
# This must be called before any other imports that might use multiprocessing
# to prevent fork-safety issues with C extensions (numpy, scipy, robotics-toolbox)
if sys.platform.startswith("linux") or sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

from parol6.server.controller import main


def main_entry():
    """Entry point for the parol6-server command."""
    main()


if __name__ == "__main__":
    main_entry()
