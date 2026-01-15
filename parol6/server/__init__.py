"""Server management modules."""

import multiprocessing

# Use spawn method on all platforms to avoid fork issues with multi-threaded processes.
# This must be done before any multiprocessing is used. On Windows/macOS this is already
# the default, but on Linux it defaults to fork which causes warnings/deadlocks.
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("spawn")
