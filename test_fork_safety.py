#!/usr/bin/env python3
"""
Minimal test case to reproduce fork-safety issues with numpy/scipy.
This simulates what happens in the PAROL6 code when multiprocessing fork is used.
"""
import multiprocessing
import sys
import numpy as np
from multiprocessing import Process

# Simulate module-level initialization (like PAROL6_ROBOT.py does)
print(f"Main process initializing numpy... (PID: {multiprocessing.current_process().pid})")

# Do some heavy numpy operations with threading (like _compute_jacobian_velocity_bound)
np.random.seed(42)
for _ in range(10):
    matrix = np.random.rand(6, 6)
    result = np.linalg.inv(matrix) @ matrix  # Matrix operations that use BLAS/LAPACK

print("Main process: numpy initialized with threading backend")


def worker_function(worker_id):
    """Worker subprocess that tries to use numpy after fork."""
    print(f"Worker {worker_id}: Starting (PID: {multiprocessing.current_process().pid})")

    try:
        # Try to use numpy - this can segfault if fork inherited corrupted thread pools
        print(f"Worker {worker_id}: Attempting numpy operations...")
        matrix = np.random.rand(10, 10)
        inv = np.linalg.inv(matrix)
        result = inv @ matrix
        print(f"Worker {worker_id}: Success! Result shape: {result.shape}")
        return True
    except Exception as e:
        print(f"Worker {worker_id}: FAILED with exception: {e}")
        return False


if __name__ == "__main__":
    print(f"\nTesting with start method: {multiprocessing.get_start_method()}")
    print("-" * 60)

    # Test with default fork method (on Linux)
    processes = []
    for i in range(3):
        p = Process(target=worker_function, args=(i,), daemon=True)
        p.start()
        processes.append(p)

    # Wait for all processes
    for i, p in enumerate(processes):
        p.join(timeout=5)
        if p.exitcode is None:
            print(f"Worker {i}: TIMEOUT (likely deadlock)")
            p.terminate()
        elif p.exitcode != 0:
            print(f"Worker {i}: CRASHED with exit code {p.exitcode}")
        else:
            print(f"Worker {i}: Completed successfully")

    print("-" * 60)
    print("Test complete. If any workers crashed or timed out, fork-safety is an issue.")
