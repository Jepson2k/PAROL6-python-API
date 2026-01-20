"""Integration test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def clean_state(server_proc, client):
    """
    Reset controller state before each integration test for isolation.

    Uses RESET command to instantly reset positions, queues, tool, errors.
    Sets LINEAR motion profile for faster test execution.
    Depends on server_proc to ensure server is ready before resetting.
    """
    client.reset()
    client.set_profile("LINEAR")
    # Use short timeouts - simulator homing is instant
    client.home(wait=True, motion_start_timeout=0.2, settle_window=0.1)
    return client
