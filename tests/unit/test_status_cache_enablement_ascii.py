"""
Integration tests for enablement detection via IK worker subprocess.
"""

import time

import numpy as np
import pytest

from parol6.config import steps_to_rad, LIMITS
from parol6.server.ik_worker_client import IKWorkerClient
import parol6.PAROL6_ROBOT as PAROL6_ROBOT


@pytest.mark.integration
def test_ik_worker_detects_joint_limits():
    """
    Test that the IK worker correctly detects when a joint is near its limits.

    Joint enablement format: [J1+, J1-, J2+, J2-, ..., J6+, J6-]
    - When near max limit: + direction disabled (0), - direction enabled (1)
    - When near min limit: + direction enabled (1), - direction disabled (0)
    """
    client = IKWorkerClient()
    client.start()

    try:
        time.sleep(0.2)
        assert client.is_alive(), "IK worker failed to start"

        # Start from home position and move J1 near its max limit
        from parol6.config import HOME_ANGLES_DEG
        qlim = PAROL6_ROBOT.robot.qlim

        q = np.deg2rad(HOME_ANGLES_DEG)
        # Delta is 0.2 degrees = 0.0035 rad, so we need to be within that
        q[0] = qlim[1, 0] - 0.001  # J1 very near max limit

        T = PAROL6_ROBOT.robot.fkine(q)
        T_matrix = T.A

        client.submit_request(q, T_matrix)

        result = None
        for _ in range(100):
            result = client.get_results_if_ready()
            if result is not None:
                break
            time.sleep(0.01)

        assert result is not None, "IK worker did not return results"
        joint_en, _, _ = result

        # J1 near max: J1+ should be disabled, J1- should be enabled
        assert joint_en[0] == 0, f"J1+ should be disabled near max limit, got {joint_en[0]}"
        assert joint_en[1] == 1, f"J1- should be enabled near max limit, got {joint_en[1]}"

        # Now test J1 near min limit
        q[0] = qlim[0, 0] + 0.001  # J1 very near min limit

        T = PAROL6_ROBOT.robot.fkine(q)
        T_matrix = T.A

        client.submit_request(q, T_matrix)

        result = None
        for _ in range(100):
            result = client.get_results_if_ready()
            if result is not None:
                break
            time.sleep(0.01)

        assert result is not None, "IK worker did not return results for min limit test"
        joint_en, _, _ = result

        # J1 near min: J1+ should be enabled, J1- should be disabled
        assert joint_en[0] == 1, f"J1+ should be enabled near min limit, got {joint_en[0]}"
        assert joint_en[1] == 0, f"J1- should be disabled near min limit, got {joint_en[1]}"

    finally:
        client.stop()


@pytest.mark.integration
def test_ik_worker_all_enabled_in_safe_position():
    """
    Test that all directions are enabled when robot is in the true center of its limits.
    """
    client = IKWorkerClient()
    client.start()

    try:
        time.sleep(0.2)
        assert client.is_alive()

        # Use home position - a known safe position
        from parol6.config import HOME_ANGLES_DEG
        q_home = np.deg2rad(HOME_ANGLES_DEG)

        T = PAROL6_ROBOT.robot.fkine(q_home)
        T_matrix = T.A

        client.submit_request(q_home, T_matrix)

        result = None
        for _ in range(50):
            result = client.get_results_if_ready()
            if result is not None:
                break
            time.sleep(0.01)

        assert result is not None
        joint_en, cart_en_wrf, cart_en_trf = result

        # All joint directions should be enabled in true center position
        assert np.all(joint_en == 1), f"All joints should be enabled at center, got {joint_en}"

    finally:
        client.stop()
