from unittest.mock import AsyncMock

from parol6 import RobotClient
from parol6.protocol.wire import QueryType, ResponseMsg


def _pose_response(matrix: list) -> ResponseMsg:
    """Create ResponseMsg with flattened pose matrix."""
    flat = []
    for row in matrix:
        flat.extend(row)
    return ResponseMsg(QueryType.POSE, flat)


def test_get_pose_rpy_identity_translation(monkeypatch):
    """
    Validate get_pose_rpy converts 4x4 pose matrix to [x,y,z,rx,ry,rz] (mm,deg).
    Use identity rotation with translation (10,20,30) mm.
    """
    client = RobotClient()

    # Identity rotation with translation in last column (row-major)
    mat = [
        [1, 0, 0, 10],
        [0, 1, 0, 20],
        [0, 0, 1, 30],
        [0, 0, 0, 1],
    ]
    response = _pose_response(mat)

    mock_request = AsyncMock(return_value=response)
    monkeypatch.setattr(client.async_client, "_request", mock_request)

    pose_rpy = client.get_pose_rpy()
    assert pose_rpy is not None
    # Translations
    assert pose_rpy[0:3] == [10, 20, 30]
    # Identity rotation -> zero Euler angles (within tolerance)
    rx, ry, rz = pose_rpy[3:6]
    assert abs(rx) < 1e-6
    assert abs(ry) < 1e-6
    assert abs(rz) < 1e-6


def test_get_pose_rpy_malformed_payload(monkeypatch):
    """
    Malformed POSE payload (wrong length) should return None.
    """
    client = RobotClient()

    # Not 16 elements - ResponseMsg with too few values
    mock_request = AsyncMock(return_value=ResponseMsg(QueryType.POSE, [1, 2, 3]))
    monkeypatch.setattr(client.async_client, "_request", mock_request)

    pose_rpy = client.get_pose_rpy()
    assert pose_rpy is None
