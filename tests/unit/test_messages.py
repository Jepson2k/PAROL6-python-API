"""
Unit tests for binary protocol message helpers.
"""

import numpy as np

from parol6.protocol.wire import (
    CmdType,
    MsgType,
    QueryType,
    decode,
    encode,
    get_command_name,
    get_command_type,
    get_response_value,
    is_error,
    is_ok,
    is_response,
    is_status,
    pack_error,
    pack_ok,
    pack_response,
    pack_status,
)


class TestPackUnpack:
    """Test packing and unpacking roundtrips."""

    def test_pack_ok(self):
        """OK is just the integer 0."""
        packed = pack_ok()
        unpacked = decode(packed)
        assert unpacked == MsgType.OK
        assert is_ok(unpacked)

    def test_pack_error(self):
        """Error is [ERROR, message]."""
        packed = pack_error("Something went wrong")
        unpacked = decode(packed)
        assert unpacked[0] == MsgType.ERROR
        assert unpacked[1] == "Something went wrong"

        is_err, msg = is_error(unpacked)
        assert is_err is True
        assert msg == "Something went wrong"

    def test_pack_response(self):
        """Response is [RESPONSE, query_type, value]."""
        packed = pack_response(QueryType.ANGLES, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        unpacked = decode(packed)

        assert unpacked[0] == MsgType.RESPONSE
        assert unpacked[1] == QueryType.ANGLES
        assert unpacked[2] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        assert is_response(unpacked)
        qt, val = get_response_value(unpacked)
        assert qt == QueryType.ANGLES
        assert val == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_pack_response_with_numpy(self):
        """Response with numpy array uses enc_hook for numpy support."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        packed = pack_response(QueryType.POSE, arr)
        unpacked = decode(packed)

        assert is_response(unpacked)
        _, val = get_response_value(unpacked)
        # msgspec enc_hook converts numpy arrays to lists
        assert val == [1.0, 2.0, 3.0]

    def test_pack_status_roundtrip(self):
        """Status broadcast with all fields."""
        pose = np.arange(16, dtype=np.float64)
        angles = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
        speeds = np.array([100, 200, 300, 400, 500, 600], dtype=np.int32)
        io = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
        gripper = np.array([1, 255, 150, 500, 3, 1], dtype=np.int32)
        joint_en = np.ones(12, dtype=np.uint8)
        cart_en_wrf = np.ones(12, dtype=np.uint8)
        cart_en_trf = np.ones(12, dtype=np.uint8)

        packed = pack_status(
            pose,
            angles,
            speeds,
            io,
            gripper,
            "MoveJointCommand",
            "EXECUTING",
            joint_en,
            cart_en_wrf,
            cart_en_trf,
        )
        unpacked = decode(packed)

        assert is_status(unpacked)
        assert unpacked[0] == MsgType.STATUS
        assert unpacked[1] == list(pose)
        assert unpacked[2] == list(angles)
        assert unpacked[6] == "MoveJointCommand"
        assert unpacked[7] == "EXECUTING"


class TestIsChecks:
    """Test is_ok, is_error, is_response, is_status."""

    def test_is_ok_true(self):
        assert is_ok(MsgType.OK) is True

    def test_is_ok_false(self):
        assert is_ok([MsgType.OK]) is False
        assert is_ok(None) is False
        assert is_ok("OK") is False
        assert is_ok(0) is False  # raw integers are not OK

    def test_is_error_true(self):
        is_err, msg = is_error([MsgType.ERROR, "test error"])
        assert is_err is True
        assert msg == "test error"

        is_err, msg = is_error((MsgType.ERROR, "tuple form"))
        assert is_err is True
        assert msg == "tuple form"

    def test_is_error_false(self):
        is_err, msg = is_error(MsgType.OK)
        assert is_err is False
        assert msg == ""

        is_err, msg = is_error([MsgType.RESPONSE, 1, 2])
        assert is_err is False

        is_err, msg = is_error(None)
        assert is_err is False

    def test_is_response_true(self):
        assert is_response([MsgType.RESPONSE, QueryType.ANGLES, []]) is True
        assert is_response((MsgType.RESPONSE, QueryType.POSE, {})) is True

    def test_is_response_false(self):
        assert is_response(MsgType.OK) is False
        assert is_response([MsgType.ERROR, "err"]) is False
        assert is_response([]) is False
        assert is_response(None) is False

    def test_is_status_true(self):
        assert (
            is_status([MsgType.STATUS, [], [], [], [], [], "", "", [], [], []]) is True
        )

    def test_is_status_false(self):
        assert is_status([MsgType.RESPONSE, 1, 2]) is False
        assert is_status(MsgType.STATUS) is False  # Must be in list/tuple
        assert is_status(None) is False


class TestGetCommandType:
    """Test get_command_type and get_command_name."""

    def test_valid_command(self):
        msg = [CmdType.HOME]
        cmd_type, params = get_command_type(msg)
        assert cmd_type == CmdType.HOME
        assert params == ()

    def test_command_with_params(self):
        msg = [CmdType.MOVEJOINT, [1, 2, 3, 4, 5, 6], 2.0, None, None]
        cmd_type, params = get_command_type(msg)
        assert cmd_type == CmdType.MOVEJOINT
        assert params == ([1, 2, 3, 4, 5, 6], 2.0, None, None)

    def test_invalid_command_type(self):
        msg = [9999]  # Invalid CmdType
        cmd_type, params = get_command_type(msg)
        assert cmd_type is None
        assert params == ()

    def test_empty_message(self):
        cmd_type, params = get_command_type([])
        assert cmd_type is None
        assert params == ()

    def test_non_list_message(self):
        cmd_type, params = get_command_type("not a list")
        assert cmd_type is None
        assert params == ()

        cmd_type, params = get_command_type(None)
        assert cmd_type is None
        assert params == ()

    def test_get_command_name(self):
        assert get_command_name([CmdType.HOME]) == "HOME"
        assert get_command_name([CmdType.MOVEJOINT, []]) == "MOVEJOINT"
        assert get_command_name([CmdType.GCODE, "G0 X0"]) == "GCODE"

    def test_get_command_name_invalid(self):
        assert get_command_name([9999]) is None
        assert get_command_name([]) is None
        assert get_command_name(None) is None


class TestGetResponseValue:
    """Test get_response_value extraction."""

    def test_valid_response(self):
        msg = [MsgType.RESPONSE, QueryType.ANGLES, [1, 2, 3, 4, 5, 6]]
        qt, val = get_response_value(msg)
        assert qt == QueryType.ANGLES
        assert val == [1, 2, 3, 4, 5, 6]

    def test_response_with_dict(self):
        msg = [
            MsgType.RESPONSE,
            QueryType.TOOL,
            {"name": "NONE", "available": ["NONE", "PNEUMATIC"]},
        ]
        qt, val = get_response_value(msg)
        assert qt == QueryType.TOOL
        assert val == {"name": "NONE", "available": ["NONE", "PNEUMATIC"]}

    def test_invalid_response(self):
        qt, val = get_response_value(MsgType.OK)
        assert qt is None
        assert val is None

        qt, val = get_response_value([MsgType.ERROR, "err"])
        assert qt is None
        assert val is None

        qt, val = get_response_value([MsgType.RESPONSE])  # Too short
        assert qt is None
        assert val is None


class TestGcodeStringEmbedding:
    """Test that GCODE commands work with msgpack string embedding."""

    def test_gcode_single_line(self):
        """Single GCODE line is embedded as string."""
        msg = (CmdType.GCODE, "G0 X100 Y50 Z10")
        packed = encode(msg)
        unpacked = decode(packed)

        cmd_type, params = get_command_type(unpacked)
        assert cmd_type == CmdType.GCODE
        assert params == ("G0 X100 Y50 Z10",)

    def test_gcode_program(self):
        """GCODE program is list of strings."""
        lines = ["G21", "G90", "G0 X0 Y0", "G1 X100 F1000"]
        msg = (CmdType.GCODE_PROGRAM, lines)
        packed = encode(msg)
        unpacked = decode(packed)

        cmd_type, params = get_command_type(unpacked)
        assert cmd_type == CmdType.GCODE_PROGRAM
        assert params[0] == lines

    def test_gcode_with_special_chars(self):
        """GCODE with comments and special characters."""
        gcode = "G0 X100 ; move to X=100"
        msg = (CmdType.GCODE, gcode)
        packed = encode(msg)
        unpacked = decode(packed)

        _, params = get_command_type(unpacked)
        assert params[0] == gcode
