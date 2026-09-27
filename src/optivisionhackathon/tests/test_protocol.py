"""Tests for the Visionnaire binary frame protocol."""

from __future__ import annotations

import struct

import pytest

from optivisionhackathon.shared.protocol import (
    MAGIC,
    ProtocolError,
    decode_frame,
    encode_frame,
)


def test_valid_packet_round_trip() -> None:
    jpeg = b"\xff\xd8\xff\xe0fake-jpeg-data\xff\xd9"
    packet = encode_frame(jpeg)
    assert packet[:4] == MAGIC
    assert struct.unpack("!I", packet[4:8])[0] == len(jpeg)
    assert decode_frame(packet) == jpeg


def test_empty_payload_round_trip() -> None:
    packet = encode_frame(b"")
    assert decode_frame(packet) == b""


def test_invalid_magic() -> None:
    jpeg = b"jpeg-bytes"
    packet = b"XXXX" + struct.pack("!I", len(jpeg)) + jpeg
    with pytest.raises(ProtocolError, match="invalid magic"):
        decode_frame(packet)


def test_truncated_header() -> None:
    with pytest.raises(ProtocolError, match="truncated header"):
        decode_frame(b"VIS")
    with pytest.raises(ProtocolError, match="truncated header"):
        decode_frame(MAGIC + b"\x00\x00")


def test_invalid_payload_length() -> None:
    """Declared length shorter than actual payload (trailing data)."""
    jpeg = b"abcdef"
    # Claim only 3 bytes but attach 6
    packet = MAGIC + struct.pack("!I", 3) + jpeg
    with pytest.raises(ProtocolError, match="invalid payload length"):
        decode_frame(packet)


def test_truncated_payload() -> None:
    jpeg = b"abcdef"
    # Claim 6 bytes but only send 3
    packet = MAGIC + struct.pack("!I", 6) + jpeg[:3]
    with pytest.raises(ProtocolError, match="truncated payload"):
        decode_frame(packet)
