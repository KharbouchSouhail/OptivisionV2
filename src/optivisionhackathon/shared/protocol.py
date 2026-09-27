"""Binary WebSocket frame protocol for Visionnaire.

Packet layout:
    Magic (4 bytes) | Payload length (uint32 BE) | JPEG payload (N bytes)
"""

from __future__ import annotations

import struct

MAGIC = b"VISH"
HEADER_SIZE = 8  # 4-byte magic + 4-byte length


class ProtocolError(ValueError):
    """Raised when a binary frame packet is malformed."""


def encode_frame(jpeg_bytes: bytes) -> bytes:
    """Encode JPEG bytes into a Visionnaire binary packet."""
    if not isinstance(jpeg_bytes, (bytes, bytearray)):
        raise TypeError("jpeg_bytes must be bytes")
    payload = bytes(jpeg_bytes)
    return MAGIC + struct.pack("!I", len(payload)) + payload


def decode_frame(packet: bytes) -> bytes:
    """Decode a Visionnaire binary packet and return the JPEG payload.

    Raises:
        ProtocolError: if the packet is truncated, has invalid magic,
            an inconsistent length, or a truncated payload.
    """
    if not isinstance(packet, (bytes, bytearray)):
        raise TypeError("packet must be bytes")

    data = bytes(packet)

    if len(data) < HEADER_SIZE:
        raise ProtocolError(
            f"truncated header: expected at least {HEADER_SIZE} bytes, got {len(data)}"
        )

    magic = data[:4]
    if magic != MAGIC:
        raise ProtocolError(f"invalid magic: expected {MAGIC!r}, got {magic!r}")

    payload_length = struct.unpack("!I", data[4:8])[0]
    payload = data[8:]

    if len(payload) < payload_length:
        raise ProtocolError(
            f"truncated payload: declared {payload_length} bytes, got {len(payload)}"
        )

    if len(payload) > payload_length:
        raise ProtocolError(
            f"invalid payload length: declared {payload_length} bytes, "
            f"got {len(payload)} (trailing data)"
        )

    return payload
