"""Security primitives for readsync.

The toolkit handles data from human participants, so three guarantees are built
in at the lowest level.

1. Pseudonymisation. Direct identifiers are never stored. A stable pseudonym is
   derived from an identifier with a keyed HMAC, so the same person maps to the
   same code within a study while the mapping cannot be reversed without the key.
2. Encryption at rest. Session data are encrypted with an authenticated cipher
   (Fernet, which is AES-128-CBC with an HMAC). Tampering is detected on read.
3. Offline operation during a session. ``NetworkGuard`` blocks outbound network
   connections for the duration of a recording, so participant data cannot leave
   the machine and experiment timing is not disturbed by network activity.

These are deliberately small, dependency-light and testable.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import socket
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

__all__ = [
    "pseudonymise",
    "new_data_key",
    "encrypt",
    "decrypt",
    "DecryptionError",
    "NetworkGuard",
    "OfflineViolation",
]


class DecryptionError(Exception):
    """Raised when ciphertext fails authentication, i.e. it was altered or the
    wrong key was supplied."""


class OfflineViolation(RuntimeError):
    """Raised when code attempts a network connection while a ``NetworkGuard`` is
    active. Surfacing this as an error makes the offline guarantee enforceable."""


def pseudonymise(identifier: str, *, key: bytes) -> str:
    """Return a stable, non-reversible pseudonym for ``identifier``.

    Uses HMAC-SHA256 so that the pseudonym depends on a secret study key. The
    same identifier yields the same pseudonym within a study, which lets sessions
    be linked, while the original identifier cannot be recovered from the
    pseudonym without the key. Store the key separately from the data, under
    stricter access control than the data themselves.
    """
    if not identifier:
        raise ValueError("identifier must be a non-empty string")
    if len(key) < 16:
        raise ValueError("key must be at least 16 bytes; use new_data_key()")
    digest = hmac.new(key, identifier.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[:16]


def new_data_key() -> bytes:
    """Generate a fresh symmetric key for encryption at rest.

    The key is a URL-safe base64 token suitable for both ``Fernet`` and, when
    sliced, for :func:`pseudonymise`. Persist it outside the data directory.
    """
    return Fernet.generate_key()


def encrypt(data: bytes, key: bytes) -> bytes:
    """Encrypt ``data`` with authenticated encryption and return the token."""
    return Fernet(key).encrypt(data)


def decrypt(token: bytes, key: bytes) -> bytes:
    """Decrypt and authenticate ``token``.

    Raises :class:`DecryptionError` if the token was altered or the key is wrong,
    which is how tampering with stored data is detected.
    """
    try:
        return Fernet(key).decrypt(token)
    except InvalidToken as exc:  # noqa: BLE001 - re-raised as a domain error
        raise DecryptionError("ciphertext failed authentication") from exc


def _is_loopback(host: str) -> bool:
    """Whether ``host`` names the local machine, parsed as an address.

    ``localhost`` and every loopback address are accepted, including the whole
    ``127.0.0.0/8`` range, ``::1`` and IPv4-mapped forms such as
    ``::ffff:127.0.0.1``. Anything that does not parse as an IP address is
    refused, so a hostname engineered to look local, ``127.attacker.com`` for
    example, stays blocked instead of being resolved over DNS.
    """
    if host == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.is_loopback:
        return True
    mapped = getattr(ip, "ipv4_mapped", None)
    return mapped is not None and bool(mapped.is_loopback)


class NetworkGuard(AbstractContextManager["NetworkGuard"]):
    """Block outbound network connections for the duration of a ``with`` block.

    Implemented by replacing ``socket.socket.connect`` and ``connect_ex`` with
    functions that raise :class:`OfflineViolation`. ``allow_loopback`` keeps
    Python-level clients of local services working, along with non-network
    transports such as Unix domain sockets, while still blocking the internet.
    Lab Streaming Layer keeps working for a different reason: ``pylsl`` binds
    the native ``liblsl`` library, whose sockets never pass through Python's
    socket layer, which is also why the guard cannot block any native-code
    transport.

    This is a guard against accidental disclosure. It intercepts connect calls
    from Python code (HTTP clients, telemetry, accidental uploads) and makes an
    offline session the default. Connectionless sends such as UDP ``sendto``,
    connections established before the guard started, and native-code transports
    are out of scope; for stronger isolation, run the session on a machine with
    networking disabled.
    """

    def __init__(self, *, allow_loopback: bool = True) -> None:
        self.allow_loopback = allow_loopback
        self._original_connect = socket.socket.connect
        self._original_connect_ex = socket.socket.connect_ex

    def _permitted(self, address: Any) -> bool:
        if not self.allow_loopback:
            return False
        if not isinstance(address, tuple):
            # Non-tuple addresses (a path string or bytes) belong to local
            # transports such as Unix domain sockets, which carry no network.
            return True
        host = str(address[0]) if address else ""
        return _is_loopback(host)

    def __enter__(self) -> NetworkGuard:
        original_connect = self._original_connect
        original_connect_ex = self._original_connect_ex

        def connect(sock: socket.socket, address: Any) -> None:
            if self._permitted(address):
                original_connect(sock, address)
                return
            raise OfflineViolation(
                f"network connection to {address!r} blocked during an offline session"
            )

        def connect_ex(sock: socket.socket, address: Any) -> int:
            if self._permitted(address):
                return original_connect_ex(sock, address)
            raise OfflineViolation(
                f"network connection to {address!r} blocked during an offline session"
            )

        socket.socket.connect = connect  # type: ignore[method-assign,assignment]
        socket.socket.connect_ex = connect_ex  # type: ignore[method-assign,assignment]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        socket.socket.connect = self._original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = self._original_connect_ex  # type: ignore[method-assign]
