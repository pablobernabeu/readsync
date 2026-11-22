"""Security primitives for readsync.

The toolkit handles data from human participants, and this module provides the
three controls it relies on. ``pseudonymise`` derives a stable code from an
identifier with HMAC-SHA256 under a study key, so a participant's sessions can be
linked while the identifier cannot be recovered without the key. ``encrypt`` and
``decrypt`` wrap Fernet, which uses AES-128 in CBC mode authenticated with
HMAC-SHA256, so an altered record is detected when it is read. ``NetworkGuard``
blocks outbound connections from Python code while a session runs. SECURITY.md
sets out what each control covers and where it stops.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import socket
import threading
import unicodedata
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
    active. Raising an error makes a blocked connection visible."""


def pseudonymise(identifier: str, *, key: bytes) -> str:
    """Return a stable, non-reversible pseudonym for ``identifier``.

    Uses HMAC-SHA256 so that the pseudonym depends on a secret study key. The
    same identifier yields the same pseudonym within a study, which lets sessions
    be linked, while the original identifier cannot be recovered from the
    pseudonym without the key. The identifier is put in Unicode NFC form and
    stripped of surrounding whitespace first, so the same identifier typed on
    different systems gives the same code. Case is kept, so lower-case
    identifiers that ignore case, such as email addresses, beforehand. Store the
    key separately from the data, under stricter access control than the data
    themselves.
    """
    identifier = unicodedata.normalize("NFC", identifier).strip()
    if not identifier:
        raise ValueError("identifier must be a non-empty string")
    if len(key) < 16:
        raise ValueError("key must be at least 16 bytes; use new_data_key()")
    digest = hmac.new(key, identifier.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[:16]


def new_data_key() -> bytes:
    """Generate a fresh symmetric key for encryption at rest.

    The key is a URL-safe base64 token for ``Fernet``, and it is also long enough
    for :func:`pseudonymise`. Use separate keys for the two purposes, and keep
    them outside the data directory.
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


# The guards active in this process. The socket methods are replaced when the
# first guard is entered and restored when the last one exits, under a lock, so
# guards may overlap in any order, including across threads.
_GUARD_LOCK = threading.Lock()
_ACTIVE_GUARDS: list[NetworkGuard] = []
_SAVED_METHODS: tuple[Any, Any] | None = None


def _blocked(address: Any) -> bool:
    """Whether any active guard refuses a connection to ``address``."""
    return any(not guard._permitted(address) for guard in list(_ACTIVE_GUARDS))


def _guarded_connect(sock: socket.socket, address: Any) -> None:
    if _blocked(address):
        raise OfflineViolation(
            f"network connection to {address!r} blocked during an offline session"
        )
    assert _SAVED_METHODS is not None
    _SAVED_METHODS[0](sock, address)


def _guarded_connect_ex(sock: socket.socket, address: Any) -> int:
    if _blocked(address):
        raise OfflineViolation(
            f"network connection to {address!r} blocked during an offline session"
        )
    assert _SAVED_METHODS is not None
    return int(_SAVED_METHODS[1](sock, address))


class NetworkGuard(AbstractContextManager["NetworkGuard"]):
    """Block outbound network connections for the duration of a ``with`` block.

    Implemented by replacing ``socket.socket.connect`` and ``connect_ex`` with
    functions that raise :class:`OfflineViolation`. ``allow_loopback`` keeps
    Python clients of local services working, along with Unix domain sockets,
    while still blocking other hosts. The replacement is installed when the
    first guard is entered and removed when the last one exits, so guards may
    overlap in any order, and while several are active a connection must be
    allowed by all of them.

    This is a guard against accidental disclosure, such as an upload or a
    telemetry call made by a dependency. DNS lookups, connectionless sends such
    as UDP ``sendto``, connections opened before the guard started, inbound
    connections, child processes and native-code transports are out of its
    scope. Lab Streaming Layer and the EyeLink link keep working for that
    reason, because their native libraries open their own sockets. A proxy
    listening on a loopback address, for example one named in ``HTTPS_PROXY``,
    would let requests leave through it, so pass ``allow_loopback=False`` on a
    machine that runs one. For stronger isolation, run the session on a machine
    disconnected from external networks.
    """

    def __init__(self, *, allow_loopback: bool = True) -> None:
        self.allow_loopback = allow_loopback

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
        global _SAVED_METHODS
        with _GUARD_LOCK:
            if not _ACTIVE_GUARDS:
                current = (socket.socket.connect, socket.socket.connect_ex)
                # If a wrapper of ours is still installed, because another
                # patch replaced it and was later undone, keep the methods saved
                # before it rather than saving the wrapper as the original.
                if current[0] is not _guarded_connect or _SAVED_METHODS is None:
                    _SAVED_METHODS = current
                socket.socket.connect = _guarded_connect  # type: ignore[method-assign,assignment]
                socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[method-assign,assignment]
            _ACTIVE_GUARDS.append(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        global _SAVED_METHODS
        with _GUARD_LOCK:
            _ACTIVE_GUARDS.remove(self)
            if not _ACTIVE_GUARDS and _SAVED_METHODS is not None:
                # Restore only if nothing else has replaced the methods since,
                # so another library's patch is never overwritten.
                if socket.socket.connect is _guarded_connect:
                    socket.socket.connect = _SAVED_METHODS[0]  # type: ignore[method-assign]
                if socket.socket.connect_ex is _guarded_connect_ex:
                    socket.socket.connect_ex = _SAVED_METHODS[1]  # type: ignore[method-assign]
