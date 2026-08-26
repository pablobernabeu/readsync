"""Tests for the security guarantees."""

from __future__ import annotations

import socket

import pytest

from readsync.security import (
    DecryptionError,
    NetworkGuard,
    OfflineViolation,
    decrypt,
    encrypt,
    new_data_key,
    pseudonymise,
)


def test_pseudonym_is_stable_and_key_dependent() -> None:
    key_a = new_data_key()
    key_b = new_data_key()
    assert pseudonymise("alice@example.org", key=key_a) == pseudonymise(
        "alice@example.org", key=key_a
    )
    assert pseudonymise("alice@example.org", key=key_a) != pseudonymise(
        "alice@example.org", key=key_b
    )
    assert pseudonymise("alice@example.org", key=key_a) != pseudonymise(
        "bob@example.org", key=key_a
    )


def test_pseudonym_rejects_empty_and_short_key() -> None:
    with pytest.raises(ValueError):
        pseudonymise("", key=new_data_key())
    with pytest.raises(ValueError):
        pseudonymise("x", key=b"too-short")


def test_encrypt_round_trip() -> None:
    key = new_data_key()
    token = encrypt(b"sensitive", key)
    assert token != b"sensitive"
    assert decrypt(token, key) == b"sensitive"


def test_decrypt_detects_tampering() -> None:
    key = new_data_key()
    token = bytearray(encrypt(b"sensitive", key))
    token[-1] ^= 0x01
    with pytest.raises(DecryptionError):
        decrypt(bytes(token), key)


def test_network_guard_blocks_outbound() -> None:
    with NetworkGuard():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(OfflineViolation):
            sock.connect(("example.org", 80))
        sock.close()


def test_network_guard_restores_connect() -> None:
    original = socket.socket.connect
    with NetworkGuard():
        pass
    assert socket.socket.connect is original


def test_pseudonymise_known_answer() -> None:
    # Pins the algorithm (HMAC-SHA256, first 16 hex characters), so an
    # accidental change cannot silently unlink previously recorded sessions.
    value = pseudonymise("participant-1", key=b"0123456789abcdef0123456789abcdef")
    assert value == "ea81c7db2c3ab38d"


def test_guard_allows_the_whole_loopback_range() -> None:
    with NetworkGuard():
        for host in ("127.0.0.2", "::ffff:127.0.0.1"):
            sock = socket.socket()
            try:
                # The attempt must reach the real connect (a closed port gives
                # a connection error), never the guard's OfflineViolation.
                sock.connect((host, 1))
            except OfflineViolation as exc:  # pragma: no cover - the guarded failure
                raise AssertionError(f"loopback host {host} was blocked") from exc
            except OSError:
                pass
            finally:
                sock.close()


def test_guard_blocks_a_loopback_looking_hostname() -> None:
    with NetworkGuard():
        sock = socket.socket()
        try:
            with pytest.raises(OfflineViolation):
                sock.connect(("127.attacker.com", 443))
        finally:
            sock.close()


def test_guard_passes_non_tuple_addresses_through() -> None:
    # Non-tuple addresses belong to local transports such as Unix domain
    # sockets; the guard must not block them. On an AF_INET socket the real
    # connect refuses the address form, which proves the call went through.
    with NetworkGuard():
        sock = socket.socket()
        try:
            try:
                sock.connect("/tmp/some-local.sock")
            except OfflineViolation as exc:  # pragma: no cover - the guarded failure
                raise AssertionError("non-tuple address was blocked") from exc
            except (OSError, TypeError):
                pass
        finally:
            sock.close()


def test_guard_blocks_connect_ex_too() -> None:
    with NetworkGuard():
        sock = socket.socket()
        try:
            with pytest.raises(OfflineViolation):
                sock.connect_ex(("93.184.216.34", 80))
        finally:
            sock.close()
