"""Tests for the encrypted, tamper-evident event log."""

from __future__ import annotations

from pathlib import Path

import pytest

from readsync.security import DecryptionError, new_data_key
from readsync.storage import EventLog, IntegrityError


def test_append_and_read_round_trip(tmp_path: Path) -> None:
    key = new_data_key()
    log = EventLog(tmp_path / "events.log", key=key)
    log.append({"type": "passage_onset", "passage": "p1"})
    log.append({"type": "gaze", "t": 0.1, "x": 120.0, "y": 540.0})
    events = log.events()
    assert [e["type"] for e in events] == ["passage_onset", "gaze"]
    assert len(log) == 2


def test_file_is_encrypted_at_rest(tmp_path: Path) -> None:
    key = new_data_key()
    log = EventLog(tmp_path / "events.log", key=key)
    log.append({"type": "note", "secret": "participant-said-hello"})
    raw = (tmp_path / "events.log").read_bytes()
    assert b"participant-said-hello" not in raw


def test_reopening_resumes_chain(tmp_path: Path) -> None:
    key = new_data_key()
    path = tmp_path / "events.log"
    EventLog(path, key=key).append({"type": "a"})
    log2 = EventLog(path, key=key)  # re-scans and verifies the existing file
    log2.append({"type": "b"})
    assert [e["type"] for e in log2.events()] == ["a", "b"]


def test_tampering_with_a_record_is_detected(tmp_path: Path) -> None:
    key = new_data_key()
    path = tmp_path / "events.log"
    log = EventLog(path, key=key)
    log.append({"type": "a"})
    log.append({"type": "b"})
    lines = path.read_bytes().splitlines()
    flipped = bytearray(lines[0])
    flipped[-1] ^= 0x01
    path.write_bytes(b"\n".join([bytes(flipped), lines[1]]) + b"\n")
    with pytest.raises((DecryptionError, IntegrityError)):
        EventLog(path, key=key).events()


def test_deleting_a_record_is_detected(tmp_path: Path) -> None:
    key = new_data_key()
    path = tmp_path / "events.log"
    log = EventLog(path, key=key)
    log.append({"type": "a"})
    log.append({"type": "b"})
    lines = path.read_bytes().splitlines()
    path.write_bytes(lines[1] + b"\n")  # drop the first record
    with pytest.raises(IntegrityError):
        EventLog(path, key=key).events()
