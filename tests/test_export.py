"""Tests for the CSV export, including the integrity-failure path."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from readsync.export import events_to_csv, log_to_csv
from readsync.security import new_data_key
from readsync.storage import EventLog, IntegrityError


def test_columns_put_common_keys_first_and_keep_the_rest(tmp_path: Path) -> None:
    events = [
        {"zeta": 1, "type": "gaze", "t": 0.1, "x": 3.0, "y": 4.0, "valid": True, "tracker_t": 0.09},
        {"type": "response", "t": 0.2, "passage": "p1", "correct": True},
    ]
    out = events_to_csv(events, tmp_path / "s.csv")
    with out.open(encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh))
    assert header[:7] == ["type", "t", "tracker_t", "passage", "x", "y", "valid"]
    assert set(header) == {
        "type", "t", "tracker_t", "passage", "x", "y", "valid", "zeta", "correct",
    }


def test_export_round_trips_values(tmp_path: Path) -> None:
    key = new_data_key()
    log = EventLog(tmp_path / "s.log", key=key)
    log.append({"type": "session_start", "participant": "p"})
    log.append({"type": "gaze", "t": 0.5, "x": 1.0, "y": 2.0, "valid": False})
    out = log_to_csv(log, tmp_path / "s.csv")
    with out.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["type"] == "session_start"
    assert rows[1]["valid"] == "False"


def test_export_refuses_a_tampered_log(tmp_path: Path) -> None:
    key = new_data_key()
    path = tmp_path / "s.log"
    log = EventLog(path, key=key)
    for index in range(3):
        log.append({"type": "gaze", "t": float(index)})
    lines = path.read_bytes().splitlines()
    # Remove an interior record, which breaks the hash chain on the next read.
    path.write_bytes(b"\n".join([lines[0], lines[2]]) + b"\n")
    with pytest.raises(IntegrityError):
        log_to_csv(log, tmp_path / "out.csv")
