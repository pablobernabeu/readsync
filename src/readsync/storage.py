"""Append-only, encrypted, tamper-evident event log.

A reading session produces a stream of events: word entries, region entries
and exits, passage boundaries, participant responses, tracker checks and gaze
samples. ``EventLog`` writes each event as one encrypted line, so the contents
are unreadable at rest and an altered record fails authentication when read.
Records form a hash chain, so deleting or reordering records inside the log is
detected. Removing the newest records leaves a valid shorter chain, so
completeness is judged by the closing ``session_end`` event, whose ``status`` is
``completed`` for a session that finished normally. A study that needs stronger
assurance can store the final chain hash separately.

The log is append-only and assumes a single writer per file. Each instance
caches the chain tail at construction, so two processes appending to one file
would fork the chain. Appended records reach the operating system at once, and
:meth:`EventLog.sync` asks it to commit them to disk, which the session does at
the end of every passage. If the machine fails mid-write, the last line can be
left incomplete, so the log no longer verifies or opens, and
:func:`recover_events` returns every record before the damage.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from .security import DecryptionError, decrypt, encrypt

__all__ = ["EventLog", "IntegrityError", "recover_events"]

_GENESIS = "0" * 64


class IntegrityError(Exception):
    """Raised when the hash chain does not verify because records inside the log
    were removed or reordered. An altered record fails authentication first and
    raises :class:`readsync.security.DecryptionError`. A truncated tail leaves a
    valid shorter chain and is judged by the closing ``session_end`` event."""


def _chain_hash(prev_hash: str, payload: bytes) -> str:
    return hashlib.sha256(prev_hash.encode("ascii") + payload).hexdigest()


class EventLog:
    """An encrypted append-only event log backed by a single file.

    Parameters
    ----------
    path:
        File to append encrypted records to. Created if absent.
    key:
        Symmetric key from :func:`readsync.security.new_data_key`.

    Each appended event is a JSON-serialisable mapping. A monotonically
    increasing ``seq`` and the previous record's chain hash are added before
    encryption, which is what makes the log tamper-evident.
    """

    def __init__(self, path: str | Path, *, key: bytes) -> None:
        self.path = Path(path)
        self._key = key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq, self._last_hash = self._scan_tail()

    def _scan_tail(self) -> tuple[int, str]:
        """Return the next sequence number and the last chain hash, verifying the
        existing file as a side effect."""
        seq = 0
        last_hash = _GENESIS
        if not self.path.exists():
            return seq, last_hash
        for record in self._iter_records():
            expected = _chain_hash(last_hash, record["_payload"])
            if record["seq"] != seq or record["prev"] != last_hash:
                raise IntegrityError(f"chain broken at record {seq}")
            last_hash = expected
            seq += 1
        return seq, last_hash

    def append(self, event: Mapping[str, Any]) -> None:
        """Append one event to the log."""
        record = {"seq": self._seq, "prev": self._last_hash, "event": dict(event)}
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        token = encrypt(payload, self._key)
        with self.path.open("ab") as fh:
            fh.write(token + b"\n")
        self._last_hash = _chain_hash(self._last_hash, payload)
        self._seq += 1

    def _iter_records(self) -> Iterator[dict[str, Any]]:
        with self.path.open("rb") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                payload = decrypt(line, self._key)
                record = json.loads(payload)
                record["_payload"] = payload
                yield record

    @property
    def head(self) -> tuple[int, str]:
        """The number of records and the chain hash of the last one.

        Storing the head separately from the log, for example in the study's
        records at the end of a session, lets a later reader check that the log
        is complete and is the file it claims to be.
        """
        return self._seq, self._last_hash

    def sync(self) -> None:
        """Ask the operating system to commit the log to disk."""
        if not self.path.exists():
            return
        with self.path.open("rb+") as fh:
            os.fsync(fh.fileno())

    def events(self) -> list[dict[str, Any]]:
        """Decrypt, verify the full chain and return the events in order.

        Raises :class:`readsync.security.DecryptionError` if a record fails
        authentication and :class:`IntegrityError` if the chain is broken.
        """
        out: list[dict[str, Any]] = []
        last_hash = _GENESIS
        for seq, record in enumerate(self._iter_records()):
            if record["seq"] != seq or record["prev"] != last_hash:
                raise IntegrityError(f"chain broken at record {seq}")
            last_hash = _chain_hash(last_hash, record["_payload"])
            out.append(record["event"])
        return out

    def __len__(self) -> int:
        return self._seq


def recover_events(path: str | Path, *, key: bytes) -> tuple[list[dict[str, Any]], str | None]:
    """Return a log's events up to the first record that fails verification.

    The second value describes the failure, or is ``None`` when the whole log
    verifies. Use this to recover a session whose log ends in a damaged record,
    for example after a power cut, since such a log cannot be opened as an
    :class:`EventLog`. Records after the failure are not returned, because they
    cannot be trusted.
    """
    out: list[dict[str, Any]] = []
    last_hash = _GENESIS
    seq = 0
    with Path(path).open("rb") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                payload = decrypt(line, key)
                record = json.loads(payload)
            except (DecryptionError, ValueError) as exc:
                return out, f"record {seq} could not be read: {exc}"
            if record.get("seq") != seq or record.get("prev") != last_hash:
                return out, f"chain broken at record {seq}"
            last_hash = _chain_hash(last_hash, payload)
            out.append(record["event"])
            seq += 1
    return out, None
