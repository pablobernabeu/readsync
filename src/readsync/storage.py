"""Append-only, encrypted, tamper-evident event log.

A reading session produces a stream of events: word and region entries and
exits, passage boundaries, participant responses, tracker checks and gaze
samples. ``EventLog`` writes each event as one encrypted line, so the file is
opaque at rest and any alteration is detected on read. Records form a hash
chain, so edits, deletion or reordering of interior records are detected.
Removing the newest records leaves a valid shorter chain, so completeness is
judged against the closing ``session_end`` event that every completed session
appends; a study needing stronger assurance records the final chain hash out of
band.

The log is append-only by design and assumes a single writer per file: each
instance caches the chain tail at construction, so two processes appending to
one file would fork the chain. It is never rewritten in place, which suits a
research record that must be auditable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from .security import decrypt, encrypt

__all__ = ["EventLog", "IntegrityError"]

_GENESIS = "0" * 64


class IntegrityError(Exception):
    """Raised when the hash chain does not verify, i.e. interior records were
    altered, removed or reordered. A truncated tail leaves a valid shorter
    chain and is judged by the presence of the closing ``session_end`` event."""


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

    def events(self) -> list[dict[str, Any]]:
        """Decrypt, verify the full chain, and return the events in order.

        Raises :class:`IntegrityError` if verification fails.
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
