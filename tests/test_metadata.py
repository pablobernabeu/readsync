"""Tests that the citation metadata agree with the package."""

from __future__ import annotations

import re
from pathlib import Path

import readsync

_CITATION = Path(__file__).resolve().parent.parent / "CITATION.cff"


def test_citation_version_matches_the_package() -> None:
    match = re.search(r"^version:\s*(\S+)\s*$", _CITATION.read_text(encoding="utf-8"), re.M)
    assert match is not None
    assert match.group(1).strip("'\"") == readsync.__version__
