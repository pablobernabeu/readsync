"""A minimal stand-in for the parts of PsychoPy that the presenter uses.

It records draws and flips, so tests can check what would be on screen at each
flip without a display.
"""

from __future__ import annotations

import sys
import types
from typing import Any


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def getTime(self) -> float:
        return self.now

    def reset(self) -> None:
        self.now = 0.0


class Window:
    def __init__(self, *, size: list[int], actual: tuple[int, int] | None = None, **_: Any) -> None:
        self.size = list(actual) if actual is not None else list(size)
        self.events: list[str] = []
        self._on_flip: list[Any] = []
        self.closed = False

    def callOnFlip(self, function: Any) -> None:
        self._on_flip.append(function)

    def flip(self) -> None:
        for function in self._on_flip:
            function()
        self._on_flip = []
        self.events.append("flip")

    def close(self) -> None:
        self.closed = True


class TextStim:
    advance_per_height = 0.6

    def __init__(self, win: Window, *, text: str, height: float = 20.0, **_: Any) -> None:
        self.win = win
        self.text = text
        self.boundingBox = (len(text) * height * self.advance_per_height, height)

    def draw(self) -> None:
        self.win.events.append(f"draw:{self.text}")


class KeyPress:
    def __init__(self, name: str, rt: float) -> None:
        self.name = name
        self.rt = rt


class Keyboard:
    def __init__(self) -> None:
        self.clock = Clock()
        self.script: list[list[KeyPress]] = []

    def clearEvents(self) -> None:
        return None

    def getKeys(self, keys: list[str], waitRelease: bool = False) -> list[KeyPress]:
        return self.script.pop(0) if self.script else []


def install(monkeypatch: Any, *, actual_size: tuple[int, int] | None = None) -> None:
    """Put the fake modules in ``sys.modules`` for the duration of a test."""
    psychopy = types.ModuleType("psychopy")
    core = types.ModuleType("psychopy.core")
    core.Clock = Clock  # type: ignore[attr-defined]
    visual = types.ModuleType("psychopy.visual")
    visual.Window = lambda **kw: Window(actual=actual_size, **kw)  # type: ignore[attr-defined]
    visual.TextStim = TextStim  # type: ignore[attr-defined]
    hardware = types.ModuleType("psychopy.hardware")
    keyboard = types.ModuleType("psychopy.hardware.keyboard")
    keyboard.Keyboard = Keyboard  # type: ignore[attr-defined]
    psychopy.core = core  # type: ignore[attr-defined]
    psychopy.visual = visual  # type: ignore[attr-defined]
    psychopy.hardware = hardware  # type: ignore[attr-defined]
    hardware.keyboard = keyboard  # type: ignore[attr-defined]
    for name, module in {
        "psychopy": psychopy,
        "psychopy.core": core,
        "psychopy.visual": visual,
        "psychopy.hardware": hardware,
        "psychopy.hardware.keyboard": keyboard,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
