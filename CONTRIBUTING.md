# Contributing

Contributions are welcome, especially new tracker and presenter backends and work on webcam gaze estimation.

## Development setup

```bash
python -m pip install -e ".[dev]"
ruff check .
mypy
pytest
cd web-demo && node --test   # when changing the browser demo
```

The core runs without research hardware or a display, so the test suite runs on any machine with Python 3.10 or later. Code that needs a device or a display sits behind protocols and imports its dependencies only when used. The tests exercise it with stand-ins for PsychoPy, pylink and Lab Streaming Layer, which check the calls readsync makes but not how the devices respond. One test runs the EyeLink backend's call sequence in pylink's dummy mode when `sr-research-pylink` is installed, and two tests of the error raised for a missing extra skip when that extra is installed.

## Ground rules

Keep the core dependency-light. New heavy or hardware-specific dependencies go behind optional extras in `pyproject.toml` and are imported only when used. Use existing packages for presentation and analysis rather than reimplementing them. Changes that touch participant data must keep the protections described in [SECURITY.md](SECURITY.md), and new data paths need tests. Never commit participant data. The `.gitignore` excludes `data/`, `export/` and `*.log`.

## Pull requests

Keep changes focused, add tests and run the checks above before opening a pull request.
