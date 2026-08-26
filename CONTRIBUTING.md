# Contributing

Contributions are welcome, especially new tracker and presenter backends and work on the webcam strand.

## Development setup

```bash
# from the repository root
python -m pip install -e ".[dev]"
ruff check .
mypy
pytest
```

The core runs with no research hardware or display, so the tests pass on any machine. Hardware paths are isolated behind protocols and lazy imports; the lines that need a device are excluded from coverage and their tests skip when the hardware or optional extra is absent.

## Ground rules

- Keep the core dependency-light. New heavy or hardware-specific dependencies go behind optional extras in `pyproject.toml` and are imported lazily.
- Do not duplicate tested external tools. Presentation builds on PsychoPy, analysis is done with the established packages, and `readsync` exports to the formats they read.
- Anything that touches participant data must preserve the security guarantees in [SECURITY.md](SECURITY.md): pseudonymisation at source, encryption at rest, the tamper-evident log, and offline operation during a session. Add tests for new data paths.
- Never commit participant data. The `.gitignore` excludes `data/`, `export/` and `*.log`.

## Pull requests

Keep changes focused and add tests. Run the lint, type-check and test commands above before opening a pull request.
