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

The core runs without research hardware or a display, so the test suite runs on any machine with Python 3.10 or later. Code that needs a device or a display sits behind protocols and imports its dependencies only when used. The tests exercise it with stand-ins for PsychoPy, pylink and Lab Streaming Layer, which check the calls readsync makes but not how the devices respond. One test runs the EyeLink backend's call sequence in pylink's dummy mode. This test runs only when the `eyelink` extra is also installed (`".[dev,eyelink]"`). The extra needs a pylink wheel for your platform and Python version, and CI installs it.

## Ground rules

Keep the core dependency-light. New heavy or hardware-specific dependencies go behind optional extras in `pyproject.toml` and are imported only when used. Use existing packages for presentation and analysis rather than reimplementing them. Changes that touch participant data must keep the protections described in [SECURITY.md](SECURITY.md), and new data paths need tests. Never commit participant data. The `.gitignore` excludes `data/`, `export/` and `*.log`.

## Pull requests

Keep changes focused, add tests and run the checks above before opening a pull request.

## Releases

To prepare a release, set the new version in `src/readsync/__init__.py`, which the package build reads, and in `CITATION.cff`, where a test checks that the two agree. Set `date-released` to the day of publication and turn the changelog's Unreleased heading into the version and date. Then, publish a GitHub release tagged with the version prefixed with `v`.

Once the repository is enabled in Zenodo's GitHub settings, Zenodo archives each published release and gives it a DOI. It takes the title, description, authors, keywords and licence from `CITATION.cff`, and the version and date from the GitHub release. Zenodo accepts a single licence there, so `license` stays `MIT` and the abstract states the licence of the stimulus files. Check the file with `cffconvert --validate` before publishing.
