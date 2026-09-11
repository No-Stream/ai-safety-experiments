"""Package marker, and not empty of purpose: it keeps this suite's ``conftest.py`` imported as
``games.tests.conftest`` rather than as a top-level ``conftest`` module. ``reward_hacking/tests``
bare-imports its own conftest by that top-level name (its layout has no ``__init__.py``, per the
pytest prepend-import note in pyproject.toml), and pytest clears ``sys.modules["conftest"]`` before
loading each package-less conftest, so a second one here left that suite importing this directory's
names and failed its collection -- observed on 2026-09-02, the day this marker was added.
``sociology/tests/__init__.py`` exists for the same reason."""
