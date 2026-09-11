"""Package marker, and deliberately not empty of purpose: it keeps this suite's ``conftest.py``
imported as ``sociology.tests.conftest`` rather than as a top-level ``conftest`` module.
``reward_hacking/tests`` bare-imports its own conftest by that top-level name (its layout has no
``__init__.py``, per the pytest prepend-import note in pyproject.toml), so a second package-less
``conftest.py`` in this repository would race it in ``sys.modules`` and break that suite's
collection -- observed, not hypothetical."""
