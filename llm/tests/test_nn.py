"""Smoke tests to confirm module health.

Real unit tests land alongside the migrated modules; for now this just
confirms the package imports cleanly under ``uv run pytest``.
"""

import spiderman_llm


def test_package_imports():
    assert spiderman_llm.__version__
