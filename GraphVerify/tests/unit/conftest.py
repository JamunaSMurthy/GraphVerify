"""Fixtures scoped to tests/unit/ only -- must never leak into tests/integration/."""
from __future__ import annotations

import pytest

from tests.fakes import install_fake_embedder


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    """
    Applied to every unit test automatically: routes every lazily-
    constructed `graphverify.embedder.Embedder(...)` to deterministic
    offline vectors instead of downloading BAAI/bge-base-en-v1.5.
    Deliberately placed here (not in tests/conftest.py) so it does not
    apply to tests/integration/, which need the real embedder.
    """
    install_fake_embedder(monkeypatch)
