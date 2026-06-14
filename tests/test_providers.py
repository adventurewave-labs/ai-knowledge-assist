"""Tests that the Google fallback LLM import is lazy (B10).

The app must import and start even when ``langchain-google-genai`` is not
installed; the dependency is only needed when the fallback is actually invoked.
"""

import builtins
import importlib

import pytest


def test_providers_imports_without_google(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "langchain_google_genai":
            raise ModuleNotFoundError("No module named 'langchain_google_genai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    import app.llm.providers as providers

    # Reloading must succeed even though the Google package is unavailable,
    # proving the import is not performed at module top level.
    reloaded = importlib.reload(providers)
    assert hasattr(reloaded, "get_primary_llm")
    assert hasattr(reloaded, "get_fallback_llm")

    # Restore a clean module state for other tests.
    monkeypatch.undo()
    importlib.reload(providers)


def test_fallback_raises_when_google_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "langchain_google_genai":
            raise ModuleNotFoundError("No module named 'langchain_google_genai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    import app.llm.providers as providers

    with pytest.raises(ModuleNotFoundError):
        providers.get_fallback_llm()
