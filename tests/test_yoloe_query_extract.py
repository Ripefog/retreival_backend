import pytest

from app.config import Settings
from app.YOLOE26 import YOLOE26LReranker


def test_settings_accepts_gemini_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    settings = Settings()

    assert settings.GEMINI_API_KEY == "gemini-test-key"
    assert settings.GOOGLE_API_KEY == "gemini-test-key"


def test_extract_classes_from_query_falls_back_cleanly():
    reranker = YOLOE26LReranker.__new__(YOLOE26LReranker)
    classes = reranker.extract_classes_from_query("A person riding a bicycle next to a dog")

    assert classes
    assert any(label.lower() in {"person", "man", "woman", "people"} for label in classes)
    assert any(label.lower() in {"bicycle", "bike"} for label in classes)
    assert any(label.lower() == "dog" for label in classes)
