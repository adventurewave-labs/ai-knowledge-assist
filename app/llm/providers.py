from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI

from app.config import settings

if TYPE_CHECKING:
    from langchain_google_genai import ChatGoogleGenerativeAI


def get_primary_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
    )


def get_fallback_llm() -> "ChatGoogleGenerativeAI":
    # Imported lazily so the app can start even if langchain-google-genai is not
    # installed; the dependency is only required when the fallback is invoked.
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.FALLBACK_LLM_MODEL,
        google_api_key=settings.GOOGLE_API_KEY or None,
        temperature=0,
    )
