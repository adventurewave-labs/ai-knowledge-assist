from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI

from app.config import settings

if TYPE_CHECKING:
    from langchain_google_genai import ChatGoogleGenerativeAI


def get_primary_llm() -> ChatOpenAI:
    if not settings.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
    )


def get_fallback_llm() -> ChatGoogleGenerativeAI:
    from langchain_google_genai import ChatGoogleGenerativeAI  # lazy import

    if not settings.GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Gemini fallback is unavailable."
        )
    return ChatGoogleGenerativeAI(
        model=settings.FALLBACK_LLM_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0,
    )
