"""Model provider helpers for ATP experiments."""

from __future__ import annotations

from typing import Any

from dotenv import load_dotenv

load_dotenv()


def build_chat_model(
    *,
    provider: str,
    model: str,
    temperature: float = 0.0,
    **kwargs: Any,
):
    """Build a chat model from a provider name."""
    provider_name = provider.lower()

    if provider_name == "tongyi":
        from langchain_community.chat_models import ChatTongyi

        return ChatTongyi(model=model, temperature=temperature, **kwargs)

    if provider_name == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "Gemini provider requires langchain_google_genai."
            ) from exc
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            **kwargs,
        )

    raise ValueError(f"Unsupported provider: {provider}")
