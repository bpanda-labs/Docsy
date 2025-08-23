# wrapper.py
import os
from langchain_openai import ChatOpenAI
from pydantic import SecretStr, Field


class ChatOpenRouter(ChatOpenAI):
    openai_api_key: SecretStr = Field(
        alias='api_key',
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY")
    )
    openai_api_base: str = "https://openrouter.ai/api/v1"

    @property
    def lc_secrets(self):
        return {"openai_api_key": "OPENROUTER_API_KEY"}
