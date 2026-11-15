# factory tạo ChatOpenAI qua OmniRoute

from functools import lru_cache
from langchain_openai import ChatOpenAI
from config import OMNI_BASE_URL, OMNI_API_KEY, MODELS

@lru_cache(maxsize=None)
def get_llm(role: str, temperature: float = 0.0, max_tokens: int | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODELS[role],
        base_url=OMNI_BASE_URL,
        api_key=OMNI_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=180,
        max_retries=2,
    )
