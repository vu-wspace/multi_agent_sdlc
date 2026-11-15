# factory tạo ChatOpenAI qua OmniRoute

from functools import lru_cache
from langchain_openai import ChatOpenAI
from .config import OMNI_BASE_URL, OMNI_API_KEY, MODELS

# max_tokens mặc định lớn: OmniRoute trả tool call bị cắt thành args={} (finish_reason vẫn là tool_calls),
# lộ ra thành ValidationError "Field required" rất khó hiểu. Cả 4 model đều nhận 64k.
@lru_cache(maxsize=None)
def get_llm(role: str, temperature: float = 0.0, max_tokens: int = 32_000) -> ChatOpenAI:
    if not (OMNI_BASE_URL and OMNI_API_KEY and MODELS[role]):
        raise RuntimeError(f"Thiếu OMNIROUTE_BASE_URL / OMNIROUTE_API_KEY / MODEL_{role.upper()} trong .env")
    return ChatOpenAI(
        model=MODELS[role],
        base_url=OMNI_BASE_URL,
        api_key=OMNI_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=180,
        max_retries=2,
    )
