# factory tạo ChatOpenAI qua OmniRoute

import logging
from functools import lru_cache

import openai
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from .config import MODELS, OMNI_API_KEY, OMNI_BASE_URL

log = logging.getLogger(__name__)

# Chỉ đổi model khi lỗi tạm thời. 400/401/403/content-filter: đổi model = gửi prompt bị từ chối sang provider khác
TRANSIENT = (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError, openai.RateLimitError)

def model_list(role: str) -> list[str]:
    """MODEL_<ROLE>=chính,dự_phòng_1,... → model chính lỗi API (vd OmniRoute 500) thì dùng model kế tiếp."""
    return [m.strip() for m in (MODELS[role] or "").split(",") if m.strip()]

# max_tokens mặc định lớn: OmniRoute trả tool call bị cắt thành args={} (finish_reason vẫn là tool_calls),
# lộ ra thành ValidationError "Field required" rất khó hiểu. Cả 4 model đều nhận 64k.
@lru_cache(maxsize=None)
def get_llm(role: str, temperature: float = 0.0, max_tokens: int = 32_000, model: str | None = None) -> ChatOpenAI:
    if not (OMNI_BASE_URL and OMNI_API_KEY and model_list(role)):
        raise RuntimeError(f"Thiếu OMNIROUTE_BASE_URL / OMNIROUTE_API_KEY / MODEL_{role.upper()} trong .env")
    return ChatOpenAI(
        model=model or model_list(role)[0],
        base_url=OMNI_BASE_URL,
        api_key=OMNI_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=180,
        max_retries=2,
    )

def check_structured(result: dict, max_tokens: int):
    """Kết quả include_raw=True → parsed; output bị cắt vì max_tokens thì báo rõ thay vì 'Field required'."""
    if result["parsing_error"] is None:
        if result["parsed"] is None:   # trả lời bằng text, không gọi tool → ValueError để with_retry thử lại
            raise ValueError("LLM không gọi tool (không có structured output)")
        return result["parsed"]
    used = (result["raw"].usage_metadata or {}).get("output_tokens", 0)
    if used >= max_tokens * 0.98:   # OmniRoute vẫn trả finish_reason=tool_calls khi bị cắt, chỉ token count là tin được
        raise RuntimeError(f"Output LLM bị cắt ({used}/{max_tokens} token, max_tokens): tăng max_tokens hoặc thu nhỏ yêu cầu") \
            from result["parsing_error"]
    raise result["parsing_error"]

def structured(role: str, schema, temperature: float = 0.0, max_tokens: int = 32_000, retries: int = 2):
    models = model_list(role)

    def checker(model):
        def check(r):
            out = check_structured(r, max_tokens)
            if model != models[0]:
                log.warning("%s: model chính %s lỗi, dùng model dự phòng %s", role, models[0], model)
            return out
        return RunnableLambda(check)

    def logged(chain, model):
        def call(inp, config):
            try:
                return chain.invoke(inp, config)
            except TRANSIENT as e:   # with_fallbacks nuốt lỗi: không ghi lại thì không biết model chính lỗi vì sao
                log.warning("%s: %s lỗi %s: %s", role, model, type(e).__name__, str(e)[:200])
                raise
        return RunnableLambda(call)

    chains = [logged(get_llm(role, temperature=temperature, max_tokens=max_tokens, model=m)
                     .with_structured_output(schema, method="function_calling", include_raw=True)
                     | checker(m), m)
              for m in models]
    if not chains:
        get_llm(role)                                     # ném lỗi "Thiếu MODEL_..." rõ ràng
    # output bị cắt / sai schema thì đổi model không giúp được
    chain = chains[0].with_fallbacks(chains[1:], exceptions_to_handle=TRANSIENT) if len(chains) > 1 else chains[0]
    # chỉ retry lỗi parse (ValueError); output bị cắt (RuntimeError) retry chỉ tốn thêm token
    return chain.with_retry(stop_after_attempt=retries, retry_if_exception_type=(ValueError,)) if retries > 1 else chain
