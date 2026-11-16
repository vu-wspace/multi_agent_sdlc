import sys

from multi_agent_sdlc.config import MODELS
from multi_agent_sdlc.llm import get_llm, model_list

sys.stdout.reconfigure(encoding="utf-8")  # Windows pipe mặc định cp1252, không in được ✔ / ✘

for role in MODELS:
    for i, model in enumerate(model_list(role)):
        tag = role if i == 0 else "  └ dự phòng"
        try:
            r = get_llm(role, model=model).invoke("Trả lời đúng 1 chữ: OK")
            print(f"✔ {tag:12} {model}: {r.content.strip()}")
        except Exception as e:
            print(f"✘ {tag:12} {model}: {type(e).__name__}: {str(e)[:100]}")
