import sys
from multi_agent_sdlc.llm import get_llm
from multi_agent_sdlc.config import MODELS

sys.stdout.reconfigure(encoding="utf-8")  # Windows pipe mặc định cp1252, không in được ✔ / ✘

for role in MODELS:
    try:
        r = get_llm(role).invoke("Trả lời đúng 1 chữ: OK")
        print(f"✔ {role:10} {r.response_metadata.get('model_name')}: {r.content.strip()}")
    except Exception as e:
        print(f"✘ {role:10} {type(e).__name__}: {str(e)[:120]}")
