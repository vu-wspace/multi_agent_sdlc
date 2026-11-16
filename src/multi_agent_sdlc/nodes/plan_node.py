from functools import cache

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..llm import structured
from ..state import SDLCState


class Task(BaseModel):
    id: str
    description: str
    files: list[str] = Field(description="File cần tạo/sửa")

class Plan(BaseModel):
    summary: str
    tech_stack: list[str]
    tasks: list[Task]
    acceptance_criteria: list[str] = Field(description="Tiêu chí kiểm thử được, dùng cho Test & Evaluate")
    risks: list[str]

@cache
def planner_structured():
    return structured("planner", Plan, temperature=0.2)

# Giới hạn của sandbox (sandbox/Dockerfile). Planner phải tuân theo; Evaluator phải biết để không đòi plan lại vô ích.
SANDBOX_CONSTRAINTS = """Ràng buộc của sandbox (không đổi được bằng cách sửa plan):
- Chỉ dùng Python; mọi file nguồn nằm trong src/. Không TypeScript/JavaScript, không DB/server thật bên ngoài.
- Chỉ dùng thư viện chuẩn (kể cả sqlite3, wsgiref) và các thư viện có sẵn: bcrypt (test: pytest, pytest-cov).
- Test chạy không có mạng; lưu trữ phải chạy được không cần hạ tầng (in-memory hoặc sqlite file)."""

PLANNER_SYSTEM = f"""Bạn là Software Architect. Phân rã yêu cầu thành các task nhỏ, độc lập,
mỗi task ứng với file cụ thể. Acceptance criteria phải đo được (test được).
{SANDBOX_CONSTRAINTS}
- Nếu yêu cầu cần thứ sandbox không có (vd Flask, JavaScript): chọn phương án thay thế gần nhất và ghi rõ vào risks."""

def plan_node(state: SDLCState) -> dict:
    user_msg = f"Yêu cầu:\n{state['requirement']}"
    if state.get("project_context"):
        user_msg += f"\n\nBối cảnh dự án:\n{state['project_context']}"

    # Nếu đây là vòng plan lại (Evaluate chấm thấp) → đưa feedback vào
    revising = bool(state.get("eval_feedback"))
    if revising:
        user_msg += f"\n\nPlan trước:\n{state['plan']}\n\nFeedback từ Evaluator:\n{state['eval_feedback']}\nHãy sửa plan."

    plan: Plan = planner_structured().invoke([
        SystemMessage(PLANNER_SYSTEM),
        HumanMessage(user_msg),
    ])

    return {
        "plan": plan.model_dump_json(indent=2),
        "acceptance_criteria": plan.acceptance_criteria,
        # đếm ở đây (lần plan lại thật sự), không phải ở evaluate
        **({"plan_revisions": state.get("plan_revisions", 0) + 1} if revising else {}),
    }
