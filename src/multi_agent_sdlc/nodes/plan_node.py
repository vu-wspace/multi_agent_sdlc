from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from ..llm import get_llm
from functools import cache
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
    return (get_llm("planner", temperature=0.2)
                      .with_structured_output(Plan, method="function_calling")
                      .with_retry(stop_after_attempt=2))

PLANNER_SYSTEM = """Bạn là Software Architect. Phân rã yêu cầu thành các task nhỏ, độc lập,
mỗi task ứng với file cụ thể. Acceptance criteria phải đo được (test được).
- Chỉ dùng Python; mọi file nguồn nằm trong src/. Không TypeScript/JavaScript, không DB/server thật.
- Chỉ dùng thư viện chuẩn và các thư viện có sẵn trong sandbox: bcrypt (test: pytest, pytest-cov).
- Lưu trữ dùng in-memory (vd: repository dạng dict) để test chạy được không cần hạ tầng."""

def plan_node(state: SDLCState) -> dict:
    user_msg = f"Yêu cầu:\n{state['requirement']}"

    # Nếu đây là vòng plan lại (Evaluate chấm thấp) → đưa feedback vào
    if state.get("eval_feedback"):
        user_msg += f"\n\nPlan trước:\n{state['plan']}\n\nFeedback từ Evaluator:\n{state['eval_feedback']}\nHãy sửa plan."

    plan: Plan = planner_structured().invoke([
        SystemMessage(PLANNER_SYSTEM),
        HumanMessage(user_msg),
    ])

    return {
        "plan": plan.model_dump_json(indent=2),
        "acceptance_criteria": plan.acceptance_criteria,
    }
