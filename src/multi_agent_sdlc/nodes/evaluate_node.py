import re, json
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from llm import get_llm

# ---------- Schema ----------
class CriterionVerdict(BaseModel):
    criterion: str
    status: Literal["met", "partial", "not_met"]
    evidence: str = Field(description="Trích file/hàm cụ thể chứng minh nhận định")

class DimensionScore(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str

class Evaluation(BaseModel):
    criteria: list[CriterionVerdict]
    correctness:    DimensionScore = Field(description="Logic đúng yêu cầu, kể cả trường hợp test chưa phủ")
    design:         DimensionScore = Field(description="Cấu trúc module, tách trách nhiệm, đúng plan")
    code_quality:   DimensionScore = Field(description="Dễ đọc, đặt tên, type hint, không trùng lặp")
    robustness:     DimensionScore = Field(description="Xử lý lỗi, validate input, edge case")
    test_adequacy:  DimensionScore = Field(description="Test có thực sự kiểm tra hành vi quan trọng không")
    issue_origin: Literal["none", "design", "implementation"] = Field(
        description="design = plan sai/thiếu, phải lập lại plan; implementation = plan ổn, code chưa đạt")
    feedback: list[str] = Field(description="Danh sách việc cần sửa, cụ thể, ưu tiên từ nặng đến nhẹ")

evaluator_structured = get_llm("evaluator").with_structured_output(Evaluation, method="function_calling")

EVALUATOR_SYSTEM = """Bạn là Principal Engineer độc lập đánh giá một deliverable.
- Đánh giá dựa trên YÊU CẦU và ACCEPTANCE CRITERIA, không dựa trên lời giải thích của người viết code.
- Test pass KHÔNG có nghĩa là đạt: hãy tìm yêu cầu bị bỏ sót hoặc test quá dễ.
- Mỗi nhận định phải có bằng chứng (tên file/hàm).
- Chấm nghiêm: 3 = chấp nhận được, 4 = tốt, 5 = xuất sắc hiếm khi có."""

WEIGHTS = {"correctness": 0.35, "design": 0.20, "code_quality": 0.15,
           "robustness": 0.15, "test_adequacy": 0.15}
PASS_THRESHOLD = 0.75

# ---------- Số liệu khách quan ----------
def parse_coverage(report: str) -> int | None:
    m = re.search(r"^TOTAL\s+\d+\s+\d+\s+(\d+)%", report, re.MULTILINE)
    return int(m.group(1)) if m else None

def run_ruff() -> dict:
    cmd = ["docker", "run", "--rm", "--network", "none", "-v", f"{WORKSPACE}:/app",
           "sdlc-sandbox", "ruff", "check", "src", "--output-format", "json", "--no-cache"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    try:
        issues = json.loads(r.stdout or "[]")
    except json.JSONDecodeError:
        issues = []
    return {"count": len(issues),
            "top": [f"{i['filename'].split('/app/')[-1]}:{i['location']['row']} {i['code']} {i['message']}"
                    for i in issues[:15]]}

# ---------- Tính điểm (deterministic) ----------
def compute_score(ev: Evaluation, coverage: int | None, lint_count: int) -> tuple[float, list[str]]:
    score = sum(getattr(ev, k).score / 5 * w for k, w in WEIGHTS.items())

    # điều chỉnh nhẹ bằng số liệu khách quan
    if coverage is not None and coverage < 70:
        score -= 0.05
    if lint_count > 10:
        score -= 0.05

    gates = []
    not_met = [c.criterion for c in ev.criteria if c.status == "not_met"]
    if not_met:
        gates.append(f"Criteria chưa đạt: {not_met}")
    if ev.correctness.score <= 2:
        gates.append("Correctness quá thấp")
    return round(max(score, 0.0), 3), gates

# ---------- Node ----------
def evaluate_node(state: SDLCState) -> dict:
    coverage = parse_coverage(state.get("test_report", ""))
    lint = run_ruff()

    msg = f"""## Yêu cầu gốc
{state['requirement']}

## Plan
{state['plan']}

## Acceptance criteria
{json.dumps(state['acceptance_criteria'], ensure_ascii=False, indent=2)}

## Source code
{read_workspace()}

## Tests
{read_tests()}

## Số liệu khách quan
- Test: PASS toàn bộ
- Coverage: {coverage if coverage is not None else 'không rõ'}%
- Ruff: {lint['count']} vấn đề
{chr(10).join(lint['top'])}"""
    # Cố ý KHÔNG đưa code_notes của Coder vào, để tránh bị "thuyết phục" bởi lời tự giải thích

    ev: Evaluation = evaluator_structured.invoke([SystemMessage(EVALUATOR_SYSTEM), HumanMessage(msg)])
    score, gates = compute_score(ev, coverage, lint["count"])
    passed = score >= PASS_THRESHOLD and not gates

    feedback = "\n".join(f"- {f}" for f in gates + ev.feedback)
    return {
        "eval_score": score,
        "eval_passed": passed,
        "eval_target": "none" if passed else ("plan" if ev.issue_origin == "design" else "code"),
        "eval_feedback": feedback,
        "eval_detail": ev.model_dump(),          # lưu lại để làm báo cáo / dashboard
        "plan_revisions": state.get("plan_revisions", 0) + (1 if state.get("eval_feedback") else 0),
    }

MAX_PLAN_REVISIONS = 1

def after_eval(s) -> Literal["review", "plan", "code", "__end__"]:
    if s["eval_passed"]:
        return "review"
    if s["eval_target"] == "plan" and s["plan_revisions"] < MAX_PLAN_REVISIONS:
        return "plan"
    if s["iterations"] < MAX_ITER:
        return "code"
    return END
