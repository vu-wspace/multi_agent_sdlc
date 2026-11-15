from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from ..llm import get_llm
import json, subprocess
from functools import cache
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import END
from ..config import MAX_ITER
from ..state import SDLCState
from .code_node import WORKSPACE, git

# ---------- Schema ----------
class ReviewComment(BaseModel):
    file: str
    line: int | None = Field(description="Dòng trong file mới; None nếu là comment chung cho file")
    severity: Literal["blocker", "major", "minor", "nit"]
    category: Literal["security", "correctness", "performance", "maintainability", "style"]
    comment: str
    suggestion: str = Field(default="", description="Code gợi ý thay thế, nếu có")

class PrevCommentStatus(BaseModel):
    id: int
    resolved: bool
    note: str = ""

class Review(BaseModel):
    summary: str
    comments: list[ReviewComment]
    previous: list[PrevCommentStatus] = Field(default_factory=list,
        description="Trạng thái các comment blocker/major của vòng trước")

@cache
def reviewer_structured():
    return (get_llm("reviewer", temperature=0)
                       .with_structured_output(Review, method="function_calling")
                       .with_retry(stop_after_attempt=2))

REVIEWER_SYSTEM = """Bạn là Staff Engineer review Pull Request.
Mức độ:
- blocker: lỗ hổng bảo mật, mất dữ liệu, crash, sai logic nghiêm trọng → không được merge
- major: bug tiềm ẩn, hiệu năng kém rõ rệt, thiết kế khó bảo trì
- minor: cải thiện nên làm nhưng không chặn merge
- nit: style, đặt tên
Quy tắc:
- Chỉ comment trên code THAY ĐỔI trong diff.
- Không lặp lại vấn đề mà ruff đã báo (style thuần).
- Mỗi comment phải chỉ đúng file/dòng và nêu cách sửa.
- Đừng hạ severity để cho qua, cũng đừng nâng nit thành major."""

# ---------- Số liệu khách quan ----------
def run_bandit() -> list[str]:
    cmd = ["docker", "run", "--rm", "--network", "none", "-v", f"{WORKSPACE}:/app:ro",
           "sdlc-sandbox", "bandit", "-r", "src", "-f", "json", "-q"]
    r = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=60)
    try:
        results = json.loads(r.stdout or "{}").get("results", [])
    except json.JSONDecodeError:
        return []
    return [f"{x['filename'].split('/app/')[-1]}:{x['line_number']} "
            f"[{x['issue_severity']}] {x['test_id']} {x['issue_text']}" for x in results]

def numbered(comments: list[dict]) -> str:
    return "\n".join(f"#{i} [{c['severity']}] {c['file']}:{c.get('line')} — {c['comment']}"
                     for i, c in enumerate(comments))

def format_feedback(comments: list[ReviewComment]) -> str:
    out = []
    for c in comments:
        if c.severity in ("blocker", "major"):
            s = f"- [{c.severity.upper()}][{c.category}] {c.file}:{c.line} — {c.comment}"
            if c.suggestion:
                s += f"\n  Gợi ý:\n```python\n{c.suggestion}\n```"
            out.append(s)
    return "\n".join(out)

def write_review_md(review: Review, approved: bool, bandit: list[str]) -> None:
    lines = [f"# Code Review — {'✅ APPROVED' if approved else '❌ CHANGES REQUESTED'}",
             "", review.summary, ""]
    for sev in ("blocker", "major", "minor", "nit"):
        items = [c for c in review.comments if c.severity == sev]
        if items:
            lines.append(f"## {sev.capitalize()} ({len(items)})")
            lines += [f"- `{c.file}:{c.line}` [{c.category}] {c.comment}" for c in items]
            lines.append("")
    if bandit:
        lines += ["## Bandit", *[f"- {b}" for b in bandit]]
    (WORKSPACE / "REVIEW.md").write_text("\n".join(lines), encoding="utf-8")

# ---------- Node ----------
def review_node(state: SDLCState) -> dict:
    full_diff = git("diff", "base", "HEAD", "--", "src", "tests")
    last_diff = state.get("diff", "")
    bandit = run_bandit()
    prev_blocking = state.get("open_review_comments", [])

    msg = f"""## Mục tiêu PR
{state['requirement']}

## Diff toàn bộ PR
``````diff
{full_diff[-50_000:]}
`````"""
    if prev_blocking:
        msg += f"""

## Comment blocker/major của vòng trước (đánh dấu resolved hay chưa)
{numbered(prev_blocking)}

## Diff vòng sửa gần nhất
````diff
{last_diff[-15_000:]}
```"""
    if bandit:
        msg += "\n\n## Bandit (security scan)\n" + "\n".join(bandit)

    review: Review = reviewer_structured().invoke([SystemMessage(REVIEWER_SYSTEM), HumanMessage(msg)])

    # Comment cũ chưa resolved → vẫn chặn
    unresolved_ids = {p.id for p in review.previous if not p.resolved}
    still_open = [c for i, c in enumerate(prev_blocking) if i in unresolved_ids]

    new_blocking = [c for c in review.comments if c.severity in ("blocker", "major")]
    bandit_high = [b for b in bandit if "[HIGH]" in b]

    approved = not new_blocking and not still_open and not bandit_high
    write_review_md(review, approved, bandit)

    feedback = format_feedback(new_blocking)
    if still_open:
        feedback += "\n\n## Chưa xử lý từ vòng trước\n" + numbered(still_open)
    if bandit_high:
        feedback += "\n\n## Bandit HIGH\n" + "\n".join(bandit_high)

    return {
        "review_approved": approved,
        "review_feedback": feedback,
        "review_summary": review.summary,
        "open_review_comments": still_open + [c.model_dump() for c in new_blocking],
        "has_blockers": any(c.severity == "blocker" for c in review.comments) or bool(bandit_high)
                        or any(c["severity"] == "blocker" for c in still_open),
    }

## Cập nhật routing
def after_review(s) -> Literal["ship", "code", "__end__"]:
    if s["review_approved"]:
        return "ship"
    if s["iterations"] < MAX_ITER:
        return "code"
    # Hết lượt: chỉ còn major thì vẫn ship (kèm cảnh báo), còn blocker thì dừng
    return END if s["has_blockers"] else "ship"
