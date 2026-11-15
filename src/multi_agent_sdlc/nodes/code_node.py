import subprocess
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from llm import get_llm

WORKSPACE = Path("./workspace").resolve()

# ---------- Schema output ----------
class FileChange(BaseModel):
    path: str = Field(description="Đường dẫn tương đối trong workspace, vd: src/auth.py")
    action: Literal["create", "update", "delete"]
    content: str = Field(default="", description="Nội dung ĐẦY ĐỦ của file (không phải diff)")

class CodeOutput(BaseModel):
    changes: list[FileChange]
    notes: str = Field(description="Giải thích ngắn đã làm gì / sửa lỗi gì")

coder_structured = (get_llm("coder", temperature=0, max_tokens=16000)
                    .with_structured_output(CodeOutput, method="function_calling")
                    .with_retry(stop_after_attempt=2))

CODER_SYSTEM = """Bạn là Senior Software Engineer. Viết code Python sạch, có type hint, chạy được ngay.
- Chỉ tạo/sửa file thuộc phạm vi plan.
- Trả về NỘI DUNG ĐẦY ĐỦ của mỗi file thay đổi.
- KHÔNG viết test (đã có agent Test riêng), chỉ viết code nguồn trong src/.
- Khi có báo cáo lỗi, sửa đúng nguyên nhân, không xóa logic để test pass."""

# ---------- Helpers ----------
def safe_path(rel: str) -> Path:
    p = (WORKSPACE / rel).resolve()
    if not p.is_relative_to(WORKSPACE):            # chặn ../../etc/passwd
        raise ValueError(f"Path ngoài workspace: {rel}")
    return p

def read_workspace(max_chars: int = 60_000) -> str:
    """Gom code hiện có (src/) để đưa vào context."""
    parts, total = [], 0
    for f in sorted(WORKSPACE.glob("src/**/*.py")):
        text = f.read_text(encoding="utf-8")
        block = f"### {f.relative_to(WORKSPACE)}\n```python\n{text}\n```"
        if total + len(block) > max_chars:
            parts.append(f"### {f.relative_to(WORKSPACE)} (bỏ qua do quá dài)")
            continue
        parts.append(block); total += len(block)
    return "\n\n".join(parts) or "(workspace trống)"

def apply_changes(changes: list[FileChange]) -> list[str]:
    touched = []
    for c in changes:
        p = safe_path(c.path)
        if c.action == "delete":
            p.unlink(missing_ok=True)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(c.content, encoding="utf-8")
        touched.append(f"{c.action}: {c.path}")
    return touched

def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=WORKSPACE, capture_output=True, text=True).stdout

def init_workspace():
    WORKSPACE.mkdir(exist_ok=True)
    if not (WORKSPACE / ".git").exists():
        git("init"); git("commit", "--allow-empty", "-m", "init"); git("tag", "base")

# ---------- Node ----------
def code_node(state: SDLCState) -> dict:
    init_workspace()
    it = state.get("iterations", 0)

    msg = f"## Plan\n{state['plan']}\n\n## Code hiện có\n{read_workspace()}"

    # Vòng sửa lỗi: ưu tiên feedback gần nhất
    if state.get("test_report") and not state.get("tests_passed", True):
        msg += f"\n\n## Test FAIL — sửa các lỗi sau\n{state['test_report'][-8000:]}"
    if state.get("review_feedback") and not state.get("review_approved", True):
        msg += f"\n\n## Reviewer yêu cầu sửa\n{state['review_feedback']}"
    if state.get("eval_feedback") and state.get("eval_target") == "code":
        msg += f"\n\n## Evaluator yêu cầu cải thiện\n{state['eval_feedback']}"

    out: CodeOutput = coder_structured.invoke([SystemMessage(CODER_SYSTEM), HumanMessage(msg)])

    touched = apply_changes(out.changes)
    git("add", "-A")
    git("commit", "-m", f"code iteration {it + 1}")
    diff = git("diff", "HEAD~1", "HEAD")

    return {
        "code_files": touched,
        "code_notes": out.notes,
        "diff": diff,                  # node Review sẽ đọc diff thay vì toàn bộ code
        "iterations": it + 1,
    }
