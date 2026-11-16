import subprocess
from functools import cache
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..llm import structured
from ..state import SDLCState

WORKSPACE = Path(__file__).resolve().parents[3] / "workspace"   # <repo>/workspace, không phụ thuộc cwd

# ---------- Schema output ----------
class FileChange(BaseModel):
    path: str = Field(description="Đường dẫn tương đối trong workspace, vd: src/auth.py")
    action: Literal["create", "update", "delete"]
    content: str = Field(default="", description="Nội dung ĐẦY ĐỦ của file (không phải diff)")

class CodeOutput(BaseModel):
    changes: list[FileChange]
    notes: str = Field(description="Giải thích ngắn đã làm gì / sửa lỗi gì")

@cache
def coder_structured():
    return structured("coder", CodeOutput, max_tokens=64_000)

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
    # chặn .git: hook / core.fsmonitor do LLM ghi sẽ chạy trên máy host ở lệnh git kế tiếp
    # (lower(): NTFS không phân biệt hoa thường, .GIT cũng là .git)
    if p.relative_to(WORKSPACE).parts[:1] and p.relative_to(WORKSPACE).parts[0].lower() == ".git":
        raise ValueError(f"Không được ghi vào .git: {rel}")
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

CODER_ALLOWED_FILES = {"README.md", "pyproject.toml"}

def coder_may_write(p: Path) -> bool:
    """Coder chỉ ghi code nguồn: test thuộc Tester; conftest / config lint có thể dùng để nới cổng chất lượng."""
    rel = p.relative_to(WORKSPACE)
    if rel.name == "conftest.py":
        return False
    return rel.parts[0] == "src" or rel.as_posix() in CODER_ALLOWED_FILES

def apply_changes(changes: list[FileChange]) -> list[str]:
    touched = []
    for c in changes:
        p = safe_path(c.path)
        if not coder_may_write(p):
            touched.append(f"skipped (không được phép): {c.path}")
            continue
        if c.action == "delete":
            p.unlink(missing_ok=True)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(c.content, encoding="utf-8")
        touched.append(f"{c.action}: {c.path}")
    return touched

# ---------- Sandbox Docker cho code do LLM viết ----------
SANDBOX = ["docker", "run", "--rm",
           "--network", "none",                                 # không cho ra mạng
           "--memory", "512m", "--cpus", "1", "--pids-limit", "128",
           "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
           "--user", "1000:1000"]                               # không chạy bằng root
DOCKER_FAILED = (125, 126, 127)   # docker run hỏng (thiếu image, daemon tắt); 137/139 = OOM/segfault của code sinh ra

def git(*args: str, check: bool = False) -> str:
    r = subprocess.run(["git", *args], cwd=WORKSPACE, capture_output=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:   # commit/tag/reset lỗi mà im lặng = release giả hoặc mất code
        raise RuntimeError(f"git {' '.join(args[:1])} thất bại ({r.returncode}): {r.stderr.strip()[:300]}")
    return r.stdout

def init_workspace():
    WORKSPACE.mkdir(exist_ok=True)
    if not (WORKSPACE / ".git").exists():
        # file sinh ra khi chạy test không được lọt vào diff mà Review / Ship đọc
        (WORKSPACE / ".gitignore").write_text(
            ".coverage\n.report.xml\n__pycache__/\n*.pyc\n.pytest_cache/\n", encoding="utf-8")
        git("init"); git("add", ".gitignore"); git("commit", "-m", "init"); git("tag", "base")

# ---------- Node ----------
def code_node(state: SDLCState) -> dict:
    init_workspace()
    it = state.get("iterations", 0)
    passes = state.get("code_passes", 0) + 1   # số lượt thật; iterations bị trả lại khi reject release

    msg = f"## Plan\n{state['plan']}\n\n## Code hiện có\n{read_workspace()}"

    # Vòng sửa lỗi: ưu tiên feedback gần nhất
    if state.get("test_report") and not state.get("tests_passed", True):
        msg += f"\n\n## Test FAIL — sửa các lỗi sau\n{state['test_report'][-8000:]}"
    if state.get("review_feedback") and not state.get("review_approved", True):
        msg += f"\n\n## Reviewer yêu cầu sửa\n{state['review_feedback']}"
    if state.get("eval_feedback") and state.get("eval_target") == "code":
        msg += f"\n\n## Evaluator yêu cầu cải thiện\n{state['eval_feedback']}"

    out: CodeOutput = coder_structured().invoke([SystemMessage(CODER_SYSTEM), HumanMessage(msg)])

    touched = apply_changes(out.changes)
    git("add", "-A")
    git("commit", "-m", f"code iteration {passes}")
    diff = git("diff", "HEAD~1", "HEAD")

    return {
        "code_files": touched,
        "code_notes": out.notes,
        "diff": diff,                  # node Review sẽ đọc diff thay vì toàn bộ code
        "iterations": it + 1,
        "code_passes": passes,
    }
