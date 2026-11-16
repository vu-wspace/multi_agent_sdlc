import subprocess
import xml.etree.ElementTree as ET
from functools import cache
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from pydantic import BaseModel, Field

from ..config import MAX_ITER, MAX_TEST_FIX
from ..llm import structured
from ..state import SDLCState
from .code_node import WORKSPACE, read_workspace, safe_path


# ---------- Schema ----------
class TestFile(BaseModel):
    path: str = Field(description="Phải nằm trong tests/, vd: tests/test_auth.py")
    content: str

class TestSuite(BaseModel):
    files: list[TestFile]
    criteria_coverage: dict[str, list[str]] = Field(
        description="Map acceptance criterion -> tên các test function kiểm tra nó")

class FailureTriage(BaseModel):
    source: Literal["code_bug", "test_bug"]
    reasoning: str
    fix_hint: str = Field(description="Gợi ý sửa ngắn gọn cho bên chịu trách nhiệm")

@cache
def tester_gen():
    return structured("tester", TestSuite)
@cache
def tester_triage():
    return structured("tester", FailureTriage, retries=1)

TESTER_SYSTEM = """Bạn là QA Engineer. Viết test pytest cho code được cung cấp.
- Mỗi acceptance criterion phải có ít nhất 1 test; thêm edge case (input rỗng, sai kiểu, biên).
- Import từ package src (vd: from src.auth import login).
- Test độc lập, không gọi mạng, không phụ thuộc thứ tự chạy.
- Test kiểm tra HÀNH VI theo yêu cầu, không sao chép logic của code.
- Chỉ Python + pytest. Sandbox không có mạng, chỉ có: pytest, pytest-cov, pytest-timeout, bcrypt.
- KHÔNG dùng pytest.skip / skipif / xfail / importorskip: test bị skip được tính là FAIL."""

# ---------- Sinh test ----------
def generate_tests(state: SDLCState, fix_hint: str = "") -> list[str]:
    criteria = "\n".join(f"- {c}" for c in state["acceptance_criteria"])
    msg = f"## Acceptance criteria\n{criteria}\n\n## Code\n{read_workspace()}"
    if fix_hint:
        msg += f"\n\n## Test cũ bị sai, sửa theo gợi ý\n{fix_hint}\n\n## Test hiện tại\n{read_tests()}"

    suite: TestSuite = tester_gen().invoke([SystemMessage(TESTER_SYSTEM), HumanMessage(msg)])

    written = []
    for f in suite.files:
        p = safe_path(f.path)
        if not p.is_relative_to(WORKSPACE / "tests"):  # so path đã resolve: "tests/../src/x.py" không lọt
            continue                                   # Tester chỉ được ghi vào tests/
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f.content, encoding="utf-8")
        written.append(f.path)
    (WORKSPACE / "tests" / "__init__.py").touch()
    (WORKSPACE / "src" / "__init__.py").touch()
    return written

def read_tests(max_chars: int = 60_000) -> str:
    text = "\n\n".join(f"### {f.relative_to(WORKSPACE)}\n```python\n{f.read_text(encoding='utf-8')}\n```"
                       for f in sorted(WORKSPACE.glob("tests/**/*.py")))
    return text[:max_chars]  # shortcut: cắt cứng, có thể cụt giữa file; tách theo file nếu LLM bị rối

# ---------- Chạy pytest trong sandbox ----------
def run_pytest(timeout: int = 120) -> dict:
    cmd = [
        "docker", "run", "--rm",
        "--network", "none",                 # code do LLM viết: không cho ra mạng
        "--memory", "512m", "--cpus", "1", "--pids-limit", "128",
        "-v", f"{WORKSPACE}:/app",
        "-v", f"{WORKSPACE / '.git'}:/app/.git:ro",  # code trong sandbox không được cài hook lên host
        "sdlc-sandbox",
        "python", "-m", "pytest", "tests", "-q",
        "--timeout=10",                      # mỗi test tối đa 10s (chống vòng lặp vô hạn)
        "--junitxml=/app/.report.xml",
        "--cov=src", "--cov-report=term",
        "-p", "no:cacheprovider",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout)
        exit_code, output = r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        exit_code, output = -1, "TIMEOUT: toàn bộ test suite vượt quá thời gian cho phép"
    return {"exit_code": exit_code, "output": output, "failures": parse_junit()}

def parse_junit() -> list[dict]:
    path = WORKSPACE / ".report.xml"
    if not path.exists():
        return []
    fails = []
    for case in ET.parse(path).iter("testcase"):
        for tag in ("failure", "error", "skipped"):  # skip = criterion chưa được kiểm tra
            node = case.find(tag)
            if node is not None:
                fails.append({
                    "test": f"{case.get('classname')}::{case.get('name')}",
                    "type": tag,
                    "message": (node.get("message") or "")[:300],
                    "trace": (node.text or "")[-1500:],
                })
    return fails

# pytest exit codes: 0=pass, 1=có test fail, 2=bị ngắt, 3=lỗi nội bộ, 4=sai cú pháp lệnh, 5=không có test nào
def format_report(res: dict) -> str:
    lines = [f"exit_code={res['exit_code']}, {len(res['failures'])} test fail"]
    for f in res["failures"][:10]:
        lines.append(f"\n[{f['type'].upper()}] {f['test']}\n{f['message']}\n{f['trace']}")
    lines.append("\n--- pytest tail ---\n" + res["output"][-2000:])
    return "\n".join(lines)

# ---------- Node ----------
def test_node(state: SDLCState) -> dict:
    # Chỉ sinh test lần đầu, hoặc khi vòng trước xác định test viết sai
    if not any(WORKSPACE.glob("tests/test_*.py")):
        generate_tests(state)
    elif state.get("failure_source") == "test_bug":
        generate_tests(state, fix_hint=state.get("test_fix_hint", ""))

    res = run_pytest()
    if res["exit_code"] >= 125:   # 125-127 = docker run hỏng (thiếu image, daemon tắt): không phải lỗi test/code
        raise RuntimeError(f"Docker không chạy được pytest: {res['output'].strip()[:300]}")
    report = format_report(res)
    passed = res["exit_code"] == 0 and not res["failures"]   # exit 0 vẫn có thể toàn test bị skip

    update = {
        "tests_passed": passed,
        "test_report": report,
        "failure_source": None,
        "test_fix_attempts": state.get("test_fix_attempts", 0),
    }
    if passed:
        return update

    # Exit 5 = không có test → chắc chắn lỗi phía test
    if res["exit_code"] == 5:
        triage = FailureTriage(source="test_bug", reasoning="Không collect được test", fix_hint="Kiểm tra tên file test_*.py và import")
    else:
        triage = tester_triage().invoke([
            SystemMessage("Phân tích test fail: lỗi do CODE sai yêu cầu hay do TEST viết sai? "
                          "Căn cứ vào acceptance criteria, không căn cứ vào code hiện tại."),
            HumanMessage(f"## Acceptance criteria\n{state['acceptance_criteria']}\n\n## Report\n{report}\n\n"
                         f"## Tests\n{read_tests()}"),
        ])

    update["failure_source"] = triage.source
    if triage.source == "test_bug":
        update["test_fix_hint"] = triage.fix_hint
        update["test_fix_attempts"] += 1
    else:
        update["test_report"] = report + f"\n\n## Phân tích của Tester\n{triage.reasoning}\nGợi ý: {triage.fix_hint}"
    return update


def after_test(s) -> Literal["evaluate", "code", "test", "__end__"]:
    if s["tests_passed"]:
        return "evaluate"
    if s["failure_source"] == "test_bug" and s["test_fix_attempts"] <= MAX_TEST_FIX:
        return "test"                     # Tester tự sửa test
    if s["iterations"] < MAX_ITER:
        return "code"                     # Coder sửa code
    return END                            # hết lượt → dừng, chờ người xử lý
