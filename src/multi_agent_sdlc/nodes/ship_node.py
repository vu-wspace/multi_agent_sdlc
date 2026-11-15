import re
from datetime import date
from typing import Literal
from pydantic import BaseModel, Field
from langgraph.types import interrupt, Command
from langchain_openai import ChatOpenAI
from llm import get_llm

# ---------- Schema ----------
class ReleaseInfo(BaseModel):
    bump: Literal["major", "minor", "patch"] = Field(
        description="major = phá vỡ API cũ; minor = thêm tính năng; patch = chỉ sửa lỗi")
    title: str = Field(description="Tiêu đề release ngắn gọn")
    highlights: list[str] = Field(description="Tính năng/thay đổi chính, viết cho người dùng cuối")
    changelog: dict[str, list[str]] = Field(
        description="Theo Keep a Changelog: keys trong Added, Changed, Fixed, Security")
    known_issues: list[str] = Field(default_factory=list)

shipper_structured = (get_llm('shipper', temperature=0.2)
                      .with_structured_output(ReleaseInfo, method="function_calling")
                      .with_retry(stop_after_attempt=2))

SHIPPER_SYSTEM = """Bạn là Release Manager. Viết release notes rõ ràng, trung thực.
- Chỉ nêu những gì THỰC SỰ có trong diff, không phóng đại.
- Mọi vấn đề review còn mở phải nằm trong known_issues.
- Chọn bump theo Semantic Versioning."""

# ---------- Version ----------
def latest_version() -> tuple[int, int, int]:
    tags = git("tag", "--list", "v*", "--sort=-v:refname").split()
    for t in tags:
        m = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", t)
        if m:
            return tuple(map(int, m.groups()))
    return (0, 0, 0)

def bump_version(cur: tuple[int, int, int], bump: str) -> str:
    major, minor, patch = cur
    if cur == (0, 0, 0):
        return "v0.1.0"                                 # release đầu tiên
    if bump == "major": return f"v{major + 1}.0.0"
    if bump == "minor": return f"v{major}.{minor + 1}.0"
    return f"v{major}.{minor}.{patch + 1}"

# ---------- File release ----------
def render_changelog_entry(version: str, info: ReleaseInfo) -> str:
    lines = [f"## [{version}] - {date.today().isoformat()}"]
    for section in ("Added", "Changed", "Fixed", "Security"):
        items = info.changelog.get(section, [])
        if items:
            lines += [f"### {section}", *[f"- {i}" for i in items]]
    return "\n".join(lines) + "\n"

def update_changelog(entry: str) -> None:
    path = WORKSPACE / "CHANGELOG.md"
    header = "# Changelog\n\n"
    old = path.read_text(encoding="utf-8").removeprefix(header) if path.exists() else ""
    path.write_text(header + entry + "\n" + old, encoding="utf-8")

def render_release_notes(version: str, info: ReleaseInfo, state: SDLCState) -> str:
    warn = ""
    if not state.get("review_approved"):
        warn = "\n> ⚠️ Release khi review chưa approve hoàn toàn (hết lượt lặp, còn issue major).\n"
    return f"""# {version} — {info.title}
{warn}
## Highlights
{chr(10).join(f'- {h}' for h in info.highlights)}

## Quality
- Tests: PASS
- Evaluation score: {state.get('eval_score')}
- Iterations: {state.get('iterations')}
- Review: {state.get('review_summary', '')}

## Known issues
{chr(10).join(f'- {k}' for k in info.known_issues) or '- Không có'}
"""

# ---------- Node: ship ----------
def ship_node(state: SDLCState) -> dict:
    open_issues = "\n".join(f"- [{c['severity']}] {c['file']}: {c['comment']}"
                            for c in state.get("open_review_comments", [])) or "Không có"
    msg = f"""## Yêu cầu
{state['requirement']}

## Plan
{state['plan']}

## Diff toàn bộ
```diff
{git('diff', 'base', 'HEAD', '--', 'src')[-40_000:]}
```

## Review summary
{state.get('review_summary', '')}

## Vấn đề còn mở
{open_issues}"""

    info: ReleaseInfo = shipper_structured.invoke([SystemMessage(SHIPPER_SYSTEM), HumanMessage(msg)])
    version = bump_version(latest_version(), info.bump)

    notes = render_release_notes(version, info, state)
    (WORKSPACE / "RELEASE_NOTES.md").write_text(notes, encoding="utf-8")
    update_changelog(render_changelog_entry(version, info))
    git("add", "-A"); git("commit", "-m", f"chore(release): {version}")

    return {"version": version, "release_notes": notes}

# ---------- Node: release (human-in-the-loop) ----------
def release_node(state: SDLCState) -> Command[Literal["code", "__end__"]]:
    decision = interrupt({                               # pipeline dừng tại đây
        "message": f"Duyệt release {state['version']}?",
        "release_notes": state["release_notes"],
        "review": (WORKSPACE / "REVIEW.md").read_text(encoding="utf-8"),
    })

    if decision.get("approved"):
        git("tag", "-a", state["version"], "-m", f"Release {state['version']}")
        return Command(goto=END, update={"released": True})

    # Từ chối: hủy commit release, quay lại Code với feedback của người duyệt
    git("reset", "--hard", "HEAD~1")
    return Command(goto="code", update={
        "released": False,
        "review_approved": False,
        "review_feedback": f"## Người duyệt yêu cầu sửa\n{decision.get('feedback', '')}",
        "iterations": max(state["iterations"] - 1, 0),   # cho thêm 1 lượt sửa
    })
