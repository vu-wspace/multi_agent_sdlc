import hashlib
import importlib
from types import SimpleNamespace
from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

run = importlib.import_module("multi_agent_sdlc.run")
cn = importlib.import_module("multi_agent_sdlc.nodes.code_node")
sn = importlib.import_module("multi_agent_sdlc.nodes.ship_node")

REQ = "login"
BASE = "run-" + hashlib.md5(REQ.encode(), usedforsecurity=False).hexdigest()[:8]


class S(TypedDict, total=False):
    requirement: str
    crash: bool


def make_app():
    def work(s):
        if s.get("crash"):
            raise RuntimeError("LLM timeout")
        return {}

    def release(s):
        interrupt({"message": "Duyệt?", "release_notes": "notes"})
        return {}
    g = StateGraph(S)
    g.add_node("work", work); g.add_node("release", release)
    g.add_edge(START, "work"); g.add_edge("work", "release"); g.add_edge("release", END)
    return g.compile(checkpointer=MemorySaver())


def cfg(tid):
    return {"configurable": {"thread_id": tid}}


# ---------- run.prepare_thread ----------
def test_new_requirement_starts_fresh_on_base_thread():
    config, mode = run.prepare_thread(make_app(), REQ)
    assert config["configurable"]["thread_id"] == BASE
    assert mode == "fresh"


def test_thread_paused_at_release_goes_to_approval():
    app = make_app()
    app.invoke({"requirement": REQ}, cfg(BASE))
    config, mode = run.prepare_thread(app, REQ)
    assert config["configurable"]["thread_id"] == BASE
    assert mode == "approve"


def test_thread_that_crashed_mid_run_continues_instead_of_indexerror():
    app = make_app()
    with pytest.raises(RuntimeError):
        app.invoke({"requirement": REQ, "crash": True}, cfg(BASE))
    config, mode = run.prepare_thread(app, REQ)
    assert config["configurable"]["thread_id"] == BASE
    assert mode == "continue"


def test_finished_thread_gets_new_suffixed_thread():
    app = make_app()
    app.invoke({"requirement": REQ}, cfg(BASE))
    app.invoke(Command(resume={"approved": True}), cfg(BASE))
    config, mode = run.prepare_thread(app, REQ)
    assert config["configurable"]["thread_id"].startswith(BASE + "-")
    assert mode == "fresh"


def test_explicit_thread_id_resumes_suffixed_run():
    app = make_app()
    app.invoke({"requirement": REQ}, cfg(BASE + "-ab12"))
    config, mode = run.prepare_thread(app, REQ, thread_id=BASE + "-ab12")
    assert config["configurable"]["thread_id"] == BASE + "-ab12"
    assert mode == "approve"


def test_requirement_whitespace_does_not_change_thread():
    a, _ = run.prepare_thread(make_app(), "login  module\n")
    b, _ = run.prepare_thread(make_app(), "login module")
    assert a["configurable"]["thread_id"] == b["configurable"]["thread_id"]


def test_project_context_is_capped(tmp_path, monkeypatch):
    f = tmp_path / "ctx.md"
    f.write_text("Bối cảnh " + "x" * 100_000, encoding="utf-8")
    monkeypatch.setattr(run, "CONTEXT_FILE", f)
    ctx = run.load_context()
    assert ctx.startswith("Bối cảnh") and len(ctx) <= 20_000


def test_missing_project_context_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "CONTEXT_FILE", tmp_path / "nope.md")
    assert run.load_context() == ""


# ---------- release_node gắn với đúng commit release ----------
@pytest.fixture
def ws(tmp_path, monkeypatch):
    w = tmp_path / "ws"
    monkeypatch.setattr(cn, "WORKSPACE", w)
    monkeypatch.setattr(sn, "WORKSPACE", w)
    cn.init_workspace()
    (w / "REVIEW.md").write_text("ok", encoding="utf-8")
    (w / "a.py").write_text("x = 1\n", encoding="utf-8")
    cn.git("add", "-A"); cn.git("commit", "-m", "code")
    (w / "RELEASE_NOTES.md").write_text("v0.1.0", encoding="utf-8")
    cn.git("add", "-A"); cn.git("commit", "-m", "chore(release): v0.1.0")
    return w


def head():
    return cn.git("rev-parse", "HEAD").strip()


def release_state():
    return {"version": "v0.1.0", "release_notes": "n", "release_sha": head(), "iterations": 2}


def other_run_commits(w):
    (w / "b.py").write_text("y = 2\n", encoding="utf-8")
    cn.git("add", "-A"); cn.git("commit", "-m", "other run")


def test_release_approve_tags_release_commit(ws, monkeypatch):
    state = release_state()
    monkeypatch.setattr(sn, "interrupt", lambda _: {"approved": True})
    sn.release_node(state)
    assert cn.git("rev-list", "-n1", "v0.1.0").strip() == state["release_sha"]


def test_release_refuses_when_workspace_head_moved(ws, monkeypatch):
    state = release_state()
    other_run_commits(ws)
    moved = head()
    for decision in ({"approved": True}, {"approved": False, "feedback": "f"}):
        monkeypatch.setattr(sn, "interrupt", lambda _, d=decision: d)
        with pytest.raises(RuntimeError, match="HEAD"):
            sn.release_node(state)
        assert head() == moved                       # không reset mất commit của run khác
        assert cn.git("tag", "-l", "v0.1.0").strip() == ""


def test_release_reject_drops_only_release_commit(ws, monkeypatch):
    state = release_state()
    monkeypatch.setattr(sn, "interrupt", lambda _: {"approved": False, "feedback": "f"})
    cmd = sn.release_node(state)
    assert head() == cn.git("rev-parse", state["release_sha"] + "~1").strip()
    assert cmd.goto == "code"


def test_ship_node_records_release_sha(ws, monkeypatch):
    info = sn.ReleaseInfo(bump="minor", title="t", highlights=[], changelog={})
    monkeypatch.setattr(sn, "shipper_structured", lambda: SimpleNamespace(invoke=lambda _: info))
    monkeypatch.setattr(sn, "render_release_notes", lambda *a: "notes")
    monkeypatch.setattr(sn, "render_changelog_entry", lambda *a: "entry")
    out = sn.ship_node({"requirement": "r", "plan": "p"})
    assert out["release_sha"] == head()


@pytest.mark.parametrize("values,expected", [
    ({"released": True}, None),
    ({"tests_passed": False, "iterations": 5}, "test vẫn fail"),
    ({"tests_passed": True, "eval_passed": False, "eval_feedback": "thiếu edge case"}, "thiếu edge case"),
    ({"tests_passed": True, "eval_passed": False, "iterations": 5}, "evaluate chưa đạt"),
    ({"tests_passed": True, "eval_passed": True, "has_blockers": True, "iterations": 5}, "review còn blocker"),
])
def test_stop_reason(values, expected):
    reason = run.stop_reason(values)
    assert reason is None if expected is None else expected in reason


def test_stop_reason_truncates_detail():
    reason = run.stop_reason({"tests_passed": False, "test_report": "\n".join(f"l{i}" for i in range(30))})
    lines = reason.splitlines()
    assert "l19" in lines and "l20" not in lines
