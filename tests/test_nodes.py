from datetime import date
from pathlib import Path

import pytest

from multi_agent_sdlc.nodes.code_node import WORKSPACE, safe_path
from multi_agent_sdlc.nodes.evaluate_node import (
    CriterionVerdict, DimensionScore, Evaluation, after_eval, compute_score, parse_coverage)
from multi_agent_sdlc.nodes.review_node import ReviewComment, after_review, format_feedback
from multi_agent_sdlc.nodes.ship_node import ReleaseInfo, bump_version, render_changelog_entry
from multi_agent_sdlc.nodes.test_node import after_test


# ---------- safe_path ----------
def test_safe_path_resolves_inside_workspace():
    assert safe_path("src/a.py") == WORKSPACE / "src" / "a.py"


@pytest.mark.parametrize("rel", ["../evil.py", "src/../../evil.py"])
def test_safe_path_rejects_escape(rel):
    with pytest.raises(ValueError):
        safe_path(rel)


# ---------- parse_coverage ----------
def test_parse_coverage_reads_total_line():
    report = "Name   Stmts   Miss  Cover\nsrc/a.py  10  2  80%\nTOTAL  10  2  80%\n"
    assert parse_coverage(report) == 80


def test_parse_coverage_reads_branch_coverage_total_line():
    report = "Name  Stmts  Miss Branch BrPart  Cover\nTOTAL  120  10  30  5  90%\n"
    assert parse_coverage(report) == 90


def test_parse_coverage_returns_none_without_total():
    assert parse_coverage("3 passed") is None


# ---------- compute_score ----------
def make_eval(score=5, correctness=None, statuses=("met",)):
    dim = lambda s: DimensionScore(score=s, reason="r")
    return Evaluation(
        criteria=[CriterionVerdict(criterion=f"c{i}", status=s, evidence="e")
                  for i, s in enumerate(statuses)],
        correctness=dim(correctness or score), design=dim(score), code_quality=dim(score),
        robustness=dim(score), test_adequacy=dim(score),
        issue_origin="none", feedback=[])


def test_compute_score_perfect_eval_scores_one_without_gates():
    assert compute_score(make_eval(5), coverage=95, lint_count=0) == (1.0, [])


def test_compute_score_penalises_low_coverage_and_lint():
    score, _ = compute_score(make_eval(5), coverage=50, lint_count=11)
    assert score == 0.9


def test_compute_score_gates_on_unmet_criterion():
    _, gates = compute_score(make_eval(5, statuses=("met", "not_met")), 95, 0)
    assert gates == ["Criteria chưa đạt: ['c1']"]


def test_compute_score_gates_on_low_correctness():
    _, gates = compute_score(make_eval(5, correctness=2), 95, 0)
    assert gates == ["Correctness quá thấp"]


# ---------- bump_version / changelog ----------
@pytest.mark.parametrize("cur,bump,expected", [
    ((0, 0, 0), "major", "v0.1.0"),
    ((1, 2, 3), "major", "v2.0.0"),
    ((1, 2, 3), "minor", "v1.3.0"),
    ((1, 2, 3), "patch", "v1.2.4"),
])
def test_bump_version(cur, bump, expected):
    assert bump_version(cur, bump) == expected


def test_render_changelog_entry_skips_empty_and_unknown_sections():
    info = ReleaseInfo(bump="minor", title="t", highlights=[],
                       changelog={"Added": ["login"], "Fixed": [], "Other": ["x"]})
    assert render_changelog_entry("v1.0.0", info) == (
        f"## [v1.0.0] - {date.today().isoformat()}\n### Added\n- login\n")


# ---------- format_feedback ----------
def test_format_feedback_keeps_only_blocking_comments():
    comments = [
        ReviewComment(file="a.py", line=3, severity="blocker", category="security",
                      comment="sql injection", suggestion="use params"),
        ReviewComment(file="b.py", line=None, severity="nit", category="style", comment="naming"),
    ]
    assert format_feedback(comments) == (
        "- [BLOCKER][security] a.py:3 — sql injection\n  Gợi ý:\n```python\nuse params\n```")


# ---------- routers ----------
@pytest.mark.parametrize("state,expected", [
    ({"tests_passed": True}, "evaluate"),
    ({"tests_passed": False, "failure_source": "test_bug", "test_fix_attempts": 2, "iterations": 3}, "test"),
    ({"tests_passed": False, "failure_source": "test_bug", "test_fix_attempts": 3, "iterations": 1}, "code"),
    ({"tests_passed": False, "failure_source": "code_bug", "test_fix_attempts": 0, "iterations": 3}, "__end__"),
])
def test_after_test(state, expected):
    assert after_test(state) == expected


@pytest.mark.parametrize("state,expected", [
    ({"eval_passed": True}, "review"),
    ({"eval_passed": False, "eval_target": "plan", "plan_revisions": 0, "iterations": 1}, "plan"),
    ({"eval_passed": False, "eval_target": "plan", "plan_revisions": 1, "iterations": 1}, "code"),
    ({"eval_passed": False, "eval_target": "code", "plan_revisions": 0, "iterations": 3}, "__end__"),
])
def test_after_eval(state, expected):
    assert after_eval(state) == expected


@pytest.mark.parametrize("state,expected", [
    ({"review_approved": True}, "ship"),
    ({"review_approved": False, "iterations": 1}, "code"),
    ({"review_approved": False, "iterations": 3, "has_blockers": True}, "__end__"),
    ({"review_approved": False, "iterations": 3, "has_blockers": False}, "ship"),
])
def test_after_review(state, expected):
    assert after_review(state) == expected


def test_graph_builds_with_release_step_and_no_static_interrupt():
    from multi_agent_sdlc.graph import build_graph
    app = build_graph()
    edges = {(e.source, e.target) for e in app.get_graph().edges}
    assert {("ship", "release"), ("release", "code"), ("release", "__end__")} <= edges
    assert not app.interrupt_before_nodes  # release_node pauses itself via interrupt()


# ---------- subprocess output decoding ----------
def test_git_decodes_utf8_output_regardless_of_windows_codepage(tmp_path, monkeypatch):
    import importlib
    cn = importlib.import_module("multi_agent_sdlc.nodes.code_node")
    monkeypatch.setattr(cn, "WORKSPACE", tmp_path)
    text = "Mật khẩu “đúng”\n"  # “ ” encode to bytes cp1252 can't decode (0x9d)
    (tmp_path / "a.txt").write_text(text, encoding="utf-8")
    cn.git("init", "-q")
    cn.git("add", "a.txt")
    cn.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x")
    assert cn.git("show", "HEAD:a.txt") == text


def test_workspace_does_not_depend_on_cwd(tmp_path, monkeypatch):
    import importlib
    import multi_agent_sdlc
    monkeypatch.chdir(tmp_path)  # e.g. running run.py from inside src/multi_agent_sdlc
    cn = importlib.reload(importlib.import_module("multi_agent_sdlc.nodes.code_node"))
    assert cn.WORKSPACE == Path(multi_agent_sdlc.__file__).resolve().parents[2] / "workspace"


# ---------- sandbox boundary: LLM output must never reach workspace/.git ----------
# A planted .git/hooks/* or core.fsmonitor in .git/config runs on the host at the next git call.
@pytest.mark.parametrize("rel", [".git", ".git/config", ".GIT/hooks/post-commit", "tests/../.git/hooks/post-commit"])
def test_safe_path_rejects_git_dir(rel):
    with pytest.raises(ValueError):
        safe_path(rel)


def test_generate_tests_only_writes_inside_tests_dir(tmp_path, monkeypatch):
    import importlib
    from types import SimpleNamespace
    cn = importlib.import_module("multi_agent_sdlc.nodes.code_node")
    tn = importlib.import_module("multi_agent_sdlc.nodes.test_node")
    for m in (cn, tn):
        monkeypatch.setattr(m, "WORKSPACE", tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text("real code", encoding="utf-8")
    suite = tn.TestSuite(files=[tn.TestFile(path="tests/test_ok.py", content="ok"),
                                tn.TestFile(path="tests/../src/auth.py", content="overwritten")],
                         criteria_coverage={})
    monkeypatch.setattr(tn, "tester_gen", lambda: SimpleNamespace(invoke=lambda _: suite))

    written = tn.generate_tests({"acceptance_criteria": ["c"]})

    assert written == ["tests/test_ok.py"]
    assert (tmp_path / "src" / "auth.py").read_text(encoding="utf-8") == "real code"


def test_pytest_sandbox_mounts_git_dir_read_only(monkeypatch):
    import importlib
    tn = importlib.import_module("multi_agent_sdlc.nodes.test_node")
    seen = {}
    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        raise tn.subprocess.TimeoutExpired(cmd, 1)
    monkeypatch.setattr(tn.subprocess, "run", fake_run)
    tn.run_pytest()
    assert f"{tn.WORKSPACE / '.git'}:/app/.git:ro" in seen["cmd"]


# ---------- LLM output budget ----------
# OmniRoute reports a cut-off tool call as finish_reason=tool_calls with args={}, which surfaces as a
# misleading "Field required" ValidationError, so every role needs an explicit, generous max_tokens.
def test_get_llm_sets_generous_default_max_tokens(monkeypatch):
    from multi_agent_sdlc import llm
    monkeypatch.setattr(llm, "OMNI_BASE_URL", "http://x")
    monkeypatch.setattr(llm, "OMNI_API_KEY", "k")
    monkeypatch.setitem(llm.MODELS, "tester", "m")
    llm.get_llm.cache_clear()
    assert llm.get_llm("tester").max_tokens >= 32000
    llm.get_llm.cache_clear()


# ---------- skipped tests must not count as passing ----------
def test_parse_junit_reports_skipped_tests(tmp_path, monkeypatch):
    import importlib
    tn = importlib.import_module("multi_agent_sdlc.nodes.test_node")
    monkeypatch.setattr(tn, "WORKSPACE", tmp_path)
    (tmp_path / ".report.xml").write_text(
        '<testsuites><testsuite><testcase classname="tests.test_a" name="test_login">'
        '<skipped type="pytest.skip" message="AUTH_TEST_ADAPTER missing">reason</skipped>'
        '</testcase><testcase classname="tests.test_a" name="test_ok"/></testsuite></testsuites>',
        encoding="utf-8")
    assert tn.parse_junit() == [{"test": "tests.test_a::test_login", "type": "skipped",
                                 "message": "AUTH_TEST_ADAPTER missing", "trace": "reason"}]


def test_test_node_fails_when_pytest_exits_zero_but_tests_were_skipped(tmp_path, monkeypatch):
    import importlib
    from types import SimpleNamespace
    tn = importlib.import_module("multi_agent_sdlc.nodes.test_node")
    monkeypatch.setattr(tn, "WORKSPACE", tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("", encoding="utf-8")  # existing tests -> no generation
    skipped = {"test": "tests.test_a::test_login", "type": "skipped", "message": "no adapter", "trace": ""}
    monkeypatch.setattr(tn, "run_pytest", lambda: {"exit_code": 0, "output": "1 skipped", "failures": [skipped]})
    triage = tn.FailureTriage(source="test_bug", reasoning="test tự skip", fix_hint="bỏ skip")
    monkeypatch.setattr(tn, "tester_triage", lambda: SimpleNamespace(invoke=lambda _: triage))

    update = tn.test_node({"acceptance_criteria": ["c"]})

    assert update["tests_passed"] is False
    assert "[SKIPPED] tests.test_a::test_login" in update["test_report"]


# ---------- workspace git ignores test artifacts ----------
def test_init_workspace_ignores_test_artifacts(tmp_path, monkeypatch):
    import importlib
    cn = importlib.import_module("multi_agent_sdlc.nodes.code_node")
    ws = tmp_path / "workspace"
    monkeypatch.setattr(cn, "WORKSPACE", ws)
    cn.init_workspace()
    for junk in (".coverage", ".report.xml", "src/__pycache__/x.pyc"):
        (ws / junk).parent.mkdir(parents=True, exist_ok=True)
        (ws / junk).write_text("x", encoding="utf-8")
    assert cn.git("status", "--porcelain") == ""
    assert "base" in cn.git("tag").split()


# ---------- review_node prompt ----------
def test_review_prompt_closes_every_diff_fence(monkeypatch):
    import importlib, re
    from types import SimpleNamespace
    rn = importlib.import_module("multi_agent_sdlc.nodes.review_node")
    monkeypatch.setattr(rn, "git", lambda *a: "+x = 1\n")
    monkeypatch.setattr(rn, "run_bandit", lambda: ["[HIGH] B602 shell=True"])
    seen = {}

    class Captured(Exception):
        pass

    def invoke(msgs):
        seen["msg"] = msgs[1].content
        raise Captured
    monkeypatch.setattr(rn, "reviewer_structured", lambda: SimpleNamespace(invoke=invoke))
    state = {"requirement": "r", "diff": "+y = 2\n", "open_review_comments": [{"file": "a.py", "line": 1, "severity": "major", "comment": "c"}]}
    with pytest.raises(Captured):
        rn.review_node(state)

    lines = seen["msg"].splitlines()
    opens = [i for i, l in enumerate(lines) if re.fullmatch(r"`{3,}diff", l)]
    assert len(opens) == 2
    for i in opens:
        fence = lines[i].removesuffix("diff")
        assert fence in lines[i + 1:], f"fence {fence!r} opened at line {i} is never closed"
    # bandit findings must be prose, not swallowed by an unclosed code block
    assert lines.index("## Bandit (security scan)") > max(lines.index(lines[i].removesuffix("diff"), i + 1) for i in opens)


# ---------- read_tests ----------
def test_read_tests_reads_utf8_and_is_capped(tmp_path, monkeypatch):
    import importlib
    tn = importlib.import_module("multi_agent_sdlc.nodes.test_node")
    monkeypatch.setattr(tn, "WORKSPACE", tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text('"""Khóa tài khoản — “đúng”"""\n', encoding="utf-8")
    (tmp_path / "tests" / "test_b.py").write_text("x = 1\n" * 50_000, encoding="utf-8")
    out = tn.read_tests()
    assert "Khóa tài khoản — “đúng”" in out
    assert len(out) <= 60_000


# ---------- checkpointer ----------
def test_checkpointer_uses_postgres_when_uri_set(monkeypatch):
    from langgraph.checkpoint.postgres import PostgresSaver
    from multi_agent_sdlc import graph
    calls = {}
    monkeypatch.setattr(graph, "POSTGRES_URI", "postgresql://u:p@localhost:5432/db")
    monkeypatch.setattr(graph.Connection, "connect", lambda uri, **kw: calls.setdefault("conn", (uri, kw)) and object())
    monkeypatch.setattr(PostgresSaver, "setup", lambda self: calls.setdefault("setup", True))
    saver = graph.make_checkpointer()
    assert isinstance(saver, PostgresSaver)
    assert calls["conn"][0] == "postgresql://u:p@localhost:5432/db"
    assert calls["conn"][1]["autocommit"] is True  # setup() chạy CREATE INDEX CONCURRENTLY, không được nằm trong transaction
    assert calls["setup"]


def test_checkpointer_falls_back_to_memory_without_uri(monkeypatch):
    from langgraph.checkpoint.memory import MemorySaver
    from multi_agent_sdlc import graph
    monkeypatch.setattr(graph, "POSTGRES_URI", None)
    assert isinstance(graph.make_checkpointer(), MemorySaver)
