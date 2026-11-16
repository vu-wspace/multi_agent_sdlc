"""Các lỗi từ đợt /ship thứ hai."""
import builtins
import hashlib
import importlib
import subprocess
import sys
from types import SimpleNamespace
from typing import TypedDict

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from multi_agent_sdlc import config, graph, llm

run = importlib.import_module("multi_agent_sdlc.run")
cn = importlib.import_module("multi_agent_sdlc.nodes.code_node")
tn = importlib.import_module("multi_agent_sdlc.nodes.test_node")
en = importlib.import_module("multi_agent_sdlc.nodes.evaluate_node")
rn = importlib.import_module("multi_agent_sdlc.nodes.review_node")
sn = importlib.import_module("multi_agent_sdlc.nodes.ship_node")


def proc(code, out=""):
    return SimpleNamespace(stdout=out, stderr="", returncode=code)


# ---------- 1. OOM (137) / segfault (139) là lỗi của code sinh ra, không phải Docker hỏng ----------
@pytest.mark.parametrize("code", [137, 139])
def test_test_node_triages_oom_instead_of_crashing(monkeypatch, tmp_path, code):
    monkeypatch.setattr(tn, "WORKSPACE", tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(tn, "run_pytest", lambda: {"exit_code": code, "output": "Killed", "failures": []})
    triage = tn.FailureTriage(source="code_bug", reasoning="OOM", fix_hint="giảm bộ nhớ")
    monkeypatch.setattr(tn, "tester_triage", lambda: SimpleNamespace(invoke=lambda _: triage))
    out = tn.test_node({"acceptance_criteria": []})
    assert out["tests_passed"] is False and out["failure_source"] == "code_bug"


@pytest.mark.parametrize("mod, fn", [(en, "run_ruff"), (rn, "run_bandit")])
def test_lint_oom_is_not_docker_failure(monkeypatch, mod, fn):
    monkeypatch.setattr(mod.subprocess, "run", lambda cmd, **kw: proc(137))
    getattr(mod, fn)()                                     # không raise


# ---------- 2. git commit / tag / reset lỗi thì phải báo ----------
@pytest.fixture
def ws(tmp_path, monkeypatch):
    w = tmp_path / "ws"
    monkeypatch.setattr(cn, "WORKSPACE", w)
    monkeypatch.setattr(sn, "WORKSPACE", w)
    cn.init_workspace()
    (w / "REVIEW.md").write_text("ok", encoding="utf-8")
    cn.git("add", "-A")
    cn.git("commit", "-m", "chore(release): v0.1.0")
    return w


def test_git_check_raises_on_failure(ws):
    with pytest.raises(RuntimeError, match="git tag"):
        cn.git("tag", "-a", "v0.1.0", "no-such-sha", "-m", "x", check=True)


def test_release_fails_loudly_when_tag_exists(ws, monkeypatch):
    sha = cn.git("rev-parse", "HEAD").strip()
    cn.git("tag", "-a", "v0.1.0", sha, "-m", "đã có từ run khác")
    monkeypatch.setattr(sn, "interrupt", lambda _: {"approved": True})
    with pytest.raises(RuntimeError, match="git tag"):
        sn.release_node({"version": "v0.1.0", "release_notes": "n", "release_sha": sha, "iterations": 1})


def test_release_without_release_sha_gives_clear_error(ws, monkeypatch):
    # thread chờ duyệt từ trước khi có release_sha
    monkeypatch.setattr(sn, "interrupt", lambda _: {"approved": True})
    with pytest.raises(RuntimeError, match="release_sha"):
        sn.release_node({"version": "v0.1.0", "release_notes": "n", "iterations": 1})


# ---------- 3. đọc được DATABASE_URL ----------
def test_config_accepts_database_url(monkeypatch):
    monkeypatch.delenv("POSTGRES_URI", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5433/db")
    assert config.postgres_uri() == "postgresql://u:p@h:5433/db"
    monkeypatch.setenv("POSTGRES_URI", "postgresql://other")
    assert config.postgres_uri() == "postgresql://other"


# ---------- 4. khoá cho cả workspace (kể cả MemorySaver) ----------
def test_workspace_lock_is_exclusive(tmp_path):
    lockfile = tmp_path / ".workspace.lock"
    first = run.lock_workspace(lockfile)
    assert first is not None
    assert run.lock_workspace(lockfile) is None           # run thứ hai bị chặn
    first.close()
    again = run.lock_workspace(lockfile)
    assert again is not None
    again.close()


# ---------- 5. Postgres mà không có khoá AES → từ chối, trừ khi opt-out rõ ràng ----------
@pytest.fixture
def fake_pg(monkeypatch):
    monkeypatch.setattr(graph, "POSTGRES_URI", "postgresql://u:p@h/db")
    monkeypatch.setattr(graph.Connection, "connect", lambda *a, **kw: object())
    monkeypatch.setattr(graph, "needs_setup", lambda conn: False)


def test_postgres_without_key_refuses(fake_pg):
    with pytest.raises(RuntimeError, match="LANGGRAPH_AES_KEY"):
        graph.make_checkpointer()


def test_postgres_plaintext_needs_explicit_opt_out(fake_pg, monkeypatch):
    from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
    monkeypatch.setenv("LANGGRAPH_ALLOW_PLAINTEXT", "1")
    assert not isinstance(graph.make_checkpointer().serde, EncryptedSerializer)


def test_aes_key_wrong_length_fails(fake_pg, monkeypatch):
    monkeypatch.setenv("LANGGRAPH_AES_KEY", "short")
    with pytest.raises(ValueError):
        graph.make_checkpointer()


def test_encrypted_serde_rejects_plaintext_rows(fake_pg, monkeypatch):
    # có khoá thì dòng không mã hoá (vd ai đó ghi thẳng vào DB) không được tin
    monkeypatch.setenv("LANGGRAPH_AES_KEY", "0123456789abcdef")
    serde = graph.make_checkpointer().serde
    with pytest.raises(ValueError, match="không mã hoá"):
        serde.loads_typed(("json", b'{"release_sha": "forged"}'))
    assert serde.loads_typed(serde.dumps_typed({"a": 1})) == {"a": 1}


# ---------- 6. fallback chỉ cho lỗi tạm thời; LLM không gọi tool → retry ----------
def raw(out_tokens, model="m"):
    return AIMessage("", usage_metadata={"input_tokens": 1, "output_tokens": out_tokens, "total_tokens": out_tokens + 1},
                     response_metadata={"model_name": model})


def fake_get_llm(behaviour, calls):
    def get_llm(role, temperature=0.0, max_tokens=32_000, model=None):
        def call(_):
            calls.append(model)
            b = behaviour[model]
            if callable(b):
                b = b()
            if isinstance(b, Exception):
                raise b
            return {"raw": raw(10, model), "parsed": b, "parsing_error": None}
        return SimpleNamespace(with_structured_output=lambda *a, **kw: RunnableLambda(call))
    return get_llm


def http_error(cls, status):
    req = httpx.Request("POST", "http://omniroute")
    return cls("x", response=httpx.Response(status, request=req), body=None)


def test_no_fallback_on_bad_request(monkeypatch):
    calls = []
    monkeypatch.setitem(llm.MODELS, "evaluator", "a,b")
    monkeypatch.setattr(llm, "get_llm", fake_get_llm({"a": http_error(openai.BadRequestError, 400), "b": "ok"}, calls))
    with pytest.raises(openai.BadRequestError):
        llm.structured("evaluator", dict, retries=1).invoke("x")
    assert calls == ["a"]                                # không gửi prompt bị từ chối sang provider khác


@pytest.mark.parametrize("cls, status", [(openai.InternalServerError, 500), (openai.RateLimitError, 429)])
def test_fallback_on_transient_errors(monkeypatch, caplog, cls, status):
    calls = []
    monkeypatch.setitem(llm.MODELS, "evaluator", "a,b")
    monkeypatch.setattr(llm, "get_llm", fake_get_llm({"a": http_error(cls, status), "b": "ok"}, calls))
    with caplog.at_level("WARNING"):
        assert llm.structured("evaluator", dict, retries=1).invoke("x") == "ok"
    assert calls == ["a", "b"]
    assert "model dự phòng b" in caplog.text            # log model nào đã trả lời thay
    assert f"a lỗi {cls.__name__}" in caplog.text      # và vì sao model chính lỗi


def test_all_models_failing_raises_and_tries_each(monkeypatch):
    calls = []
    monkeypatch.setitem(llm.MODELS, "evaluator", "a,b")
    monkeypatch.setattr(llm, "get_llm", fake_get_llm(
        {"a": http_error(openai.InternalServerError, 500), "b": http_error(openai.InternalServerError, 502)}, calls))
    with pytest.raises(openai.InternalServerError):
        llm.structured("evaluator", dict, retries=1).invoke("x")
    assert calls == ["a", "b"]


@pytest.mark.parametrize("value", [None, "", " , "])
def test_empty_model_var_gives_clear_error(monkeypatch, value):
    monkeypatch.setitem(llm.MODELS, "evaluator", value)
    monkeypatch.setattr(llm, "OMNI_BASE_URL", "http://x")
    monkeypatch.setattr(llm, "OMNI_API_KEY", "k")
    llm.get_llm.cache_clear()
    with pytest.raises(RuntimeError, match="MODEL_EVALUATOR"):
        llm.structured("evaluator", dict)
    llm.get_llm.cache_clear()


def test_retry_only_parse_errors(monkeypatch):
    calls = []
    results = iter([ValueError("Field required"), "ok"])
    monkeypatch.setitem(llm.MODELS, "evaluator", "a")
    monkeypatch.setattr(llm, "get_llm", fake_get_llm({"a": lambda: next(results)}, calls))
    assert llm.structured("evaluator", dict, retries=2).invoke("x") == "ok"
    assert calls == ["a", "a"]

    calls.clear()
    monkeypatch.setattr(llm, "get_llm", fake_get_llm({"a": RuntimeError("bị cắt")}, calls))
    with pytest.raises(RuntimeError):
        llm.structured("evaluator", dict, retries=2).invoke("x")
    assert calls == ["a"]


def test_no_tool_call_is_retryable_error():
    with pytest.raises(ValueError, match="không gọi tool"):
        llm.check_structured({"raw": raw(10), "parsed": None, "parsing_error": None}, 32_000)


@pytest.mark.parametrize("tokens, usage, err", [(31_500, True, RuntimeError), (100, False, ValueError)])
def test_truncation_threshold(tokens, usage, err):
    r = raw(tokens) if usage else AIMessage("")
    with pytest.raises(err):
        llm.check_structured({"raw": r, "parsed": None, "parsing_error": ValueError("Field required")}, 32_000)


# ---------- 7. sandbox: hardening + container timeout bị dọn ----------
def docker_cmds(monkeypatch, mod, fn, code=0):
    cmds = []
    monkeypatch.setattr(mod.subprocess, "run", lambda cmd, **kw: (cmds.append(cmd), proc(code))[1])
    getattr(mod, fn)()
    return cmds


@pytest.mark.parametrize("mod, fn", [(tn, "run_pytest"), (en, "run_ruff"), (rn, "run_bandit")])
def test_sandbox_flags(monkeypatch, tmp_path, mod, fn):
    monkeypatch.setattr(tn, "WORKSPACE", tmp_path)
    s = " ".join(docker_cmds(monkeypatch, mod, fn)[0])
    for flag in ("--network none", "--memory 512m", "--cpus 1", "--pids-limit 128",
                 "--cap-drop ALL", "--security-opt no-new-privileges", "--user 1000:1000"):
        assert flag in s, (fn, flag)


def test_pytest_timeout_removes_container(monkeypatch, tmp_path):
    monkeypatch.setattr(tn, "WORKSPACE", tmp_path)
    cmds = []

    def fake(cmd, **kw):
        cmds.append(cmd)
        if cmd[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(cmd, 1)
        return proc(0)
    monkeypatch.setattr(tn.subprocess, "run", fake)
    res = tn.run_pytest(timeout=1)
    name = cmds[0][cmds[0].index("--name") + 1]
    assert ["docker", "rm", "-f", name] in cmds
    assert res["exit_code"] == -1


def test_pytest_ignores_stale_report(monkeypatch, tmp_path):
    monkeypatch.setattr(tn, "WORKSPACE", tmp_path)
    (tmp_path / ".report.xml").write_text(
        '<testsuite><testcase classname="t" name="old"><failure message="cũ"/></testcase></testsuite>', encoding="utf-8")
    monkeypatch.setattr(tn.subprocess, "run", lambda cmd, **kw: proc(2, "collection error"))
    assert tn.run_pytest()["failures"] == []


# ---------- 8. LLM không tự nới cổng chất lượng ----------
def test_ruff_ignores_workspace_config_and_bandit_ignores_nosec(monkeypatch):
    assert "--isolated" in docker_cmds(monkeypatch, en, "run_ruff")[0]
    assert "--ignore-nosec" in docker_cmds(monkeypatch, rn, "run_bandit")[0]


def test_pytest_neutralises_addopts(monkeypatch, tmp_path):
    monkeypatch.setattr(tn, "WORKSPACE", tmp_path)
    cmd = docker_cmds(monkeypatch, tn, "run_pytest")[0]
    assert cmd[cmd.index("-o") + 1] == "addopts="


@pytest.mark.parametrize("path", ["conftest.py", "src/conftest.py", "tests/test_x.py", "ruff.toml", ".github/x.yml"])
def test_coder_cannot_write_outside_allowed_paths(ws, path):
    touched = cn.apply_changes([cn.FileChange(path=path, action="create", content="x")])
    assert not (ws / path).exists()
    assert touched == [f"skipped (không được phép): {path}"]


@pytest.mark.parametrize("path", ["src/app/x.py", "README.md", "pyproject.toml"])
def test_coder_can_write_allowed_paths(ws, path):
    cn.apply_changes([cn.FileChange(path=path, action="create", content="x")])
    assert (ws / path).read_text(encoding="utf-8") == "x"


# ---------- 9. checked() nhận update dạng list tuple ----------
def test_checked_accepts_tuple_list_update():
    assert graph.checked(lambda s: Command(update=[("plan", "p")]))({}).update == [("plan", "p")]
    with pytest.raises(KeyError, match="nope"):
        graph.checked(lambda s: Command(update=[("nope", 1)]))({})


# ---------- 10–11. run.main(): --purge, khoá, luồng duyệt ----------
class RS(TypedDict, total=False):
    requirement: str
    project_context: str
    iterations: int
    version: str
    released: bool


def release_app():
    def release(s):
        d = interrupt({"message": "Duyệt?", "release_notes": "notes"})
        return {"released": bool(d.get("approved")), "version": "v0.1.0"}
    g = StateGraph(RS)
    g.add_node("work", lambda s: {})
    g.add_node("release", release)
    g.add_edge(START, "work")
    g.add_edge("work", "release")
    g.add_edge("release", END)
    return g.compile(checkpointer=MemorySaver())


def tid(req):
    return "run-" + hashlib.md5(req.encode(), usedforsecurity=False).hexdigest()[:8]


@pytest.fixture
def main_env(monkeypatch, tmp_path):
    app = release_app()
    monkeypatch.setattr(run, "build_graph", lambda: app)
    monkeypatch.setattr(run, "LOCK_FILE", tmp_path / ".workspace.lock")
    monkeypatch.setattr(run, "CONTEXT_FILE", tmp_path / "none.md")

    def feed(*answers, argv=()):
        it = iter(answers)
        monkeypatch.setattr(builtins, "input", lambda p="": next(it))
        monkeypatch.setattr(sys, "argv", ["run", *argv])
    return app, feed


def test_main_purge_deletes_thread(main_env):
    app, feed = main_env
    app.invoke({"requirement": "x"}, {"configurable": {"thread_id": "old"}})
    feed(argv=["--purge", "old"])
    run.main()
    assert not app.get_state({"configurable": {"thread_id": "old"}}).values


def test_main_unknown_thread_exits(main_env):
    app, feed = main_env
    feed(argv=["--thread", "run-nope"])
    with pytest.raises(SystemExit):
        run.main()


def test_main_exits_when_workspace_locked(main_env, monkeypatch):
    app, feed = main_env
    feed("login", "")
    monkeypatch.setattr(run, "lock_workspace", lambda *a: None)
    monkeypatch.setattr(run, "run", lambda *a: pytest.fail("không được chạy khi workspace đang bị khoá"))
    with pytest.raises(SystemExit):
        run.main()


def test_main_reject_then_approve_and_cleanup(main_env, monkeypatch):
    app, feed = main_env
    resumes = []
    real_run = run.run

    def spy(a, inp, config):
        if isinstance(inp, Command):
            resumes.append(inp.resume)
        return real_run(a, inp, config)
    monkeypatch.setattr(run, "run", spy)

    feed("login", "", "n", "thêm log")              # graph giả: reject → END, released=False
    run.main()
    assert resumes == [{"approved": False, "feedback": "thêm log"}]
    assert app.get_state({"configurable": {"thread_id": tid("login")}}).values   # không release → giữ lại

    resumes.clear()
    feed("login2", "", "y")
    run.main()
    assert resumes == [{"approved": True}]
    assert not app.get_state({"configurable": {"thread_id": tid("login2")}}).values   # release → xoá checkpoint


# ---------- 12. PostgresSaver lưu str/int/bool thẳng vào JSONB (bỏ qua serde) → phải ép qua blob khi mã hoá ----------
def test_encrypted_saver_routes_primitives_through_serde(fake_pg, monkeypatch):
    from langgraph.checkpoint.postgres import PostgresSaver
    monkeypatch.setenv("LANGGRAPH_AES_KEY", "0123456789abcdef")
    saver = graph.make_checkpointer()
    seen = {}
    monkeypatch.setattr(PostgresSaver, "put", lambda self, c, cp, m, v: seen.setdefault("cv", cp["channel_values"]))
    saver.put({}, {"channel_values": {"requirement": "BÍ MẬT", "iterations": 3, "plan": None, "files": ["a"]}}, {}, {})
    cv = seen["cv"]
    for k in ("requirement", "iterations", "plan"):
        assert not isinstance(cv[k], (str, int, float, bool, type(None))), k   # không còn bị inline
    assert cv["files"] == ["a"]
    for v in (cv["requirement"], cv["iterations"], cv["plan"]):
        assert saver.serde.loads_typed(saver.serde.dumps_typed(v)) == v.value   # serde gỡ lớp bọc rồi mới mã hoá


def test_main_db_unreachable_gives_clear_message(monkeypatch):
    import psycopg

    def boom():
        raise psycopg.OperationalError('database "multi_agent_sdlc" does not exist')
    monkeypatch.setattr(run, "build_graph", boom)
    monkeypatch.setattr(sys, "argv", ["run"])
    with pytest.raises(SystemExit, match="POSTGRES_URI/DATABASE_URL"):
        run.main()


@pytest.mark.parametrize("exc", [EOFError, KeyboardInterrupt])
def test_main_input_eof_or_ctrl_c_exits_cleanly(main_env, monkeypatch, exc):
    app, feed = main_env
    feed()
    def boom(p=""):
        raise exc
    monkeypatch.setattr(builtins, "input", boom)
    with pytest.raises(SystemExit) as e:
        run.main()
    assert "Đã dừng" in str(e.value.code)
    assert run.lock_workspace(run.LOCK_FILE) is not None   # khoá đã được nhả
