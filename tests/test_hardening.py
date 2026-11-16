import importlib
from datetime import date
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from multi_agent_sdlc import graph, llm

run = importlib.import_module("multi_agent_sdlc.run")
tn = importlib.import_module("multi_agent_sdlc.nodes.test_node")
en = importlib.import_module("multi_agent_sdlc.nodes.evaluate_node")
rn = importlib.import_module("multi_agent_sdlc.nodes.review_node")
sn = importlib.import_module("multi_agent_sdlc.nodes.ship_node")


# ---------- key state không khai báo → lỗi ngay lúc chạy (AST scan bỏ sót key build động) ----------
def test_checked_node_rejects_undeclared_key():
    key = "plan" + "_typo"                          # key tạo lúc runtime, AST scan không thấy
    with pytest.raises(KeyError, match="plan_typo"):
        graph.checked(lambda s: {key: 1})({})


def test_checked_node_checks_command_update_and_passes_valid():
    assert graph.checked(lambda s: {"plan": "p"})({}) == {"plan": "p"}
    with pytest.raises(KeyError, match="nope"):
        graph.checked(lambda s: Command(goto="code", update={"nope": 1}))({})


def test_build_graph_wraps_every_node():
    app = graph.build_graph()
    for name in ("plan", "code", "test", "evaluate", "review", "ship", "release"):
        assert getattr(app.builder.nodes[name].runnable.func, "__wrapped__", None) is not None, name


# ---------- output LLM bị cắt do max_tokens ----------
def raw(out_tokens):
    return AIMessage("", usage_metadata={"input_tokens": 1, "output_tokens": out_tokens, "total_tokens": out_tokens + 1})


def test_truncated_structured_output_gets_clear_error():
    with pytest.raises(RuntimeError, match="max_tokens"):
        llm.check_structured({"raw": raw(32_000), "parsed": None, "parsing_error": ValueError("Field required")}, 32_000)


def test_untruncated_parse_error_is_reraised_and_ok_passes():
    with pytest.raises(ValueError, match="Field required"):
        llm.check_structured({"raw": raw(100), "parsed": None, "parsing_error": ValueError("Field required")}, 32_000)
    assert llm.check_structured({"raw": raw(100), "parsed": "ok", "parsing_error": None}, 32_000) == "ok"


# ---------- Docker lỗi không bị tính là test_bug ----------
def test_docker_failure_stops_instead_of_triaging(monkeypatch, tmp_path):
    monkeypatch.setattr(tn, "WORKSPACE", tmp_path)
    (tmp_path / "tests").mkdir(); (tmp_path / "tests" / "test_a.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(tn, "run_pytest", lambda: {"exit_code": 125, "output": "Unable to find image 'sdlc-sandbox'", "failures": []})
    monkeypatch.setattr(tn, "tester_triage", lambda: pytest.fail("không được gọi LLM triage"))
    with pytest.raises(RuntimeError, match="Docker"):
        tn.test_node({"acceptance_criteria": []})


# ---------- ruff / bandit có giới hạn tài nguyên như pytest ----------
@pytest.mark.parametrize("mod, fn", [(en, "run_ruff"), (rn, "run_bandit")])
def test_lint_sandboxes_are_resource_limited(monkeypatch, mod, fn):
    seen = {}
    monkeypatch.setattr(mod.subprocess, "run", lambda cmd, **kw: seen.setdefault("cmd", cmd) and SimpleNamespace(stdout="", stderr="", returncode=0))
    getattr(mod, fn)()
    for flag in ("--memory", "--cpus", "--pids-limit"):
        assert flag in seen["cmd"], (fn, flag)


# ---------- changelog không phụ thuộc giờ chạy test ----------
def test_changelog_entry_uses_given_date():
    info = sn.ReleaseInfo(bump="minor", title="t", highlights=[], changelog={"Added": ["login"]})
    assert sn.render_changelog_entry("v1.0.0", info, today=date(2026, 1, 2)).startswith("## [v1.0.0] - 2026-01-02")


# ---------- hai process cùng thread ----------
def test_lock_is_noop_for_memory_saver():
    assert run.lock_thread(SimpleNamespace(checkpointer=MemorySaver()), "t") is True


def test_lock_uses_postgres_advisory_lock(monkeypatch):
    from langgraph.checkpoint.postgres import PostgresSaver
    seen = {}

    class Conn:
        def execute(self, sql, params):
            seen["sql"], seen["params"] = sql, params
            return SimpleNamespace(fetchone=lambda: {"locked": False})
    saver = PostgresSaver.__new__(PostgresSaver); saver.conn = Conn()
    assert run.lock_thread(SimpleNamespace(checkpointer=saver), "run-abc") is False
    assert "pg_try_advisory_lock" in seen["sql"] and seen["params"] == ("run-abc",)


# ---------- dọn thread sau khi release ----------
def test_finish_deletes_released_thread_only():
    saver = MemorySaver()
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class S(TypedDict, total=False):
        released: bool
    g = StateGraph(S); g.add_node("n", lambda s: {}); g.add_edge(START, "n"); g.add_edge("n", END)
    app = g.compile(checkpointer=saver)
    for tid, released in (("a", True), ("b", False)):
        app.invoke({"released": released}, {"configurable": {"thread_id": tid}})
        run.finish(app, {"configurable": {"thread_id": tid}})
    assert not app.get_state({"configurable": {"thread_id": "a"}}).values
    assert app.get_state({"configurable": {"thread_id": "b"}}).values   # giữ lại để xem vì sao dừng


# ---------- mã hoá checkpoint ----------
def test_checkpointer_encrypts_when_key_set(monkeypatch):
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
    monkeypatch.setattr(graph, "POSTGRES_URI", "postgresql://u:p@h/db")
    monkeypatch.setenv("LANGGRAPH_AES_KEY", "0123456789abcdef")
    monkeypatch.setattr(graph.Connection, "connect", lambda *a, **kw: object())
    monkeypatch.setattr(PostgresSaver, "setup", lambda self: None)
    monkeypatch.setattr(graph, "needs_setup", lambda conn: True)
    assert isinstance(graph.make_checkpointer().serde, EncryptedSerializer)


@pytest.mark.parametrize("mod, fn", [(en, "run_ruff"), (rn, "run_bandit")])
def test_lint_docker_failure_is_not_reported_as_clean(monkeypatch, mod, fn):
    # trước: JSON rỗng → [] → bandit "không có lỗi" dù không chạy được
    monkeypatch.setattr(mod.subprocess, "run", lambda cmd, **kw: SimpleNamespace(stdout="", stderr="Unable to find image", returncode=125))
    with pytest.raises(RuntimeError, match="Docker"):
        getattr(mod, fn)()


# ---------- setup() chỉ chạy khi schema thiếu/cũ → role runtime không cần quyền CREATE ----------
class FakeConn:
    def __init__(self, table, version):
        self.table, self.version = table, version

    def execute(self, sql, *a):
        row = {"t": self.table} if "to_regclass" in sql else {"v": self.version}
        return SimpleNamespace(fetchone=lambda: row)


@pytest.mark.parametrize("table, version, expected", [
    (None, None, True),                       # DB mới
    ("checkpoint_migrations", 3, True),       # langgraph nâng cấp, có migration mới
    ("checkpoint_migrations", None, True),
    ("checkpoint_migrations", "LATEST", False),
])
def test_needs_setup(table, version, expected):
    from langgraph.checkpoint.postgres import PostgresSaver
    if version == "LATEST":
        version = len(PostgresSaver.MIGRATIONS) - 1
    assert graph.needs_setup(FakeConn(table, version)) is expected


def test_checkpointer_skips_setup_when_schema_current(monkeypatch):
    from langgraph.checkpoint.postgres import PostgresSaver
    monkeypatch.setattr(graph, "POSTGRES_URI", "postgresql://u:p@h/db")
    monkeypatch.setattr(graph.Connection, "connect", lambda *a, **kw: object())
    monkeypatch.setattr(graph, "needs_setup", lambda conn: False)
    monkeypatch.setattr(PostgresSaver, "setup", lambda self: pytest.fail("không được chạy DDL"))
    graph.make_checkpointer()


def test_ruff_ignores_exec_bit_rules(monkeypatch):
    # Docker Desktop trên Windows mount mọi file thành 0777 → EXE002 giả, kéo điểm evaluate xuống
    seen = {}
    monkeypatch.setattr(en.subprocess, "run", lambda cmd, **kw: seen.setdefault("cmd", cmd) and SimpleNamespace(stdout="", stderr="", returncode=0))
    en.run_ruff()
    i = seen["cmd"].index("--extend-ignore")
    assert seen["cmd"][i + 1] == "EXE"


def test_shipper_sees_test_diff(monkeypatch):
    # trước: chỉ diff src → shipper ghi vào release notes là "không có file test nào"
    seen = {}
    monkeypatch.setattr(sn, "git", lambda *a: seen.setdefault("diff", a) and "" if a[0] == "diff" else "")

    class Stop(Exception):
        pass
    monkeypatch.setattr(sn, "shipper_structured", lambda: SimpleNamespace(invoke=lambda _: (_ for _ in ()).throw(Stop())))
    with pytest.raises(Stop):
        sn.ship_node({"requirement": "r", "plan": "p"})
    assert "tests" in seen["diff"]


# ---------- model dự phòng khi model chính lỗi API (vd OmniRoute trả 500) ----------
def fake_llm_factory(behaviour, calls):
    """get_llm giả: behaviour[model] = Exception để ném, hoặc giá trị parsed để trả."""
    from langchain_core.runnables import RunnableLambda

    def get_llm(role, temperature=0.0, max_tokens=32_000, model=None):
        def call(_):
            calls.append(model)
            b = behaviour[model]
            if isinstance(b, Exception):
                raise b
            return {"raw": raw(10), "parsed": b, "parsing_error": None}
        return SimpleNamespace(with_structured_output=lambda *a, **kw: RunnableLambda(call))
    return get_llm


def api_error():
    import httpx
    import openai
    return openai.APIConnectionError(request=httpx.Request("POST", "http://omniroute"))


def test_model_list_supports_fallbacks(monkeypatch):
    monkeypatch.setitem(llm.MODELS, "evaluator", "agy/gemini, cx/gpt ")
    assert llm.model_list("evaluator") == ["agy/gemini", "cx/gpt"]


def test_structured_falls_back_to_next_model_on_api_error(monkeypatch):
    calls = []
    monkeypatch.setitem(llm.MODELS, "evaluator", "agy/gemini,cx/gpt")
    monkeypatch.setattr(llm, "get_llm", fake_llm_factory({"agy/gemini": api_error(), "cx/gpt": "ok"}, calls))
    assert llm.structured("evaluator", dict, retries=1).invoke("x") == "ok"
    assert calls == ["agy/gemini", "cx/gpt"]


def test_structured_does_not_fall_back_on_non_api_error(monkeypatch):
    # output bị cắt / lỗi logic: đổi model không giải quyết được, đừng đốt thêm token
    calls = []
    monkeypatch.setitem(llm.MODELS, "evaluator", "agy/gemini,cx/gpt")
    monkeypatch.setattr(llm, "get_llm", fake_llm_factory({"agy/gemini": RuntimeError("bị cắt"), "cx/gpt": "ok"}, calls))
    with pytest.raises(RuntimeError, match="bị cắt"):
        llm.structured("evaluator", dict, retries=1).invoke("x")
    assert calls == ["agy/gemini"]


def test_ssl_uses_os_trust_store():
    # mạng công ty chặn HTTPS bằng CA nội bộ (Windows tin, certifi thì không) → LangSmith/OmniRoute lỗi SSL
    import ssl

    import multi_agent_sdlc.config  # noqa: F401
    assert ssl.SSLContext.__module__.startswith("truststore")
