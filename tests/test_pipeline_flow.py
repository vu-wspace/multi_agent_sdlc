"""Chạy graph thật (build_graph: router, checked(), interrupt, git thật) với LLM + Docker giả, kiểm tra đường đi."""
import importlib
from types import SimpleNamespace

import pytest
from langgraph.types import Command

from multi_agent_sdlc import graph
from multi_agent_sdlc.config import MAX_ITER

cn = importlib.import_module("multi_agent_sdlc.nodes.code_node")
tn = importlib.import_module("multi_agent_sdlc.nodes.test_node")
en = importlib.import_module("multi_agent_sdlc.nodes.evaluate_node")
rn = importlib.import_module("multi_agent_sdlc.nodes.review_node")
sn = importlib.import_module("multi_agent_sdlc.nodes.ship_node")
pn = importlib.import_module("multi_agent_sdlc.nodes.plan_node")

PASS = {"exit_code": 0, "output": "TOTAL 10 0 95%", "failures": []}
FAIL = {"exit_code": 1, "output": "TOTAL 10 5 50%", "failures": [{"type": "failure", "test": "t::x", "message": "assert", "trace": ""}]}


def evaluation(ok=True, origin="implementation", test_score=None):
    dim = lambda s: en.DimensionScore(score=s, reason="r")
    s = 5 if ok else 2
    return en.Evaluation(criteria=[en.CriterionVerdict(criterion="c", status="met" if ok else "not_met", evidence="e")],
                         correctness=dim(s), design=dim(s), code_quality=dim(s), robustness=dim(s),
                         test_adequacy=dim(test_score or s), issue_origin="none" if ok else origin,
                         feedback=[] if ok else ["sửa cái này"])


def review(blocker=False):
    comments = [rn.ReviewComment(file="src/app.py", line=1, severity="blocker", category="security",
                                 comment="lỗ hổng")] if blocker else []
    return rn.Review(summary="s", comments=comments, previous=[])


class Script:
    """Hàng đợi kết quả cho từng lần gọi; phần tử cuối dùng lại khi hết."""
    def __init__(self, *items):
        self.items, self.calls = list(items), 0

    def __call__(self, *_):
        item = self.items[min(self.calls, len(self.items) - 1)]
        self.calls += 1
        return item() if callable(item) else item


@pytest.fixture
def pipe(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    for m in (cn, tn, en, rn, sn):
        monkeypatch.setattr(m, "WORKSPACE", ws)
    s = SimpleNamespace(
        plan=Script(pn.Plan(summary="s", tech_stack=["python"], tasks=[], acceptance_criteria=["c"], risks=[])),
        pytest=Script(PASS), triage=Script(tn.FailureTriage(source="code_bug", reasoning="r", fix_hint="h")),
        eval=Script(evaluation()), review=Script(review()), tests_written=0, code_calls=0)

    def coder(_):
        s.code_calls += 1
        return cn.CodeOutput(changes=[cn.FileChange(path="src/app.py", action="create",
                                                    content=f"VERSION = {s.code_calls}\n")], notes="n")

    def tester(_):
        s.tests_written += 1
        return tn.TestSuite(files=[tn.TestFile(path="tests/test_app.py", content=f"# gen {s.tests_written}\n")],
                            criteria_coverage={"c": ["test_x"]})

    llm = lambda fn: (lambda: SimpleNamespace(invoke=fn))
    monkeypatch.setattr(pn, "planner_structured", llm(s.plan))
    monkeypatch.setattr(cn, "coder_structured", llm(coder))
    monkeypatch.setattr(tn, "tester_gen", llm(tester))
    monkeypatch.setattr(tn, "tester_triage", llm(s.triage))
    monkeypatch.setattr(tn, "run_pytest", s.pytest)
    monkeypatch.setattr(en, "evaluator_structured", llm(s.eval))
    monkeypatch.setattr(en, "run_ruff", lambda: {"count": 0, "top": []})
    monkeypatch.setattr(rn, "reviewer_structured", llm(s.review))
    monkeypatch.setattr(rn, "run_bandit", lambda: [])
    monkeypatch.setattr(sn, "shipper_structured", llm(lambda _: sn.ReleaseInfo(
        bump="minor", title="t", highlights=["h"], changelog={"Added": ["login"]})))

    app = graph.build_graph()
    cfg = {"configurable": {"thread_id": "flow"}, "recursion_limit": 80}

    def run(inp):
        path = []
        for ev in app.stream(inp, cfg, stream_mode="updates"):
            path += [n for n in ev if n != "__interrupt__"]
            if "__interrupt__" in ev:
                path.append("<chờ duyệt>")
        return path

    s.run = lambda: run({"requirement": "login", "iterations": 0})
    s.resume = lambda **d: run(Command(resume=d))
    s.state = lambda: app.get_state(cfg).values
    s.git = cn.git
    return s


HAPPY = ["plan", "code", "test", "evaluate", "review", "ship", "<chờ duyệt>"]


def test_happy_path_then_approve(pipe):
    assert pipe.run() == HAPPY
    assert pipe.resume(approved=True) == ["release"]
    st = pipe.state()
    assert st["released"] and st["version"] == "v0.1.0"
    assert pipe.git("rev-list", "-n1", "v0.1.0").strip() == st["release_sha"]
    assert pipe.tests_written == 1


def test_code_bug_goes_back_to_coder(pipe):
    pipe.pytest.items = [FAIL, PASS]
    assert pipe.run() == ["plan", "code", "test", "code", "test", "evaluate", "review", "ship", "<chờ duyệt>"]
    assert pipe.tests_written == 1                              # test giữ nguyên, chỉ code sửa


def test_test_bug_goes_back_to_tester(pipe):
    pipe.pytest.items = [FAIL, PASS]
    pipe.triage.items = [tn.FailureTriage(source="test_bug", reasoning="r", fix_hint="sửa assert")]
    assert pipe.run() == ["plan", "code", "test", "test", "evaluate", "review", "ship", "<chờ duyệt>"]
    assert pipe.tests_written == 2 and pipe.code_calls == 1


def test_tester_gives_up_after_max_fixes_then_coder(pipe):
    pipe.pytest.items = [FAIL, FAIL, FAIL, PASS]
    pipe.triage.items = [tn.FailureTriage(source="test_bug", reasoning="r", fix_hint="h")]
    path = pipe.run()
    assert path[:5] == ["plan", "code", "test", "test", "test"]   # MAX_TEST_FIX = 2 lần sửa test
    assert path[5] == "code"


def test_design_issue_replans_once(pipe):
    pipe.eval.items = [evaluation(ok=False, origin="design"), evaluation(ok=False, origin="design"), evaluation()]
    path = pipe.run()
    assert path[:7] == ["plan", "code", "test", "evaluate", "plan", "code", "test"]
    assert path[7:10] == ["evaluate", "code", "test"]           # lần 2 vẫn "design" nhưng hết lượt plan → code
    assert path.count("plan") == 2


def test_replan_regenerates_tests_for_new_criteria(pipe):
    # plan mới có acceptance criteria mới → test cũ không còn khớp, phải sinh lại
    pipe.plan.items = [pn.Plan(summary="s", tech_stack=[], tasks=[], acceptance_criteria=["c"], risks=[]),
                       pn.Plan(summary="s", tech_stack=[], tasks=[], acceptance_criteria=["c", "c2 mới"], risks=[])]
    pipe.eval.items = [evaluation(ok=False, origin="design"), evaluation()]
    pipe.run()
    assert pipe.tests_written == 2


def test_inadequate_tests_go_to_tester_not_only_coder(pipe):
    # Evaluator chê test (test_adequacy thấp); Coder không được ghi tests/ → Tester phải bổ sung
    pipe.eval.items = [evaluation(ok=False, test_score=2), evaluation()]
    pipe.run()
    assert pipe.tests_written == 2


def test_review_blocker_goes_back_to_code(pipe):
    pipe.review.items = [review(blocker=True), review()]
    path = pipe.run()
    assert path == ["plan", "code", "test", "evaluate", "review", "code", "test", "evaluate", "review", "ship",
                    "<chờ duyệt>"]


def test_reject_goes_back_to_code_and_drops_release_commit(pipe):
    pipe.run()
    sha = pipe.state()["release_sha"]
    path = pipe.resume(approved=False, feedback="đổi tên hàm")
    assert path[:2] == ["release", "code"] and path[-1] == "<chờ duyệt>"
    assert sha not in pipe.git("log", "--format=%H")               # commit release bị bỏ
    assert pipe.git("tag", "-l").split() == ["base"]
    assert pipe.resume(approved=True) == ["release"]
    assert pipe.state()["released"]


def test_stops_after_max_iterations(pipe):
    pipe.pytest.items = [FAIL]
    path = pipe.run()
    assert path == ["plan"] + ["code", "test"] * MAX_ITER   # hết MAX_ITER lượt rồi END
    assert not pipe.state().get("released")
