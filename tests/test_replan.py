"""Vì sao pipeline quay lại plan: đếm plan_revisions đúng, và Evaluator không đòi plan lại vì ràng buộc sandbox."""
import importlib
from types import SimpleNamespace

from multi_agent_sdlc.nodes.evaluate_node import after_eval

en = importlib.import_module("multi_agent_sdlc.nodes.evaluate_node")
pn = importlib.import_module("multi_agent_sdlc.nodes.plan_node")


def make_eval(origin):
    dim = lambda s: en.DimensionScore(score=s, reason="r")
    return en.Evaluation(criteria=[], correctness=dim(2), design=dim(2), code_quality=dim(2),
                         robustness=dim(2), test_adequacy=dim(2), issue_origin=origin, feedback=["sửa x"])


def run_eval(monkeypatch, tmp_path, state, origin):
    monkeypatch.setattr(en, "WORKSPACE", tmp_path)
    monkeypatch.setattr(en, "run_ruff", lambda: {"count": 0, "top": []})
    monkeypatch.setattr(en, "read_workspace", lambda: "")
    monkeypatch.setattr(en, "read_tests", lambda: "")
    monkeypatch.setattr(en, "evaluator_structured", lambda: SimpleNamespace(invoke=lambda _: make_eval(origin)))
    return {**state, **en.evaluate_node(state)}


BASE = {"requirement": "r", "plan": "p", "acceptance_criteria": [], "test_report": "TOTAL 10 0 95%",
        "iterations": 1}


def test_design_issue_after_code_feedback_still_gets_one_replan(monkeypatch, tmp_path):
    # Lỗi cũ: plan_revisions tăng mỗi lần evaluate có feedback trước đó, dù chưa hề plan lại
    s = run_eval(monkeypatch, tmp_path, BASE, "implementation")      # lần 1: lỗi code
    assert after_eval(s) == "code"
    s = run_eval(monkeypatch, tmp_path, {**s, "iterations": 2}, "design")   # lần 2: lỗi thiết kế
    assert after_eval(s) == "plan"


def test_plan_revisions_counted_by_plan_node(monkeypatch):
    plan = pn.Plan(summary="s", tech_stack=[], tasks=[], acceptance_criteria=["a"], risks=[])
    monkeypatch.setattr(pn, "planner_structured", lambda: SimpleNamespace(invoke=lambda _: plan))
    first = pn.plan_node({"requirement": "r"})
    assert first.get("plan_revisions", 0) == 0                       # plan đầu không tính là sửa
    again = pn.plan_node({"requirement": "r", "plan": "p", "eval_feedback": "- sai", "plan_revisions": 0})
    assert again["plan_revisions"] == 1
    s = {"eval_passed": False, "eval_target": "plan", "iterations": 1, **again}
    assert after_eval(s) == "code"                                   # hết lượt plan lại → sang code


def test_evaluator_knows_sandbox_constraints():
    # Evaluator không biết sandbox chỉ cho stdlib → chấm "design" vì thiếu Flask/JS → plan lại vô ích
    assert pn.SANDBOX_CONSTRAINTS in pn.PLANNER_SYSTEM
    assert pn.SANDBOX_CONSTRAINTS in en.EVALUATOR_SYSTEM
    assert "không phải lỗi design" in en.EVALUATOR_SYSTEM
