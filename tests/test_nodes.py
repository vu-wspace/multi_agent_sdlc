from datetime import date

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


def test_graph_builds_without_env():
    from multi_agent_sdlc.main import app
    assert app is not None
