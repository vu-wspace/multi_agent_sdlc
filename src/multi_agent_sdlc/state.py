from typing import TypedDict


# total=False: keys appear as the pipeline progresses. Every key a node returns must be
# declared here, or LangGraph silently drops it (guarded by tests/test_state_contract.py).
class SDLCState(TypedDict, total=False):
    requirement: str
    project_context: str          # specs/project_context.md, run.py nạp vào
    # plan
    plan: str
    acceptance_criteria: list[str]
    # code
    code_files: list[str]
    code_notes: str
    diff: str
    iterations: int
    # test
    tests_passed: bool
    test_report: str
    failure_source: str | None
    test_fix_hint: str
    test_fix_attempts: int
    retest_hint: str              # khác rỗng → Tester sinh lại test (plan đổi / Evaluator chê test)
    # evaluate
    eval_score: float
    eval_passed: bool
    eval_target: str
    eval_feedback: str
    eval_detail: dict
    plan_revisions: int
    # review
    review_approved: bool
    review_feedback: str
    review_summary: str
    open_review_comments: list[dict]
    has_blockers: bool
    # ship / release
    version: str
    release_notes: str
    release_sha: str              # commit release mà người duyệt được xem
    released: bool
