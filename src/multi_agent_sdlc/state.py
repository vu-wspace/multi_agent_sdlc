from typing import TypedDict


class SDLCState(TypedDict):
    requirement: str
    plan: str
    code: str
    test_report: str
    tests_passed: bool
    eval_score: float
    review_approved: bool
    iterations: int
    release_notes: str
