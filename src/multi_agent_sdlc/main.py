from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from .nodes import plan_node, code_node, test_node, evaluate_node, review_node, ship_node

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

MAX_ITER = 3

def after_test(s) -> Literal["evaluate", "code", "__end__"]:
    if s["tests_passed"]: return "evaluate"
    return "code" if s["iterations"] < MAX_ITER else END

def after_eval(s) -> Literal["review", "plan", "code"]:
    if s["eval_score"] >= 0.8: return "review"
    return "plan" if s["eval_score"] < 0.5 else "code"   # lỗi thiết kế → plan lại

def after_review(s) -> Literal["ship", "code"]:
    return "ship" if s["review_approved"] or s["iterations"] >= MAX_ITER else "code"

g = StateGraph(SDLCState)
for name, fn in [
    ("plan", plan_node),
    ("code", code_node),
    ("test", test_node),
    ("evaluate", evaluate_node),
    ("review", review_node),
    ("ship", ship_node)]:
    g.add_node(name, fn)

g.add_edge(START, "plan")
g.add_edge("plan", "code")
g.add_edge("code", "test")

g.add_conditional_edges("test", after_test)
g.add_conditional_edges("evaluate", after_eval)
g.add_conditional_edges("review", after_review)

g.add_edge("ship", END)

app = g.compile(checkpointer=MemorySaver(), interrupt_before=["ship"])  # người duyệt trước Release
