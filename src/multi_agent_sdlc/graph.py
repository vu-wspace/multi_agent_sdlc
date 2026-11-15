from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from .nodes import plan_node, code_node, test_node,\
    evaluate_node, review_node, ship_node
from .nodes.ship_node import release_node
from .nodes.test_node import after_test
from .nodes.evaluate_node import after_eval
from .nodes.review_node import after_review
from .state import SDLCState

def build_graph():
    g = StateGraph(SDLCState)

    g.add_node("plan", plan_node)
    g.add_node("code", code_node)
    g.add_node("test", test_node)
    g.add_node("evaluate", evaluate_node)
    g.add_node("review", review_node)
    g.add_node("ship", ship_node)
    g.add_node("release", release_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "code")
    g.add_edge("code", "test")

    g.add_conditional_edges("test", after_test)          # → evaluate | code | test | END
    g.add_conditional_edges("evaluate", after_eval)      # → review | plan | code | END
    g.add_conditional_edges("review", after_review)      # → ship | code | END

    g.add_edge("ship", "release")
    # release tự điều hướng bằng Command

    # Không dùng interrupt_before: release_node tự dừng bằng interrupt() kèm payload cho người duyệt
    return g.compile(checkpointer=MemorySaver())
