import os
from functools import wraps

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.graph import START, StateGraph
from langgraph.types import Command
from psycopg import Connection
from psycopg.rows import dict_row

from .config import POSTGRES_URI
from .nodes import (
    code_node,
    evaluate_node,
    plan_node,
    review_node,
    ship_node,
    test_node,
)
from .nodes.evaluate_node import after_eval
from .nodes.review_node import after_review
from .nodes.ship_node import release_node
from .nodes.test_node import after_test
from .state import SDLCState


def needs_setup(conn) -> bool:
    """Schema checkpoint thiếu hoặc cũ hơn bản langgraph đang cài; nếu đủ thì role runtime không cần quyền CREATE."""
    if conn.execute("SELECT to_regclass('checkpoint_migrations') AS t").fetchone()["t"] is None:
        return True
    v = conn.execute("SELECT max(v) AS v FROM checkpoint_migrations").fetchone()["v"]
    return v is None or v < len(PostgresSaver.MIGRATIONS) - 1

def make_checkpointer():
    if not POSTGRES_URI:
        return MemorySaver()                 # test / chạy thử không cần DB
    # autocommit + dict_row là bắt buộc với PostgresSaver; conn sống suốt process
    conn = Connection.connect(POSTGRES_URI, autocommit=True, prepare_threshold=0, row_factory=dict_row,
                              connect_timeout=5)   # DB tắt / sai URI → báo lỗi sau 5s thay vì treo
    # LANGGRAPH_AES_KEY (16/24/32 byte) → mã hoá state trong DB; không đặt thì lưu plaintext
    serde = EncryptedSerializer.from_pycryptodome_aes() if os.environ.get("LANGGRAPH_AES_KEY") else None
    saver = PostgresSaver(conn, serde=serde)
    if needs_setup(conn):
        saver.setup()                        # DDL: chỉ lần đầu / sau khi nâng langgraph (cần quyền CREATE)
    return saver

STATE_KEYS = frozenset(SDLCState.__annotations__)

def checked(fn):
    """LangGraph lặng lẽ bỏ key không khai báo trong SDLCState → báo lỗi ngay lúc chạy."""
    @wraps(fn)
    def node(state):
        out = fn(state)
        update = out.update if isinstance(out, Command) else out
        bad = set(update or {}) - STATE_KEYS
        if bad:
            raise KeyError(f"{fn.__name__} trả về key chưa khai báo trong SDLCState: {sorted(bad)}")
        return out
    return node

def build_graph():
    g = StateGraph(SDLCState)

    g.add_node("plan", checked(plan_node))
    g.add_node("code", checked(code_node))
    g.add_node("test", checked(test_node))
    g.add_node("evaluate", checked(evaluate_node))
    g.add_node("review", checked(review_node))
    g.add_node("ship", checked(ship_node))
    g.add_node("release", checked(release_node))

    g.add_edge(START, "plan")
    g.add_edge("plan", "code")
    g.add_edge("code", "test")

    g.add_conditional_edges("test", after_test)          # → evaluate | code | test | END
    g.add_conditional_edges("evaluate", after_eval)      # → review | plan | code | END
    g.add_conditional_edges("review", after_review)      # → ship | code | END

    g.add_edge("ship", "release")
    # release tự điều hướng bằng Command

    # Không dùng interrupt_before: release_node tự dừng bằng interrupt() kèm payload cho người duyệt
    return g.compile(checkpointer=make_checkpointer())
