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


class Boxed:
    """Bọc str/int/bool/None để PostgresSaver không inline chúng vào JSONB (chỗ đó không qua serde, không mã hoá)."""
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

class StrictEncryptedSerializer(EncryptedSerializer):
    """Đã bật mã hoá thì không tin dòng plaintext (vd ai đó ghi thẳng vào DB một release_sha giả)."""
    def dumps_typed(self, obj):
        return super().dumps_typed(obj.value if isinstance(obj, Boxed) else obj)

    def loads_typed(self, data):
        if "+" not in data[0]:
            raise ValueError(f"Checkpoint không mã hoá ({data[0]}) trong khi đã bật LANGGRAPH_AES_KEY")
        return super().loads_typed(data)

class EncryptedPostgresSaver(PostgresSaver):
    """langgraph lưu giá trị nguyên thuỷ thẳng vào cột JSONB → bỏ qua mã hoá. Ép mọi channel qua bảng blob."""
    def put(self, config, checkpoint, metadata, new_versions):
        cv = {k: Boxed(v) if v is None or isinstance(v, (str, int, float, bool)) else v
              for k, v in checkpoint["channel_values"].items()}
        return super().put(config, {**checkpoint, "channel_values": cv}, metadata, new_versions)

def make_serde():
    if os.environ.get("LANGGRAPH_AES_KEY"):   # 16/24/32 ký tự; sai độ dài → ValueError ngay lúc khởi động
        return StrictEncryptedSerializer.from_pycryptodome_aes()
    if os.environ.get("LANGGRAPH_ALLOW_PLAINTEXT") == "1":
        return None
    raise RuntimeError("Đã đặt POSTGRES_URI/DATABASE_URL nhưng thiếu LANGGRAPH_AES_KEY: checkpoint sẽ lưu plaintext. "
                       "Đặt khoá, hoặc LANGGRAPH_ALLOW_PLAINTEXT=1 nếu chấp nhận")

def needs_setup(conn) -> bool:
    """Schema checkpoint thiếu hoặc cũ hơn bản langgraph đang cài; nếu đủ thì role runtime không cần quyền CREATE."""
    if conn.execute("SELECT to_regclass('checkpoint_migrations') AS t").fetchone()["t"] is None:
        return True
    v = conn.execute("SELECT max(v) AS v FROM checkpoint_migrations").fetchone()["v"]
    return v is None or v < len(PostgresSaver.MIGRATIONS) - 1

def make_checkpointer():
    if not POSTGRES_URI:
        return MemorySaver()                 # test / chạy thử không cần DB
    serde = make_serde()                     # lỗi khoá báo trước khi tốn công kết nối
    # autocommit + dict_row là bắt buộc với PostgresSaver; conn sống suốt process
    conn = Connection.connect(POSTGRES_URI, autocommit=True, prepare_threshold=0, row_factory=dict_row,
                              connect_timeout=5)   # DB tắt / sai URI → báo lỗi sau 5s thay vì treo
    saver = (EncryptedPostgresSaver if serde else PostgresSaver)(conn, serde=serde)
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
        keys = [k for k, _ in update] if isinstance(update, (list, tuple)) else list(update or {})
        bad = set(keys) - STATE_KEYS
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
