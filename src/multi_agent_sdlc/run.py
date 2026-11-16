import argparse
import hashlib
import sys
import uuid
from pathlib import Path

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command

from multi_agent_sdlc.graph import build_graph

# Windows: stdout là pipe thì Python dùng cp1252 → ✔ / tiếng Việt gây UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

CONTEXT_FILE = Path(__file__).resolve().parents[2] / "specs" / "project_context.md"   # <repo>/specs, không phụ thuộc cwd
MAX_CONTEXT = 20_000   # file được lưu vào mọi checkpoint và gửi cho LLM: không để secret trong đó

def ask_requirement() -> str:
    print("Nhập requirement (dòng trống để kết thúc):")
    lines = []
    while True:
        line = input("> ")
        if not line.strip():
            if lines:
                break
            continue            # chưa nhập gì thì hỏi tiếp
        lines.append(line)
    return "\n".join(lines)

def load_context() -> str:
    return CONTEXT_FILE.read_text(encoding="utf-8")[:MAX_CONTEXT] if CONTEXT_FILE.exists() else ""

def prepare_thread(app, requirement: str, thread_id: str | None = None) -> tuple[dict, str]:
    """Chọn thread và cách chạy: "fresh" (input mới), "approve" (đang chờ duyệt), "continue" (crash giữa chừng)."""
    if not thread_id:
        norm = " ".join(requirement.split())             # khác khoảng trắng vẫn cùng thread
        thread_id = "run-" + hashlib.md5(norm.encode(), usedforsecurity=False).hexdigest()[:8]
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}

    snap = app.get_state(config)
    if snap.next:
        paused = any(t.interrupts for t in snap.tasks)
        return config, "approve" if paused else "continue"
    if snap.values:
        # thread đã chạy xong: chạy lại trên nó sẽ kế thừa test_fix_attempts, plan_revisions... cũ
        config["configurable"]["thread_id"] = f"{thread_id}-{uuid.uuid4().hex[:4]}"
    return config, "fresh"

def lock_thread(app, thread_id: str) -> bool:
    """Giữ advisory lock Postgres cho thread đến hết process; False nếu process khác đang chạy thread này."""
    cp = app.checkpointer
    if not isinstance(cp, PostgresSaver):
        return True                                       # MemorySaver: state riêng từng process
    row = cp.conn.execute("SELECT pg_try_advisory_lock(hashtext(%s)) AS locked", (thread_id,)).fetchone()
    return row["locked"]

def finish(app, config) -> None:
    """Release xong thì xoá checkpoint (chứa diff, plan...); thread dừng vì lỗi / hết lượt thì giữ để xem lại."""
    if app.get_state(config).values.get("released"):
        app.checkpointer.delete_thread(config["configurable"]["thread_id"])

def show_payload(payload: dict) -> None:
    print("\n" + "=" * 60 + f"\n{payload['message']}\n")
    print(payload["release_notes"])

def run(app, inp, config) -> bool:
    for event in app.stream(inp, config, stream_mode="updates"):
        for node, update in event.items():
            if node == "__interrupt__":
                show_payload(update[0].value)
                return True
            print(f"✔ {node}")
    return False

def main():
    ap = argparse.ArgumentParser(description="Chạy pipeline SDLC")
    ap.add_argument("--thread", help="thread_id cần chạy tiếp (in ra ở lần chạy trước)")
    args = ap.parse_args()

    app = build_graph()
    print("checkpointer:", type(app.checkpointer).__name__)

    requirement = ""
    if args.thread:
        config, mode = prepare_thread(app, "", args.thread)
        if mode == "fresh":
            sys.exit(f"thread {args.thread} không có gì để chạy tiếp")
    else:
        requirement = ask_requirement()
        config, mode = prepare_thread(app, requirement)
    thread_id = config["configurable"]["thread_id"]
    if not lock_thread(app, thread_id):
        sys.exit(f"thread {thread_id} đang được chạy ở process khác")

    if mode == "approve":
        print(f"\nthread_id = {thread_id} đang chờ duyệt release, tiếp tục từ đó.")
        show_payload(next(i for t in app.get_state(config).tasks for i in t.interrupts).value)
        waiting = True
    elif mode == "continue":
        print(f"\nthread_id = {thread_id} dừng giữa chừng lần trước, chạy tiếp...\n")
        waiting = run(app, None, config)
    else:
        print(f"\nthread_id = {thread_id}\nĐang chạy pipeline...\n")
        waiting = run(app, {"requirement": requirement,
                            "project_context": load_context(),
                            "iterations": 0}, config)

    while waiting:
        if input("\nApprove? (y/n): ").strip().lower() == "y":
            waiting = run(app, Command(resume={"approved": True}), config)
        else:
            fb = input("Feedback: ")
            waiting = run(app, Command(resume={"approved": False, "feedback": fb}), config)

    print(f"\nDone. Version: {app.get_state(config).values.get('version')}  (thread_id = {thread_id})")
    finish(app, config)

if __name__ == "__main__":
    main()
