import argparse
import hashlib
import os
import sys
import uuid
from pathlib import Path

import psycopg
from langgraph.types import Command

from multi_agent_sdlc.config import MAX_ITER
from multi_agent_sdlc.graph import build_graph

# Windows: stdout là pipe thì Python dùng cp1252 → ✔ / tiếng Việt gây UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

CONTEXT_FILE = Path(__file__).resolve().parents[2] / "specs" / "project_context.md"   # <repo>/specs, không phụ thuộc cwd
LOCK_FILE = Path(__file__).resolve().parents[2] / ".workspace.lock"   # ngoài workspace/ để không lọt vào git add -A
_workspace_lock = None   # giữ handle đến hết process
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

def lock_workspace(path: Path = LOCK_FILE):
    """Khoá độc quyền cho workspace/ (dùng chung mọi thread, mọi checkpointer). None nếu run khác đang giữ.

    OS tự nhả khoá khi process chết, nên không có khoá "treo" sau crash.
    """
    f = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f

def finish(app, config) -> None:
    """Release xong thì xoá checkpoint (chứa diff, plan...); thread dừng vì lỗi / hết lượt thì giữ để xem lại."""
    if app.get_state(config).values.get("released"):
        app.checkpointer.delete_thread(config["configurable"]["thread_id"])

def stop_reason(v: dict) -> str | None:
    """Thread kết thúc mà chưa release → vì sao (router trả END không để lại dấu vết gì)."""
    if v.get("released"):
        return None
    if not v.get("tests_passed"):
        why, field = "test vẫn fail", "test_report"
    elif not v.get("eval_passed"):
        why, field = "evaluate chưa đạt", "eval_feedback"
    else:
        why, field = "review còn blocker", "review_feedback"
    # in luôn chi tiết: MemorySaver mất khi thoát, Postgres thì bị mã hoá → không xem lại được từ checkpoint
    detail = "\n".join((v.get(field) or "(trống)").splitlines()[:20])
    passes = v.get("code_passes", v.get("iterations"))   # checkpoint cũ chưa có code_passes
    return f"Dừng trước release: {why} sau {passes} lượt code (MAX_ITER={MAX_ITER})\n{field}:\n{detail}"

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
    ap.add_argument("--purge", metavar="THREAD", help="xoá checkpoint của một thread (crash / hết lượt / bỏ dở)")
    args = ap.parse_args()

    try:
        app = build_graph()
    except psycopg.OperationalError as e:
        reason = str(e).strip().splitlines()[-1]
        sys.exit(f"Không kết nối được Postgres (POSTGRES_URI/DATABASE_URL trong .env): {reason}\n"
                 "→ tạo database / bật server, hoặc bỏ biến đó để dùng MemorySaver")
    print("checkpointer:", type(app.checkpointer).__name__)

    if args.purge:
        app.checkpointer.delete_thread(args.purge)
        print(f"Đã xoá checkpoint của thread {args.purge}")
        return

    global _workspace_lock
    _workspace_lock = lock_workspace(LOCK_FILE)
    if _workspace_lock is None:
        sys.exit("workspace/ đang được một run khác dùng (xem .workspace.lock); đợi run đó xong rồi chạy lại")
    try:
        drive(app, args)
    except (EOFError, KeyboardInterrupt):   # Ctrl+Z / Ctrl+C khi đang nhập: thread vẫn còn trong checkpoint
        sys.exit("\nĐã dừng. Chạy lại cùng requirement (hoặc --thread <id>) để tiếp tục.")
    finally:
        _workspace_lock.close()
        _workspace_lock = None

def drive(app, args) -> None:
    """Chạy / chạy tiếp / duyệt một thread; gọi khi đang giữ khoá workspace."""
    requirement = ""
    if args.thread:
        config, mode = prepare_thread(app, "", args.thread)
        if mode == "fresh":
            sys.exit(f"thread {args.thread} không có gì để chạy tiếp")
    else:
        requirement = ask_requirement()
        config, mode = prepare_thread(app, requirement)
    thread_id = config["configurable"]["thread_id"]

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

    values = app.get_state(config).values
    if reason := stop_reason(values):
        print(f"\n{reason}")
    print(f"\nDone. Version: {values.get('version')}  (thread_id = {thread_id})")
    finish(app, config)

if __name__ == "__main__":
    main()
