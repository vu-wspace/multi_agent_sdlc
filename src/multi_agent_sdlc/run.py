from langgraph.types import Command
from graph import build_graph

app = build_graph()
config = {"configurable": {"thread_id": "feature-login-001"}, "recursion_limit": 60}

def run(inp):
    for event in app.stream(inp, config, stream_mode="updates"):
        for node, update in event.items():
            if node == "__interrupt__":
                payload = update[0].value
                print("\n" + "=" * 60 + f"\n{payload['message']}\n")
                print(payload["release_notes"])
                return True                              # đang chờ duyệt
            print(f"✔ {node}: {list(update.keys()) if update else ''}")
    return False

waiting = run({"requirement": "Viết module đăng nhập: email + mật khẩu, hash bằng bcrypt, khóa tài khoản sau 5 lần sai."})

while waiting:
    ans = input("\nApprove? (y/n): ").strip().lower()
    if ans == "y":
        waiting = run(Command(resume={"approved": True}))
    else:
        fb = input("Feedback: ")
        waiting = run(Command(resume={"approved": False, "feedback": fb}))

print("Done. Final state:", app.get_state(config).values.get("version"))
