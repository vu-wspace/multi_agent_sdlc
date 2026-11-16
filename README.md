# multi-agent-sdlc

Pipeline nhiều agent viết bằng LangGraph, đi hết vòng đời phần mềm: **plan → code → test → evaluate → review → ship → release**. Mỗi vai trò dùng một LLM riêng, gọi qua proxy OmniRoute. Code được sinh vào repo git `workspace/`. Test, lint và quét bảo mật đều chạy trong sandbox Docker không có mạng. Pipeline dừng lại để người duyệt trước khi release.

## Pipeline

```mermaid
flowchart TD
    START([START]) --> plan[Plan<br/><small>planner: tách task + acceptance criteria</small>]
    plan --> code[Code<br/><small>coder: ghi src/, commit</small>]
    code --> test[Test<br/><small>tester: sinh test, chạy pytest trong sandbox</small>]

    test -->|pass| evaluate[Evaluate<br/><small>evaluator + ruff + coverage</small>]
    test -->|test sai, còn lượt sửa test| test
    test -->|code sai, còn lượt| code
    test -->|hết lượt| STOP([END])

    evaluate -->|score ≥ 0.75, không vướng gate| review[Review<br/><small>reviewer + bandit</small>]
    evaluate -->|lỗi ở plan, còn lượt sửa plan| plan
    evaluate -->|lỗi ở code, còn lượt| code
    evaluate -->|hết lượt| STOP

    review -->|approved| ship[Ship<br/><small>shipper: version, CHANGELOG, release notes</small>]
    review -->|có blocker/major, còn lượt| code
    review -->|hết lượt, chỉ còn major| ship
    review -->|hết lượt, còn blocker| STOP

    ship --> release{{Release<br/><small>interrupt: chờ người duyệt</small>}}
    release -->|approve: tag vX.Y.Z| DONE([END])
    release -->|reject + feedback: hủy commit release| code

    classDef human fill:#fde68a,stroke:#b45309,color:#000;
    class release human;
```

Giới hạn vòng lặp nằm trong `src/multi_agent_sdlc/config.py`:

| Biến | Giá trị | Ý nghĩa |
|---|---|---|
| `MAX_ITER` | 3 | Số lượt Coder được sửa (dùng chung cho test, evaluate, review) |
| `MAX_TEST_FIX` | 2 | Số lần Tester tự sửa test khi triage kết luận lỗi nằm ở test |
| `MAX_PLAN_REVISIONS` | 1 | Số lần Evaluator được đẩy về Plan |

Nếu người duyệt từ chối release, Coder được thêm 1 lượt sửa.

Reviewer và Evaluator khá khắt khe, nên với yêu cầu lớn pipeline dễ hết 3 lượt trước khi tới release. Khi đó tăng `MAX_ITER`, hoặc hạ `PASS_THRESHOLD` (0.75) trong `nodes/evaluate_node.py`.

## Các node

| Node | File | Việc làm | Ghi vào state |
|---|---|---|---|
| plan | `nodes/plan_node.py` | Tách yêu cầu thành task và acceptance criteria. Chỉ Python, chỉ stdlib và bcrypt | `plan`, `acceptance_criteria` |
| code | `nodes/code_node.py` | Sinh hoặc sửa file trong `workspace/src`, rồi commit | `code_files`, `diff`, `iterations` |
| test | `nodes/test_node.py` | Sinh `tests/`, chạy pytest trong sandbox. Test bị skip cũng tính là fail. Khi fail, triage xem lỗi nằm ở code hay ở test | `tests_passed`, `test_report`, `failure_source` |
| evaluate | `nodes/evaluate_node.py` | Chấm điểm theo các tiêu chí, ruff và coverage | `eval_score`, `eval_passed`, `eval_target` |
| review | `nodes/review_node.py` | Review diff của cả PR, chạy bandit, ghi `REVIEW.md` | `review_approved`, `has_blockers` |
| ship | `nodes/ship_node.py` | Tăng version, ghi `CHANGELOG.md` và `RELEASE_NOTES.md`, commit release | `version`, `release_notes`, `release_sha` |
| release | `nodes/ship_node.py` | `interrupt()` để chờ người duyệt. Approve thì gắn tag lên `release_sha`, reject thì bỏ commit release và quay về code | `released` |

Mọi key của state được khai báo trong `state.py` (`SDLCState`). LangGraph lặng lẽ bỏ qua key không được khai báo, nên có 2 lớp chặn:

- `tests/test_state_contract.py` quét code (AST) tìm key được dùng mà chưa khai báo.
- `graph.checked()` bọc mọi node: node nào trả về key chưa khai báo thì báo lỗi ngay lúc chạy.

Mọi lời gọi LLM đi qua `llm.structured()`:

- Output bị cắt vì `max_tokens` thì báo rõ "bị cắt (x/y token)" và không thử lại.
- Lỗi API thì chuyển sang model dự phòng nếu có.

## Cài đặt

Cần có: Python 3.13+, [uv](https://docs.astral.sh/uv/), Docker và git.

```bash
uv sync
docker build -t sdlc-sandbox sandbox/
```

Tạo file `.env`:

```env
OMNIROUTE_BASE_URL=...
OMNIROUTE_API_KEY=...
MODEL_PLANNER=...
MODEL_CODER=...
MODEL_TESTER=...
MODEL_EVALUATOR=...
MODEL_REVIEWER=...
MODEL_SHIPPER=...
# tuỳ chọn: lưu checkpoint vào Postgres (bỏ trống → MemorySaver, mất khi tắt process)
POSTGRES_URI=postgresql://user:pass@localhost:5432/sdlc
# tuỳ chọn: mã hoá checkpoint bằng AES (khoá 16/24/32 ký tự); mất khoá = không đọc lại được checkpoint cũ
LANGGRAPH_AES_KEY=...
# tuỳ chọn: gửi trace lên LangSmith (gồm prompt, code sinh ra, diff)
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=multi_agent_sdlc
```

Postgres:

- Nếu Postgres không chạy trên máy này, thêm `?sslmode=verify-full` vào URI.
- Nên dùng một role riêng, chỉ có quyền trên database này, vì checkpoint chứa toàn bộ state: diff, plan, release notes.
- Bảng checkpoint chỉ được tạo ở lần chạy đầu tiên, hoặc sau khi nâng cấp langgraph. Chạy lần đầu bằng role admin. Sau đó role chạy hằng ngày chỉ cần quyền `SELECT/INSERT/UPDATE/DELETE` trên các bảng `checkpoint*`.

Python dùng kho chứng chỉ của Windows/OS (thư viện `truststore`), nên LangSmith và OmniRoute vẫn kết nối được khi mạng công ty kiểm tra HTTPS bằng CA nội bộ.

Mỗi `MODEL_<ROLE>` có thể khai báo thêm model dự phòng, cách nhau bằng dấu phẩy, ví dụ `MODEL_EVALUATOR=agy/gemini-3.8-flash-medium,cx/gpt-6.1-sol`. Khi model chính lỗi API (5xx, mất kết nối), pipeline tự chuyển sang model kế tiếp. Lỗi output bị cắt hoặc sai schema thì không chuyển, vì đổi model không giải quyết được.

Kiểm tra từng model, kể cả các model dự phòng, có trả lời được không:

```bash
uv run python -m multi_agent_sdlc.check_models
```

## Chạy

Chạy trong terminal của bạn, vì bước release cần nhập `y/n`:

```bash
uv run python -m multi_agent_sdlc.run
```

Script hỏi requirement (nhiều dòng, dòng trống để kết thúc). Nếu có file `specs/project_context.md`, nội dung của nó (tối đa 20k ký tự) được gửi kèm cho Planner. Đừng để secret trong file này, vì nó được lưu vào checkpoint và gửi cho LLM.

Đến bước release, pipeline in release notes ra rồi hỏi `Approve? (y/n)`. Nếu trả lời `n`, bạn nhập feedback và Coder sẽ sửa tiếp.

**Chạy tiếp sau khi tắt** (chỉ có tác dụng khi đặt `POSTGRES_URI`). Khi khởi động, script in ra checkpointer đang dùng và `thread_id`. Mỗi requirement có một `thread_id` cố định:

- Nếu lần trước đang chờ duyệt, script in lại release notes và hỏi duyệt tiếp.
- Nếu lần trước bị crash giữa chừng (LLM lỗi, Ctrl+C), script chạy tiếp từ node bị dừng.
- Nếu lần trước đã chạy xong, script tạo thread mới có hậu tố (`run-xxxxxxxx-abcd`). Muốn quay lại một thread có hậu tố thì truyền id của nó: `uv run python -m multi_agent_sdlc.run --thread run-xxxxxxxx-abcd`.

Mỗi thread chỉ được một process chạy tại một thời điểm, nhờ advisory lock của Postgres. Thread release xong thì checkpoint bị xoá. Thread dừng vì lỗi hoặc hết lượt thì được giữ lại để xem.

Lúc approve hoặc reject, `release_node` kiểm tra HEAD của `workspace/` còn đúng commit release lúc chuẩn bị không. Nếu đã có run khác commit chồng lên, nó dừng lại, không tag và không reset.

## Code đã sinh (`workspace/`)

`workspace/` là một repo git riêng, có tag `base` ở commit đầu tiên và một tag `vX.Y.Z` cho mỗi lần release. Pipeline **không xoá** workspace cũ: nếu bạn chạy một requirement mới, Coder sẽ viết chồng lên code đang có. Muốn bắt đầu một project mới thì xoá hoặc đổi tên `workspace/` trước khi chạy.

Xem code đã sinh:

```bash
git -C workspace log --oneline
git -C workspace diff base HEAD -- src
```

Cách chạy phụ thuộc vào thứ đã được sinh ra. Hướng dẫn nằm trong `workspace/README.md` nếu Coder có viết. Code do LLM viết, nên chạy trong sandbox thay vì chạy thẳng trên máy. Các lệnh dưới đây dùng cho cmd và chạy từ thư mục gốc repo.

Chạy test:

```bat
docker run --rm --network none -v "%cd%\workspace:/app" -v "%cd%\workspace\.git:/app/.git:ro" sdlc-sandbox python -m pytest tests -q -p no:cacheprovider
```

Chạy một app web, ví dụ app todo hiện tại (`src/todo`, cổng 8000), rồi mở http://127.0.0.1:8000:

```bat
docker run --rm -it -p 127.0.0.1:8000:8000 -e PYTHONPATH=/app/src -e TODO_DB=/data/todo.db -v "%cd%\workspace:/app:ro" -v todo-data:/data sdlc-sandbox python -c "from todo.__main__ import main; main(host='0.0.0.0')"
```

- Phải truyền `host='0.0.0.0'` vì bên trong container server chỉ nghe `127.0.0.1`, nên máy bạn không truy cập được.
- Dữ liệu nằm trong volume Docker `todo-data`, nên vẫn còn sau khi tắt container.

Với một thư viện không có entry point, chạy một script thử qua stdin:

```bat
docker run --rm -i --network none -v "%cd%\workspace:/app:ro" sdlc-sandbox python - < my_script.py
```

Trong script, thêm `sys.path.insert(0, "/app")` trước khi import `src...`.

## An toàn

- `safe_path` chặn mọi đường dẫn đi ra ngoài `workspace/` và chặn ghi vào `.git`, vì một hook do LLM ghi vào đó sẽ chạy trên máy thật.
- Tester chỉ được ghi vào `workspace/tests/`.
- pytest, ruff và bandit chạy với `--network none` và giới hạn tài nguyên `--memory 512m --cpus 1 --pids-limit 128`. `.git` được mount ở chế độ chỉ đọc.
- Docker không chạy được (exit 125–127, ví dụ thiếu image hoặc daemon tắt) thì pipeline dừng và báo lỗi. Nó không coi đó là test fail hay "không có lỗi bảo mật".
- ruff bỏ qua nhóm luật `EXE`, vì Docker Desktop trên Windows mount mọi file thành `0777` và sinh lỗi EXE002 giả.
- Trước khi approve hoặc reject, `release_node` kiểm tra HEAD của `workspace/` còn đúng `release_sha`, để không tag nhầm hoặc xoá commit của run khác.

## Test của pipeline

```bash
uv run pytest -q
```

Test không gọi LLM thật, không kết nối Postgres thật và không gửi trace lên LangSmith (`tests/conftest.py`).
