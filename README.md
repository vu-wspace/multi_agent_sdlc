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
| `MAX_ITER` | 5 | Số lượt Coder được sửa (dùng chung cho test, evaluate, review) |
| `MAX_TEST_FIX` | 2 | Số lần Tester tự sửa test khi triage kết luận lỗi nằm ở test |
| `MAX_PLAN_REVISIONS` | 1 | Số lần Evaluator được đẩy về Plan |

Nếu người duyệt từ chối release, Coder được thêm 1 lượt sửa.

Reviewer và Evaluator khá khắt khe, nên với yêu cầu lớn pipeline dễ hết 5 lượt trước khi tới release. Khi đó tăng `MAX_ITER`, hoặc hạ `PASS_THRESHOLD` (0.75) trong `nodes/evaluate_node.py`.

## Các node

| Node | File | Việc làm | Ghi vào state |
|---|---|---|---|
| plan | `nodes/plan_node.py` | Tách yêu cầu thành task và acceptance criteria. Chỉ Python, chỉ stdlib và bcrypt | `plan`, `acceptance_criteria` |
| code | `nodes/code_node.py` | Sinh hoặc sửa file trong `workspace/src`, rồi commit | `code_files`, `diff`, `iterations` |
| test | `nodes/test_node.py` | Sinh `tests/` lần đầu; sinh lại khi plan đổi hoặc Evaluator chấm `test_adequacy` ≤ 2 (`retest_hint`), vì Coder không được ghi `tests/`. Chạy pytest trong sandbox. Test bị skip cũng tính là fail. Khi fail, triage xem lỗi nằm ở code hay ở test | `tests_passed`, `test_report`, `failure_source` |
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
# tuỳ chọn: lưu checkpoint vào Postgres (bỏ trống → MemorySaver, mất khi tắt process).
# Đọc POSTGRES_URI, nếu không có thì DATABASE_URL. Database phải được tạo sẵn.
POSTGRES_URI=postgresql://user:pass@localhost:5432/sdlc
# bắt buộc khi dùng Postgres: khoá mã hoá AES (16/24/32 ký tự). Mất khoá = không đọc lại được checkpoint cũ.
# Sinh khoá: python -c "import secrets; print(secrets.token_urlsafe(24))"
LANGGRAPH_AES_KEY=...
# chỉ khi thật sự chấp nhận lưu state plaintext:
# LANGGRAPH_ALLOW_PLAINTEXT=1
# tuỳ chọn: gửi trace lên LangSmith (gồm prompt, code sinh ra, diff)
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=multi_agent_sdlc
```

Postgres:

- Nếu Postgres không chạy trên máy này, thêm `?sslmode=verify-full` vào URI.
- Nên dùng một role riêng, chỉ có quyền trên database này, vì checkpoint chứa toàn bộ state: diff, plan, release notes.
- Bảng checkpoint chỉ được tạo ở lần chạy đầu tiên, hoặc sau khi nâng cấp langgraph. Chạy lần đầu bằng role admin. Sau đó role chạy hằng ngày chỉ cần quyền `SELECT/INSERT/UPDATE/DELETE` trên các bảng `checkpoint*`.

Mã hoá checkpoint:

- Khi dùng Postgres mà thiếu `LANGGRAPH_AES_KEY`, chương trình từ chối chạy, trừ khi bạn đặt `LANGGRAPH_ALLOW_PLAINTEXT=1`.
- Mã hoá phủ **mọi** giá trị state, kể cả chuỗi và số. Mặc định langgraph lưu thẳng các giá trị này vào cột JSONB mà không mã hoá, nên code ép chúng đi qua bảng blob.
- Khi đã bật khoá, dòng checkpoint không mã hoá sẽ bị từ chối, để không ai cài được state giả qua database.

Dữ liệu ra khỏi máy:

- **LangSmith**: khi `LANGSMITH_TRACING=true`, toàn bộ prompt (requirement, `specs/project_context.md`, code, diff, kết quả test, REVIEW.md) được gửi lên LangSmith. Bản này **không** được mã hoá AES bảo vệ. Với project nhạy cảm, đặt `LANGSMITH_TRACING=false`, hoặc `LANGSMITH_HIDE_INPUTS=true` và `LANGSMITH_HIDE_OUTPUTS=true`.
- **`truststore`**: Python dùng kho chứng chỉ của Windows/OS, nên LangSmith và OmniRoute vẫn kết nối được khi mạng công ty kiểm tra HTTPS bằng CA nội bộ. Đây là đánh đổi có chủ ý: proxy công ty đọc được API key và prompt. Hostname vẫn được kiểm tra như bình thường.
- **Model dự phòng**: chỉ được dùng khi model chính gặp lỗi tạm thời (mất kết nối, timeout, 5xx, 429). Lỗi 400/401/403 hoặc bị content filter chặn thì **không** chuyển, để prompt bị từ chối không bị gửi sang nhà cung cấp khác. Mỗi lần chuyển đều được ghi log. Chỉ khai báo model dự phòng từ nhà cung cấp mà bạn chấp nhận chính sách dữ liệu.

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

Mỗi lần chỉ một run được dùng `workspace/`, nhờ file lock `.workspace.lock` ở thư mục gốc repo. Khoá này áp dụng cho mọi thread và mọi checkpointer. Run thứ hai sẽ thoát kèm thông báo. Hệ điều hành tự nhả khoá khi process chết, nên không có khoá bị treo.

**Khi pipeline dừng trước release** (hết `MAX_ITER` lượt), script in lý do và 20 dòng đầu của phần chi tiết tương ứng, rồi mới in `Done.`:

```
Dừng trước release: evaluate chưa đạt sau 5 lượt code (MAX_ITER=5)
eval_feedback:
...
```

Lý do là một trong ba: `test vẫn fail` (in `test_report`), `evaluate chưa đạt` (in `eval_feedback`), `review còn blocker` (in `review_feedback`). Phần này được in ngay vì sau đó không xem lại được: MemorySaver mất khi tắt process, còn checkpoint Postgres thì đã mã hoá. Con số "sau N lượt" là số lượt code thật (`code_passes`), kể cả các lượt được trả lại khi reject release, nên có thể lớn hơn `MAX_ITER`.

Thread release xong thì checkpoint bị xoá. Thread dừng vì lỗi hoặc hết lượt thì được giữ lại để xem. Xoá một thread bằng tay:

```bash
uv run python -m multi_agent_sdlc.run --purge run-xxxxxxxx
```

**Lỗi thường gặp:**

- `openai.APIConnectionError: Connection error` kèm `WinError 10061 ... actively refused it` ngay ở node `plan`: OmniRoute chưa chạy, hoặc `OMNIROUTE_BASE_URL` trỏ sai cổng. Kiểm tra bằng `curl <OMNIROUTE_BASE_URL>/models`.
- Log `<role>: model chính ... lỗi, dùng model dự phòng ...` lặp lại mỗi lượt: model chính đang lỗi tạm thời hoặc sai tên trên OmniRoute. Chạy `check_models` (ở trên) để xem.

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
- Coder chỉ được ghi `src/**`, `README.md` và `pyproject.toml`, và không được ghi `conftest.py`. File ngoài danh sách bị bỏ qua và ghi vào `code_files`.
- LLM không tự nới được cổng chất lượng: ruff chạy `--isolated` (bỏ qua config trong workspace), bandit chạy `--ignore-nosec`, pytest chạy `-o addopts=`.
- pytest, ruff và bandit chạy với `--network none --memory 512m --cpus 1 --pids-limit 128 --cap-drop ALL --security-opt no-new-privileges --user 1000:1000`. `.git` được mount ở chế độ chỉ đọc.
- pytest bị timeout thì container bị `docker rm -f`, để code của LLM không chạy tiếp. `.report.xml` cũ bị xoá trước mỗi lần chạy.
- Docker không chạy được (exit 125–127, ví dụ thiếu image hoặc daemon tắt) thì pipeline dừng và báo lỗi. Exit 137/139, tức code sinh ra bị kill vì hết bộ nhớ hoặc segfault, được tính là test fail và gửi về Coder.
- `git commit`, `git tag` và `git reset` ở bước ship/release lỗi thì pipeline dừng, không báo release thành công giả.
- ruff bỏ qua nhóm luật `EXE`, vì Docker Desktop trên Windows mount mọi file thành `0777` và sinh lỗi EXE002 giả.
- Trước khi approve hoặc reject, `release_node` kiểm tra HEAD của `workspace/` còn đúng `release_sha`, để không tag nhầm hoặc xoá commit của run khác.

## Test của pipeline

```bash
uv run pytest -q
```

Test không gọi LLM thật, không kết nối Postgres thật và không gửi trace lên LangSmith (`tests/conftest.py`).

- Test router và test luồng dùng hằng số `MAX_ITER`, nên đổi giới hạn trong `config.py` không cần sửa test.
- Chạy test trong một git worktree (ví dụ `.claude/worktrees/...`) bằng venv của repo gốc thì phải thêm `PYTHONPATH=src`, vì bản cài editable trỏ về code của repo gốc chứ không phải worktree.