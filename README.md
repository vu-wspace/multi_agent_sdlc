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

## Các node

| Node | File | Việc làm | Ghi vào state |
|---|---|---|---|
| plan | `nodes/plan_node.py` | Tách yêu cầu thành task và acceptance criteria. Chỉ Python, chỉ stdlib và bcrypt | `plan`, `acceptance_criteria` |
| code | `nodes/code_node.py` | Sinh hoặc sửa file trong `workspace/src`, rồi commit | `code_files`, `diff`, `iterations` |
| test | `nodes/test_node.py` | Sinh `tests/`, chạy pytest trong sandbox. Test bị skip cũng tính là fail. Khi fail, triage xem lỗi nằm ở code hay ở test | `tests_passed`, `test_report`, `failure_source` |
| evaluate | `nodes/evaluate_node.py` | Chấm điểm theo các tiêu chí, ruff và coverage | `eval_score`, `eval_passed`, `eval_target` |
| review | `nodes/review_node.py` | Review diff của cả PR, chạy bandit, ghi `REVIEW.md` | `review_approved`, `has_blockers` |
| ship | `nodes/ship_node.py` | Tăng version, ghi `CHANGELOG.md` và `RELEASE_NOTES.md`, commit release | `version`, `release_notes` |
| release | `nodes/ship_node.py` | `interrupt()` để chờ người duyệt. Approve thì gắn tag, reject thì quay về code | `released` |

Mọi key của state được khai báo trong `state.py` (`SDLCState`). LangGraph lặng lẽ bỏ qua key không được khai báo, nên `tests/test_state_contract.py` có kiểm tra để bắt lỗi này.

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
```

Kiểm tra từng model có trả lời được không:

```bash
uv run python -m multi_agent_sdlc.check_models
```

## Chạy

Chạy trong terminal của bạn, vì bước release cần nhập `y/n`:

```bash
uv run python -m multi_agent_sdlc.run
```

Yêu cầu đầu vào hiện đang ghi cứng trong `run.py`. Đến bước release, pipeline in release notes ra rồi hỏi `Approve? (y/n)`. Nếu trả lời `n`, bạn nhập feedback và Coder sẽ sửa tiếp.

Kết quả nằm trong `workspace/`, là một repo git riêng có tag `base` ở commit đầu tiên:

```bash
git -C workspace log --oneline
git -C workspace diff base HEAD -- src
```

Chạy test của code đã sinh trong sandbox (cmd):

```bat
docker run --rm --network none -v "%cd%\workspace:/app" -v "%cd%\workspace\.git:/app/.git:ro" sdlc-sandbox python -m pytest tests -q
```

Thử module đăng nhập mẫu bằng `demo_auth.py` (cmd):

```bat
docker run --rm -i --network none -v "%cd%\workspace:/app:ro" sdlc-sandbox python - < demo_auth.py
```

## An toàn

- `safe_path` chặn mọi đường dẫn đi ra ngoài `workspace/` và chặn ghi vào `.git`, vì một hook do LLM ghi vào đó sẽ chạy trên máy thật.
- Tester chỉ được ghi vào `workspace/tests/`.
- pytest, ruff và bandit chạy với `--network none`. `.git` được mount ở chế độ chỉ đọc.

## Test của pipeline

```bash
uv run pytest -q
```
