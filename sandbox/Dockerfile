FROM python:3.14-slim

# Test chạy với --network none nên không cài được gì lúc chạy:
# mọi thư viện mà code sinh ra được phép dùng phải có sẵn ở đây.
RUN pip install --no-cache-dir pytest==8.* pytest-cov pytest-timeout ruff bandit bcrypt

WORKDIR /app
