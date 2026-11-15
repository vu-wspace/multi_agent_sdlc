FROM python:3.14-slim

RUN pip install --no-cache-dir pytest==8.* pytest-cov pytest-timeout
RUN pip install --no-cache-dir pytest==8.* pytest-cov pytest-timeout ruff
RUN pip install --no-cache-dir pytest==8.* pytest-cov pytest-timeout ruff bandit

WORKDIR /app
