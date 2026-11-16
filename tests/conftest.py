import pytest


@pytest.fixture(autouse=True)
def _no_real_postgres(monkeypatch):
    # config.py nạp .env khi import; có POSTGRES_URI thì build_graph() trong test sẽ nối DB thật
    from multi_agent_sdlc import graph
    monkeypatch.setattr(graph, "POSTGRES_URI", None)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")   # test không gửi trace ra ngoài
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
