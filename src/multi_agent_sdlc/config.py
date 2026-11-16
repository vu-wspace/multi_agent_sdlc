# đọc .env, map vai trò → model

import os

import truststore
from dotenv import load_dotenv  # pip install python-dotenv

# Dùng kho chứng chỉ của OS thay cho certifi: mạng công ty kiểm tra HTTPS bằng CA nội bộ
# mà Windows tin còn certifi thì không → LangSmith báo CERTIFICATE_VERIFY_FAILED
truststore.inject_into_ssl()

load_dotenv()

OMNI_BASE_URL = os.environ.get("OMNIROUTE_BASE_URL")
OMNI_API_KEY  = os.environ.get("OMNIROUTE_API_KEY")
POSTGRES_URI  = os.environ.get("POSTGRES_URI")   # không đặt → dùng MemorySaver

MODELS = {
    "planner":   os.environ.get("MODEL_PLANNER"),
    "coder":     os.environ.get("MODEL_CODER"),
    "tester":    os.environ.get("MODEL_TESTER"),
    "evaluator": os.environ.get("MODEL_EVALUATOR"),
    "reviewer":  os.environ.get("MODEL_REVIEWER"),
    "shipper":   os.environ.get("MODEL_SHIPPER"),
}

MAX_ITER = 3
MAX_TEST_FIX = 2
MAX_PLAN_REVISIONS = 1
