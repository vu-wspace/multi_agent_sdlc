# đọc .env, map vai trò → model

import os
from dotenv import load_dotenv          # pip install python-dotenv

load_dotenv()

OMNI_BASE_URL = os.environ["OMNIROUTE_BASE_URL"]
OMNI_API_KEY  = os.environ["OMNIROUTE_API_KEY"]

MODELS = {
    "planner":   os.environ["MODEL_PLANNER"],
    "coder":     os.environ["MODEL_CODER"],
    "tester":    os.environ["MODEL_TESTER"],
    "evaluator": os.environ["MODEL_EVALUATOR"],
    "reviewer":  os.environ["MODEL_REVIEWER"],
    "shipper":   os.environ["MODEL_SHIPPER"],
}

MAX_ITER = 3
MAX_TEST_FIX = 2
MAX_PLAN_REVISIONS = 1
