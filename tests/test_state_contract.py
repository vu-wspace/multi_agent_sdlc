"""LangGraph silently drops keys a node returns that SDLCState doesn't declare,
so every key a node reads or writes must be declared there."""
import ast
from pathlib import Path

import multi_agent_sdlc
from multi_agent_sdlc.state import SDLCState

PKG = Path(multi_agent_sdlc.__file__).parent
STATE_ARGS = {"state", "s"}


def _str(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _keys_used(fn: ast.FunctionDef) -> set[str]:
    params = {a.arg for a in fn.args.args} & STATE_ARGS
    returned = {id(r.value) for r in ast.walk(fn) if isinstance(r, ast.Return) and r.value}
    keys = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in params | {"update"}:
            keys.add(_str(n.slice))
        elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get"
              and isinstance(n.func.value, ast.Name) and n.func.value.id in params):
            keys.add(_str(n.args[0]))
        elif isinstance(n, ast.keyword) and n.arg == "update" and isinstance(n.value, ast.Dict):
            keys |= {_str(k) for k in n.value.keys}
        elif isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict) and any(
                isinstance(t, ast.Name) and t.id == "update" for t in n.targets):
            keys |= {_str(k) for k in n.value.keys}
        elif isinstance(n, ast.Dict) and id(n) in returned:
            keys |= {_str(k) for k in n.keys}
    return keys - {None}


def test_every_state_key_used_by_nodes_is_declared():
    used = {}
    for f in (PKG / "nodes").glob("*.py"):
        for fn in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(fn, ast.FunctionDef) and {a.arg for a in fn.args.args} & STATE_ARGS:
                for k in _keys_used(fn):
                    used.setdefault(k, []).append(f"{f.name}:{fn.name}")
    missing = {k: v for k, v in used.items() if k not in SDLCState.__annotations__}
    assert not missing, f"Undeclared state keys (LangGraph drops them): {missing}"


def test_contract_scan_sees_known_keys():
    # guard against the scanner silently finding nothing
    tree = ast.parse((PKG / "nodes" / "plan_node.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "plan_node")
    assert {"requirement", "plan", "acceptance_criteria", "eval_feedback"} <= _keys_used(fn)


def test_graph_uses_node_routers():
    from importlib import import_module

    from multi_agent_sdlc import graph
    # import_module: nodes/__init__ shadows these module names with the node functions
    mod = lambda name: import_module(f"multi_agent_sdlc.nodes.{name}")
    assert graph.after_test is mod("test_node").after_test
    assert graph.after_eval is mod("evaluate_node").after_eval
    assert graph.after_review is mod("review_node").after_review
