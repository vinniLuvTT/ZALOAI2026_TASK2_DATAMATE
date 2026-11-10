"""Deep Agents backend against a mock OpenAI-compatible server (scripted tool calls)."""
import json, shutil, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.test_smoke import make_toy
from dm.agent import Solver
from dm.config import Config
from dm.prep import prepare_business
from dm.tools import SQLRunner, ToolBox


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n_tool = sum(1 for m in body["messages"] if m["role"] == "tool")
        names = {t["function"]["name"] for t in body.get("tools", [])}
        assert {"sql_query", "submit_answer", "find_entity"} <= names, names
        if n_tool == 0:
            call = ("sql_query", {"sql": "select count(*) from lines_valid"})
        else:
            call = ("submit_answer", {"answer_type": "number", "answer": "7"})
        msg = {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{n_tool}", "type": "function", "function": {"name": call[0], "arguments": json.dumps(call[1])}}]}
        if n_tool >= 2:
            msg = {"role": "assistant", "content": "done"}
        out = {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
               "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        data = json.dumps(out).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)


def test_deepagents():
    srv = HTTPServer(("127.0.0.1", 18765), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    root = Path("/tmp/dm_da"); shutil.rmtree(root, ignore_errors=True); root.mkdir()
    biz = make_toy(root)
    ctx = prepare_business("biz_toy", biz, root / "w")
    cfg = Config(); cfg.backend = "deepagents"; cfg.llm_urls = "http://127.0.0.1:18765/v1"; cfg.max_steps = 6
    tb = ToolBox(cfg, ctx, "", None, None, "", SQLRunner(ctx.sql_db, 20), {})
    q = {"id": "q1", "business_id": "biz_toy", "question": "x", "as_of_date": "2026-03-04"}
    res = Solver(cfg).solve_sample(q, tb, "QUESTION: x", 0)
    print("answer:", res.answer, "| error:", res.error, "| steps:", res.steps)
    assert res.answer == {"type": "number", "answer": 7}, (res.answer, res.error)
    print("DEEPAGENTS TEST PASSED")

if __name__ == "__main__":
    test_deepagents()
