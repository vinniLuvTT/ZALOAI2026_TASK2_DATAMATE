"""Runner end-to-end with a scripted fake LLM (no GPU, no Cube): prep -> N samples -> vote -> submission -> scoring."""
import json, shutil, sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.test_smoke import make_toy
from dm import scoring
from dm.config import Config
from dm.fewshot import FewShotBank
from dm.pipeline import Runner, Split


def test_runner():
    root = Path("/tmp/dm_runner"); shutil.rmtree(root, ignore_errors=True); root.mkdir()
    make_toy(root)
    qs = [{"id": f"q{i}", "business_id": "biz_toy", "question": q, "as_of_date": "2026-03-04"}
          for i, q in enumerate(["tổng số dòng hàng hợp lệ?", "sp bán chạy nhất?", "lợi nhuận tháng này?"])]
    (root / "questions.jsonl").write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in qs), encoding="utf-8")

    class Fake:
        """Answers by question text: 1st call runs SQL, 2nd submits."""
        def __init__(self, *a, **k):
            self.state = {}
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        def create(self, **kw):
            user = kw["messages"][1]["content"]
            n = sum(1 for m in kw["messages"] if m["role"] == "tool")
            if "lợi nhuận" in user.split("QUESTION:")[1].split("\n")[0]:
                call = ("submit_answer", {"answer_type": "unanswerable", "reason": "no cost column"})
            elif n == 0:
                call = ("sql_query", {"sql": "select count(*) from lines_valid"})
            else:
                call = ("submit_answer", {"answer_type": "number", "answer": "7"})
            tc = SimpleNamespace(id=f"c{n}", function=SimpleNamespace(name=call[0], arguments=json.dumps(call[1])))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[tc]))])
    import openai; openai.OpenAI = Fake

    cfg = Config(); cfg.backend = "loop"; cfg.use_cube = False; cfg.n_samples = 3; cfg.early_stop = 3
    cfg.work_dir = root / "work"; cfg.concurrency = 2
    out = root / "out"
    recs = Runner(cfg, Split(root), out, FewShotBank()).run()
    lines = [json.loads(l) for l in (out / "submission.jsonl").read_text().splitlines()]
    assert len(lines) == 3, lines
    by = {l["id"]: l for l in lines}
    assert by["q0"] == {"id": "q0", "type": "number", "answer": 7}, by["q0"]
    assert by["q2"]["type"] == "unanswerable", by["q2"]
    assert (out / "logs" / "q0.json").exists()
    # resume: second run does nothing new
    assert Runner(cfg, Split(root), out, None).run() == []
    refs = {"q0": {"type": "number", "answer": 7}, "q1": {"type": "string", "answer": "Trà đào"}, "q2": {"type": "unanswerable", "answer": None}}
    rep = scoring.summarize(recs, refs, 1.0)
    assert rep["correct"] == 2 and rep["wrong"] == 1, rep
    print("RUNNER TEST PASSED", rep["PA"])

if __name__ == "__main__":
    test_runner()
