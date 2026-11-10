"""
Smoke test without GPU / Cube / vLLM:
  toy multi-table business -> prep -> views -> ToolBox (sql, entity, calendar) -> scripted LLM loop
  -> verify/vote -> scoring.
Run:  python -m pytest -q tests/test_smoke.py      (or: python tests/test_smoke.py)
"""
from __future__ import annotations

import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm import scoring, verify  # noqa: E402
from dm.agent import Solver, func_to_schema  # noqa: E402
from dm.config import Config  # noqa: E402
from dm.cube_gen import build_models  # noqa: E402
from dm.prep import prepare_business  # noqa: E402
from dm.timeutil import Calendar, normalize_question, time_context  # noqa: E402
from dm.tools import SQLRunner, ToolBox  # noqa: E402


def make_toy(root: Path) -> Path:
    random.seed(0)
    biz = root / "data" / "biz_toy"
    biz.mkdir(parents=True)
    prods = pd.DataFrame({
        "Mã SP": [f"SP{i:03d}" for i in range(1, 7)],
        "Tên sản phẩm": ["Trà đào", "Trà sữa trân châu", "Cà phê sữa", "Bạc xỉu", "Sting dâu 330ml", "Bánh mì"],
        "Giá bán": [30000, 35000, 25000, 28000, 12000, 20000],
    })
    prods.to_csv(biz / "san_pham.csv", index=False)
    orders, lines = [], []
    d0 = date(2026, 1, 1)
    for i in range(1, 401):
        dt = d0 + timedelta(days=random.randint(0, 62))
        status = random.choices(["Hoàn thành", "Đã hủy", "Trả hàng"], [90, 6, 4])[0]
        orders.append({"Mã đơn": f"HD{i:04d}", "Ngày bán": dt.strftime("%d/%m/%Y"), "Trạng thái": status,
                       "Chi nhánh": random.choice(["Q1", "Q3"])})
        for _ in range(random.randint(1, 3)):
            p = prods.sample(1).iloc[0]
            q = random.randint(1, 4)
            lines.append({"Mã đơn": f"HD{i:04d}", "Mã SP": p["Mã SP"], "Số lượng": q, "Đơn giá": p["Giá bán"],
                          "Thành tiền": q * p["Giá bán"]})
    pd.DataFrame(orders).to_excel(biz / "don_hang.xlsx", index=False)
    pd.DataFrame(lines).to_csv(biz / "chi_tiet_don.csv", index=False)
    (biz / "README.md").write_text("Quán trà sữa nhỏ. don_hang: đơn; chi_tiet_don: dòng hàng.", encoding="utf-8")
    (root / "rulebook.md").write_text("revenue = sum of line amounts, no cancelled orders.", encoding="utf-8")
    return biz


def test_end_to_end(tmp_path=None):
    tmp = Path(tmp_path or "/tmp/dm_smoke")
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    biz = make_toy(tmp)

    # ---- prep ----------------------------------------------------------
    ctx = prepare_business("biz_toy", biz, tmp / "work")
    assert ctx.ok, ctx.notes
    assert ctx.fact_table == "chi_tiet_don", ctx.fact_table
    assert {"qty", "line_amount", "date", "status", "order_id"} <= set(ctx.roles), ctx.roles
    sql = SQLRunner(ctx.sql_db, 20)
    cols, rows, total = sql.run("select count(*), sum(amount) from lines_valid", 10)
    assert rows[0][0] > 0 and rows[0][1] > 0
    # cancelled / returned orders must be excluded
    _, r_all, _ = sql.run("select count(distinct order_id) from lines_all", 10)
    _, r_val, _ = sql.run("select count(distinct order_id) from lines_valid", 10)
    assert r_val[0][0] < r_all[0][0], (r_val, r_all)
    # dates parsed from dd/mm/yyyy strings
    _, r, _ = sql.run("select min(dt), max(dt) from lines_valid", 10)
    assert str(r[0][0]).startswith("2026-0"), r

    # ---- cube model generation ----------------------------------------
    cubes, catalog = build_models(ctx)
    assert any(c["name"] == "sales" for c in cubes) and "sales.revenue" in catalog

    # ---- tools ----------------------------------------------------------
    cfg = Config()
    cfg.max_steps = 6
    tb = ToolBox(cfg, ctx, "rulebook text", None, None, catalog, sql, {})
    assert "Trà sữa trân châu" in tb.find_entity("tra sua tran chau", "product")
    assert "gia" in tb.find_columns("gia ban").lower()
    assert "SQL error" in tb.sql_query("drop table base")
    out = tb.sql_query("select product, sum(qty) q from lines_valid group by 1 order by q desc, product limit 3")
    assert "rows" in out, out
    assert "INVALID" in tb.submit_answer("number", "abc")
    assert "recorded" in tb.submit_answer("number", "1.234.567")
    assert tb.final == {"type": "number", "answer": 1234567}

    # ---- time -----------------------------------------------------------
    tc = time_context("2026-03-04")           # Wednesday
    assert tc["last_week"] == ("2026-02-23", "2026-03-01"), tc["last_week"]
    assert tc["this_month_to_date"] == ("2026-03-01", "2026-03-04")
    assert tc["same_period_last_year__this_month_to_date"] == ("2025-03-01", "2025-03-04")
    assert "sản phẩm" in normalize_question("sp bán dc nhiều nhất tuần rồi là gì")

    # ---- calendar (synthetic) ------------------------------------------
    cal_rows = []
    for solar, lm, ld in [("2026-02-02", 12, 16), ("2026-02-09", 12, 23), ("2026-02-17", 1, 1),
                          ("2026-02-26", 1, 10)]:
        cal_rows.append({"solar_date": solar, "lunar_month": lm, "lunar_day": ld, "is_leap": 0})
    pd.DataFrame(cal_rows).to_csv(tmp / "calendar.csv", index=False)
    cal = Calendar.load(tmp / "calendar.csv")
    assert cal is not None
    w = cal.tet_window(2026)
    assert w == (date(2026, 2, 9), date(2026, 2, 26)), w

    # ---- scripted tool-calling loop (fake OpenAI client) ---------------
    class FakeClient:
        def __init__(self, *a, **k):
            self.n = 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kw):
            self.n += 1
            if self.n == 1:
                call = ("sql_query", {"sql": "select sum(amount) from lines_valid"})
            else:
                call = ("submit_answer", {"answer_type": "number", "answer": "42"})
            tc = SimpleNamespace(id=f"c{self.n}", function=SimpleNamespace(name=call[0], arguments=json.dumps(call[1])))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[tc]))])

    import openai
    openai.OpenAI = FakeClient  # type: ignore
    cfg.backend = "loop"
    tb2 = ToolBox(cfg, ctx, "", None, None, catalog, sql, {})
    q = {"id": "q1", "business_id": "biz_toy", "question": "tổng doanh thu?", "as_of_date": "2026-03-04"}
    res = Solver(cfg).solve_sample(q, tb2, "user msg", 0)
    assert res.answer == {"type": "number", "answer": 42}, (res.answer, res.error)
    assert res.sql, "sql history should be recorded"

    # schema generation for the loop backend
    names = {func_to_schema(f)["function"]["name"] for f in tb2.functions()}
    assert {"sql_query", "submit_answer", "find_entity"} <= names

    # ---- verify / vote / scoring ---------------------------------------
    a = {"type": "string", "answer": "Trà đào"}
    b = {"type": "string", "answer": "tra đào!"}
    assert verify.canon(a) == verify.canon({"type": "string", "answer": "TRÀ ĐÀO"})
    v = verify.vote([a, a, {"type": "string", "answer": "Bánh mì"}], 3, 0.5, "mặt hàng nào bán chạy nhất")
    assert v["final"]["answer"] == "Trà đào"
    v = verify.vote([a, {"type": "string", "answer": "Bánh mì"}, None], 3, 0.5, "mặt hàng nào bán chạy nhất")
    assert v["final"]["type"] == "abstain"            # tie
    assert scoring.match({"type": "string", "answer": "sp0042"}, {"type": "string", "answer": "Sting", "alias": ["SP0042"]}) == "correct"
    assert scoring.match({"type": "number", "answer": 10.004}, {"type": "number", "answer": 10}) == "correct"
    assert scoring.match({"type": "unanswerable"}, {"type": "number", "answer": 1}) == "wrong"
    assert scoring.match({"type": "abstain"}, {"type": "number", "answer": 1}) == "abstain"
    t_ref = {"type": "table", "answer": {"columns": ["a", "b"], "rows": [["x", 1], ["y", 2]]}}
    t_prd = {"type": "table", "answer": {"columns": ["B", "A"], "rows": [[2, "y"], [1, "x"]]}}
    assert scoring.match(t_prd, t_ref) == "correct"
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    test_end_to_end()
