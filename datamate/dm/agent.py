"""
Solver: builds the prompt, runs one agent sample (Deep Agents SDK or a plain tool-calling loop),
returns the submitted answer + trace. N samples + voting happen in pipeline.py.

Backends
  "deepagents" : deepagents.create_deep_agent over a LangChain ChatOpenAI pointed at local vLLM
  "loop"       : minimal OpenAI tool-calling loop (no extra deps; use it as the A/B baseline)
"""
from __future__ import annotations

import inspect
import json
import time
import typing
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .config import Config
from .timeutil import normalize_question, time_context_text
from .tools import ToolBox

SYSTEM_PROMPT = """You are DataMate, a careful data analyst. You answer ONE Vietnamese question about ONE small \
business using its spreadsheets/databases (already loaded into DuckDB). Questions may be informal, abbreviated or \
without accents. Accuracy matters more than speed: a wrong answer is penalized, so say `unanswerable` when the data \
truly cannot support the question.

WORKFLOW
1. Apply the RULEBOOK defaults unless the question explicitly overrides them (e.g. says "kể cả đơn hủy").
2. Use the PRE-RESOLVED time ranges given in the user message. Never compute dates by hand. For Tết / lunar dates use \
tet_window / lunar_to_solar. For "cùng kỳ năm ngoái" use shift_range_years.
3. Resolve names with find_entity (fuzzy) before filtering; use the exact spelling returned.
4. Prefer cube_query (cube `sales`) for plain aggregates over valid sales lines. Use sql_query for everything else: \
stock/expiry/debts/inventory, comparisons, rankings with tie-breaks, custom rules, or when Cube errors. \
Views: lines_valid (default valid lines with standardized `dt`, `qty`, `amount`), lines_all (adds `is_cancelled`, \
`is_returned_order`), orders_valid, auxiliary views (`inventory_now`, `debts_open`, `cost_view`, `_unit_conv` when available), \
and raw tables.
5. Rankings: sort by the metric DESC (criterion in the question, else rulebook default), ties by code ASC; \
if fewer than k candidates exist return all of them.
6. Sanity-check results (non-empty, plausible magnitude, right period). Re-query if something looks off.
7. unanswerable: call find_columns for the missing concept (e.g. 'gia von', 'loi nhuan', 'han dung') first. If the \
needed column/table really does not exist, submit unanswerable with a short reason. Do not invent proxies.
8. FINISH by calling submit_answer exactly once. Types: number (plain number, VND integer, no units/separators), \
percent (e.g. -12.45, not a fraction), string (exact name as stored), date (YYYY-MM-DD; a month question -> YYYY-MM), \
boolean, list (JSON array in requested order), table ({"columns":[...],"rows":[[...]]}), unanswerable.
TOOL CALL FORMAT: Always invoke tools using tool calling or JSON object: {"name": "<tool_name>", "arguments": {<args>}}
Use as few tool calls as needed (hard limit {max_steps}). Be decisive."""


def build_user_message(q: dict, ctx, rulebook: str, cfg: Config, fewshot_text: str, compact_schema: str,
                       cube_catalog: str, use_cube: bool) -> str:
    parts = [
        f"QUESTION: {q['question']}",
        f"(normalized abbreviations: {normalize_question(q['question'])})",
        "",
        time_context_text(q["as_of_date"]),
        "",
        "RULEBOOK:" if len(rulebook) <= cfg.rulebook_inline_chars else "RULEBOOK (truncated; call get_rulebook for the rest):",
        rulebook[: cfg.rulebook_inline_chars] or "(none)",
        "",
        compact_schema,
    ]
    if use_cube and cube_catalog:
        parts += ["", "CUBE CATALOG (member names for cube_query):", cube_catalog[:3500]]
    if fewshot_text:
        parts += ["", fewshot_text]
    parts += ["", "Now answer the question. End with submit_answer."]
    return "\n".join(parts)


def compact_schema(ctx) -> str:
    lines = ["DATA OVERVIEW (call get_schema for samples / README):"]
    if ctx.ok:
        lines.append("lines_valid columns: " + ", ".join(f"{c}" for c, _ in ctx.lines_cols[:70]))
        lines.append(f"roles: {json.dumps(ctx.roles, ensure_ascii=False)}")
        if getattr(ctx, "aux_views", None):
            lines.append("auxiliary views: " + ", ".join(ctx.aux_views))
    for t, cols in ctx.raw_cols.items():
        lines.append(f"raw table {t}: " + ", ".join(c for c, _ in cols[:40]))
    for n in ctx.notes:
        if n.startswith(("amount", "qty", "status", "WARNING", "PREP", "returned", "joined", "built _unit_conv", "auxiliary")):
            lines.append(f"note: {n}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# JSON schema from a python function (for the loop backend)
# ----------------------------------------------------------------------------
def _json_type(tp) -> dict:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        return _json_type(non_none[0]) if non_none else {}
    if tp is str:
        return {"type": "string"}
    if tp is bool:
        return {"type": "boolean"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if origin in (list, typing.List) or tp is list:
        return {"type": "array", "items": _json_type(args[0]) if args else {}}
    if origin in (dict, typing.Dict) or tp is dict:
        return {"type": "object"}
    return {}   # Any


def func_to_schema(fn: Callable) -> dict:
    sig = inspect.signature(fn)
    hints = typing.get_type_hints(fn)
    props, required = {}, []
    for name, p in sig.parameters.items():
        props[name] = _json_type(hints.get(name, str))
        if p.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "function", "function": {
        "name": fn.__name__, "description": (inspect.getdoc(fn) or "")[:1500],
        "parameters": {"type": "object", "properties": props, "required": required}}}


# ----------------------------------------------------------------------------
def _extract_tool_calls_from_text(content: str, allowed_names: set[str]) -> list[tuple[str, dict]]:
    if not content:
        return []
    res = []
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        idx = content.find("{", pos)
        if idx == -1:
            break
        try:
            obj, end = decoder.raw_decode(content, idx)
            if isinstance(obj, dict):
                name = obj.get("name") or obj.get("tool") or obj.get("function")
                if name in allowed_names:
                    args = obj.get("arguments") or obj.get("parameters") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    elif not isinstance(args, dict):
                        args = {}
                    res.append((name, args))
            pos = max(end, idx + 1)
        except json.JSONDecodeError:
            pos = idx + 1
    return res


@dataclass
class SampleResult:
    answer: Optional[dict] = None
    steps: int = 0
    error: str = ""
    elapsed: float = 0.0
    trace: list = field(default_factory=list)
    sql: list = field(default_factory=list)


class Solver:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _url(self, qid: str, sample: int) -> str:
        urls = self.cfg.llm_url_list
        return urls[(zlib.crc32(qid.encode()) + sample) % len(urls)]

    def _extra_body(self) -> dict:
        return {"chat_template_kwargs": {"enable_thinking": self.cfg.enable_thinking}}

    # ---- plain tool-calling loop -----------------------------------------
    def _run_loop(self, tb: ToolBox, system: str, user: str, qid: str, sample: int,
                  temperature: float, seed: int) -> int:
        from openai import OpenAI
        client = OpenAI(base_url=self._url(qid, sample), api_key=self.cfg.llm_api_key, timeout=self.cfg.llm_timeout,
                        max_retries=1)
        fns = {f.__name__: f for f in tb.functions()}
        tools = [func_to_schema(f) for f in fns.values()]
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        steps, nudges = 0, 0
        while steps < self.cfg.max_steps and tb.final is None:
            resp = client.chat.completions.create(
                model=self.cfg.llm_model, messages=messages, tools=tools, tool_choice="auto",
                temperature=temperature, seed=seed, max_tokens=2048, extra_body=self._extra_body())
            msg = resp.choices[0].message
            steps += 1
            entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}

            calls: list[tuple[str, str, dict]] = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {}
                    calls.append((tc.id, tc.function.name, args))
            elif msg.content:
                extracted = _extract_tool_calls_from_text(msg.content, set(fns.keys()))
                for i, (fn_name, fn_args) in enumerate(extracted):
                    calls.append((f"call_fallback_{steps}_{i}", fn_name, fn_args))

            if calls:
                entry["tool_calls"] = [
                    {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                    for cid, name, args in calls
                ]
            messages.append(entry)

            if not calls:
                nudges += 1
                if nudges > 2:
                    break
                messages.append({"role": "user", "content": "Use the tools. When done, call submit_answer."})
                continue

            for cid, name, args in calls:
                try:
                    out = fns[name](**args) if name in fns else f"Unknown tool {name}"
                except Exception as e:
                    out = f"Tool error: {type(e).__name__}: {e}"
                messages.append({"role": "tool", "tool_call_id": cid, "content": str(out)[:5000]})
        if tb.final is None:  # force a final decision
            messages.append({"role": "user", "content": "Out of steps. Call submit_answer now with your best answer, "
                                                        "or unanswerable if the data cannot support it."})
            try:
                resp = client.chat.completions.create(
                    model=self.cfg.llm_model, messages=messages, tools=[func_to_schema(fns["submit_answer"])],
                    tool_choice={"type": "function", "function": {"name": "submit_answer"}},
                    temperature=temperature, seed=seed, max_tokens=1024, extra_body=self._extra_body())
                sub_msg = resp.choices[0].message
                if sub_msg.tool_calls:
                    for tc in sub_msg.tool_calls:
                        fns["submit_answer"](**json.loads(tc.function.arguments or "{}"))
                elif sub_msg.content:
                    sub_calls = _extract_tool_calls_from_text(sub_msg.content, {"submit_answer"})
                    for _, fn_args in sub_calls:
                        fns["submit_answer"](**fn_args)
            except Exception:
                pass
        return steps

    # ---- Deep Agents SDK --------------------------------------------------
    def _run_deepagents(self, tb: ToolBox, system: str, user: str, qid: str, sample: int,
                        temperature: float, seed: int) -> int:
        from deepagents import create_deep_agent
        from langchain_core.tools import StructuredTool
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=self.cfg.llm_model, base_url=self._url(qid, sample), api_key=self.cfg.llm_api_key,
                         temperature=temperature, seed=seed, timeout=self.cfg.llm_timeout, max_retries=1,
                         max_tokens=2048, extra_body=self._extra_body())
        lc_tools = [StructuredTool.from_function(f, name=f.__name__, description=inspect.getdoc(f) or f.__name__)
                    for f in tb.functions()]
        try:
            agent = create_deep_agent(model=llm, tools=lc_tools, system_prompt=system)
        except TypeError:                       # older SDK versions call it `instructions`
            agent = create_deep_agent(model=llm, tools=lc_tools, instructions=system)
        steps = 0
        try:
            out = agent.invoke({"messages": [{"role": "user", "content": user}]},
                               config={"recursion_limit": self.cfg.max_steps * 2 + 10})
            steps = sum(1 for m in out.get("messages", []) if getattr(m, "type", "") == "ai")
        except Exception as e:  # recursion limit etc. -> keep whatever submit_answer recorded
            if tb.final is None:
                raise
        return steps

    # ---- public -----------------------------------------------------------
    def solve_sample(self, q: dict, tb: ToolBox, user_msg: str, sample: int) -> SampleResult:
        cfg = self.cfg
        temperature = 0.0 if sample == 0 else cfg.sample_temperature
        seed = cfg.seed + sample
        system = SYSTEM_PROMPT.replace("{max_steps}", str(cfg.max_steps))
        t0 = time.time()
        res = SampleResult()
        try:
            run = self._run_deepagents if cfg.backend == "deepagents" else self._run_loop
            res.steps = run(tb, system, user_msg, q["id"], sample, temperature, seed)
        except Exception as e:
            res.error = f"{type(e).__name__}: {str(e)[:300]}"
        res.answer = tb.final
        res.elapsed = round(time.time() - t0, 2)
        res.trace = tb.trace
        res.sql = tb.sql_history
        return res
