"""Central config. Every knob can be overridden by env var DM_<NAME>."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default, cast=str):
    v = os.getenv(f"DM_{name}")
    if v is None:
        return default
    if cast is bool:
        return v.lower() in ("1", "true", "yes")
    return cast(v)


@dataclass
class Config:
    # ---- paths -------------------------------------------------------------
    work_dir: Path = field(default_factory=lambda: Path(_env("WORK", "/tmp/dm_work")))
    out_dir: Path = field(default_factory=lambda: Path(_env("OUT", "/result")))
    project_root: Path = Path(__file__).resolve().parent.parent

    # ---- LLM (vLLM, OpenAI-compatible). Comma-separated URLs = replicas ----
    llm_urls: str = _env("LLM_URLS", "http://127.0.0.1:8080/v1")
    llm_model: str = _env("LLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
    llm_api_key: str = _env("LLM_KEY", "EMPTY")
    enable_thinking: bool = _env("THINKING", False, bool)
    llm_timeout: int = _env("LLM_TIMEOUT", 180, int)

    # ---- agent -------------------------------------------------------------
    backend: str = _env("BACKEND", "loop")   # "loop" (recommended for Qwen/local vLLM) | "deepagents"
    use_cube: bool = _env("USE_CUBE", True, bool)
    n_samples: int = _env("N_SAMPLES", 5, int)
    sample_temperature: float = _env("TEMP", 0.6, float)
    max_steps: int = _env("MAX_STEPS", 12, int)
    concurrency: int = _env("CONC", 8, int)
    n_fewshot: int = _env("FEWSHOT", 3, int)
    early_stop: int = _env("EARLY_STOP", 3, int)        # stop sampling when this many samples all agree (0 = off)
    refine_roles_llm: bool = _env("REFINE_ROLES", False, bool)
    fewshot_bank: str = _env("BANK", "")
    rulebook_inline_chars: int = _env("RULEBOOK_INLINE", 7000, int)

    # ---- verification ------------------------------------------------------
    agree_threshold: float = _env("AGREE", 0.5, float)  # vote share needed to answer
    # PA formula lives in the competition's image; set lambda to match it.
    # PA = (correct - penalty * wrong) / N   (penalty=1 => wrong costs as much as right gains)
    wrong_penalty: float = _env("PENALTY", 1.0, float)

    # ---- SQL tool ----------------------------------------------------------
    sql_timeout_s: int = _env("SQL_TIMEOUT", 30, int)
    max_rows: int = _env("MAX_ROWS", 40, int)

    # ---- Cube --------------------------------------------------------------
    cube_port: int = _env("CUBE_PORT", 4000, int)
    cube_start_timeout: int = _env("CUBE_START_TIMEOUT", 120, int)
    node_bin: str = _env("NODE", "node")

    # ---- time budget (competition limit is 6h incl. model loading) ---------
    time_budget_s: int = _env("BUDGET", 5 * 3600 + 20 * 60, int)

    seed: int = _env("SEED", 1234, int)

    @property
    def llm_url_list(self) -> list[str]:
        return [u.strip() for u in self.llm_urls.split(",") if u.strip()]
