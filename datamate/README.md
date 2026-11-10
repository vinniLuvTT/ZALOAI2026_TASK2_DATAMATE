# DataMate pipeline (Zalo AI Challenge 2026)

```
question + business_id + as_of_date
  -> prep (pandas -> DuckDB): load csv/xlsx/sqlite, header detection, number/date cleaning, role mapping,
     fact-table + join detection -> flat `base` -> views lines_all / lines_valid / orders_valid (rulebook defaults)
  -> Cube semantic layer (YAML generated from templates, DuckDB driver, one Cube process per business)
  -> agent (Deep Agents SDK, or plain loop) with tools:
       get_schema, get_rulebook, find_columns, find_entity (fuzzy names), cube_query, sql_query (read-only),
       tet_window / lunar_to_solar / solar_to_lunar / shift_range_years, submit_answer
  -> N samples (sample 0 greedy, others T=0.6), early stop when unanimous
  -> verify: type coercion, canonical vote, abstain on low agreement / ties / suspect type
  -> submission.jsonl (+ logs/<id>.json)
```

Model: Qwen3-8B served by vLLM (one replica per GPU, deterministic round-robin). 8B < 15B limit; declare it in the README you submit.

## Setup
```bash
pip install -r requirements.txt
npm install --omit=dev            # needs network: downloads Cube native + duckdb node binaries
bash -c 'source scripts/serve_vllm.sh'   # or run vllm yourself; set DM_LLM_URLS / DM_LLM_MODEL
```

## Workflow
```bash
python run.py prep --biz train/data/biz_001          # inspect schema, roles, notes (check these first!)
python run.py calendar --file train/calendar.csv     # check Tết windows parse
python run.py bank --train train --out fewshot_bank.jsonl
python run.py eval --split train --out out/a --businesses biz_001,biz_002 --limit 30
DM_USE_CUBE=0 python run.py eval ...                 # A/B: SQL only
DM_BACKEND=loop python run.py eval ...               # A/B: Deep Agents vs plain loop
python run.py eval ... --harvest                     # save SQL of correct runs as few-shot examples
python run.py predict --split public_test --out out/public
```
`report.json` has PA, per-type / per-business breakdown and a threshold calibration table
(`DM_AGREE` = vote share needed to answer; pick the best PA row).

## Verify before trusting (could not be tested in the build sandbox)
1. **Cube + DuckDB**: `@cubejs-backend/native` and `duckdb` download binaries at `npm install`. Check
   `cube.log` in `/tmp/dm_work/<biz>/`. If it fails, the run silently falls back to SQL only.
   Env vars for the DuckDB file path (`CUBEJS_DB_NAME`, `CUBEJS_DB_DUCKDB_DATABASE_PATH`) and the `model/` layout
   follow the docs but are unverified; fix in `dm/cube_server.py` if the Cube server reports a DB error.
2. **PA formula** is an image in the task text. Set `DM_PENALTY` so `PA = (correct - penalty*wrong)/N` matches it.
3. **calendar.csv** column names are guessed in `dm/timeutil.py::Calendar.load`.
4. **Rulebook per split**: defaults baked into `lines_valid` (`dm/prep.py::RuleOptions`). Read each split's
   rulebook; the agent also sees it in the prompt and can use `lines_all` for variants.
5. Weekday answers (Thứ Hai ...) and other vocabulary: check train answers and the rulebook.

## Knobs (env, prefix `DM_`)
`BACKEND` deepagents|loop, `USE_CUBE`, `N_SAMPLES`, `TEMP`, `MAX_STEPS`, `CONC`, `AGREE`, `EARLY_STOP`, `THINKING`,
`REFINE_ROLES` (LLM role refinement), `FEWSHOT`, `BUDGET` (seconds; remaining questions become abstain), `LLM_URLS`.

## Tests (no GPU needed)
`python tests/test_smoke.py && python tests/test_runner.py && python tests/test_deepagents.py`
