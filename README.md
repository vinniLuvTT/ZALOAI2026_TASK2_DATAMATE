# DataMate — Baseline v3 Engine for Vietnamese Tabular QA

Zalo AI Challenge 2026 — Subtask 2: DataMate.

Deterministic canonical business normalizer and hybrid rule-compiled / semantic execution pipeline for natural-language Vietnamese tabular question answering across heterogeneous data sources (CSV, Excel, SQLite).

---

## 1. Repository Structure

```
.
|-- configs/
|   |-- baseline.yaml            # Pipeline execution configuration
|   `-- compiled_rulebook.yaml   # Extracted business rules and synonyms
|-- docs/
|   |-- ARCHITECTURE.md          # Architectural decisions and pipeline details
|   |-- COMPETITION_CONTRACT.md  # Official schema, metric, and formatting specification
|   |-- DATASET_FINDINGS.md      # Edge-case profile across train and public test businesses
|   `-- REPO_AUDIT.md            # Codebase audit notes
|-- reports/
|   |-- baseline_metrics_v3.json # Official scorer report on 3,000 train questions
|   |-- error_analysis.md        # Detailed breakdown of remaining edge-case errors
|   `-- experiment_log.csv       # Versioned score tracking
|-- scripts/
|   |-- evaluate.py              # Score calculation against ground truth
|   |-- profile_data.py          # Data profiling utility for biz schemas
|   `-- run_baseline.py          # Inference runner generating submission.jsonl
|-- src/
|   |-- canonical.py             # Business data normalization and deduplication
|   |-- entity_linker.py         # Fuzzy and exact entity resolution
|   |-- evaluator.py             # Official evaluation metric and scoring logic
|   |-- excel_fast.py            # High-performance Excel loader
|   |-- executor.py              # Canonical query executor
|   |-- __init__.py
|   |-- ir.py                    # Typed intermediate representation (QueryPlanIR)
|   |-- llm_client.py            # LLM backend client (vLLM / HuggingFace)
|   |-- loader.py                # Multi-format data loader (CSV, XLSX, SQLite)
|   |-- parser.py                # Rule-compiled semantic question parser
|   |-- rule_engine.py           # Domain rulebook interpreter
|   |-- runner.py                # Batch pipeline orchestrator
|   |-- serializer.py            # Contract-compliant answer formatter
|   |-- temporal.py              # Vietnamese temporal expression resolver
|   |-- value_parser.py          # Currency, percentage, and date parsing
|   `-- verifier.py              # Evidence consistency and abstention verifier
`-- tests/
    |-- integration/             # End-to-end integration tests
    `-- unit/                    # Unit tests for core modules
```

---

## 2. Requirements and Setup

Python 3.10+ is required.

Install standard dependencies:
```bash
pip install pandas numpy openpyxl xlrd python-dateutil pyyaml pytest
```

---

## 3. Usage

### Run Inference
Generate predictions for a dataset split:
```bash
python scripts/run_baseline.py \
    --data-dir train \
    --output submission.jsonl \
    --config configs/baseline.yaml
```

To run on public test data:
```bash
python scripts/run_baseline.py \
    --data-dir public_test \
    --output submission.jsonl \
    --config configs/baseline.yaml
```

### Run Evaluation
Score predictions against ground truth labels:
```bash
python scripts/evaluate.py \
    --pred submission.jsonl \
    --gold train/answers.jsonl
```

### Run Tests
Execute the unit and integration test suite:
```bash
pytest tests/ -v
```

---

## 4. Benchmark Performance (Baseline v3)

Evaluated on the full 3,000 training questions using the official competition evaluator (`src/evaluator.py`):

- Total Score: +2748
- Normalized Score: 0.9160 (91.60%)
- Correct: 2,874 / 3,000 (95.80%)
- Wrong Value: 112 (3.73%)
- Wrong Type: 14 (0.47%)
- Format Error: 0 (0.00%)
- Abstain: 0
- Throughput: ~94 QPS (entire 3,000-question run completes in under 32 seconds on CPU)
