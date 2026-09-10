# Convenience targets. Every script also runs directly with `python scripts/<name>.py`.
# On Windows PowerShell, set $env:PYTHONUTF8=1 first (tweets contain emoji).
PY ?= python
export PYTHONUTF8 = 1

.PHONY: eval quick test data index silver agent baselines judge human-sample agreement all

## Reproduce headline numbers from committed predictions + judge outputs (seconds, no model needed)
eval:
	$(PY) scripts/evaluate.py
	$(PY) scripts/agreement_report.py

## Live smoke run: 15 golden tweets through the full agent with Ollama (a few minutes)
quick:
	$(PY) scripts/run_pipeline.py --set golden --limit 15 --out outputs/predictions_quick.jsonl
	$(PY) scripts/demo.py "my premium got charged twice this month, what gives?"

test:
	$(PY) -m pytest -q tests

## Full rebuild from the raw Kaggle CSV (data/raw/twcs/twcs.csv must exist)
data:
	$(PY) scripts/build_dataset.py
index:
	$(PY) scripts/build_index.py
silver:
	$(PY) scripts/label_silver.py 1000
agent:
	$(PY) scripts/run_pipeline.py --set golden
baselines:
	$(PY) scripts/run_baselines.py
judge:
	$(PY) scripts/run_judge.py
human-sample:
	$(PY) scripts/sample_for_human_rating.py
agreement:
	$(PY) scripts/agreement_report.py

all: data index silver agent baselines judge eval
