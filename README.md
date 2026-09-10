# SpotifyCares support agent: triage, grounded replies, and the proof

An AI first-response agent for **@SpotifyCares**, built from the Kaggle *Customer Support on Twitter* dump.
For every incoming customer tweet it (1) classifies the intent into one of 8 data-derived classes,
(2) decides whether to auto-handle or escalate to a human, with a stated reason, and (3) drafts a reply
grounded in how SpotifyCares actually resolved the most similar past tweets.

Everything runs on a laptop with local models (Ollama: `qwen2.5:3b` as the agent, `llama3.1:8b` as the judge),
and every model call is cached in the repo, so the headline numbers reproduce in seconds without a GPU or API key.

* **Report** (framing, results vs. baselines, failure analysis, what is misleading, next week): [REPORT.md](REPORT.md)
* **Decision log** (15 non-obvious decisions and why): [DECISIONS.md](DECISIONS.md)
* **Golden set and labelling guide**: [data/golden/](data/golden/) (`golden_set.jsonl`, `LABELING_GUIDE.md`)

## Headline results (200-example hand-labelled golden set)

| system | intent acc (95% CI) | macro-F1 | escalation precision | escalation recall | automation rate | unsafe-auto rate | judge: acceptable | human: acceptable (n=20) |
|---|---|---|---|---|---|---|---|---|
| **agent** (LLM triage + policy + RAG draft) | **0.76** (0.69-0.81) | **0.75** | **0.95** | **0.89** | 0.68 | **0.040** | **0.75** (mean 4.08/5) | **0.95** |
| simple (TF-IDF/LR + 1-NN reply + rules) | 0.66 (0.59-0.72) | 0.67 | 0.83 | 0.83 | 0.65 | 0.060 | 0.60 (mean 3.97/5) | 0.50 |
| trivial (majority intent, canned reply, always escalate) | 0.16 | 0.03 | 0.35 | 1.00 | 0.00 | 0.000 | 0.50 (mean 3.61/5) | 0.35 |
| trivial (never escalate) | 0.16 | 0.03 | 0.00 | 0.00 | 1.00 | 0.350 | n/a | n/a |

*unsafe-auto rate* = share of all messages that a human should have taken but the system auto-handled. It is the number
that matters for trust; automation rate is what it buys. **End-to-end trust** (intent right, escalation right, and
reply judged acceptable, all at once): agent 0.58, simple 0.41, trivial 0.01. **Judge vs. human** on 60 blind ratings:
Spearman 0.52, Cohen's kappa 0.51 on "acceptable". Full tables, per-class F1, calibration and the judge-vs-human
agreement study are in `outputs/results.md` and `outputs/agreement.json` after `make eval`, and discussed in the report.

## Reproduce in under 15 minutes

```bash
pip install -r requirements.txt            # ~2 min; torch CPU wheel is fine
make eval                                  # seconds: recomputes every metric from committed predictions + judge outputs
make test                                  # 11 unit tests for cleaning, escalation rules, post-processing
```

`make eval` needs nothing but Python: it reads `outputs/predictions_*.jsonl` and `outputs/judge_*.jsonl` (committed) and
writes `outputs/results.md`, `outputs/results.json`, `outputs/agreement.json`.

To see the live path work (a few minutes), install [Ollama](https://ollama.com), then:

```bash
ollama pull qwen2.5:3b                     # 1.9 GB agent model
make quick                                 # 15 golden tweets end-to-end, then one demo message with the decision trace
python scripts/demo.py "someone changed my email and now I'm locked out"
```

Because prompts and decoding are deterministic and cached, `make quick` on the golden ids returns the committed
predictions instantly; delete `outputs/llm_cache/` (or `LLM_CACHE=0`) to force fresh generations.

Optional: `LLM_PROVIDER=openai AGENT_MODEL=gpt-4o-mini JUDGE_MODEL=gpt-4o` switches providers (needs `OPENAI_API_KEY`).
All scripts print `PYTHONUTF8`-safe output; on Windows PowerShell set `$env:PYTHONUTF8=1`.

### Full rebuild from raw data (~1 hour on a laptop GPU, not needed for the numbers)

```bash
# 1. put the Kaggle CSV at data/raw/twcs/twcs.csv  (kaggle datasets download thoughtvector/customer-support-on-twitter)
make data        # reconstruct 25,918 (customer opener -> first SpotifyCares reply) pairs          ~3 min
make index       # embed them with all-MiniLM-L6-v2, excluding golden/dev ids                       ~2 min
make silver      # 1,000 LLM silver labels for the TF-IDF baseline                                  ~20 min
make agent       # run the agent over the 200 golden tweets                                          ~10 min
make baselines   # trivial + simple baselines                                                        ~1 min
make judge       # llama3.1:8b grades 600 replies (3 systems x 200)                                 ~50 min
make eval
```

## What the agent does

```
tweet ──► triage (LLM, JSON) ──► escalation policy (rules) ──► retrieve top-5 similar past cases ──► draft reply (LLM)
          intent, language,        regex hard triggers ->          MiniLM cosine over 25.6k             SpotifyCares voice,
          needs_account_action,    language -> LLM risk ->         historical pairs, held-out            playbook line,
          risk, confidence         intent policy -> auto           ids excluded                          [LINK] tokens only
```

* **Triage** (`src/agent/triage.py`): one JSON call with 14 hand-written few-shots. Non-English messages become intent
  `other` by rule, matching the labelling guide.
* **Escalation** (`src/agent/escalation.py`): deterministic. Order: regex hard triggers (hacked, sue, cancel threats,
  "talk to a human", private data posted, login failures) -> language -> LLM risk flag -> intent policy
  (account access always; billing unless it is a pure public question; data-loss recovery) -> auto. Every decision
  records a reason code and its source, which `scripts/demo.py` prints.
* **Drafting** (`src/agent/drafting.py`): prompt = tweet + intent + per-intent playbook (from `configs/intents.yaml`) +
  handling instruction + 5 real (tweet, SpotifyCares reply) exemplars. Post-processing replaces any URL with `[LINK]`,
  strips agent initials, and caps length.
* **Baselines** (`src/agent/baselines.py`): trivial and simple, described in the table above.

## Intents (defined from the data)

`technical_issue`, `service_outage`, `account_access`, `billing_subscription`, `content_availability`,
`feature_feedback`, `general_question`, `other`. Definitions, default actions and the brand's typical resolution per
intent are in [configs/intents.yaml](configs/intents.yaml); tie-break rules are in the labelling guide.

## Golden set

200 customer thread-openers: 100 uniform-random and 100 keyword-enriched so that rare intents have support, labelled by
hand with intent, escalate (bool) and reason after writing the guide. A disjoint 40-example dev set absorbed all prompt
iteration. Sampling, rules and the final label distribution: [data/golden/LABELING_GUIDE.md](data/golden/LABELING_GUIDE.md).

## Evaluation harness

* `scripts/evaluate.py`: intent accuracy / macro-F1 with bootstrap CIs, per-class P/R/F1, confusion; escalation precision,
  recall, automation rate, unsafe-auto rate, reason agreement; reply proxies (invented URLs, PII requests outside DM,
  initials, length); results on the uniform-random subset; confidence calibration; end-to-end trust rate.
* `src/eval/judge.py`: LLM-as-judge rubric (grounded / helpful / tone 1-5, safe, overall, acceptable) using a different
  model family than the agent, with the brand's real reply and three real precedents as evidence.
* `scripts/agreement_report.py`: judge vs. 60 blind human ratings (20 per system): Spearman on 1-5, Cohen's kappa on
  "acceptable", false-accept / false-reject counts.

## Repo layout

```
configs/intents.yaml          intent taxonomy + per-intent playbook
data/golden/                  golden_set.jsonl, dev_set.jsonl, LABELING_GUIDE.md, human_reply_ratings.jsonl, raw label files
data/silver/                  1,000 LLM silver labels (training data for the simple baseline)
src/agent/                    data.py, llm.py, retrieval.py, triage.py, escalation.py, drafting.py, pipeline.py, baselines.py
src/eval/                     metrics.py, judge.py, agreement.py
scripts/                      numbered pipeline steps + demo.py
outputs/                      predictions_*.jsonl, judge_*.jsonl, results.md/json, agreement.json, llm_cache/
tests/                        unit tests
```

## Borrowed / cited

* Dataset: Axelbrooke, *Customer Support on Twitter* (Kaggle, thoughtvector/customer-support-on-twitter), CC BY-NC-SA 4.0.
* Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (Reimers & Gurevych, 2019) via the `sentence-transformers` library.
* Models: Qwen2.5-3B-Instruct (Alibaba), Llama 3.1 8B Instruct (Meta), served by Ollama.
* Libraries: pandas, scikit-learn (TF-IDF, logistic regression), numpy, scipy (Spearman), pytest.
* Cohen's kappa and bootstrap CI are implemented from their textbook definitions in `src/eval/`.
* Prompts, rules, taxonomy, labels and all code in `src/` and `scripts/` were written for this assignment; no external
  prompt templates or repositories were copied.
