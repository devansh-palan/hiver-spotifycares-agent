# SpotifyCares support agent

A first-response agent for @SpotifyCares, built from the Kaggle "Customer Support on Twitter" dump. Given one incoming
customer tweet it classifies the intent (8 classes I defined from the data), decides whether to auto-handle or escalate
to a human with a stated reason, and drafts a reply based on how SpotifyCares actually resolved the most similar past
tweets.

It runs entirely on a laptop with local models through Ollama (`qwen2.5:3b` as the agent, `llama3.1:8b` as the judge).
Every model call is cached in the repo, so the numbers below reproduce in seconds without a GPU or an API key.

- Report (framing, results vs baselines, failure analysis, what's misleading, next week): [REPORT.md](REPORT.md)
- Decision log, 15 non-obvious choices and why: [DECISIONS.md](DECISIONS.md)
- Golden set and labelling guide: [data/golden/](data/golden/) (`golden_set.jsonl`, `LABELING_GUIDE.md`)

## Headline results (200 hand-labelled examples)

| system | intent acc (95% CI) | macro-F1 | escalation precision | escalation recall | automation rate | unsafe-auto rate | judge: acceptable | human: acceptable (n=20) |
|---|---|---|---|---|---|---|---|---|
| agent (LLM triage + rules + retrieval-grounded draft) | 0.76 (0.69-0.81) | 0.75 | 0.95 | 0.89 | 0.68 | 0.040 | 0.75 (mean 4.08/5) | 0.95 |
| simple (TF-IDF/LR + 1-NN reply + rules) | 0.66 (0.59-0.72) | 0.67 | 0.83 | 0.83 | 0.65 | 0.060 | 0.60 (mean 3.97/5) | 0.50 |
| trivial (majority intent, canned reply, always escalate) | 0.16 | 0.03 | 0.35 | 1.00 | 0.00 | 0.000 | 0.50 (mean 3.61/5) | 0.35 |
| trivial (never escalate) | 0.16 | 0.03 | 0.00 | 0.00 | 1.00 | 0.350 | n/a | n/a |

Unsafe-auto rate is the share of all messages that a human should have taken but the system auto-handled. That's the
number I care about most; automation rate is what it buys you. End-to-end trust (intent right, escalation right, and the
reply judged acceptable, all on the same tweet): agent 0.58, simple 0.41, trivial 0.01. Judge vs human on 60 blind
ratings: Spearman 0.52, Cohen's kappa 0.51 on "acceptable". Per-class F1, the confusion matrix, calibration and the
full agreement study are in `outputs/results.md` and `outputs/agreement.json` after `make eval`, and discussed in the
report.

## Reproduce in under 15 minutes

```bash
pip install -r requirements.txt            # ~2 min, CPU torch is fine
make eval                                  # seconds: recomputes every metric from the committed predictions + judge outputs
make test                                  # 11 unit tests for cleaning, escalation rules, post-processing
```

`make eval` only needs Python. It reads `outputs/predictions_*.jsonl` and `outputs/judge_*.jsonl` (committed) and writes
`outputs/results.md`, `outputs/results.json` and `outputs/agreement.json`.

To see the live path work, install [Ollama](https://ollama.com) and then:

```bash
ollama pull qwen2.5:3b                     # 1.9 GB
make quick                                 # 15 golden tweets end to end, then one demo message with the full decision trace
python scripts/demo.py "someone changed my email and now I'm locked out"
```

Prompts and decoding are deterministic and cached, so `make quick` on the golden ids returns the committed predictions
instantly. Delete `outputs/llm_cache/` (or set `LLM_CACHE=0`) to force fresh generations. The retrieval index is
committed too, so the demo works straight after cloning.

If you'd rather use OpenAI: `LLM_PROVIDER=openai AGENT_MODEL=gpt-4o-mini JUDGE_MODEL=gpt-4o` with `OPENAI_API_KEY` set.
On Windows PowerShell set `$env:PYTHONUTF8=1` first, tweets are full of emoji.

### Full rebuild from raw data (about an hour on a laptop GPU, not needed for the numbers)

```bash
# put the Kaggle CSV at data/raw/twcs/twcs.csv  (kaggle datasets download thoughtvector/customer-support-on-twitter)
make data        # reconstruct 25,918 (customer opener -> first SpotifyCares reply) pairs     ~3 min
make index       # embed them with all-MiniLM-L6-v2, excluding golden/dev ids                  ~2 min
make silver      # 1,000 LLM silver labels for the TF-IDF baseline                             ~20 min
make agent       # run the agent over the 200 golden tweets                                     ~10 min
make baselines   # trivial + simple baselines                                                   ~1 min
make judge       # llama3.1:8b grades 600 replies (3 systems x 200)                            ~50 min
make eval
```

## How the agent works

```
tweet ──► triage (LLM, JSON) ──► escalation policy (rules) ──► retrieve top-5 similar past cases ──► draft reply (LLM)
          intent, language,        regex hard triggers ->          MiniLM cosine over 25.6k             SpotifyCares voice,
          needs_account_action,    language -> LLM risk ->         historical pairs, held-out            playbook line,
          risk, confidence         intent policy -> auto           ids excluded                          [LINK] tokens only
```

Triage (`src/agent/triage.py`) is one JSON call with 14 hand-written few-shots. Non-English messages get intent `other`
by rule, which is how I labelled them too.

Escalation (`src/agent/escalation.py`) is deterministic. Order: regex hard triggers (hacked, sue, cancel threats, "talk
to a human", private data posted, login failures), then language, then the LLM's risk flag, then intent policy (account
access always, billing unless it's a pure public question, data-loss recovery), else auto. Every decision records a
reason code and where it came from; `scripts/demo.py` prints that.

Drafting (`src/agent/drafting.py`) builds a prompt from the tweet, the intent and its playbook line from
`configs/intents.yaml`, the handling instruction, and 5 real (tweet, SpotifyCares reply) exemplars. Post-processing
replaces any URL with `[LINK]`, strips agent initials and caps the length.

Baselines are in `src/agent/baselines.py`.

## Intents

`technical_issue`, `service_outage`, `account_access`, `billing_subscription`, `content_availability`,
`feature_feedback`, `general_question`, `other`. I defined these after reading a few hundred customer tweets. Definitions,
default actions and the brand's usual resolution per intent are in [configs/intents.yaml](configs/intents.yaml). The
tie-break rules are in the labelling guide.

## Golden set

200 customer thread-openers, 100 uniform-random and 100 keyword-enriched so the rare intents have enough support. I
labelled each one by hand with intent, escalate (bool) and a reason, after writing the guide. A separate 40-example dev
set took all the prompt iteration. Sampling, rules and the final distribution are in
[data/golden/LABELING_GUIDE.md](data/golden/LABELING_GUIDE.md).

## Evaluation harness

- `scripts/evaluate.py`: intent accuracy and macro-F1 with bootstrap CIs, per-class P/R/F1, confusion matrix; escalation
  precision, recall, automation rate, unsafe-auto rate, reason agreement; reply proxies (invented URLs, PII requests
  outside DM, initials, length); results on the uniform-random subset; confidence calibration; end-to-end trust rate.
- `src/eval/judge.py`: the LLM-as-judge rubric (grounded / helpful / tone on 1-5, safe, overall, acceptable). It uses a
  different model family than the agent and is shown the brand's real reply plus three real precedents.
- `scripts/agreement_report.py`: judge vs 60 blind human ratings (20 per system). Spearman on the 1-5 scores, Cohen's
  kappa on "acceptable", false-accept and false-reject counts.

## Repo layout

```
configs/intents.yaml          intent taxonomy + per-intent playbook
data/golden/                  golden_set.jsonl, dev_set.jsonl, LABELING_GUIDE.md, human_reply_ratings.jsonl, raw label files
data/silver/                  1,000 LLM silver labels (training data for the simple baseline)
data/processed/index/         MiniLM embeddings + metadata for retrieval
src/agent/                    data.py, llm.py, retrieval.py, triage.py, escalation.py, drafting.py, pipeline.py, baselines.py
src/eval/                     metrics.py, judge.py, agreement.py
scripts/                      pipeline steps in order, plus demo.py
outputs/                      predictions_*.jsonl, judge_*.jsonl, results.md/json, agreement.json, llm_cache/
tests/                        unit tests
```

## Borrowed / cited

- Dataset: Axelbrooke, Customer Support on Twitter (Kaggle, thoughtvector/customer-support-on-twitter), CC BY-NC-SA 4.0.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (Reimers and Gurevych, 2019) via the `sentence-transformers` library.
- Models: Qwen2.5-3B-Instruct (Alibaba) and Llama 3.1 8B Instruct (Meta), served with Ollama.
- Libraries: pandas, scikit-learn (TF-IDF, logistic regression), numpy, scipy (Spearman), pytest.
- Cohen's kappa and the bootstrap CI are implemented from their standard definitions in `src/eval/`.
- Prompts, rules, taxonomy, labels and all the code in `src/` and `scripts/` were written for this assignment. No
  external prompt templates or repositories were copied.
