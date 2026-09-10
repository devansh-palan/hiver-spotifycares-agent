"""Step 7: LLM-as-judge over every system's replies on the golden set.

Usage: python scripts/run_judge.py [--systems agent simple trivial_always] [--limit N]
Writes outputs/judge_<system>.jsonl. Calls are cached, so re-runs are instant.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent.llm import get_llm  # noqa: E402
from src.agent.retrieval import Retriever  # noqa: E402
from src.eval.judge import judge_reply  # noqa: E402

PRED_FILES = {"agent": "outputs/predictions_agent_golden.jsonl", "simple": "outputs/predictions_simple.jsonl",
              "trivial_always": "outputs/predictions_trivial_always.jsonl"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=list(PRED_FILES))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    gold = {json.loads(l)["id"]: json.loads(l) for l in open("data/golden/golden_set.jsonl", encoding="utf-8")}
    retriever = Retriever.load()
    precedent_cache = {}
    llm = get_llm("judge", num_ctx=2048)
    for system in args.systems:
        preds = [json.loads(l) for l in open(PRED_FILES[system], encoding="utf-8")]
        if args.limit:
            preds = preds[: args.limit]
        out = Path(f"outputs/judge_{system}.jsonl")
        t0 = time.time()
        with open(out, "w", encoding="utf-8") as f:
            for i, p in enumerate(preds, 1):
                g = gold[p["id"]]
                if p["id"] not in precedent_cache:
                    precedent_cache[p["id"]] = retriever.search(g["customer_text"], k=3)
                j = judge_reply(llm, customer_text=g["customer_text"], gold_intent=g["intent"], gold_escalate=g["escalate"],
                                gold_reason=g["escalation_reason"], reference_reply=g["brand_reply"],
                                precedent=precedent_cache[p["id"]], reply=p["reply"])
                f.write(json.dumps({"id": p["id"], "system": system, "reply": p["reply"], **j}, ensure_ascii=False) + "\n")
                if i % 25 == 0 or i == len(preds):
                    print(f"[{system}] {i}/{len(preds)} {time.time()-t0:.0f}s", flush=True)
        print("wrote", out, llm.stats)
