"""Step 5: run the SupportAgent over the golden (or dev) set and save predictions.

Usage: python scripts/run_pipeline.py [--set golden|dev] [--limit N] [--out path]
All LLM calls are cached under outputs/llm_cache, so re-running is instant once cached.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent.pipeline import SupportAgent  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="golden", choices=["golden", "dev"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(f"data/golden/{args.set}_set.jsonl", encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    out = Path(args.out or f"outputs/predictions_agent_{args.set}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    agent = SupportAgent()
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            o = agent.handle(r["customer_text"]).to_dict()
            o["id"] = r["id"]
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
            if i % 25 == 0 or i == len(rows):
                print(f"{i}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
    print("wrote", out, agent.llm.stats)
