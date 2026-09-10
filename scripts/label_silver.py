"""Step 4: LLM silver labels for the simple baseline's training set.

1,000 historical openers (disjoint from golden/dev) are labelled with the same triage
prompt the agent uses. The TF-IDF baseline is therefore a distillation of the agent's
classifier; it cannot know anything the agent does not.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_index import held_out_ids  # noqa: E402
from src.agent import triage  # noqa: E402
from src.agent.data import load_pairs  # noqa: E402
from src.agent.llm import get_llm  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
OUT = Path("data/silver/silver_labels.jsonl")

if __name__ == "__main__":
    pairs = load_pairs()
    pool = pairs[~pairs.opener_id.isin(held_out_ids())]
    sample = pool.sample(N, random_state=7)
    llm = get_llm("agent")
    t0 = time.time()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for i, (_, r) in enumerate(sample.iterrows(), 1):
            tri = triage.triage(llm, r.customer_text)
            f.write(json.dumps({"opener_id": int(r.opener_id), "customer_text": r.customer_text, "brand_reply": r.brand_reply,
                                **{k: tri[k] for k in ("intent", "confidence", "language", "needs_account_action", "risk")}},
                               ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(f"{i}/{N} {time.time()-t0:.0f}s", flush=True)
    print("done", llm.stats)
