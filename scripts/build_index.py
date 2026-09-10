"""Step 3: embed historical pairs (minus golden/dev ids) into the retrieval index."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent.data import load_pairs  # noqa: E402
from src.agent.retrieval import INDEX_DIR, Retriever  # noqa: E402


def held_out_ids() -> set[int]:
    ids = set()
    for name in ("golden_set.jsonl", "dev_set.jsonl", "candidates.jsonl", "dev_candidates.jsonl"):
        p = Path("data/golden") / name
        if p.exists():
            ids |= {json.loads(l)["opener_id"] for l in open(p, encoding="utf-8")}
    return ids


if __name__ == "__main__":
    t0 = time.time()
    pairs = load_pairs()
    ex = held_out_ids()
    r = Retriever.build(pairs, exclude_ids=ex, out_dir=INDEX_DIR)
    print(f"indexed {len(r.meta)} pairs (excluded {len(ex)} held-out ids) in {time.time()-t0:.0f}s -> {INDEX_DIR}")
    print("dm_redirect share:", round(float(r.meta.dm_redirect.mean()), 3))
    for q in ["my premium got charged twice", "app keeps crashing on iphone x", "is spotify down right now"]:
        print("\nQ:", q)
        for e in r.search(q, k=3):
            print(f"  {e['score']:.2f} | {e['customer_text'][:70]} -> {e['brand_reply'][:90]}")
