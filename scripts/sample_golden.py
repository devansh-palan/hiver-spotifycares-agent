"""Step 2: draw golden-set candidates (200) and a dev set (40) for hand labelling.

Writes data/golden/candidates.jsonl and data/golden/dev_candidates.jsonl with empty label fields.
Labels are then filled in by hand (data/golden/golden_set.jsonl). See LABELING_GUIDE.md.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent.data import load_pairs  # noqa: E402

SEED = 2024
BUCKETS = {
    "outage": r"\bdown\b|outage|is (spotify|it) (down|broken)|anyone else|not just me|what'?s going on",
    "account": r"hack|log ?in|login|password|locked out|email (was |has been )?changed|sign ?in|can'?t (get )?(in|access)",
    "billing": r"charg|refund|payment|bill|subscri|student|family|trial|discount|cancel|£|\$|invoice|premium",
    "content": r"\badd\b|album|release|missing|removed|available in|region|explicit|catalog|bring back",
    "tech": r"crash|won'?t play|not playing|skip|glitch|bug|error|freez|offline|download|connect|sonos|alexa|google home|\bcar\b|ps4|xbox|watch|bluetooth",
    "feedback": r"feature|suggest|please (stop|remove|bring)|would be (nice|great)|wish|shuffle|\bads?\b|advert",
    "question": r"^(how|what|when|where|why|does|do|is|are|can|will)\b",
    "short_or_nonascii": None,  # filled programmatically
}


def bucket_of(text: str) -> list[str]:
    t = text.lower()
    hits = [name for name, rx in BUCKETS.items() if rx and re.search(rx, t)]
    non_ascii = sum(ord(ch) > 127 for ch in text) / max(len(text), 1)
    if len(text) < 40 or non_ascii > 0.15:
        hits.append("short_or_nonascii")
    return hits


def draw(pairs: pd.DataFrame, n_uniform: int, n_bucket_total: int, rng: np.random.Generator, exclude: set):
    pool = pairs[~pairs.opener_id.isin(exclude)]
    uniform = pool.sample(n_uniform, random_state=int(rng.integers(1 << 31)))
    chosen = {int(i): "uniform" for i in uniform.opener_id}
    per_bucket = int(np.ceil(n_bucket_total / len(BUCKETS)))
    rest = pool[~pool.opener_id.isin(chosen)]
    rest = rest.assign(buckets=rest.customer_text.map(bucket_of))
    for name in BUCKETS:
        cand = rest[rest.buckets.map(lambda b: name in b) & ~rest.opener_id.isin(chosen)]
        take = cand.sample(min(per_bucket, len(cand)), random_state=int(rng.integers(1 << 31)))
        for i in take.opener_id:
            if len(chosen) < n_uniform + n_bucket_total:
                chosen[int(i)] = name
    rows = pairs[pairs.opener_id.isin(chosen)].copy()
    rows["sampling_bucket"] = rows.opener_id.map(chosen)
    # shuffle so labelling order is not grouped by bucket
    return rows.sample(frac=1, random_state=int(rng.integers(1 << 31)))


def to_records(rows: pd.DataFrame, prefix: str):
    out = []
    for k, (_, r) in enumerate(rows.iterrows()):
        out.append(
            {
                "id": f"{prefix}{k:03d}",
                "opener_id": int(r.opener_id),
                "created_at": str(r.created_at),
                "sampling_bucket": r.sampling_bucket,
                "customer_text": r.customer_text,
                "brand_reply": r.brand_reply,
                "intent": "",
                "escalate": None,
                "escalation_reason": "",
                "notes": "",
            }
        )
    return out


if __name__ == "__main__":
    pairs = load_pairs()
    rng = np.random.default_rng(SEED)
    golden = draw(pairs, 100, 100, rng, exclude=set())
    dev = draw(pairs, 20, 20, rng, exclude=set(golden.opener_id))
    Path("data/golden").mkdir(parents=True, exist_ok=True)
    with open("data/golden/candidates.jsonl", "w", encoding="utf-8") as f:
        for rec in to_records(golden, "g"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open("data/golden/dev_candidates.jsonl", "w", encoding="utf-8") as f:
        for rec in to_records(dev, "d"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("golden", len(golden), golden.sampling_bucket.value_counts().to_dict())
    print("dev", len(dev), dev.sampling_bucket.value_counts().to_dict())
