"""Merge hand labels (labels_golden_part*.json) into candidates.jsonl -> golden_set.jsonl.

Each label entry is [intent, escalate, escalation_reason, notes]. Fails loudly on any
missing or invalid label so the golden set is always complete and schema-valid.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
INTENTS = set(yaml.safe_load(open(ROOT / "configs/intents.yaml"))["intents"])
REASONS = {
    "none", "account_access_needed", "billing_or_refund", "security_fraud", "churn_or_anger_risk",
    "non_english", "repeat_contact_or_frustration", "legal_or_pr_risk", "needs_account_lookup", "sensitive_situation",
}


def merge(cand_path: Path, label_glob: str, out_path: Path):
    labels = {}
    for p in sorted((ROOT / "data/golden").glob(label_glob)):
        labels.update(json.load(open(p, encoding="utf-8")))
    rows = [json.loads(l) for l in open(cand_path, encoding="utf-8")]
    out = []
    for r in rows:
        if r["id"] not in labels:
            sys.exit(f"missing label for {r['id']}")
        intent, esc, reason, notes = labels[r["id"]]
        assert intent in INTENTS, (r["id"], intent)
        assert isinstance(esc, bool), (r["id"], esc)
        assert reason in REASONS, (r["id"], reason)
        assert (reason == "none") == (not esc), f"{r['id']}: escalate={esc} but reason={reason}"
        r.update(intent=intent, escalate=esc, escalation_reason=reason, notes=notes)
        out.append(r)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{out_path}: {len(out)} rows")
    print("  intents:", dict(Counter(r["intent"] for r in out).most_common()))
    print("  escalate:", dict(Counter(r["escalate"] for r in out)))
    print("  reasons:", dict(Counter(r["escalation_reason"] for r in out).most_common()))
    return out


if __name__ == "__main__":
    merge(ROOT / "data/golden/candidates.jsonl", "labels_golden_part*.json", ROOT / "data/golden/golden_set.jsonl")
    if (ROOT / "data/golden/labels_dev.json").exists():
        merge(ROOT / "data/golden/dev_candidates.jsonl", "labels_dev.json", ROOT / "data/golden/dev_set.jsonl")
