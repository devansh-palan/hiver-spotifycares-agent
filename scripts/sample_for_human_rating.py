"""Step 9a: draw a blind sample of replies for hand rating (judge-agreement study).

20 replies per system (agent / simple / trivial_always), shuffled with a fixed seed and with the
system name hidden, so the human rates without knowing which system produced the reply.
Writes data/golden/human_rating_candidates.jsonl; the rater fills `overall` (1-5), `acceptable`
(bool) and `note` into data/golden/human_reply_ratings.jsonl using the same rubric as the judge.
"""
import json
import random
from pathlib import Path

SYSTEMS = {"agent": "outputs/predictions_agent_golden.jsonl", "simple": "outputs/predictions_simple.jsonl",
           "trivial_always": "outputs/predictions_trivial_always.jsonl"}
PER_SYSTEM = 20
SEED = 11

if __name__ == "__main__":
    gold = {json.loads(l)["id"]: json.loads(l) for l in open("data/golden/golden_set.jsonl", encoding="utf-8")}
    rng = random.Random(SEED)
    ids = sorted(gold)
    picks = []
    for system, path in SYSTEMS.items():
        preds = {json.loads(l)["id"]: json.loads(l) for l in open(path, encoding="utf-8")}
        for i in rng.sample(ids, PER_SYSTEM):
            picks.append({"id": i, "system": system, "reply": preds[i]["reply"]})
    rng.shuffle(picks)
    out = Path("data/golden/human_rating_candidates.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for k, p in enumerate(picks):
            g = gold[p["id"]]
            f.write(json.dumps({"rating_id": f"r{k:02d}", "id": p["id"], "system": p["system"], "customer_text": g["customer_text"],
                                "gold_intent": g["intent"], "gold_escalate": g["escalate"], "gold_reason": g["escalation_reason"],
                                "reference_reply": g["brand_reply"], "reply": p["reply"], "overall": None, "acceptable": None, "note": ""},
                               ensure_ascii=False) + "\n")
    print("wrote", out, len(picks))
