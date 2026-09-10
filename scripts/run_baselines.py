"""Step 6: run the trivial and simple baselines over the golden set.

Trivial : majority silver-label intent, one canned reply, always-escalate (and never-escalate, reported separately).
Simple  : TF-IDF + LR trained on data/silver/silver_labels.jsonl, 1-NN historical reply, regex + intent escalation.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent.baselines import SimpleBaseline, TrivialBaseline  # noqa: E402
from src.agent.retrieval import Retriever  # noqa: E402


def dump(rows, outs, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({"id": r["id"], "customer_text": r["customer_text"], "intent": o.intent,
                                "intent_confidence": o.intent_confidence, "escalate": o.escalate,
                                "escalation_reason": o.escalation_reason, "reply": o.reply}, ensure_ascii=False) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    rows = [json.loads(l) for l in open("data/golden/golden_set.jsonl", encoding="utf-8")]
    silver = [json.loads(l) for l in open("data/silver/silver_labels.jsonl", encoding="utf-8")]
    majority = Counter(s["intent"] for s in silver).most_common(1)[0][0]
    print("silver size", len(silver), "majority intent", majority, Counter(s["intent"] for s in silver).most_common())

    Path("outputs").mkdir(exist_ok=True)
    for policy in ("always", "never"):
        tb = TrivialBaseline(majority, policy=policy)
        dump(rows, [tb.handle(r["customer_text"]) for r in rows], Path(f"outputs/predictions_trivial_{policy}.jsonl"))

    retriever = Retriever.load()
    sb = SimpleBaseline(retriever).fit([s["customer_text"] for s in silver], [s["intent"] for s in silver])
    dump(rows, [sb.handle(r["customer_text"]) for r in rows], Path("outputs/predictions_simple.jsonl"))
