"""Prompt-iteration check on the 40-example dev set (never on the golden set).

Runs triage + escalation policy only (no drafting) and prints intent accuracy,
escalation precision/recall and every disagreement.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent import escalation, triage  # noqa: E402
from src.agent.llm import get_llm  # noqa: E402

if __name__ == "__main__":
    rows = [json.loads(l) for l in open("data/golden/dev_set.jsonl", encoding="utf-8")]
    llm = get_llm("agent")
    t0 = time.time()
    ok_i = ok_e = 0
    tp = fp = fn = 0
    for r in rows:
        tri = triage.triage(llm, r["customer_text"])
        dec = escalation.decide(r["customer_text"], tri)
        hit_i = tri["intent"] == r["intent"]
        hit_e = dec.escalate == r["escalate"]
        ok_i += hit_i
        ok_e += hit_e
        tp += dec.escalate and r["escalate"]
        fp += dec.escalate and not r["escalate"]
        fn += (not dec.escalate) and r["escalate"]
        if not (hit_i and hit_e):
            print(f"{r['id']} gold=({r['intent']},{r['escalate']}) pred=({tri['intent']},{dec.escalate}:{dec.reason}/{dec.source}) "
                  f"conf={tri['confidence']:.2f} naa={tri['needs_account_action']} risk={tri['risk']} | {r['customer_text'][:90]}")
    n = len(rows)
    print(f"\nintent acc {ok_i}/{n}={ok_i/n:.2f}  escalation acc {ok_e}/{n}={ok_e/n:.2f}  "
          f"esc P={tp/max(tp+fp,1):.2f} R={tp/max(tp+fn,1):.2f}  time {time.time()-t0:.0f}s  llm={llm.stats}")
