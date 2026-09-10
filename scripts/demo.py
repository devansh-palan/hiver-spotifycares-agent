"""Handle one customer message end to end and print the full decision trace.

Usage: python scripts/demo.py "my premium got charged twice this month"
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent.pipeline import SupportAgent  # noqa: E402

if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "is spotify down?? nothing is loading"
    agent = SupportAgent()
    out = agent.handle(text)
    print(f"\nCUSTOMER : {text}")
    print(f"INTENT   : {out.intent} (confidence {out.intent_confidence:.2f}) - {out.triage.get('summary','')}")
    print(f"DECISION : {'ESCALATE' if out.escalate else 'AUTO-HANDLE'} [{out.escalation_reason}] via {out.escalation_source}: {out.escalation_detail}")
    print(f"REPLY    : {out.reply}")
    print(f"GROUNDING: exemplar ids {out.exemplar_ids} (cosine {out.exemplar_scores})")
    print(f"LATENCY  : {out.latency_s}s   llm={json.dumps(agent.llm.stats)}")
