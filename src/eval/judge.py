"""LLM-as-judge for reply quality.

The judge is a *different, larger* model than the agent (llama3.1:8b vs qwen2.5:3b by default)
to reduce self-preference bias. It grades one reply at a time against (a) the gold intent and
handling decision, (b) the brand's real reply to the same tweet (reference), and (c) three
real replies to similar tweets (precedent). Scores are 1-5 plus a binary `acceptable`.
Judge/human agreement is measured in src/eval/agreement.py.
"""
from __future__ import annotations

from ..agent.llm import LLM

SYSTEM = (
    "You are a strict quality auditor for customer-support replies written for @SpotifyCares, Spotify's support "
    "account on Twitter. You grade one drafted reply. Be harsh on generic replies and on anything invented. "
    "Return a single strict JSON object and nothing else."
)

RUBRIC = """Score each criterion from 1 (bad) to 5 (excellent):
- grounded: the reply is consistent with the reference reply and precedent and with Spotify's known behaviour. Invented URLs (anything other than the literal token [LINK]), invented prices, dates, features, refunds or promises -> 1 or 2.
- helpful: it moves THIS customer's specific problem forward: the right troubleshooting step or diagnostic question, the right information, or (when escalation is correct) a clear request to DM the right details. A generic reply that could be sent to any tweet -> 2 or lower. Troubleshooting a customer who should have been escalated, or escalating a simple question -> 2 or lower.
- tone: matches the SpotifyCares voice: friendly, human, concise, tweet-length, greeting at the start, no corporate boilerplate, no lecturing.
- safe: true only if it never asks for passwords or payment details, never asks for private details outside DM, never promises a refund or a fix date, and never asks a possibly-hacked user to post details publicly.
- overall: 1-5 holistic quality.
- acceptable: true if a Spotify support lead would let it go out without edits.
Return {"grounded": int, "helpful": int, "tone": int, "safe": bool, "overall": int, "acceptable": bool, "critique": "one short sentence"}"""


def build_prompt(customer_text: str, gold_intent: str, gold_escalate: bool, gold_reason: str, reference_reply: str,
                 precedent: list[dict], reply: str) -> str:
    handling = f"ESCALATE to a human (reason: {gold_reason})" if gold_escalate else "AUTO-HANDLE with a public reply"
    lines = [f'Customer tweet: "{customer_text}"', f"Gold intent: {gold_intent}", f"Gold handling: {handling}",
             f'Reference (SpotifyCares\' real reply to this tweet): "{reference_reply}"',
             "Precedent (real SpotifyCares replies to similar tweets):"]
    for i, ex in enumerate(precedent[:3], 1):
        lines.append(f'{i}. "{ex["customer_text"][:160]}" -> "{ex["brand_reply"][:200]}"')
    lines += ["", f'Reply to grade: "{reply}"', "", RUBRIC]
    return "\n".join(lines)


def normalise(raw: dict) -> dict:
    def _i(k):
        try:
            return int(max(1, min(5, round(float(raw.get(k, 1))))))
        except (TypeError, ValueError):
            return 1

    return {"grounded": _i("grounded"), "helpful": _i("helpful"), "tone": _i("tone"), "overall": _i("overall"),
            "safe": bool(raw.get("safe", False)), "acceptable": bool(raw.get("acceptable", False)),
            "critique": str(raw.get("critique", ""))[:300], "parse_error": bool(raw.get("_parse_error", False))}


def judge_reply(llm: LLM, **kw) -> dict:
    return normalise(llm.chat_json(SYSTEM, build_prompt(**kw), max_tokens=160, tag="judge"))
