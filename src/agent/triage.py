"""LLM triage: intent + structured risk fields for one incoming tweet.

One call returns a JSON object with intent, language, needs_account_action, risk,
confidence and a one-line summary. The escalation policy (escalation.py) is a
deterministic function of these fields plus regex triggers, so every decision is
explainable.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .llm import LLM

INTENTS_PATH = Path("configs/intents.yaml")
_cfg = yaml.safe_load(open(INTENTS_PATH, encoding="utf-8"))
INTENTS: list[str] = list(_cfg["intents"].keys())
INTENT_DESC: dict[str, str] = {k: " ".join(v["description"].split()) for k, v in _cfg["intents"].items()}
RISKS = {"none", "security", "legal", "churn", "sensitive", "wants_human"}

SYSTEM = (
    "You are the first-line triage assistant for SpotifyCares, Spotify's customer support account on Twitter. "
    "You read one incoming customer tweet and return a single strict JSON object. No text outside the JSON."
)

# Few-shot examples are hand-written (not taken from the golden or dev sets).
FEW_SHOT = [
    ("my downloads keep vanishing every week on android, so annoying",
     {"intent": "technical_issue", "language": "en", "needs_account_action": False, "risk": "none", "confidence": 0.9,
      "summary": "Offline downloads are repeatedly removed on Android."}),
    ("is spotify down?? nothing loads for me or my friends",
     {"intent": "service_outage", "language": "en", "needs_account_action": False, "risk": "none", "confidence": 0.9,
      "summary": "Suspects a platform-wide outage."}),
    ("someone changed my email and now I'm locked out of my account",
     {"intent": "account_access", "language": "en", "needs_account_action": True, "risk": "security", "confidence": 0.95,
      "summary": "Account taken over; email changed; locked out."}),
    ("you charged me twice this month, I want my money back",
     {"intent": "billing_subscription", "language": "en", "needs_account_action": True, "risk": "none", "confidence": 0.95,
      "summary": "Double charge, refund requested."}),
    ("when does the 3 months for $0.99 offer end?",
     {"intent": "billing_subscription", "language": "en", "needs_account_action": False, "risk": "none", "confidence": 0.85,
      "summary": "Asks about intro offer end date (public information)."}),
    ("please add the new drake album already!!",
     {"intent": "content_availability", "language": "en", "needs_account_action": False, "risk": "none", "confidence": 0.9,
      "summary": "Wants an album added to the catalogue."}),
    ("shuffle is so bad, it plays the same 20 songs out of 500",
     {"intent": "feature_feedback", "language": "en", "needs_account_action": False, "risk": "none", "confidence": 0.85,
      "summary": "Complaint about shuffle design."}),
    ("how many songs can I download for offline?",
     {"intent": "general_question", "language": "en", "needs_account_action": False, "risk": "none", "confidence": 0.9,
      "summary": "Asks about the offline download limit."}),
    ("thanks for the help yesterday, you guys rock",
     {"intent": "other", "language": "en", "needs_account_action": False, "risk": "none", "confidence": 0.9,
      "summary": "Thanks only."}),
    ("app crashes every day, I'm switching to apple music if this isn't fixed",
     {"intent": "technical_issue", "language": "en", "needs_account_action": False, "risk": "churn", "confidence": 0.9,
      "summary": "Daily crashes; threatens to switch to a competitor."}),
    ("no puedo iniciar sesión en mi cuenta premium",
     {"intent": "account_access", "language": "es", "needs_account_action": True, "risk": "none", "confidence": 0.85,
      "summary": "Cannot log in (Spanish)."}),
    ("I already DMed you twice and nobody answers",
     {"intent": "other", "language": "en", "needs_account_action": False, "risk": "wants_human", "confidence": 0.8,
      "summary": "Chasing an unanswered DM thread."}),
    ("can I reorder the songs in a playlist on android? can't find the option",
     {"intent": "general_question", "language": "en", "needs_account_action": False, "risk": "none", "confidence": 0.85,
      "summary": "Asks how to do something in the app today."}),
    ("I no longer have access to the email on my premium account, can you move it?",
     {"intent": "account_access", "language": "en", "needs_account_action": True, "risk": "none", "confidence": 0.9,
      "summary": "Lost access to the account email."}),
]

GUIDANCE = (
    "Tie-breaks: a question about how to do something, or whether it is possible today, is general_question even if "
    "the answer is 'not possible'; feature_feedback is a suggestion or a complaint about the design. "
    "A trial, plan or Premium status that looks wrong for THIS user needs_account_action=true. "
    "Cannot log in / lost access to the account email / hacked is account_access even if money is mentioned."
)


def build_user_prompt(text: str) -> str:
    import json

    lines = ["Classify the customer tweet below.", "", "INTENTS (choose exactly one):"]
    for k in INTENTS:
        lines.append(f"- {k}: {INTENT_DESC[k]}")
    lines += [
        "",
        "Also decide:",
        '- language: "en" if the tweet is written in English, otherwise the ISO-639-1 code (e.g. "id", "es", "tl").',
        "- needs_account_action: true only if resolving this requires looking up, verifying, or changing THIS user's "
        "account, subscription, payment, or saved data (wrong charge, Premium not active, cannot log in, family invite "
        "failing, recover a deleted playlist). false if public information or generic troubleshooting steps are enough.",
        '- risk: "none", "security" (hacked / unauthorised access / fraud), "legal" (threatens to sue or involve lawyers), '
        '"churn" (threatens to cancel or switch to a competitor, or abuses the support team), "sensitive" (bereavement, '
        'health, distress), "wants_human" (asks for a human, says support ignored them, or refers to an existing DM).',
        "- confidence: 0 to 1 for the intent.",
        "- summary: one short sentence describing the problem.",
        "",
        GUIDANCE,
        "",
        "Examples:",
    ]
    for t, j in FEW_SHOT:
        lines.append(f'Tweet: "{t}" -> {json.dumps(j, ensure_ascii=False)}')
    lines += ["", f'Tweet: "{text}"', "Return only the JSON object."]
    return "\n".join(lines)


def normalise(raw: dict) -> dict:
    """Coerce a model response into a valid triage record (never raises)."""
    out = {
        "intent": raw.get("intent") if raw.get("intent") in INTENTS else "other",
        "language": str(raw.get("language", "en")).lower()[:5] or "en",
        "needs_account_action": bool(raw.get("needs_account_action", False)),
        "risk": raw.get("risk") if raw.get("risk") in RISKS else "none",
        "confidence": float(raw.get("confidence", 0.0) or 0.0),
        "summary": str(raw.get("summary", ""))[:200],
        "parse_error": bool(raw.get("_parse_error", False)),
        "raw_intent": raw.get("intent"),
    }
    out["confidence"] = max(0.0, min(1.0, out["confidence"]))
    # Labelling guide: non-English messages are intent `other` (routed by language, not topic).
    if out["language"] not in ("en", "eng", "english"):
        out["intent_before_language_rule"] = out["intent"]
        out["intent"] = "other"
    return out


def triage(llm: LLM, text: str) -> dict:
    raw = llm.chat_json(SYSTEM, build_user_prompt(text), max_tokens=160, tag="triage")
    return normalise(raw)
