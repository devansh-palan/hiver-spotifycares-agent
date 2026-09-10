"""Retrieval-grounded reply drafting in the SpotifyCares voice.

The model sees the customer tweet, the triage result, the escalation decision and the
k most similar historical (tweet -> real SpotifyCares reply) pairs. It is told to imitate
how the brand resolved those cases, never to invent URLs/prices/dates (support links are
written as the literal token [LINK], exactly as in the exemplars), and never to sign with
agent initials.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from .llm import LLM

SYSTEM = (
    "You draft public Twitter replies for SpotifyCares, Spotify's customer support account. "
    "Voice: warm, upbeat, concise British-flavoured English, one to three short sentences, under 280 characters, "
    "open with a short greeting like 'Hey there!' or 'Hi!'. No hashtags, at most one emoji, no bullet points. "
    "Ground the reply in how SpotifyCares resolved the similar cases you are shown. "
    "Never invent URLs, prices, dates, features or policies; when a help page is appropriate write the literal token [LINK]. "
    "Never ask for passwords, card numbers or payment details. Ask for the account's username or email address only via DM. "
    "Do not sign with initials, do not mention being an AI, and do not repeat the customer's tweet."
)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
# agent initials copied from exemplars: "... backstage /KL [LINK]" or "... /NJ"
SIG_RE = re.compile(r"\s/[A-Za-z]{1,3}(?=\s*(\[LINK\])?\s*$)")

_cfg = yaml.safe_load(open(Path("configs/intents.yaml"), encoding="utf-8"))["intents"]
PLAYBOOK = {k: v["typical_resolution"] for k, v in _cfg.items()}


def _handling_text(escalate: bool, reason: str, detail: str) -> str:
    if escalate:
        return (
            f"Handling decision: ESCALATE to a human agent (reason: {reason}; {detail}). "
            "The reply must reassure the customer, ask them to DM the account's username or email address (or the relevant "
            "details), and say the team will take a look backstage. Do not troubleshoot, do not promise refunds or outcomes. "
            "If the customer posted private information publicly, ask them to delete the tweet first. "
            "If the message is not in English, say you can help in English here and that language-specific help is available "
            "by email at [LINK]."
        )
    return (
        "Handling decision: AUTO-HANDLE. The reply should resolve the issue or move it forward: give the standard "
        "troubleshooting step or ask the one diagnostic question SpotifyCares usually asks (device, OS, app version), "
        "share the standard information, or acknowledge feedback and say it will be passed to the right team."
    )


def build_user_prompt(text: str, triage: dict, escalate: bool, reason: str, detail: str, exemplars: list[dict]) -> str:
    lines = [f'Customer tweet: "{text}"', f"Intent: {triage['intent']} ({triage.get('summary','')})",
             f"SpotifyCares playbook for this intent: {PLAYBOOK.get(triage['intent'], '')}",
             _handling_text(escalate, reason, detail), "",
             "How SpotifyCares resolved similar cases (real past replies, most similar first). Use the ones that match the "
             "intent above; ignore neighbours that are about a different problem:"]
    for i, ex in enumerate(exemplars, 1):
        lines.append(f'{i}. Customer: "{ex["customer_text"][:220]}"\n   SpotifyCares: "{ex["brand_reply"][:260]}"')
    lines += ["", "Write the reply now. Output the reply text only."]
    return "\n".join(lines)


def postprocess(reply: str) -> str:
    r = reply.strip().strip('"').strip()
    r = re.sub(r"^(reply|spotifycares)\s*:\s*", "", r, flags=re.I)
    r = URL_RE.sub("[LINK]", r)  # the model must not invent links
    r = SIG_RE.sub("", r)
    r = re.sub(r"\s+", " ", r).strip()
    if len(r) > 320:  # keep tweet-sized; cut at sentence boundary if possible
        cut = r[:320]
        r = cut[: cut.rfind(".") + 1] if "." in cut[100:] else cut
    return r


def draft(llm: LLM, text: str, triage: dict, escalate: bool, reason: str, detail: str, exemplars: list[dict]) -> str:
    raw = llm.chat(SYSTEM, build_user_prompt(text, triage, escalate, reason, detail, exemplars), max_tokens=140, tag="draft")
    return postprocess(raw)
