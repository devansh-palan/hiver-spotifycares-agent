"""Escalation policy: deterministic rules over (regex triggers, LLM triage fields, intent).

Order matters and mirrors the labelling guide: hard triggers first, then language,
then intent-level policy. Every decision carries a reason code and a human-readable detail.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Regex hard triggers on the raw customer text. Deliberately tight: "what the hack" must not fire.
TRIGGERS = {
    "security_fraud": re.compile(
        r"\b(hacked|hacker|hijack\w*|unauthori[sz]ed|fraud\w*|stolen account|"
        r"someone (else )?(has been |is |was |got |apparently got )?(using|on|in|accessing|access to|hacked) my account)\b", re.I),
    "legal_or_pr_risk": re.compile(r"\b(sue|suing|lawsuit|lawyer|attorney|legal action|solicitor|small claims|trading standards)\b", re.I),
    "churn_or_anger_risk": re.compile(
        r"\b(going to (have to )?cancel|about to cancel|gonna cancel|will cancel|thinking (of|about) (quitting|switching|leaving|cancel\w*)|"
        r"switch(ing)? to (apple|tidal|deezer|youtube|amazon|google|pandora|soundcloud)|quit(ting)? (the service|spotify)|"
        r"seriously thinking about quitting|(away|close) (from|to) cancel\w*|cancel\w* (my )?(sub\w*|premium|membership) and)\b", re.I),
    "repeat_contact_or_frustration": re.compile(
        r"\b((talk|speak) to (a |an )?(human|person|real person|actual person|someone)|need customer service|"
        r"no ?(one|body) (is )?(respond\w*|answer\w*|repl\w*)|still waiting|already (contacted|emailed|dm'?d|messaged|tweeted)|"
        r"useless reply|dm me\b|(check|read) (your|the|my) dms?\b|sent (you )?a dm|dropped a dm|just dm'?d|"
        r"(replied|responded) to (your|the) dm|please read dm)\b", re.I),
}
PRIVATE_DATA = re.compile(r"__email__|[\w.+-]+@[\w-]+\.\w{2,}|\buser ?name\s*:\s*\S+", re.I)
# Login / identity problems always need a human, whatever the classifier says.
ACCESS_PROBLEM = re.compile(
    r"\b(forgot(ten)? (my |the )?(email|password|login)|can'?t (log ?in|sign ?in|login|access my account|get (back )?in(to)?)|"
    r"cannot (log ?in|sign ?in|login)|unable to (log ?in|sign ?in|login)|locked out|reset (my |the )?password|password reset|"
    r"lost access to|no longer have access to|don'?t (know|have) (my |the )?(email|password))\b", re.I)
# Billing: escalate unless it is a pure public-information question (no problem signal in the text).
BILLING_PROBLEM = re.compile(
    r"\b(not work\w*|fail\w*|error|can'?t|cannot|won'?t|doesn'?t|didn'?t|charg\w*|paid|refund\w*|still|wrong|issue|problem|"
    r"trial|invite|invitation|member|stuck|declin\w*|expired|renew\w*|cancel\w*|remove|gift card|code)\b", re.I)
MONEY = re.compile(r"\b(charg\w*|refund\w*|pay\w*|paid|money|invoice|bill\w*|\$|£|€|reimburs\w*|debit\w*)", re.I)
DATA_LOSS = re.compile(r"\b(playlist|library|saved songs)\b.*\b(gone|deleted|removed|disappeared|missing|wiped|lost)\b|"
                       r"\b(lost|deleted|removed|wiped)\b.*\b(playlist|library)\b", re.I)
RISK_TO_REASON = {
    "security": "security_fraud",
    "legal": "legal_or_pr_risk",
    "churn": "churn_or_anger_risk",
    "sensitive": "sensitive_situation",
    "wants_human": "repeat_contact_or_frustration",
}
ALWAYS_ESCALATE = {"account_access"}


@dataclass
class Decision:
    escalate: bool
    reason: str  # code from the labelling guide, "none" when auto-handled
    detail: str  # human-readable explanation
    source: str  # "regex" | "llm_risk" | "language" | "intent_policy" | "auto"


def decide(text: str, triage: dict) -> Decision:
    intent = triage["intent"]
    # 1. hard regex triggers
    for reason, rx in TRIGGERS.items():
        m = rx.search(text)
        if m:
            return Decision(True, reason, f"trigger phrase: '{m.group(0)}'", "regex")
    if PRIVATE_DATA.search(text):
        return Decision(True, "account_access_needed", "customer posted private account data publicly; a human should reply and ask them to delete it", "regex")
    m = ACCESS_PROBLEM.search(text)
    if m:
        return Decision(True, "account_access_needed", f"login/identity problem: '{m.group(0)}'", "regex")
    # 2. language
    if intent == "other" and triage.get("intent_before_language_rule"):
        return Decision(True, "non_english", f"message not in English ({triage['language']}); route to language support", "language")
    # 3. LLM-detected risk
    risk = triage.get("risk", "none")
    if risk in RISK_TO_REASON:
        return Decision(True, RISK_TO_REASON[risk], f"triage flagged risk={risk}: {triage.get('summary','')}", "llm_risk")
    # 4. intent-level policy
    if intent in ALWAYS_ESCALATE:
        return Decision(True, "account_access_needed", "account access problems need identity verification", "intent_policy")
    if intent == "billing_subscription":
        problem = BILLING_PROBLEM.search(text)
        if triage.get("needs_account_action", False) or problem:
            reason = "billing_or_refund" if MONEY.search(text) else "account_access_needed"
            why = f"problem signal '{problem.group(0)}'" if problem else "triage says the account must be checked"
            return Decision(True, reason, f"billing/plan issue that needs the customer's account ({why})", "intent_policy")
        return Decision(False, "none", "billing question answerable from public plan information", "auto")
    if intent == "technical_issue" and triage.get("needs_account_action", False) and DATA_LOSS.search(text):
        return Decision(True, "account_access_needed", "lost playlist/library recovery needs an account lookup", "intent_policy")
    if intent == "general_question" and triage.get("needs_account_action", False) and MONEY.search(text):
        return Decision(True, "needs_account_lookup", "question depends on this customer's account/billing state", "intent_policy")
    # 5. default: auto-handle
    return Decision(False, "none", f"{intent}: standard public resolution applies", "auto")
