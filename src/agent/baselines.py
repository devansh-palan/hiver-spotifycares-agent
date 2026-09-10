"""Two baselines the agent has to beat.

TrivialBaseline  - majority-class intent, one canned reply, a fixed escalation policy
                   (always-escalate or never-escalate).
SimpleBaseline   - TF-IDF + logistic regression intent classifier trained on LLM silver labels,
                   nearest-neighbour reply (the real SpotifyCares reply to the most similar past
                   tweet), and an intent-level + regex escalation policy (no LLM at all).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .escalation import PRIVATE_DATA, TRIGGERS
from .retrieval import Retriever

CANNED = "Hey there! Sorry to hear that. Can you DM us your account's username or email address? We'll take a look backstage."
NAME_RE = re.compile(r"^(hey|hi|hello|hiya)\s+[A-Z][\w'-]+[,.!]?\s*", re.I)


def strip_name(reply: str) -> str:
    """'Hi Kyle! Fingers crossed...' -> 'Hi! Fingers crossed...' (we never know the customer's name)."""
    m = NAME_RE.match(reply)
    if m:
        word = m.group(0).split()[1].strip(",.!")
        if word.lower() not in {"there", "all", "guys", "folks", "again"}:
            return re.sub(r"^(hey|hi|hello|hiya)\s+\S+[,.!]?", lambda mm: mm.group(1).capitalize() + "!", reply, flags=re.I).strip()
    return reply


@dataclass
class BaselineOutput:
    intent: str
    escalate: bool
    escalation_reason: str
    reply: str
    intent_confidence: float = 0.0


class TrivialBaseline:
    def __init__(self, majority_intent: str, policy: str = "always"):
        assert policy in {"always", "never"}
        self.majority_intent = majority_intent
        self.policy = policy

    def handle(self, text: str) -> BaselineOutput:
        esc = self.policy == "always"
        return BaselineOutput(self.majority_intent, esc, "account_access_needed" if esc else "none", CANNED, 1.0)


class SimpleBaseline:
    """No LLM: TF-IDF/LR intent, 1-NN reply, regex + intent escalation."""

    ESCALATE_INTENTS = {"account_access", "billing_subscription"}

    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.clf = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
            ("lr", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
        ])

    def fit(self, texts: list[str], labels: list[str]):
        self.clf.fit(texts, labels)
        return self

    def handle(self, text: str) -> BaselineOutput:
        proba = self.clf.predict_proba([text])[0]
        idx = proba.argmax()
        intent = self.clf.classes_[idx]
        reason, esc = "none", False
        for r, rx in TRIGGERS.items():
            if rx.search(text):
                esc, reason = True, r
                break
        if not esc and PRIVATE_DATA.search(text):
            esc, reason = True, "account_access_needed"
        if not esc and intent in self.ESCALATE_INTENTS:
            esc, reason = True, "account_access_needed"
        nn = self.retriever.search(text, k=1, diverse=False)
        reply = strip_name(nn[0]["brand_reply"]) if nn else CANNED
        return BaselineOutput(intent, esc, reason, reply, float(proba[idx]))
