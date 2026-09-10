"""Automated metrics for intent, escalation and reply proxies, with bootstrap CIs."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np

INTENTS = ["technical_issue", "service_outage", "account_access", "billing_subscription",
           "content_availability", "feature_feedback", "general_question", "other"]


# ----------------------------------------------------------------------------- intent
def _prf(gold: list[str], pred: list[str], label: str) -> tuple[float, float, float, int]:
    tp = sum(g == label and p == label for g, p in zip(gold, pred))
    fp = sum(g != label and p == label for g, p in zip(gold, pred))
    fn = sum(g == label and p != label for g, p in zip(gold, pred))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, tp + fn


def macro_f1(gold: list[str], pred: list[str], labels=INTENTS) -> float:
    present = [l for l in labels if any(g == l for g in gold)]
    return float(np.mean([_prf(gold, pred, l)[2] for l in present])) if present else 0.0


def accuracy(gold: list, pred: list) -> float:
    return float(np.mean([g == p for g, p in zip(gold, pred)])) if gold else 0.0


def bootstrap_ci(gold: list, pred: list, fn, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(gold)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(fn([gold[i] for i in idx], [pred[i] for i in idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def intent_report(gold: list[str], pred: list[str]) -> dict:
    per_class = {}
    for l in INTENTS:
        p, r, f, n = _prf(gold, pred, l)
        per_class[l] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "support": n}
    conf = defaultdict(Counter)
    for g, p in zip(gold, pred):
        conf[g][p] += 1
    acc = accuracy(gold, pred)
    lo, hi = bootstrap_ci(gold, pred, accuracy)
    mf = macro_f1(gold, pred)
    mlo, mhi = bootstrap_ci(gold, pred, macro_f1)
    return {"n": len(gold), "accuracy": round(acc, 3), "accuracy_ci95": [round(lo, 3), round(hi, 3)],
            "macro_f1": round(mf, 3), "macro_f1_ci95": [round(mlo, 3), round(mhi, 3)],
            "per_class": per_class, "confusion": {g: dict(c) for g, c in conf.items()}}


# ------------------------------------------------------------------------- escalation
def escalation_report(gold: list[bool], pred: list[bool], gold_reason: list[str] | None = None,
                      pred_reason: list[str] | None = None) -> dict:
    n = len(gold)
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    fn = sum(g and (not p) for g, p in zip(gold, pred))
    tn = n - tp - fp - fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    def _rec(g, p):
        t = sum(a and b for a, b in zip(g, p))
        d = sum(g)
        return t / d if d else 0.0

    rlo, rhi = bootstrap_ci(gold, pred, _rec)
    out = {"n": n, "accuracy": round((tp + tn) / n, 3), "precision": round(prec, 3), "recall": round(rec, 3),
           "recall_ci95": [round(rlo, 3), round(rhi, 3)], "f1": round(f1, 3),
           "automation_rate": round((tn + fn) / n, 3),  # share the system would auto-handle
           "unsafe_auto_rate": round(fn / n, 3),  # should have escalated but auto-handled (the number that matters)
           "over_escalation_rate": round(fp / n, 3), "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    if gold_reason is not None and pred_reason is not None:
        both = [(gr, pr) for g, p, gr, pr in zip(gold, pred, gold_reason, pred_reason) if g and p]
        out["reason_match_rate_on_tp"] = round(float(np.mean([a == b for a, b in both])), 3) if both else None
        out["reason_confusion"] = {f"{a}->{b}": c for (a, b), c in Counter(both).most_common(12) if a != b}
    return out


# ------------------------------------------------------------------------ reply proxies
URL_RE = re.compile(r"https?://|www\.", re.I)
GREETING_RE = re.compile(r"^(hey|hi|hello|hiya)\b", re.I)
PASSWORD_RE = re.compile(r"\bpassword\b|card number|cvv|bank details", re.I)
PUBLIC_PII_RE = re.compile(r"(reply|tweet|post|share|send)\w*\s(us\s)?(with\s)?(your|the)\s(email|username)", re.I)
DM_RE = re.compile(r"\bdm\b|direct message", re.I)
INITIALS_RE = re.compile(r"\s/[A-Z]{1,3}$")
AI_RE = re.compile(r"\b(as an ai|language model|i am an ai|i'm an ai)\b", re.I)


def reply_proxies(replies: list[str], escalate_pred: list[bool] | None = None) -> dict:
    n = len(replies)
    stats = {
        "n": n,
        "mean_chars": round(float(np.mean([len(r) for r in replies])), 1),
        "pct_over_280": round(float(np.mean([len(r) > 280 for r in replies])), 3),
        "greeting_rate": round(float(np.mean([bool(GREETING_RE.match(r)) for r in replies])), 3),
        "invented_url_rate": round(float(np.mean([bool(URL_RE.search(r)) for r in replies])), 3),
        "asks_password_or_card": round(float(np.mean([bool(PASSWORD_RE.search(r)) for r in replies])), 3),
        "asks_pii_publicly": round(float(np.mean([bool(PUBLIC_PII_RE.search(r)) and not DM_RE.search(r) for r in replies])), 3),
        "signs_initials": round(float(np.mean([bool(INITIALS_RE.search(r)) for r in replies])), 3),
        "mentions_ai": round(float(np.mean([bool(AI_RE.search(r)) for r in replies])), 3),
        "empty": sum(len(r.strip()) == 0 for r in replies),
    }
    if escalate_pred is not None:
        esc = [r for r, e in zip(replies, escalate_pred) if e]
        stats["escalated_replies_mention_dm"] = round(float(np.mean([bool(DM_RE.search(r)) for r in esc])), 3) if esc else None
    return stats
