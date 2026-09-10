"""Judge vs human agreement on a blind-rated subset of replies."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = float(np.mean(a == b))
    pe = float(np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b)))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement(human: list[dict], judge: list[dict]) -> dict:
    """Both lists aligned by (id, system). Human rows: overall (1-5), acceptable (bool)."""
    h_over = [h["overall"] for h in human]
    j_over = [j["overall"] for j in judge]
    h_acc = [bool(h["acceptable"]) for h in human]
    j_acc = [bool(j["acceptable"]) for j in judge]
    rho, p = spearmanr(h_over, j_over)
    diff = np.abs(np.asarray(h_over) - np.asarray(j_over))
    return {
        "n": len(human),
        "spearman_overall": round(float(rho), 3), "spearman_p": round(float(p), 4),
        "exact_agreement_overall": round(float(np.mean(diff == 0)), 3),
        "within_1_agreement_overall": round(float(np.mean(diff <= 1)), 3),
        "cohen_kappa_acceptable": round(cohen_kappa(h_acc, j_acc), 3),
        "raw_agreement_acceptable": round(float(np.mean(np.asarray(h_acc) == np.asarray(j_acc))), 3),
        "human_acceptable_rate": round(float(np.mean(h_acc)), 3),
        "judge_acceptable_rate": round(float(np.mean(j_acc)), 3),
        "judge_false_accepts": int(sum((not h) and j for h, j in zip(h_acc, j_acc))),
        "judge_false_rejects": int(sum(h and (not j) for h, j in zip(h_acc, j_acc))),
        "mean_human_overall": round(float(np.mean(h_over)), 2), "mean_judge_overall": round(float(np.mean(j_over)), 2),
    }
