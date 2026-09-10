"""Step 8: compute every metric from saved predictions (no model needed) -> outputs/results.json + results.md.

Systems: agent (LLM pipeline), simple (TF-IDF/LR + 1-NN + rules), trivial_always / trivial_never.
Reply quality comes from outputs/judge_<system>.jsonl when present.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.metrics import escalation_report, intent_report, reply_proxies  # noqa: E402

SYSTEMS = {"agent": "outputs/predictions_agent_golden.jsonl", "simple": "outputs/predictions_simple.jsonl",
           "trivial_always": "outputs/predictions_trivial_always.jsonl", "trivial_never": "outputs/predictions_trivial_never.jsonl"}


def load(path):
    p = Path(path)
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else None


def judge_summary(rows):
    if not rows:
        return None
    return {"n": len(rows), "acceptable_rate": round(float(np.mean([r["acceptable"] for r in rows])), 3),
            "safe_rate": round(float(np.mean([r["safe"] for r in rows])), 3),
            "mean_overall": round(float(np.mean([r["overall"] for r in rows])), 2),
            "mean_grounded": round(float(np.mean([r["grounded"] for r in rows])), 2),
            "mean_helpful": round(float(np.mean([r["helpful"] for r in rows])), 2),
            "mean_tone": round(float(np.mean([r["tone"] for r in rows])), 2),
            "parse_errors": sum(r.get("parse_error", False) for r in rows)}


def evaluate(system: str, gold: list[dict]) -> dict | None:
    preds = load(SYSTEMS[system])
    if preds is None:
        return None
    pm = {p["id"]: p for p in preds}
    ids = [g["id"] for g in gold if g["id"] in pm]
    g_int = [next(x for x in gold if x["id"] == i)["intent"] for i in ids]
    gmap = {g["id"]: g for g in gold}
    p_int = [pm[i]["intent"] for i in ids]
    g_esc = [gmap[i]["escalate"] for i in ids]
    p_esc = [bool(pm[i]["escalate"]) for i in ids]
    g_rea = [gmap[i]["escalation_reason"] for i in ids]
    p_rea = [pm[i]["escalation_reason"] for i in ids]
    res = {"intent": intent_report(g_int, p_int), "escalation": escalation_report(g_esc, p_esc, g_rea, p_rea),
           "reply_proxies": reply_proxies([pm[i]["reply"] for i in ids], p_esc)}
    # production-distribution view: uniform-random subset only
    uni = [i for i in ids if gmap[i]["sampling_bucket"] == "uniform"]
    if uni:
        res["uniform_subset"] = {
            "n": len(uni),
            "intent_accuracy": intent_report([gmap[i]["intent"] for i in uni], [pm[i]["intent"] for i in uni])["accuracy"],
            "escalation": {k: v for k, v in escalation_report([gmap[i]["escalate"] for i in uni], [bool(pm[i]["escalate"]) for i in uni]).items()
                           if k in ("accuracy", "precision", "recall", "automation_rate", "unsafe_auto_rate")},
        }
    jr = load(f"outputs/judge_{system}.jsonl")
    res["judge"] = judge_summary(jr)
    if jr:
        jm = {j["id"]: j for j in jr}
        # end-to-end "trustworthy" = intent right AND escalation right AND judge says acceptable
        e2e = [(pm[i]["intent"] == gmap[i]["intent"]) and (bool(pm[i]["escalate"]) == gmap[i]["escalate"]) and jm[i]["acceptable"]
               for i in ids if i in jm]
        res["end_to_end_trust_rate"] = round(float(np.mean(e2e)), 3) if e2e else None
        # judge acceptability by gold intent / by escalation correctness
        by_int = {}
        for i in ids:
            if i in jm:
                by_int.setdefault(gmap[i]["intent"], []).append(jm[i]["acceptable"])
        res["judge_acceptable_by_intent"] = {k: round(float(np.mean(v)), 2) for k, v in sorted(by_int.items())}
    # confidence calibration for the agent
    if system == "agent":
        bins = {}
        for i in ids:
            c = pm[i].get("intent_confidence", 0.0)
            b = "<0.7" if c < 0.7 else ("0.7-0.85" if c < 0.85 else ">=0.85")
            bins.setdefault(b, []).append(pm[i]["intent"] == gmap[i]["intent"])
        res["confidence_calibration"] = {b: {"n": len(v), "accuracy": round(float(np.mean(v)), 3)} for b, v in sorted(bins.items())}
        res["escalation_source_counts"] = dict(Counter(pm[i].get("escalation_source", "?") for i in ids))
    return res


def md_table(results: dict) -> str:
    rows = ["| system | intent acc (95% CI) | macro-F1 | esc. precision | esc. recall | automation rate | unsafe-auto rate | judge acceptable | judge overall | end-to-end trust |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for s, r in results.items():
        if not r:
            continue
        i, e, j = r["intent"], r["escalation"], r.get("judge")
        rows.append(f"| {s} | {i['accuracy']:.2f} ({i['accuracy_ci95'][0]:.2f}-{i['accuracy_ci95'][1]:.2f}) | {i['macro_f1']:.2f} | "
                    f"{e['precision']:.2f} | {e['recall']:.2f} | {e['automation_rate']:.2f} | {e['unsafe_auto_rate']:.3f} | "
                    f"{(str(j['acceptable_rate']) if j else 'n/a')} | {(str(j['mean_overall']) if j else 'n/a')} | "
                    f"{r.get('end_to_end_trust_rate', 'n/a')} |")
    return "\n".join(rows)


if __name__ == "__main__":
    gold = [json.loads(l) for l in open("data/golden/golden_set.jsonl", encoding="utf-8")]
    results = {s: evaluate(s, gold) for s in SYSTEMS}
    Path("outputs").mkdir(exist_ok=True)
    json.dump(results, open("outputs/results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    md = ["# Results on the 200-example golden set", "", md_table(results), ""]
    a = results.get("agent")
    if a:
        md += ["## Agent: per-intent F1", "", "| intent | P | R | F1 | support |", "|---|---|---|---|---|"]
        for k, v in a["intent"]["per_class"].items():
            md.append(f"| {k} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} | {v['support']} |")
        from src.eval.metrics import INTENTS
        short = {k: k.split("_")[0][:7] for k in INTENTS}
        md += ["", "## Agent: confusion matrix (rows = gold, columns = predicted)", "",
               "| gold \\ pred | " + " | ".join(short[k] for k in INTENTS) + " |", "|---|" + "---|" * len(INTENTS)]
        for g in INTENTS:
            row = a["intent"]["confusion"].get(g, {})
            md.append(f"| {g} | " + " | ".join(str(row.get(p, 0)) if row.get(p, 0) else "." for p in INTENTS) + " |")
        md += ["", "## Agent: uniform-random subset (production distribution)", "", f"`{json.dumps(a.get('uniform_subset'))}`", "",
               "## Agent: confidence calibration", "", f"`{json.dumps(a.get('confidence_calibration'))}`", "",
               "## Agent: escalation decision sources", "", f"`{json.dumps(a.get('escalation_source_counts'))}`", "",
               "## Agent: reply proxies", "", f"`{json.dumps(a['reply_proxies'])}`", ""]
        if a.get("judge"):
            md += ["## Agent: judge acceptability by gold intent", "", f"`{json.dumps(a.get('judge_acceptable_by_intent'))}`", ""]
    open("outputs/results.md", "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))
