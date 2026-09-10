"""Step 9b: judge-vs-human agreement on the blind-rated subset -> outputs/agreement.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.agreement import agreement  # noqa: E402

if __name__ == "__main__":
    human = [json.loads(l) for l in open("data/golden/human_reply_ratings.jsonl", encoding="utf-8")]
    human = [h for h in human if h.get("overall") is not None]
    judge = {}
    for system in {h["system"] for h in human}:
        p = Path(f"outputs/judge_{system}.jsonl")
        if not p.exists():
            print(f"(no judge output yet for {system}; skipping its ratings)")
            continue
        for l in open(p, encoding="utf-8"):
            j = json.loads(l)
            judge[(j["id"], system)] = j
    pairs = [(h, judge[(h["id"], h["system"])]) for h in human if (h["id"], h["system"]) in judge]
    res = {"overall": agreement([h for h, _ in pairs], [j for _, j in pairs])}
    for system in sorted({h["system"] for h in human}):
        sub = [(h, j) for h, j in pairs if h["system"] == system]
        if len(sub) >= 5:
            res[system] = agreement([h for h, _ in sub], [j for _, j in sub])
    disagreements = [{"rating_id": h["rating_id"], "system": h["system"], "human": h["overall"], "judge": j["overall"],
                      "human_ok": h["acceptable"], "judge_ok": j["acceptable"], "reply": h["reply"][:140], "judge_critique": j["critique"], "note": h.get("note", "")}
                     for h, j in pairs if h["acceptable"] != j["acceptable"] or abs(h["overall"] - j["overall"]) >= 2]
    res["disagreements"] = disagreements
    Path("outputs").mkdir(exist_ok=True)
    json.dump(res, open("outputs/agreement.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "disagreements"}, indent=1))
    print(f"{len(disagreements)} disagreements (acceptable flip or |diff|>=2)")
