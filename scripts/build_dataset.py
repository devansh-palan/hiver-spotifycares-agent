"""Step 1: reconstruct SpotifyCares (customer opener -> first brand reply) pairs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.data import PROCESSED_PATH, build_pairs  # noqa: E402

if __name__ == "__main__":
    pairs = build_pairs()
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(PROCESSED_PATH, index=False)
    print(f"wrote {len(pairs)} pairs -> {PROCESSED_PATH}")
    print(pairs[["customer_text", "brand_reply"]].head(8).to_string())
