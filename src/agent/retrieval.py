"""Dense retrieval over historical (customer opener -> SpotifyCares reply) pairs.

Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-d, cosine). 25k rows embed in
well under a minute on a laptop GPU and in a few minutes on CPU. The index excludes every
golden/dev opener so evaluation never retrieves its own answer.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

INDEX_DIR = Path("data/processed/index")
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DM_ONLY_RE = re.compile(r"(sent|send|replied|responded|shoot|fire|drop)\w*\s.*\bdm\b|\bdm\b.*(way|there|us)", re.I)


def is_dm_redirect(reply: str) -> bool:
    """True when the brand reply only moves the conversation to DM (no public resolution)."""
    return bool(DM_ONLY_RE.search(reply)) and len(reply) < 170


class Retriever:
    def __init__(self, meta: pd.DataFrame, emb: np.ndarray, model=None):
        self.meta = meta.reset_index(drop=True)
        self.emb = emb
        self._model = model

    # ------------------------------------------------------------------ build / load
    @classmethod
    def build(cls, pairs: pd.DataFrame, exclude_ids: set[int], out_dir: Path = INDEX_DIR, batch_size: int = 256):
        from sentence_transformers import SentenceTransformer

        keep = pairs[~pairs.opener_id.isin(exclude_ids)].copy()
        keep = keep[keep.brand_reply.str.len() >= 20]
        # drop exact duplicate customer messages (retweets / copy-paste storms) to keep neighbours diverse
        keep = keep.drop_duplicates(subset=["customer_text"]).reset_index(drop=True)
        keep["dm_redirect"] = keep.brand_reply.map(is_dm_redirect)
        model = SentenceTransformer(EMB_MODEL)
        emb = model.encode(keep.customer_text.tolist(), batch_size=batch_size, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "emb.npy", emb)
        keep[["opener_id", "customer_text", "brand_reply", "dm_redirect", "created_at"]].to_parquet(out_dir / "meta.parquet", index=False)
        return cls(keep, emb, model)

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR):
        meta = pd.read_parquet(index_dir / "meta.parquet")
        emb = np.load(index_dir / "emb.npy")
        return cls(meta, emb)

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(EMB_MODEL)
        return self._model

    # ------------------------------------------------------------------ query
    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    def search(self, text: str, k: int = 5, *, diverse: bool = True, pool: int = 30) -> list[dict]:
        q = self.embed([text])[0]
        scores = self.emb @ q
        top = np.argpartition(-scores, min(pool, len(scores) - 1))[:pool]
        top = top[np.argsort(-scores[top])]
        out, seen = [], set()
        for i in top:
            row = self.meta.iloc[int(i)]
            sig = re.sub(r"\W+", " ", row.brand_reply.lower())[:60]
            if diverse and sig in seen:
                continue
            seen.add(sig)
            out.append(
                {"opener_id": int(row.opener_id), "customer_text": row.customer_text, "brand_reply": row.brand_reply,
                 "dm_redirect": bool(row.dm_redirect), "score": float(scores[i])}
            )
            if len(out) >= k:
                break
        return out
