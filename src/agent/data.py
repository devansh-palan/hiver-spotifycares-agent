"""Thread reconstruction and text cleaning for one brand from the TWCS dataset.

The Kaggle file is a flat table of tweets linked by `in_response_to_tweet_id`
(parent) and `response_tweet_id` (comma-separated children). We rebuild
(customer opener -> first brand reply) pairs, which is the unit the agent is
trained/evaluated on: first-contact triage.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

BRAND = "SpotifyCares"
RAW_PATH = Path("data/raw/twcs/twcs.csv")
PROCESSED_PATH = Path("data/processed/spotify_pairs.parquet")

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"(^|\s)@\w+")
# SpotifyCares agents sign off with a slash + initials, e.g. "... /NJ" or "/MT".
SIGNATURE_RE = re.compile(r"\s*/[A-Za-z]{1,3}\s*$")
WS_RE = re.compile(r"\s+")


def clean_text(text: str, *, strip_signature: bool = False) -> str:
    """Normalise a tweet: unescape HTML, drop @mentions and URLs, fix mojibake."""
    if not isinstance(text, str):
        return ""
    t = html.unescape(text)
    t = t.replace("�", "'")  # the CSV has broken curly quotes -> U+FFFD
    t = URL_RE.sub(" [LINK] ", t)
    t = MENTION_RE.sub(" ", t)
    if strip_signature:
        t = SIGNATURE_RE.sub("", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def _split_ids(value) -> list[int]:
    if not isinstance(value, str) or not value:
        return []
    out = []
    for part in value.split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out


@dataclass
class ThreadIndex:
    tweets: pd.DataFrame  # indexed by tweet_id

    def text(self, tid: int) -> str:
        return self.tweets.at[tid, "text"] if tid in self.tweets.index else ""


def build_pairs(raw_path: Path = RAW_PATH, brand: str = BRAND, max_context_turns: int = 4) -> pd.DataFrame:
    """Return one row per (customer opener, first brand reply) for `brand`.

    Columns: opener_id, reply_id, created_at, customer_text, brand_reply,
    customer_raw, brand_raw, thread_len (number of tweets in the thread
    that we could follow), followups (customer messages after the first reply).
    """
    df = pd.read_csv(raw_path, dtype={"response_tweet_id": "string"})
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].astype("Int64")
    tweets = df.set_index("tweet_id")

    brand_rows = df[df.author_id == brand]
    # first-level pairs: brand reply whose parent is a customer tweet
    parent_ids = brand_rows["in_response_to_tweet_id"].dropna().astype(int)
    parents = tweets.reindex(parent_ids.unique())
    parents = parents[(parents.inbound == True)]  # noqa: E712
    # openers only: the customer tweet starts the thread
    openers = parents[parents["in_response_to_tweet_id"].isna()]

    rows = []
    for opener_id, op in openers.iterrows():
        children = _split_ids(op["response_tweet_id"])
        brand_children = [c for c in children if c in tweets.index and tweets.at[c, "author_id"] == brand]
        if not brand_children:
            continue
        # earliest brand reply by tweet id order is unreliable; use created_at
        brand_children.sort(key=lambda c: pd.to_datetime(tweets.at[c, "created_at"]))
        reply_id = brand_children[0]
        # follow the thread a little for context / follow-up statistics
        followups = []
        cur = reply_id
        depth = 0
        while depth < max_context_turns:
            kids = _split_ids(tweets.at[cur, "response_tweet_id"]) if cur in tweets.index else []
            kids = [k for k in kids if k in tweets.index]
            if not kids:
                break
            cur = kids[0]
            if tweets.at[cur, "inbound"]:
                followups.append(clean_text(tweets.at[cur, "text"]))
            depth += 1
        rows.append(
            {
                "opener_id": int(opener_id),
                "reply_id": int(reply_id),
                "created_at": op["created_at"],
                "customer_raw": op["text"],
                "brand_raw": tweets.at[reply_id, "text"],
                "customer_text": clean_text(op["text"]),
                "brand_reply": clean_text(tweets.at[reply_id, "text"], strip_signature=True),
                "thread_len": 2 + depth,
                "followups": " ||| ".join(followups),
            }
        )
    out = pd.DataFrame(rows)
    out["created_at"] = pd.to_datetime(out["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce")
    out = out.sort_values("created_at").reset_index(drop=True)
    # drop empty / trivially short customer messages (pure mentions, emoji only)
    out = out[out.customer_text.str.len() >= 8].reset_index(drop=True)
    return out


def load_pairs(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python scripts/build_dataset.py` first")
    return pd.read_parquet(path)
