"""Provider-agnostic chat wrapper with an on-disk cache.

Every call is cached by a hash of (provider, model, prompt, decoding params) under
outputs/llm_cache/. The cache is committed to the repo, so `python scripts/run_pipeline.py`
reproduces the headline numbers without a GPU or an API key. Delete the cache directory
(or set LLM_CACHE=0) to force live calls.

Providers:
  ollama  (default)  local models, e.g. qwen2.5:3b (agent) and llama3.1:8b (judge)
  openai             set OPENAI_API_KEY; models e.g. gpt-4o-mini / gpt-4o
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULTS = {
    "ollama": {"agent": "qwen2.5:3b", "judge": "llama3.1:8b"},
    "openai": {"agent": "gpt-4o-mini", "judge": "gpt-4o"},
}
CACHE_DIR = Path(os.environ.get("LLM_CACHE_DIR", "outputs/llm_cache"))


def parse_json_loose(text: str) -> dict:
    """Parse JSON from a model response, tolerating code fences and leading prose."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"_parse_error": True, "_raw": text[:500]}


@dataclass
class LLM:
    role: str = "agent"
    provider: str | None = None
    model: str | None = None
    temperature: float = 0.0
    num_ctx: int = 4096
    seed: int = 0
    use_cache: bool = True
    stats: dict = field(default_factory=lambda: {"calls": 0, "cache_hits": 0, "latency_s": 0.0})

    def __post_init__(self):
        self.provider = self.provider or os.environ.get("LLM_PROVIDER", "ollama")
        env_key = "AGENT_MODEL" if self.role == "agent" else "JUDGE_MODEL"
        self.model = self.model or os.environ.get(env_key, DEFAULTS[self.provider][self.role])
        if os.environ.get("LLM_CACHE", "1") == "0":
            self.use_cache = False
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ cache
    def _key(self, system: str, user: str, json_mode: bool, max_tokens: int) -> str:
        blob = json.dumps(
            [self.provider, self.model, system, user, json_mode, max_tokens, self.temperature, self.seed],
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]

    def _cache_path(self, key: str) -> Path:
        return CACHE_DIR / f"{key}.json"

    # ------------------------------------------------------------------ calls
    def chat(self, system: str, user: str, *, json_mode: bool = False, max_tokens: int = 300, tag: str = "") -> str:
        key = self._key(system, user, json_mode, max_tokens)
        path = self._cache_path(key)
        if self.use_cache and path.exists():
            self.stats["cache_hits"] += 1
            return json.load(open(path, encoding="utf-8"))["response"]
        t0 = time.time()
        text = self._call(system, user, json_mode, max_tokens)
        dt = time.time() - t0
        self.stats["calls"] += 1
        self.stats["latency_s"] += dt
        json.dump(
            {"provider": self.provider, "model": self.model, "tag": tag, "latency_s": round(dt, 3),
             "system": system, "user": user, "json_mode": json_mode, "response": text},
            open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
        )
        return text

    def chat_json(self, system: str, user: str, *, max_tokens: int = 300, tag: str = "") -> dict:
        return parse_json_loose(self.chat(system, user, json_mode=True, max_tokens=max_tokens, tag=tag))

    def _call(self, system: str, user: str, json_mode: bool, max_tokens: int) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_err = None
        for attempt in range(3):
            try:
                if self.provider == "ollama":
                    import ollama

                    r = ollama.chat(
                        model=self.model,
                        messages=messages,
                        format="json" if json_mode else "",
                        options={"temperature": self.temperature, "num_predict": max_tokens,
                                 "num_ctx": self.num_ctx, "seed": self.seed},
                        keep_alive="20m",
                    )
                    return r["message"]["content"]
                if self.provider == "openai":
                    from openai import OpenAI

                    client = OpenAI()
                    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
                    r = client.chat.completions.create(
                        model=self.model, messages=messages, temperature=self.temperature,
                        max_tokens=max_tokens, seed=self.seed, **kwargs,
                    )
                    return r.choices[0].message.content or ""
                raise ValueError(f"unknown provider {self.provider}")
            except Exception as e:  # noqa: BLE001 - retry any transport error
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after retries: {last_err}")


def get_llm(role: str = "agent", **kw) -> LLM:
    return LLM(role=role, **kw)
