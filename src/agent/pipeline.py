"""SupportAgent: triage -> escalation policy -> grounded reply, for one incoming tweet."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from . import drafting, escalation, triage as triage_mod
from .llm import LLM, get_llm
from .retrieval import Retriever


@dataclass
class AgentOutput:
    customer_text: str
    intent: str
    intent_confidence: float
    escalate: bool
    escalation_reason: str
    escalation_detail: str
    escalation_source: str
    reply: str
    triage: dict
    exemplar_ids: list = field(default_factory=list)
    exemplar_scores: list = field(default_factory=list)
    latency_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class SupportAgent:
    def __init__(self, llm: LLM | None = None, retriever: Retriever | None = None, k: int = 5):
        self.llm = llm or get_llm("agent")
        self.retriever = retriever or Retriever.load()
        self.k = k

    def handle(self, text: str) -> AgentOutput:
        t0 = time.time()
        tri = triage_mod.triage(self.llm, text)
        dec = escalation.decide(text, tri)
        exemplars = self.retriever.search(text, k=self.k)
        reply = drafting.draft(self.llm, text, tri, dec.escalate, dec.reason, dec.detail, exemplars)
        return AgentOutput(
            customer_text=text,
            intent=tri["intent"],
            intent_confidence=tri["confidence"],
            escalate=dec.escalate,
            escalation_reason=dec.reason,
            escalation_detail=dec.detail,
            escalation_source=dec.source,
            reply=reply,
            triage=tri,
            exemplar_ids=[e["opener_id"] for e in exemplars],
            exemplar_scores=[round(e["score"], 3) for e in exemplars],
            latency_s=round(time.time() - t0, 2),
        )
