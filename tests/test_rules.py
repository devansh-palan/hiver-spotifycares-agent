"""Unit tests for the deterministic parts: cleaning, escalation rules, triage normalisation, reply post-processing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import escalation  # noqa: E402
from src.agent.data import clean_text  # noqa: E402
from src.agent.drafting import postprocess  # noqa: E402
from src.agent.triage import normalise  # noqa: E402
from src.eval.metrics import escalation_report, intent_report  # noqa: E402


def tri(intent="technical_issue", **kw):
    base = {"intent": intent, "language": "en", "needs_account_action": False, "risk": "none", "confidence": 0.9, "summary": ""}
    base.update(kw)
    return normalise(base)


def test_clean_text_strips_mentions_links_and_signature():
    t = clean_text("@SpotifyCares @115712 my app crashes https://t.co/abc &amp; more /NJ", strip_signature=True)
    assert t == "my app crashes [LINK] & more"


def test_what_the_hack_does_not_trigger_security():
    d = escalation.decide("What the hack just happened to my Release Radar?", tri("feature_feedback"))
    assert not d.escalate


def test_hacked_triggers_security_regardless_of_intent():
    d = escalation.decide("my account got hacked and the email changed", tri("feature_feedback"))
    assert d.escalate and d.reason == "security_fraud" and d.source == "regex"


def test_login_problem_is_hard_trigger():
    d = escalation.decide("I forgot my email and password for premium", tri("billing_subscription"))
    assert d.escalate and d.reason == "account_access_needed"


def test_billing_public_question_is_auto():
    d = escalation.decide("when does the 3 months for 9.99 offer end?", tri("billing_subscription"))
    assert not d.escalate


def test_billing_problem_is_escalated_even_if_llm_says_no_account_action():
    d = escalation.decide("my family invite keeps failing", tri("billing_subscription", needs_account_action=False))
    assert d.escalate


def test_non_english_maps_to_other_and_escalates():
    t = tri("account_access", language="id")
    assert t["intent"] == "other"
    d = escalation.decide("misi min kok premium ga bisa", t)
    assert d.escalate and d.reason == "non_english"


def test_churn_phrase():
    d = escalation.decide("been down all morning, 2 seconds away from cancelling my subscription", tri("service_outage"))
    assert d.escalate and d.reason == "churn_or_anger_risk"


def test_private_data_posted_publicly():
    d = escalation.decide("Premium dropped to free. User name: ronancremin. fix it", tri("billing_subscription"))
    assert d.escalate and d.reason == "account_access_needed"


def test_postprocess_replaces_invented_urls_and_strips_initials():
    r = postprocess('"Hey there! Try the steps at https://support.spotify.com/xyz and let us know /NJ"')
    assert "[LINK]" in r and "http" not in r and not r.endswith("/NJ")


def test_metrics_shapes():
    ir = intent_report(["other", "technical_issue"], ["other", "other"])
    assert ir["accuracy"] == 0.5 and "confusion" in ir
    er = escalation_report([True, False, True], [True, True, False])
    assert er["tp"] == 1 and er["fp"] == 1 and er["fn"] == 1 and er["unsafe_auto_rate"] == round(1 / 3, 3)
