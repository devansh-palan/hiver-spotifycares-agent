# Golden set: sampling and labelling guide

## What is in the set
One row per **customer thread-opener** to @SpotifyCares (the first message a customer sends, before any
brand reply), with the brand's real first reply attached for reference. The agent is evaluated on
first-contact triage: intent, escalation decision, and reply draft.

## Sampling (scripts/sample_golden.py, seed 2024)
* Source: 25,918 reconstructed (opener -> first SpotifyCares reply) pairs from the Kaggle TWCS dump.
* 100 examples drawn uniformly at random ("uniform" bucket). This mirrors the production distribution.
* 100 examples drawn from 8 regex keyword buckets (~12-13 each) so that rarer intents (outage, account access,
  billing) have enough support to measure per-class F1. The regexes are deliberately loose; bucket membership
  is only a sampling aid and was **not** used as a label.
* A separate 40-example **dev set** (same procedure, disjoint ids) was used for prompt iteration.
  The 200-example golden set was frozen before any prompt was written against it.
* Golden and dev ids are excluded from the retrieval index and from the silver-label training data.

## Labelling procedure
Every example was read in full and labelled by hand with three fields, following the rules below.
Ambiguous cases carry a `notes` field explaining the call. The brand's real reply was visible during
labelling and used as a tie-breaker only when the customer text alone was ambiguous.

### intent (exactly one of 8; see configs/intents.yaml)
Tie-break rules, in priority order:
1. If the user cannot get into the account or reports compromise -> `account_access`, even if money is also mentioned.
2. If money, a charge, a refund, or a paid plan's status/eligibility is the problem -> `billing_subscription`.
3. "Is it down / is it just me / what's going on" with no personal detail -> `service_outage`.
   A persistent personal malfunction (crashes weekly, my downloads vanish) -> `technical_issue`.
4. A complaint about *design* (shuffle is bad, too many suggested tracks, ads are annoying) -> `feature_feedback`.
   A complaint that something *stopped working* -> `technical_issue`.
5. Catalogue requests / missing music -> `content_availability`. Country launch questions -> `general_question`.
6. Non-English, thanks-only, chit-chat, DM follow-ups, spam, or too vague -> `other`.
   Praise *about a feature* ("iPhone X support looks amazing") is `feature_feedback`; a bare thank-you is `other`.
7. "When will you support the iPhone X / Apple Watch" (an unreleased capability) -> `feature_feedback`;
   "is there a way to do X today" -> `general_question`; "when do you launch in <country>" -> `general_question`.
8. Sign-up failures are `technical_issue` (there is no account to access yet).
9. A vague "I have an issue with my account" is `account_access`; a vague "I have a billing question" is `billing_subscription`.

### escalate (bool): "should a human take this rather than an automated first reply?"
Escalate = True when a correct resolution needs any of: account lookup or identity verification, a money decision,
handling of a security incident, a judgement call on an angry / churning / legal / press situation, or a language
the public agent does not serve. Concretely:
* `account_access` -> always True.
* `billing_subscription` -> True, except pure public-information questions (offer end date, does a discount exist).
* `technical_issue`, `service_outage`, `content_availability`, `feature_feedback`, `general_question` -> False
  unless a hard trigger below fires.
* `other` -> True for non-English and for "pick up my DM / I need customer service" messages (a human must take the
  private thread); False for thanks / chit-chat / vague (the safe automated move is a clarifying reply).
* A vague "I have a billing question" with no detail -> False (the correct automated move is "fire away").

Hard triggers (True regardless of intent): hacked / fraud / unauthorised activity; explicit refund demand;
threat to cancel or switch service, or abuse directed at the team (mild swearing at ads is not abuse); legal threat,
even if hyperbolic ("I WILL SUE"); says they already contacted support and were ignored; explicitly asks for a human;
posts private data (email, username, card) publicly; a sensitive personal situation (bereavement, health) that needs
a human to review tone.

### escalation_reason (one of)
`none`, `account_access_needed`, `billing_or_refund`, `security_fraud`, `churn_or_anger_risk`, `non_english`,
`repeat_contact_or_frustration`, `legal_or_pr_risk`, `sensitive_situation`, `needs_account_lookup` (for general
questions that turn out to depend on the user's account).

## Final label distribution (200)
feature_feedback 45, technical_issue 32, billing_subscription 31, account_access 30, content_availability 25,
general_question 20, other 13, service_outage 4. Escalate: 70 True / 130 False. `service_outage` has only four
examples, so its per-class F1 is not meaningful; it is reported but not used in any headline claim.
