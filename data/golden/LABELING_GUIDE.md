# Golden set: how I sampled and labelled it

## What's in the set

One row per customer thread-opener to @SpotifyCares, i.e. the first message a customer sends before the brand has
replied, with the brand's real first reply attached for reference. The agent is evaluated on first-contact triage:
intent, escalation decision, reply draft.

## Sampling (scripts/sample_golden.py, seed 2024)

- Source: the 25,918 (opener -> first SpotifyCares reply) pairs I reconstructed from the Kaggle TWCS dump.
- 100 examples drawn uniformly at random. This is the "uniform" bucket and mirrors real traffic.
- 100 examples drawn from 8 loose regex keyword buckets, about 12-13 each, so that the rarer intents (outage, account
  access, billing) have enough support to measure per-class F1. The regexes are deliberately sloppy. Bucket membership
  was only a sampling aid and was never used as a label.
- A separate 40-example dev set, same procedure, disjoint ids. That's where all the prompt iteration happened. The
  200-example golden set was frozen before I wrote any prompt.
- Golden and dev ids are excluded from the retrieval index and from the silver-label training data.

## How I labelled

I read every example in full and filled in three fields by hand, following the rules below. Where a call was a
judgement call there's a `notes` field explaining it. The brand's real reply was visible while labelling; I used it as
a tie-breaker only when the customer text alone was ambiguous.

### intent (exactly one of 8, see configs/intents.yaml)

Tie-break rules, in priority order:

1. If the user can't get into the account or reports it being compromised -> `account_access`, even if money is also
   mentioned.
2. If money, a charge, a refund, or the status/eligibility of a paid plan is the problem -> `billing_subscription`.
3. "Is it down / is it just me / what's going on" with no personal detail -> `service_outage`. A persistent personal
   malfunction (crashes weekly, my downloads vanish) -> `technical_issue`.
4. A complaint about design (shuffle is bad, too many suggested tracks, ads are annoying) -> `feature_feedback`. A
   complaint that something stopped working -> `technical_issue`.
5. Catalogue requests and missing music -> `content_availability`. Country launch questions -> `general_question`.
6. Non-English, thanks-only, chit-chat, DM follow-ups, spam, or just too vague -> `other`. Praise about a specific
   feature ("iPhone X support looks amazing") is `feature_feedback`; a bare thank-you is `other`.
7. "When will you support the iPhone X / Apple Watch" (something that doesn't exist yet) -> `feature_feedback`. "Is
   there a way to do X today" -> `general_question`. "When do you launch in <country>" -> `general_question`.
8. Sign-up failures are `technical_issue`. There's no account to access yet.
9. A vague "I have an issue with my account" is `account_access`. A vague "I have a billing question" is
   `billing_subscription`.

### escalate (bool): should a human take this instead of an automated first reply?

True when resolving it properly needs any of: an account lookup or identity check, a decision about money, handling a
security incident, a judgement call on an angry / churning / legal / press situation, or a language the public agent
doesn't serve. In practice:

- `account_access` -> always True.
- `billing_subscription` -> True, except pure public-information questions (when does the offer end, is there a
  military discount).
- `technical_issue`, `service_outage`, `content_availability`, `feature_feedback`, `general_question` -> False unless
  a hard trigger below fires.
- `other` -> True for non-English and for "pick up my DM / I need customer service" messages (a human has to take the
  private thread). False for thanks / chit-chat / vague, where the safe automated move is a clarifying reply.
- A vague "I have a billing question" with no detail -> False. The right automated reply is "fire away".

Hard triggers (True regardless of intent): hacked / fraud / unauthorised activity; an explicit refund demand; a threat
to cancel or switch, or abuse aimed at the team (mild swearing at the ads doesn't count); a legal threat, even a
hyperbolic one ("I WILL SUE"); says they already contacted support and were ignored; explicitly asks for a human; posts
private data (email, username, card) publicly; a sensitive personal situation (bereavement, health) where a human should
check the tone.

### escalation_reason (one of)

`none`, `account_access_needed`, `billing_or_refund`, `security_fraud`, `churn_or_anger_risk`, `non_english`,
`repeat_contact_or_frustration`, `legal_or_pr_risk`, `sensitive_situation`, `needs_account_lookup` (for general
questions that turn out to depend on the user's account).

## Final label distribution (200)

feature_feedback 45, technical_issue 32, billing_subscription 31, account_access 30, content_availability 25,
general_question 20, other 13, service_outage 4. Escalate: 70 True / 130 False. service_outage only has four examples,
so its per-class F1 isn't meaningful. It's reported but not used in any headline claim.
