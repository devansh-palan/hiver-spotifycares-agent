# Report: an AI first-response agent for @SpotifyCares

## 1. Problem framing

**The brand.** SpotifyCares is Spotify's support handle on Twitter. In the Kaggle dump it has 43,265 replies, from
which I reconstructed 25,918 threads that start with a customer tweet and get a SpotifyCares reply (99.9% from
Oct-Dec 2017). Unlike AppleSupport (52% of replies are "DM us"), SpotifyCares gives a public, substantive first reply
about two thirds of the time: diagnostic questions ("what device, OS and app version?"), the standard licensing line for
missing music, the "Downloads unexpectedly removed" help page, "vote for this idea here". That makes "grounded in how the
brand historically resolved it" testable rather than aspirational.

**What "good" means here.** A first-contact triage system for a brand like this is good if:

1. It never quietly auto-handles something a human must own: account takeovers, money, login failures, legal or
   churn threats, sensitive situations, other languages. The metric is the **unsafe-auto rate** (should-escalate but
   auto-handled, as a share of all traffic). The bar I set is under 5%, and every miss must be explainable.
2. Subject to (1), it auto-handles as much as possible (**automation rate**), because SpotifyCares' public replies to
   outages, catalogue requests, feature feedback and how-to questions are already templated and a machine can send them.
3. Its drafts are ones a support lead would send without edits: the brand's voice, the brand's actual resolution for that
   problem, no invented links, prices, dates or promises, and no request for private details outside DM.
4. Every decision is explainable in one line, so a lead can tune one rule instead of re-prompting.

Intent accuracy matters only insofar as it drives (1)-(3); it is reported because it is the comparable number, not
because it is the goal.

**What I chose not to build.**

* Multi-turn conversation handling. 72% of reconstructed threads have no customer follow-up at all; the first reply is
  where routing value sits. The pipeline accepts a single message; carrying DM state would need a different dataset.
* Actual DM handling or account actions. The agent decides that a human is needed and drafts the public hand-off; it
  never pretends to look "backstage".
* A fine-tuned classifier as the main system. A distilled TF-IDF model is the *simple baseline*, and it gets within
  10 points of the LLM on intent; a fine-tuned small encoder is the obvious next step (section 7), not this submission.
* Real URLs. Every link in the dump is a dead t.co shortener, so links are the literal token `[LINK]` end to end.
* Customer names. The brand greets people by their Twitter display name; the dump anonymises authors, so drafts use
  "Hey there!" and the human-rating rubric does not penalise that.

## 2. System in one paragraph

`tweet -> triage -> escalation policy -> retrieval -> draft`. Triage is one JSON call to `qwen2.5:3b` with 14
hand-written few-shots, returning intent, language, `needs_account_action`, a `risk` flag, confidence and a summary.
The escalation policy is deterministic: regex hard triggers (hacked / hijacked / sue / "thinking about quitting" /
"talk to a human" / login failures / private data posted), then language, then the LLM risk flag, then intent policy
(account access always; billing unless it is a pure public question; data-loss recovery), else auto. Retrieval is
MiniLM cosine over 25.6k historical (tweet, reply) pairs with golden and dev ids removed. The draft prompt contains the
tweet, the intent and its playbook line from `configs/intents.yaml`, the handling instruction, and the five nearest real
exemplars; post-processing removes any URL and any copied agent initials. Full decision log: `DECISIONS.md`.

## 3. Golden set and judge

**Golden set** (`data/golden/golden_set.jsonl`, 200 rows). 100 uniform-random thread openers plus 100 drawn from eight
loose keyword buckets so rare intents have support (bucket membership was never used as a label). Each row was read in
full and labelled with intent, `escalate` and a reason code following `LABELING_GUIDE.md`, which was written first and
amended with tie-break rules as they came up (e.g. "praise about a feature is feedback, a bare thank-you is other";
"a DM follow-up needs a human to pick up the private thread"). Distribution: feature_feedback 45, technical_issue 32,
billing 31, account_access 30, content 25, general_question 20, other 13, outage 4; 70 escalate / 130 auto.
A disjoint 40-example **dev set** absorbed all three rounds of prompt and rule iteration; the golden set was frozen
before the first prompt was written.

**Judge.** `llama3.1:8b`, a different and larger model family than the agent, grades one reply at a time given the gold
intent and handling decision, the brand's *real* reply to that tweet (which the agent never sees), and three real
precedents. It returns grounded / helpful / tone (1-5), safe, overall and a binary `acceptable` ("a support lead would
send it without edits"). **Human agreement:** 60 replies (20 per system) were shuffled with the system hidden and rated
by hand on the same rubric. Agreement numbers are in section 4.3.

## 4. Results

### 4.1 Versus baselines (200 golden examples)

| system | intent acc (95% CI) | macro-F1 | esc. precision | esc. recall | automation | unsafe-auto | judge acceptable | judge overall | human acceptable (n=20) |
|---|---|---|---|---|---|---|---|---|---|
| **agent** | **0.76** (0.69-0.81) | **0.75** | **0.95** | **0.89** | 0.68 | **0.040** | **0.75** | **4.08** | **0.95** |
| simple: TF-IDF/LR + 1-NN reply + rules | 0.66 (0.59-0.72) | 0.67 | 0.83 | 0.83 | 0.65 | 0.060 | 0.60 | 3.97 | 0.50 |
| trivial: majority + canned + always escalate | 0.16 | 0.03 | 0.35 | 1.00 | 0.00 | 0.000 | 0.50 | 3.61 | 0.35 |
| trivial: never escalate | 0.16 | 0.03 | 0.00 | 0.00 | 1.00 | 0.350 | n/a | n/a | n/a |

The agent beats the simple baseline by 10 points of intent accuracy (the CIs barely overlap), and at a *higher*
automation rate (0.68 vs 0.65) it makes fewer unsafe calls (8 vs 12 of 200). The trivial policies bracket the
problem: "always escalate" is safe and useless, "never" automates everything and misses all 70 human-needed cases.

**Per-intent F1 (agent):** technical 0.77, outage 0.86 (n=4), account_access 0.81, billing 0.81, content 0.79,
feedback 0.74, **general_question 0.45**, other 0.75. General questions are the weak class (recall 0.35): the model
prefers a topical label ("when do you launch in Romania" -> billing; "how do I submit music" -> feature_feedback).

**Production-distribution view (uniform-random half, n=100):** intent accuracy 0.74, escalation precision 0.94,
recall 0.87, automation 0.65, unsafe-auto 0.05. Slightly worse than the enriched set because the uniform half has more
vague and off-topic tweets.

**Where escalation decisions come from (agent, 200):** auto 135, intent policy 31, regex 25, LLM risk 5, language 4.
Only 5 of 65 escalations depend on the LLM's risk judgement; the rest are auditable rules. Reason codes match the gold
reason on 95% of true positives (59/62); the mismatches are `billing_or_refund` vs `account_access_needed`, which
route to the same queue in practice. Recall by gold reason: account_access_needed 32/35, billing_or_refund 10/11,
security_fraud 8/9, repeat_contact 6/6, non_english 3/3, legal 1/1, churn_or_anger 2/4, sensitive_situation 0/1.

**End-to-end trust rate** (intent right AND escalation right AND judge-acceptable): **0.58** for the agent vs
**0.41** for the simple baseline. This is the number I would quote to a support lead: on roughly six tweets in ten
the agent gets everything right at once; on the rest at least one of the three outputs needs a human's eyes.

### 4.2 Reply quality

Judge scores (1-5, agent, n=200): grounded 4.16, helpful 3.83, tone 4.88; safe on 100%; distribution of overall
2/3/4/5 = 3/48/78/71. Helpfulness is the weak dimension: the judge's most frequent critique is "generic, does not
address the specific question", concentrated in feedback and general-question replies.
Hard proxies on the agent's 200 drafts: invented URLs 0% (by construction), asks for password or card 0.5% (one reply
*acknowledges* a password-reset problem, it does not request one), asks for private details outside DM 0%, signs with
initials 0%, over 280 characters 0%, greeting present 100%; 97% of escalation replies mention DM.
Judge acceptability by gold intent (agent): outage 1.00 (n=4), billing 0.94, account_access 0.93, technical 0.84,
content 0.64, other 0.62, feedback 0.60, general_question 0.50. The escalation hand-off replies are near-perfect
because the brand's own pattern ("DM us your username or email, we'll take a look backstage") is short and consistent;
the public-resolution replies are where quality varies.

### 4.3 Does the judge agree with a human?

On the 60 blind ratings (20 per system): Spearman 0.52 between human and judge 1-5 scores, 80% of scores within one
point, Cohen's kappa **0.51** on the binary "acceptable" (raw agreement 77%; 7 judge false accepts, 7 false rejects).
Human and judge accept the same overall share (60%), and both rank the systems the same way (agent > simple > trivial).
Kappa 0.51 is "moderate": good enough to rank systems and to spot the failure modes below, not good enough to trust a
single reply's verdict. Per-system: trivial kappa 0.79 (its defects are obvious), simple 0.20, agent -0.08 (see below).
Full numbers: `outputs/agreement.json`.
Where they disagree, the pattern is consistent and instructive. **Judge false rejects** are the brand's own templated
lines: "we're launching regularly in countries around the world", "we'll have it as soon as it's available to us", the
Indonesian-support line, the "Downloads unexpectedly removed" page. The judge calls them "generic" even though they are
exactly what SpotifyCares sends, because its rubric rewards tweet-specific wording. **Judge false accepts** are replies
that copy a precedent perfectly, initials included ("... backstage /TC"), or that troubleshoot a design question; the
judge scores form-faithfulness, the human scores whether the customer got an answer. The initials defect is caught by
the automated proxy (`signs_initials`), which is why the proxies and the judge are reported side by side. Within the
agent's own 20 ratings agreement is poor (kappa -0.08) simply because both raters accept 17-19 of 20; the kappa is only
meaningful across systems. The human column in 4.1 is the smaller but more trustworthy number.

## 5. Failure analysis: top five failure modes

1. **`general_question` is treated as a residual class (13 of 20 missed).** "Is there any way to add more than 5 family
   members?" -> billing; "when will Spotify be available in the UAE?" -> content; "I need to submit my music, what is
   required?" -> feature_feedback. Hypothesis: a 3B model keys on the topic noun, and my definition of the class is
   "answerable from public info", a property of the *answer*, not the *question*. Fix: either fold country-launch and
   artist-tooling into topical classes, or give the classifier the retrieved precedents so it can see that the brand
   answered publicly.

2. **Implicit account problems phrased as product complaints slip to auto (3 of 8 unsafe-auto cases).** "Why am I
   listening to adverts I can't skip when I have premium?" (Premium not active), "Why don't I have Hulu anymore?"
   (lost bundle), "all my songs got replaced by rap music" (likely takeover). The model returns `needs_account_action:
   false` and no risk flag because nothing in the text says "account". Hypothesis: the few-shots teach explicit signals
   only. Fix: add "symptom -> account state" examples and a rule that Premium-feature complaints from self-declared
   Premium users escalate.

3. **Retrieval drift when the nearest neighbours share words but not the problem.** An ads complaint on the web player
   pulled web-player bug threads and the draft became "try an incognito window"; "Please read DM" pulled a country-launch
   reply (human rating 1/5). The playbook line in the prompt reduced but did not remove this. Hypothesis: MiniLM
   similarity on 100-character tweets is dominated by surface tokens. Fix: filter exemplars by predicted intent (needs
   intent labels on the index; 1k silver labels exist, 25k would take ~8 laptop-hours), or re-rank with the triage summary.

4. **Unflagged abuse, hyperbole and sensitive context.** "get your shit together", "WTF? You high." were auto-handled
   with a clarifying reply (gold: churn/anger, human review). The miscarriage-ads tweet was classified `other` with no
   `sensitive` flag and got an empathetic but automated acknowledgement. Hypothesis: the risk field is only activated by
   explicit phrases the few-shots contain. Fix: a separate, tiny "tone and sensitivity" classifier, or make the risk field
   a checklist rather than a single enum.

5. **Drafts that assert state the agent cannot know.** "We've just replied to your DM" to a first-contact billing
   complaint (human 2/5, twice), "We've got your issue sorted out" on a trial-length question, and copied agent initials
   before the fix. Hypothesis: the model imitates the exemplars' *form* ("we've replied to your DM" is 36% of the
   corpus) even when the instruction says to *ask* for a DM. Fix: exclude DM-redirect-only exemplars when the decision is
   escalate-and-request-DM, and a post-check that rejects claims of past actions.

A sixth, minor mode: the language detector flagged "Yo, how does a guy register for ?" as non-English (one false
escalation), and the "1:" thread-numbering prefix from split brand replies leaked into one draft.

## 6. What is misleading about my headline number

* **0.76 intent accuracy is against labels I wrote myself, with a guide I also wrote.** A second annotator would not
  agree with me 100% of the time; on the ambiguous pairs (general_question vs feature_feedback, technical vs billing when
  a Premium user reports a symptom) my own tie-break rules changed while labelling. Inter-annotator agreement was not
  measured, so part of the 24% error is label noise of unknown size, and part of the 76% is the model learning my
  conventions through the few-shots.
* **The golden set is half enriched.** Keyword buckets over-represent account and billing tweets, where the agent is
  strong. On the uniform half the numbers drop (0.74 / unsafe 0.05), and that half is the honest production estimate.
  `service_outage` has four examples; its F1 of 0.86 is one tweet away from 0.67.
* **Escalation recall of 0.89 hides that the misses are the expensive ones.** The eight unsafe-auto cases include a
  probable account takeover and a bereaved customer. A single missed takeover costs more than dozens of over-escalated
  feature requests. Recall should be read per reason: 8/9 on `security_fraud` (the miss is the one takeover with no
  explicit wording), 0/1 on `sensitive_situation`, 2/4 on `churn_or_anger_risk`.
* **The simple baseline is distilled from the agent.** Its training labels came from the same prompt, so the 10-point
  gap measures the value of running the LLM at inference time, not the LLM versus an independently trained classifier.
* **The judge and the human are not independent of the system's author**, and blinding was partial: the trivial
  baseline's canned reply is recognisable. The human acceptable rate for the agent (0.95, n=20) is also above the
  judge's full-set rate, which suggests the 20-example sample was lucky, not that the agent is better than the judge says.
* **Confidence is uninformative.** The model reports >= 0.85 on 198 of 200 tweets; accuracy in that bin is 0.76. Any
  "route low-confidence to a human" idea is dead on arrival with this model.
* **"75% judge-acceptable" is only 25 points above a canned reply.** The trivial baseline's single sentence ("Sorry to
  hear that. Can you DM us your username or email? We'll take a look backstage.") is judged acceptable on 50% of tweets,
  because it is the *correct* reply for the 35% that need escalation and the judge forgives it on some others. The
  agent's real margin over "always ask for a DM" is in the auto-handled 65%, where it is acceptable 60-84% of the time
  depending on intent and the canned reply almost never is. The end-to-end trust rate (0.58 vs 0.41 vs 0.01) is the
  column that separates the systems; the judge column alone flatters the trivial baseline.
* **Latency and cost are laptop numbers.** ~2.5 s per tweet on a 4 GB GPU with a 3B model; the judge takes ~8 s per reply.
  Cached outputs make the repo reproduce in seconds, which is convenient, not evidence of speed.
* **The golden set was looked at once mid-run.** After the first 75 predictions I noticed copied initials and one
  retrieval-drift draft and changed the drafting post-processing and prompt (dev-validated) before re-running. That is a
  drafting-only change and no label or triage rule was touched, but it is a leak of information from test to design.

## 7. With one more week

1. **Measure label noise**: have a second person label 100 golden rows; report Cohen's kappa per field and re-state all
   headline numbers with that ceiling.
2. **Intent-filtered retrieval**: silver-label the full 25.6k index (8 laptop-hours, or minutes with an API) and
   retrieve within the predicted intent, then re-judge; failure modes 1 and 3 should move most.
3. **A fine-tuned MiniLM classifier** on 5k silver + 200 gold (cross-validated) as the intent model, keeping the LLM only
   for the draft. Expected: better calibration, 50 ms latency, and a real confidence signal for routing.
4. **Risk checklist instead of an enum**, plus a small hand-labelled "sensitive / abusive" set (~100 tweets), targeting
   failure mode 4.
5. **Reply post-checks as unit tests**: reject drafts that claim past actions ("we've replied to your DM"), that ask for
   diagnostics when the decision is escalate, or that contain no action at all; measure the judge's kappa again.
6. **Human-in-the-loop evaluation on a fresh 100-tweet slice** from a different month to check that the numbers survive
   distribution shift (the dump is 99.9% Q4 2017).
