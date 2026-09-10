# SpotifyCares support agent: report

## 1. Problem framing

I picked SpotifyCares, and the reason was mostly practical. The biggest accounts in the dump (AmazonHelp, AppleSupport)
answer more than half their tweets with some version of "DM us", and you can't ground a reply in a DM you can't see.
SpotifyCares has 43,265 replies. When I rebuilt the threads I got 25,918 that start with a customer tweet and get a
SpotifyCares answer, almost all from October to December 2017. Roughly two thirds of those answers actually say
something: ask for the device and app version, explain that a missing album is a licensing thing, point at the
"Downloads unexpectedly removed" help page, tell people to go vote for a feature. So "reply the way the brand
historically resolved it" is something I can check rather than just claim.

### What "good" means for this brand

I ended up with four things, in order of importance.

First, the agent must not quietly handle something a human has to own. Hacked accounts, anything involving money,
people who can't log in, legal threats, people threatening to cancel, sensitive situations, tweets in other languages.
The metric for this is what I call the unsafe-auto rate: the share of all traffic where a human should have taken it and
the system didn't hand it over. I wanted this under 5%, and I wanted to be able to explain every miss.

Second, given the first, handle as much as possible automatically. A lot of SpotifyCares' public replies (outages,
catalogue requests, feature feedback, how-to questions) are already templated. A machine can send those.

Third, the drafts have to be sendable. Brand voice, the resolution the brand actually uses for that kind of problem, no
made-up links or prices or promises, and never asking for private details in public.

Fourth, every routing decision should be explainable in one line, so whoever runs the support queue can change a rule
instead of fiddling with a prompt.

Intent accuracy is in the results because it's the number everyone can compare, but it isn't the goal. It only matters
as far as it feeds the first three.

### What I didn't build

- Multi-turn handling. 72% of the threads I reconstructed have no customer follow-up at all. The first reply is where
  the routing decision happens, so the pipeline takes one message.
- Anything that actually touches an account. The agent decides a human is needed and drafts the public hand-off. It
  never pretends to have looked something up "backstage".
- A fine-tuned classifier as the main model. The TF-IDF baseline gets within 10 points of the LLM on intent, so a
  fine-tuned small encoder is clearly the next thing to try (section 7). I didn't get to it.
- Real URLs. Every link in the dump is a dead t.co shortener. Links are the literal token `[LINK]` all the way through.
- Customer names. SpotifyCares greets people by name; the dump anonymises authors. Drafts say "Hey there!" and I didn't
  penalise that when rating.

## 2. How it works

`tweet -> triage -> escalation policy -> retrieval -> draft`

Triage is one JSON call to `qwen2.5:3b` (running locally in Ollama) with 14 few-shot examples I wrote by hand. It
returns intent, language, whether the account needs to be touched, a risk flag, a confidence, and a one-line summary.

Escalation is plain code, not a model decision. Regex hard triggers go first (hacked, hijacked, sue, "thinking about
quitting", "talk to a human", can't log in, private data posted in the tweet). Then language. Then the risk flag from
triage. Then intent rules: account access always escalates, billing escalates unless it's a pure public question,
data-loss recovery escalates. Anything left is auto.

Retrieval is cosine similarity with MiniLM over 25.6k historical (tweet, reply) pairs, with every golden and dev tweet
removed from the index.

The draft prompt gets the tweet, the intent plus its playbook line from `configs/intents.yaml`, the handling
instruction, and the five nearest real exemplars. After generation I strip any URL and any agent initials the model
copied.

The longer version of why each of those choices was made is in `DECISIONS.md`.

## 3. Golden set and judge

The golden set is 200 customer thread-openers. 100 were sampled uniformly at random. The other 100 came from eight
loose keyword buckets (outage, account, billing and so on) so the rarer intents would have enough examples to measure.
Bucket membership was only used for sampling, never as a label.

I wrote the labelling guide first, then read every tweet and labelled intent, escalate (yes/no) and a reason code. The
guide grew tie-break rules as I went, things like "praise about a specific feature is feedback, a bare thank-you is
other" and "someone chasing an existing DM needs a human to pick that thread up". Final distribution: feature_feedback
45, technical_issue 32, billing 31, account_access 30, content 25, general_question 20, other 13, outage 4. 70 escalate,
130 auto.

There's a separate 40-example dev set with the same sampling. All three rounds of prompt and rule tweaking happened on
dev. The golden set was frozen before I wrote the first prompt.

The judge is `llama3.1:8b`, a different and bigger model family than the agent, on purpose. For each reply it sees the
gold intent and handling decision, the brand's actual reply to that very tweet (which the agent never sees), and three
real replies to similar tweets. It scores grounded, helpful and tone on 1-5, plus safe, overall, and a yes/no
"acceptable" meaning a support lead would send it without edits.

To check the judge, I rated 60 replies myself (20 per system) with the system name hidden and the order shuffled, using
the same rubric. Agreement numbers are in 4.3.

## 4. Results

### 4.1 Against the baselines (200 golden examples)

| system | intent acc (95% CI) | macro-F1 | esc. precision | esc. recall | automation | unsafe-auto | judge acceptable | judge overall | human acceptable (n=20) |
|---|---|---|---|---|---|---|---|---|---|
| agent | 0.76 (0.69-0.81) | 0.75 | 0.95 | 0.89 | 0.68 | 0.040 | 0.75 | 4.08 | 0.95 |
| simple: TF-IDF/LR + 1-NN reply + rules | 0.66 (0.59-0.72) | 0.67 | 0.83 | 0.83 | 0.65 | 0.060 | 0.60 | 3.97 | 0.50 |
| trivial: majority + canned + always escalate | 0.16 | 0.03 | 0.35 | 1.00 | 0.00 | 0.000 | 0.50 | 3.61 | 0.35 |
| trivial: never escalate | 0.16 | 0.03 | 0.00 | 0.00 | 1.00 | 0.350 | n/a | n/a | n/a |

The agent is 10 points ahead of the simple baseline on intent, and the confidence intervals only just touch. More
importantly it automates slightly more (0.68 vs 0.65) while making fewer unsafe calls (8 vs 12 out of 200). The two
trivial policies are the corners: always-escalate is safe and pointless, never-escalate handles everything and misses
all 70 cases that needed a person.

Per-intent F1 for the agent: technical 0.77, outage 0.86 (only 4 examples), account_access 0.81, billing 0.81, content
0.79, feedback 0.74, general_question 0.45, other 0.75. General questions are the weak spot, recall is 0.35. The model
reaches for a topical label instead ("when do you launch in Romania" went to billing, "how do I submit my music" went to
feature_feedback).

On the uniform-random half only (n=100, the closest thing I have to the real traffic mix): intent accuracy 0.74,
escalation precision 0.94, recall 0.87, automation 0.65, unsafe-auto 0.05. A bit worse than the full set, because the
random half has more vague and off-topic tweets.

Where the escalation decisions came from: auto 135, intent policy 31, regex 25, LLM risk flag 5, language 4. So only 5
of the 65 escalations rest on the LLM's judgement of risk. The rest are rules you can read. The reason code matched the
gold reason on 59 of 62 true positives; the three mismatches are billing_or_refund vs account_access_needed, which go to
the same queue anyway. Recall by gold reason: account_access_needed 32/35, billing_or_refund 10/11, security_fraud 8/9,
repeat_contact 6/6, non_english 3/3, legal 1/1, churn_or_anger 2/4, sensitive_situation 0/1.

End-to-end trust rate, which I define as intent right and escalation right and judge says acceptable, all on the same
tweet: 0.58 for the agent, 0.41 for the simple baseline. If I had to give a support lead one number it would be this
one. About six tweets in ten come out fully right. On the other four something needs a human's eyes.

### 4.2 Reply quality

Judge scores for the agent's 200 drafts: grounded 4.16, helpful 3.83, tone 4.88, safe on all 200. Overall scores were
3 twos, 48 threes, 78 fours, 71 fives. Helpfulness is the weak dimension. The judge's most common complaint is some
version of "generic, doesn't address the specific question", mostly on feedback and general-question replies.

The hard checks I run on every draft: invented URLs 0% (impossible by construction), asks for a password or card 0.5%
(one reply mentions a password reset problem, it doesn't ask for one), asks for private details outside DM 0%, signs
with initials 0%, over 280 characters 0%, greeting present 100%. 97% of escalation replies mention DM.

Judge acceptability by gold intent: outage 1.00 (n=4), billing 0.94, account_access 0.93, technical 0.84, content 0.64,
other 0.62, feedback 0.60, general_question 0.50. The hand-off replies are close to perfect because the brand's own
pattern is short and always the same ("DM us your username or email, we'll take a look backstage"). The public replies,
where the agent actually has to say something, are where it gets uneven.

### 4.3 Does the judge agree with a human?

On the 60 blind ratings: Spearman 0.52 between my 1-5 scores and the judge's, 80% within one point, Cohen's kappa 0.51
on the yes/no "acceptable" (77% raw agreement, 7 judge false accepts, 7 false rejects). We accepted the same overall
share (60%) and ranked the systems the same way. 0.51 is moderate. Good enough to rank systems and to find the failure
modes below, not good enough to trust the verdict on any single reply. Per system: trivial 0.79 (its problems are
obvious), simple 0.20, agent -0.08. Full numbers in `outputs/agreement.json`.

The disagreements have a pattern. The judge's false rejects are almost all the brand's own template lines: "we're
launching regularly in countries around the world", "we'll have it as soon as it's available to us", the
Indonesian-support line, the "Downloads unexpectedly removed" page. The judge calls them generic. They are generic, but
they're also exactly what SpotifyCares sends, so I marked them acceptable. The judge's false accepts go the other way:
replies that copy a precedent word for word, initials included ("...backstage /TC"), or that troubleshoot a design
question. The judge rewards fidelity to the precedent; I was rating whether the customer got an answer. The initials
problem is caught by the automated proxy anyway, which is why I report the proxies next to the judge rather than
instead of it. The agent's own kappa of -0.08 looks alarming, but it's just that we both accepted 17 to 19 of its 20
replies, so there's almost no disagreement to measure. The human column in 4.1 is the smaller sample but the more
trustworthy one.

## 5. Failure analysis: the top five

1. general_question gets used as a leftover bucket (13 of 20 missed). "Is there any way to add more than 5 family
   members?" went to billing. "When will Spotify be available in the UAE?" went to content. "I need to submit my music,
   what is required?" went to feature_feedback. My guess is that a 3B model keys on the topic noun, and my definition of
   the class ("answerable from public information") is a property of the answer, not the question, which is a hard thing
   to ask a classifier to see. I'd either fold country-launch and artist-tooling questions into the topical classes, or
   give the classifier the retrieved precedents so it can see that the brand answered publicly last time.

2. Account problems described as product complaints get auto-handled (3 of the 8 unsafe-auto cases). "Why am I
   listening to adverts I can't skip when I have premium?" is a Premium-not-active problem. "Why don't I have Hulu
   anymore?" is a lost bundle. "All my songs got replaced by rap music" is probably a takeover. In all three the model
   said needs_account_action false and no risk, because nothing in the text says "account". The few-shots only teach
   explicit signals. I'd add symptom-to-account-state examples, and a plain rule: a self-declared Premium user
   complaining about a Premium feature escalates.

3. Retrieval pulls neighbours that share words but not the problem. An ads complaint about the web player retrieved
   web-player bug threads and the draft told the customer to try an incognito window. "Please read DM" retrieved a
   country-launch reply and got a 1 out of 5 from me. Adding the playbook line to the prompt reduced this but didn't
   kill it. MiniLM similarity on 100-character tweets is mostly surface tokens. The fix is to filter exemplars by
   predicted intent, which needs intent labels on the whole index. I have 1k silver labels; the full 25k is about 8
   hours on this laptop.

4. Abuse, hyperbole and sensitive context don't trip the risk flag. "get your shit together" and "WTF? You high." got
   a polite clarifying reply (gold said churn/anger, send to a human). The tweet about baby ads after a miscarriage was
   classified other with no sensitive flag and got an empathetic but still automated acknowledgement. The risk field
   only fires on phrases similar to the few-shots. I think this wants either a small separate tone-and-sensitivity
   classifier or a checklist instead of a single enum.

5. Drafts that claim things the agent can't know. "We've just replied to your DM" sent to a first-contact billing
   complaint (I gave it 2/5, twice). "We've got your issue sorted out" on a trial-length question. Copied agent
   initials, before I added the strip. The model imitates the shape of the exemplars, and "we've replied to your DM" is
   36% of the corpus, even when the instruction says to ask for a DM. I'd drop the DM-redirect-only exemplars when the
   decision is escalate-and-ask-for-DM, and add a post-check that rejects claims of past actions.

There's a sixth, minor one: the language detector flagged "Yo, how does a guy register for ?" as non-English (one
false escalation), and the "1:" prefix from split brand replies leaked into one draft.

## 6. What is misleading about my headline number

- 0.76 intent accuracy is measured against labels I wrote, using a guide I also wrote. Another annotator wouldn't agree
  with me every time. On the ambiguous pairs (general_question vs feature_feedback, technical vs billing when a Premium
  user describes a symptom) my own tie-breaks shifted while I was labelling. I didn't measure inter-annotator
  agreement, so some unknown chunk of the 24% error is label noise, and some of the 76% is the model picking up my
  conventions through the few-shots.
- Half the golden set is keyword-enriched. That over-represents account and billing tweets, which is where the agent
  is strongest. On the uniform half the numbers drop to 0.74 accuracy and 0.05 unsafe-auto, and that half is the honest
  production estimate. service_outage has four examples; its F1 of 0.86 is one tweet away from 0.67.
- Escalation recall of 0.89 hides that the misses are the expensive ones. The eight unsafe-auto cases include a likely
  account takeover and a bereaved customer. One missed takeover costs more than dozens of over-escalated feature
  requests. Read recall per reason: 8/9 on security_fraud (the miss is the one with no explicit wording), 0/1 on
  sensitive_situation, 2/4 on churn_or_anger.
- The simple baseline was trained on labels produced by the agent's own prompt. So the 10-point gap is the value of
  running the LLM at inference time, not LLM vs an independently trained classifier.
- I am the human rater, and I built the system. Blinding was also only partial: the trivial baseline's canned reply is
  easy to recognise. The agent's human acceptable rate (0.95 on 20) is above the judge's rate on all 200, which more
  likely means the 20 were a kind sample than that the agent is better than the judge thinks.
- The confidence score is useless. The model says 0.85 or higher on 198 of 200 tweets, and accuracy in that bin is
  0.76. Any plan to route low-confidence cases to a human is dead on arrival with this model.
- 75% judge-acceptable is only 25 points above a canned reply. The trivial baseline's one sentence ("Sorry to hear
  that. Can you DM us your username or email? We'll take a look backstage.") gets judged acceptable half the time,
  because it is the right reply for the 35% that need escalation, and the judge lets it slide on a few more. The agent's
  real margin is in the 65% it auto-handles, where it's acceptable 60 to 84% of the time depending on intent and the
  canned reply almost never is. The end-to-end trust column (0.58 vs 0.41 vs 0.01) is the one that actually separates
  the systems. The judge column on its own flatters the trivial baseline.
- Latency and cost are laptop numbers. About 2.5 seconds per tweet on a 4 GB GPU with a 3B model, about 8 seconds per
  reply for the judge. The repo reproduces in seconds because the outputs are cached, which is convenient but says
  nothing about speed.
- I looked at the golden set once mid-run. After the first 75 predictions I noticed copied initials and one
  retrieval-drift draft, and changed the drafting post-processing and prompt (checked on dev) before re-running. That
  only touched drafting, no labels or triage rules changed, but it is still information leaking from the test set into
  the design and I should say so.

## 7. With one more week

1. Measure the label noise. Get a second person to label 100 golden rows, report kappa per field, and restate every
   headline number with that as the ceiling.
2. Intent-filtered retrieval. Silver-label the full 25.6k index (8 laptop-hours, or minutes with an API), retrieve
   within the predicted intent, re-judge. Failure modes 1 and 3 should move the most.
3. Fine-tune a MiniLM classifier on 5k silver plus the 200 gold (cross-validated) and use it for intent, keeping the LLM
   only for the draft. I'd expect better calibration, 50 ms latency, and a confidence score that means something.
4. Turn the risk field into a checklist, and hand-label a small (~100) sensitive/abusive set to test it. Aimed at
   failure mode 4.
5. Reply post-checks as unit tests: reject drafts that claim past actions, that ask for diagnostics when the decision is
   escalate, or that contain no action at all. Then re-measure the judge's kappa.
6. Run everything on a fresh 100-tweet slice from a different month, with a human in the loop, to see whether the
   numbers survive a distribution shift. The dump is 99.9% Q4 2017.
