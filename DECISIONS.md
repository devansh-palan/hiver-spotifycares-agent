# Decision log

The choices that weren't obvious, roughly in the order I made them, and why.

1. SpotifyCares rather than AppleSupport or AmazonHelp. The biggest brands answer with "DM us" more than half the time
   (AppleSupport is at 52%), which leaves nothing to ground a reply in. SpotifyCares has 43k replies, only 31% of them
   are DM redirects, and the rest are real troubleshooting, catalogue and policy answers. That makes "grounded in how
   the brand resolved it" something I can actually test.

2. The unit of work is first-contact triage (customer opener, first brand reply), not whole threads. 72% of the
   reconstructed threads have no customer follow-up at all. The first reply is where the routing value is. Labelling
   multi-turn state would have roughly halved the golden set for the same effort. Follow-ups are in the "not built" list.

3. Eight intents, decided by reading around 300 openers before writing any code. Five classes merged things that get
   handled differently (an outage vs. a personal bug). Twelve or more gave me classes with under five golden examples.
   I kept service_outage even though it's rare because its resolution is unique and fully automatable.

4. Non-English messages get intent "other", not their topic. SpotifyCares routes by language before topic ("we have
   Indonesian support via email"), so language is the decision that matters. The model still predicts the topic; a
   post-processing rule maps it to "other" so the failure analysis can see both.

5. Escalation is a deterministic policy over the LLM's fields plus regex, not an LLM yes/no. Every decision carries a
   reason code and a source (regex, llm_risk, language, intent_policy, auto), so a support lead can adjust one rule
   without touching a prompt. The 3B model was inconsistent about "does this need an account lookup", so I made billing
   escalate by default with a narrow public-question exception. That took dev recall from 0.80 to 1.00 without any prompt
   change.

6. The regex hard triggers are deliberately narrow. "hack" on its own fires on "what the hack happened to my Release
   Radar", which is a real golden example. The triggers match hacked / hijacked / unauthorised / "someone ... using my
   account", explicit cancel threats, legal words, and "talk to a human" phrasings. False positives are cheap for a human
   but they eat the automation rate, so I went for precision.

7. A 40-example dev set, disjoint from golden, for all prompt iteration. Three rounds happened on dev (tie-break
   guidance, two extra few-shots, the billing rule). The golden set was frozen before the first prompt was written. I
   did look at it once, after the first 75 predictions, which led to a drafting-only change (item 9). That's disclosed in
   the report's "misleading" section.

8. The retrieval index excludes every golden and dev opener. Otherwise the agent could pull the brand's real reply to
   the exact tweet it's being graded on. I also dropped exact-duplicate customer texts so the top-5 neighbours aren't
   five copies of the same tweet.

9. The reply prompt includes a per-intent playbook line from configs/intents.yaml and tells the model to ignore
   off-intent neighbours. Dense retrieval matches surface words: an ads complaint on the web player pulled web-player bug
   threads and the draft turned into browser troubleshooting. The playbook line is a cheap guardrail, and it came out of
   the same reading of the data that produced the taxonomy.

10. Drafts never contain a real URL or an agent signature. Every link in the dump is a dead t.co shortener, and a 3B
    model will happily invent plausible support URLs if you let it. So links are the literal token [LINK] end to end, and
    post-processing replaces any URL the model emits (invented_url_rate is zero by construction; the judge penalises
    invented policies, prices and dates instead). The "/NJ" initials in the exemplars are real people. An automated reply
    signing as one of them would be deceptive, so a regex strips those too.

11. The judge is a different, larger model family (llama3.1:8b) than the agent (qwen2.5:3b). Same-model judging
    inflates scores through self-preference. The judge also gets the brand's real reply to the tweet as a reference,
    which the agent never sees, plus three real precedents, so "grounded" is scored against evidence rather than the
    judge's imagination.

12. The human ratings for judge agreement are blind and balanced across systems. 20 replies per system, shuffled with
    the system name hidden, so I couldn't favour the agent. Agreement is reported as Spearman on the 1-5 scores and
    Cohen's kappa on the binary "acceptable", because that's the number the acceptable-rate headline rests on.

13. The baselines are meant to bracket the problem, not to be easy to beat. The simple baseline is trained on 1,000 LLM
    silver labels rather than golden labels, which makes it a distillation of the agent's own classifier: the honest
    "cheap production alternative", and its gap to the agent is the value of running the LLM at inference time. Two
    trivial escalation policies are reported (always and never) because they're the safe-but-useless and reckless
    corners. The agent is only interesting if it beats both on unsafe-auto rate at a given automation rate.

14. The golden set is 50% uniform-random and 50% keyword-enriched. Uniform sampling alone gave me fewer than five
    outage examples and about ten account-access ones. The enriched half gives per-class support; the uniform half is
    reported separately as the production-distribution number. The bucket regexes were never used as labels.

15. All LLM calls are cached on disk and the cache is committed. The full pipeline takes about 25 minutes on a laptop
    GPU and the judging about an hour and a half. With the cache, `make eval` reproduces every number in seconds and
    `make quick` shows the live path works in minutes. A reviewer with an OpenAI key can set LLM_PROVIDER=openai instead.
