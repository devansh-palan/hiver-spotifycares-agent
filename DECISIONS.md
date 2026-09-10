# Decision log

Non-obvious choices, in roughly the order they were made, with the reason for each.

1. **Brand: SpotifyCares, not AppleSupport or AmazonHelp.** The biggest brands reply with "DM us" more than half the time
   (AppleSupport 52%), which leaves nothing to ground a reply on. SpotifyCares (43k replies) redirects to DM only 31% of the
   time and otherwise gives real troubleshooting, catalogue and policy answers, so "grounded in how the brand resolved it"
   is testable.

2. **Unit of work = first-contact triage (customer opener -> first brand reply), not full multi-turn threads.** 72% of the
   reconstructed threads have no customer follow-up at all, the first reply is where routing value is concentrated, and
   labelling multi-turn state would have halved the golden set for the same effort. Follow-up handling is listed under
   "not built".

3. **Eight intents, defined by reading ~300 openers before writing any code.** Fewer classes (5) merged things that are
   handled differently (outage vs. personal bug); more (12+) produced classes with <5 golden examples. `service_outage` was
   kept despite low support because its resolution is unique and fully automatable.

4. **Non-English messages are intent `other`, not their topic.** SpotifyCares routes by language before topic ("we have
   Indonesian support via email"), so language is the operative decision. The model still predicts the topic; a
   post-processing rule maps it to `other` so the failure analysis can see both.

5. **Escalation is a deterministic policy over LLM fields + regex, not an LLM yes/no.** Every decision carries a reason code
   and a source (`regex`, `llm_risk`, `language`, `intent_policy`, `auto`), so a support lead can audit and adjust a single
   rule without re-prompting. The 3B model was inconsistent on "does this need an account lookup"; making billing
   escalate-by-default with a narrow public-question exception fixed dev recall from 0.80 to 1.00 without touching the prompt.

6. **Regex hard triggers are deliberately narrow.** `hack` alone would fire on "what the hack happened to my Release Radar"
   (a real golden example). Triggers match `hacked|hijacked|unauthorised|someone ... using my account`, explicit cancel
   threats, legal words, and "talk to a human" phrasings. False positives are cheap for a human but they erode the
   automation rate, so precision was preferred.

7. **A 40-example dev set, disjoint from the golden set, for all prompt iteration.** Three iterations happened on dev
   (adding tie-break guidance, two few-shots, and the billing rule). The golden set was frozen before the first prompt was
   written and was only inspected once, after the first 75 predictions, which led to a drafting-only change (item 9).
   That inspection is disclosed in the report's "misleading" section.

8. **Retrieval index excludes every golden and dev opener.** Otherwise the agent could retrieve the brand's real reply to the
   very tweet it is being graded on. Exact-duplicate customer texts are also dropped so the top-5 neighbours are diverse.

9. **Reply prompt includes a per-intent playbook line from `configs/intents.yaml` and tells the model to ignore off-intent
   neighbours.** Dense retrieval matches surface words: an ads complaint on the web player retrieved web-player bug threads
   and the draft became browser troubleshooting. The playbook is a cheap guardrail derived from the same data reading that
   produced the taxonomy.

10. **Drafts never contain a real URL or an agent signature.** All links in the dump are dead t.co shorteners and a 3B
    model invents plausible support URLs if allowed, so links are the literal token `[LINK]` end to end and
    post-processing replaces any URL the model emits (`invented_url_rate` is zero by construction; the judge penalises
    invented policies, prices and dates instead). The "/NJ" initials in exemplars are real humans; an automated reply
    signing as one would be deceptive, so a regex strips them from drafts too.

11. **Judge is a different, larger model family (llama3.1:8b) than the agent (qwen2.5:3b).** Same-model judging inflates
    scores through self-preference. The judge also receives the brand's real reply to the tweet as a reference, which is
    unavailable to the agent, plus three real precedent replies, so "grounded" is measured against evidence rather than
    the judge's imagination.

12. **Human ratings for judge agreement are blind and system-balanced.** 20 replies per system are shuffled with the system
    name hidden before rating, so the human cannot favour the agent. Agreement is reported as Spearman on 1-5 and
    Cohen's kappa on the binary "acceptable", which is the number the acceptable-rate headline depends on.

13. **Baselines are designed to bracket the problem, not to be easy to beat.** The simple baseline is trained on 1,000
    LLM silver labels rather than golden labels, which makes it a distillation of the agent's classifier: the honest
    "cheap production alternative", whose gap to the agent is the value of the LLM at inference time. Two trivial
    escalation policies are reported (always / never) because they are the safe-but-useless and reckless corners; the
    agent is only interesting if it beats both on unsafe-auto rate at a given automation rate.

14. **Golden set is 50% uniform-random and 50% keyword-enriched.** Uniform sampling alone gave <5 examples of outage and
    ~10 of account access. The enriched half gives per-class support; the uniform half is reported separately as the
    production-distribution number. Bucket regexes were never used as labels.

15. **All LLM calls are cached on disk and the cache is committed.** Local models on a laptop GPU take ~25 minutes for the
    full pipeline and ~1.5 hours for judging. With the cache, `make eval` reproduces every number in seconds and
    `make quick` proves the live path works in minutes. A reviewer with an OpenAI key can set `LLM_PROVIDER=openai` instead.
