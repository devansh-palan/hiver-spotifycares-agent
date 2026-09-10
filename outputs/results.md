# Results on the 200-example golden set

| system | intent acc (95% CI) | macro-F1 | esc. precision | esc. recall | automation rate | unsafe-auto rate | judge acceptable | judge overall | end-to-end trust |
|---|---|---|---|---|---|---|---|---|---|
| agent | 0.76 (0.69-0.81) | 0.75 | 0.95 | 0.89 | 0.68 | 0.040 | 0.745 | 4.08 | 0.58 |
| simple | 0.66 (0.59-0.72) | 0.67 | 0.83 | 0.83 | 0.65 | 0.060 | 0.6 | 3.97 | 0.41 |
| trivial_always | 0.16 (0.12-0.21) | 0.03 | 0.35 | 1.00 | 0.00 | 0.000 | 0.495 | 3.61 | 0.01 |
| trivial_never | 0.16 (0.12-0.21) | 0.03 | 0.00 | 0.00 | 1.00 | 0.350 | n/a | n/a | n/a |

## Agent: per-intent F1

| intent | P | R | F1 | support |
|---|---|---|---|---|
| technical_issue | 0.80 | 0.75 | 0.77 | 32 |
| service_outage | 1.00 | 0.75 | 0.86 | 4 |
| account_access | 0.95 | 0.70 | 0.81 | 30 |
| billing_subscription | 0.70 | 0.97 | 0.81 | 31 |
| content_availability | 0.75 | 0.84 | 0.79 | 25 |
| feature_feedback | 0.75 | 0.73 | 0.74 | 45 |
| general_question | 0.64 | 0.35 | 0.45 | 20 |
| other | 0.63 | 0.92 | 0.75 | 13 |

## Agent: confusion matrix (rows = gold, columns = predicted)

| gold \ pred | technic | service | account | billing | content | feature | general | other |
|---|---|---|---|---|---|---|---|---|
| technical_issue | 24 | . | 1 | 1 | 3 | 3 | . | . |
| service_outage | 1 | 3 | . | . | . | . | . | . |
| account_access | 2 | . | 21 | 4 | . | . | 1 | 2 |
| billing_subscription | . | . | . | 30 | . | 1 | . | . |
| content_availability | . | . | . | 1 | 21 | 1 | . | 2 |
| feature_feedback | 2 | . | . | 3 | 2 | 33 | 3 | 2 |
| general_question | 1 | . | . | 4 | 2 | 5 | 7 | 1 |
| other | . | . | . | . | . | 1 | . | 12 |

## Agent: uniform-random subset (production distribution)

`{"n": 100, "intent_accuracy": 0.74, "escalation": {"accuracy": 0.93, "precision": 0.943, "recall": 0.868, "automation_rate": 0.65, "unsafe_auto_rate": 0.05}}`

## Agent: confidence calibration

`{"0.7-0.85": {"n": 2, "accuracy": 0.5}, ">=0.85": {"n": 198, "accuracy": 0.758}}`

## Agent: escalation decision sources

`{"intent_policy": 31, "auto": 135, "llm_risk": 5, "language": 4, "regex": 25}`

## Agent: reply proxies

`{"n": 200, "mean_chars": 133.7, "pct_over_280": 0.0, "greeting_rate": 1.0, "invented_url_rate": 0.0, "asks_password_or_card": 0.005, "asks_pii_publicly": 0.0, "signs_initials": 0.0, "mentions_ai": 0.0, "empty": 0, "escalated_replies_mention_dm": 0.969}`

## Agent: judge acceptability by gold intent

`{"account_access": 0.93, "billing_subscription": 0.94, "content_availability": 0.64, "feature_feedback": 0.6, "general_question": 0.5, "other": 0.62, "service_outage": 1.0, "technical_issue": 0.84}`
