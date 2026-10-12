# RingGuard: Cold-Start Fraud Ring Detection

**Question.** A brand-new account has no transaction history, so a behavioral fraud model has nothing to score.
How do you catch fraud on day zero?

**Answer tested here.** Link accounts through shared devices into a graph, then score new accounts on KYC and
graph features with an unsupervised model, before any transaction exists.

**Data: fully synthetic.** `src/data_generator.py` simulates 3,590 accounts, 3,590 device records and 22,331
transactions over 90 days across ACH, wire, RTP and check, with three injected fraud patterns: synthetic-identity
rings sharing devices, account takeover, and mule/kiting rings. It is seeded, so every run produces identical
files. Because the fraud patterns are planted by the same code that evaluates them, the numbers below show the
method works when the pattern is present; they are not estimates of real-world performance.

## Results (`results/metrics_report.json`)

| Check | Result |
|---|---|
| Ring detection (NetworkX connected components on the account-device graph) | 20 components flagged out of 3,470; **20 of 20** are injected fraud rings |
| Cold-start model (Isolation Forest, KYC + graph features), cold-start accounts | **AUC 0.9996**, recall 100% at precision 89.1% (n = 390, 90 fraud) |
| Same model, all accounts | AUC 0.800: it misses account takeover on established accounts, which it was not built for |
| Supervised XGBoost, behavioral features only, on cold-start accounts | **AUC 0.500**: no history means no signal |
| Supervised XGBoost, behavioral + KYC + graph | AUC 1.000 on the test split. A perfect score is a property of the planted, cleanly separable synthetic patterns, not something to expect on real data |

Bugs found and fixed along the way: two rounds of time leakage (features computed past the decision time, and a
counterparty-degree feature that leaked across accounts' cutoffs) and one giant connected component that swallowed
the legitimate population.

## Explanations for reviewers

`src/llm_explainer.py` writes a plain-language reason for each top-flagged account
(`results/sample_explanations.json`). It calls the Claude API when `ANTHROPIC_API_KEY` is set and otherwise uses a
deterministic template; the committed output used the template.

## Run

```
pip install -r requirements.txt
python src/pipeline.py      # about a minute; regenerates data/ and results/
```
