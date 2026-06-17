# Deployment notes

## Enabling natural-language explanations

`src/llm_explainer.py` works out of the box with a deterministic, rule-based
explanation generator -- no API key required, which is what produced
`results/sample_explanations.json` in this repo.

To upgrade to natural-language explanations from Claude, set an API key
before running the pipeline:

```
export ANTHROPIC_API_KEY=sk-ant-...
python src/pipeline.py
```

No code changes needed -- `llm_explainer._call_claude()` checks for the key
at runtime and falls back to the template generator automatically if it's
missing or if the API call fails for any reason, so the pipeline never
breaks because of network/credential issues.

## Running on a schedule

For a real deployment, `pipeline.py` would be split into two parts: a
nightly batch job (data ingestion, graph rebuild, model refresh) and a
real-time scoring path that loads the fitted cold-start model/scaler and
scores a single new account or transaction against it without re-running
the whole pipeline. This repo keeps them together for clarity and because
it's evaluated as a single coherent run.
