"""
RingGuard explanation layer ("Human-in-the-Loop" validation).

The JD calls for a human-in-the-loop validator of AI-driven strategy logic:
someone who can audit *why* the model flagged something, in plain language,
for safety/transparency/false-positive review. This module generates that
explanation for every flagged account.

It will call the real Claude API (claude-sonnet-4-6) if ANTHROPIC_API_KEY is
set in the environment. If it isn't (e.g. running this project locally
without a key configured yet), it falls back to a deterministic, rule-based
reason-code generator built from the same feature contributions -- so the
pipeline always runs end to end, and swapping in a real key just upgrades
the explanation from templated to natural language. See deploy/README for
how to set the key.
"""
import os
import json
import urllib.request


def _template_explanation(row):
    reasons = []
    if row.get("device_degree", 1) >= 3:
        reasons.append(
            f"shares a device fingerprint with {int(row['device_degree']) - 1} other account(s)"
        )
    if row.get("component_size", 1) >= 4:
        reasons.append(
            f"belongs to a connected network of {int(row['component_size'])} linked accounts/devices"
        )
    if row.get("account_age_at_sim_start_days", 999) < 7:
        reasons.append("was opened within the last 7 days (no behavioral history yet)")
    if row.get("max_amount", 0) > 3000 and row.get("account_age_at_sim_start_days", 999) < 14:
        reasons.append(
            f"moved a large amount (${row['max_amount']:,.0f}) very early in the account's life"
        )
    if row.get("txn_per_active_day", 0) > 3:
        reasons.append("transacted at unusually high velocity")

    if not reasons:
        reasons.append("scored above the population baseline on a combination of weaker signals")

    risk_col = "coldstart_risk_score" if "coldstart_risk_score" in row else "supervised_risk_score"
    score = row.get(risk_col, None)
    score_txt = f" (risk score {score:.2f})" if score is not None else ""
    return (
        f"Account {row.get('account_id', 'unknown')}{score_txt} was flagged because it "
        + "; ".join(reasons) + "."
    )


def _call_claude(prompt, model="claude-sonnet-4-6", max_tokens=200):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return "".join(
                block.get("text", "") for block in data.get("content", [])
                if block.get("type") == "text"
            ).strip()
    except Exception as e:
        return None


def explain_flagged_account(row):
    """Returns (explanation_text, source) where source is 'llm' or 'template'."""
    template = _template_explanation(row)
    prompt = (
        "You are a fraud analyst assistant. In 2 short sentences, explain to "
        "a human reviewer why this account was flagged as risky, in plain "
        "language a non-technical compliance officer would understand. "
        f"Facts: {template}"
    )
    llm_text = _call_claude(prompt)
    if llm_text:
        return llm_text, "llm"
    return template, "template"


def explain_top_flags(table, score_col, top_n=10):
    top = table.sort_values(score_col, ascending=False).head(top_n)
    out = []
    for _, row in top.iterrows():
        text, source = explain_flagged_account(row)
        out.append({
            "account_id": row["account_id"],
            "score": float(row[score_col]),
            "explanation": text,
            "explanation_source": source,
        })
    return out
