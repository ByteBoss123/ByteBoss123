"""Promo discount optimizer (mixed-integer program, PuLP/CBC).

Chooses one discount depth (0/10/20/30%) per product to maximize projected incremental revenue,
subject to a promo budget (money given away on units sold at the discount). Demand response uses the
same constant-elasticity form as the model: qty_new = qty0 * (price_new / price0) ** elasticity.

Baseline per product = its most recent full month in the panel (Nov 2011 for most products).
Comparison = the best *uniform* discount that fits the same budget (scanned in 0.5-point steps).
Uncertainty = the chosen plan re-scored under 4,000 posterior elasticity draws.

Input : clean/model_input_top300.csv, clean/elasticities.csv, clean/beta_draws.npz
Output: clean/optimizer_solution.csv, results/optimizer_summary.json
"""
import json
import os

import numpy as np
import pandas as pd
import pulp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGET = 5000.0          # GBP
LEVELS = [0.0, 0.10, 0.20, 0.30]

panel = pd.read_csv(os.path.join(ROOT, "clean", "model_input_top300.csv"), dtype={"StockCode": str})
el = pd.read_csv(os.path.join(ROOT, "clean", "elasticities.csv"), dtype={"StockCode": str}).set_index("StockCode")
post = np.load(os.path.join(ROOT, "clean", "beta_draws.npz"))
draws = pd.DataFrame(post["beta"], index=post["products"].astype(str))

last = panel.sort_values("month").groupby("StockCode").tail(1).set_index("StockCode")
data = pd.DataFrame({"price0": last["price"], "qty0": last["qty"], "month0": last["month"]}).join(el["elasticity_mean"])


def response(p0, q0, e, d):
    p1 = p0 * (1 - d)
    q1 = q0 * (1 - d) ** e
    return p1 * q1 - p0 * q0, q1 * (p0 - p1)  # incremental revenue, promo cost


opts = []
for code, r in data.iterrows():
    for d in LEVELS:
        inc, cost = response(r.price0, r.qty0, r.elasticity_mean, d)
        opts.append((code, d, inc, cost))
opts = pd.DataFrame(opts, columns=["StockCode", "discount", "incremental_revenue", "promo_cost"])

prob = pulp.LpProblem("promo", pulp.LpMaximize)
x = {(r.StockCode, r.discount): pulp.LpVariable(f"x_{i}", cat="Binary") for i, r in enumerate(opts.itertuples())}
prob += pulp.lpSum(x[(r.StockCode, r.discount)] * r.incremental_revenue for r in opts.itertuples())
for code in data.index:
    prob += pulp.lpSum(x[(code, d)] for d in LEVELS) == 1
prob += pulp.lpSum(x[(r.StockCode, r.discount)] * r.promo_cost for r in opts.itertuples()) <= BUDGET
prob.solve(pulp.PULP_CBC_CMD(msg=0))
status = pulp.LpStatus[prob.status]

chosen = opts[[x[(r.StockCode, r.discount)].value() > 0.5 for r in opts.itertuples()]].set_index("StockCode")
chosen.to_csv(os.path.join(ROOT, "clean", "optimizer_solution.csv"))

# Best uniform discount within the same budget
best_uniform = (0.0, 0.0, 0.0)
for d in np.arange(0.005, 0.305, 0.005):
    inc, cost = response(data.price0, data.qty0, data.elasticity_mean, d)
    if cost.sum() <= BUDGET and inc.sum() > best_uniform[1]:
        best_uniform = (round(float(d), 3), float(inc.sum()), float(cost.sum()))

# Re-score the chosen plan under posterior uncertainty in elasticity
d_vec = chosen["discount"].reindex(data.index).values
E = draws.loc[data.index].values                       # products x draws
p0, q0 = data.price0.values[:, None], data.qty0.values[:, None]
q1 = q0 * (1 - d_vec[:, None]) ** E
inc_draws = (p0 * (1 - d_vec[:, None]) * q1 - p0 * q0).sum(axis=0)
cost_draws = (q1 * p0 * d_vec[:, None]).sum(axis=0)

summary = {
    "solver_status": status,
    "budget_gbp": BUDGET,
    "baseline_month_counts": data["month0"].value_counts().to_dict(),
    "baseline_revenue_gbp": round(float((data.price0 * data.qty0).sum()), 2),
    "optimized_incremental_revenue_gbp": round(float(chosen["incremental_revenue"].sum()), 2),
    "optimized_promo_cost_gbp": round(float(chosen["promo_cost"].sum()), 2),
    "optimized_discount_counts": {str(k): int(v) for k, v in chosen["discount"].value_counts().sort_index().items()},
    "best_uniform_discount": best_uniform[0],
    "best_uniform_incremental_revenue_gbp": round(best_uniform[1], 2),
    "best_uniform_promo_cost_gbp": round(best_uniform[2], 2),
    "optimized_vs_uniform_pct": round(100 * (chosen["incremental_revenue"].sum() / best_uniform[1] - 1), 1) if best_uniform[1] else None,
    "posterior_incremental_revenue_p5_p50_p95": [round(float(np.percentile(inc_draws, q)), 2) for q in (5, 50, 95)],
    "posterior_promo_cost_p5_p50_p95": [round(float(np.percentile(cost_draws, q)), 2) for q in (5, 50, 95)],
    "posterior_share_cost_over_budget": round(float((cost_draws > BUDGET).mean()), 3),
    "posterior_share_incremental_revenue_negative": round(float((inc_draws < 0).mean()), 3),
}
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
with open(os.path.join(ROOT, "results", "optimizer_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
