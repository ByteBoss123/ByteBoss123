"""
Promo discount optimizer.

Given per-product Bayesian elasticity estimates and current price/qty baselines,
choose a discount depth (0/10/20/30%) per product to maximize projected
incremental revenue, subject to a total promo budget constraint
(budget = total $ given away in discounts on units sold at the discounted price).

Demand response uses the constant-elasticity form consistent with the
log-log Bayesian model: qty_new = qty_current * (price_new / price_current) ** elasticity
"""
import pandas as pd
import numpy as np
import pulp

data = pd.read_csv('clean/optimizer_input.csv')

DISCOUNT_LEVELS = [0.0, 0.10, 0.20, 0.30]
BUDGET = 5000.0  # promo budget in same currency units as unit_price (GBP)

rows = []
for _, r in data.iterrows():
    sc = r['StockCode']
    elast = r['elasticity_mean']
    p0 = r['current_price']
    q0 = r['current_qty']
    baseline_revenue = p0 * q0
    for d in DISCOUNT_LEVELS:
        p_new = p0 * (1 - d)
        q_new = q0 * (p_new / p0) ** elast
        revenue_new = p_new * q_new
        incremental_revenue = revenue_new - baseline_revenue
        promo_cost = q_new * (p0 - p_new)  # $ given away on units sold at discount
        rows.append({
            'StockCode': sc, 'discount': d, 'price_new': p_new, 'qty_new': q_new,
            'incremental_revenue': incremental_revenue, 'promo_cost': promo_cost,
        })

opts = pd.DataFrame(rows)
opts.to_csv('clean/optimizer_options.csv', index=False)

# ---- MIP formulation ----
prob = pulp.LpProblem("PromoOptimization", pulp.LpMaximize)

# binary decision var: x[sc, d] = 1 if product sc gets discount d
x = {}
for _, r in opts.iterrows():
    x[(r['StockCode'], r['discount'])] = pulp.LpVariable(
        f"x_{r['StockCode']}_{int(r['discount']*100)}", cat='Binary'
    )

# objective: maximize total incremental revenue
prob += pulp.lpSum(
    x[(r['StockCode'], r['discount'])] * r['incremental_revenue']
    for _, r in opts.iterrows()
)

# constraint: exactly one discount level chosen per product
for sc in opts['StockCode'].unique():
    prob += pulp.lpSum(x[(sc, d)] for d in DISCOUNT_LEVELS) == 1

# constraint: total promo cost <= budget
prob += pulp.lpSum(
    x[(r['StockCode'], r['discount'])] * r['promo_cost']
    for _, r in opts.iterrows()
) <= BUDGET

prob.solve(pulp.PULP_CBC_CMD(msg=0))

print("Solver status:", pulp.LpStatus[prob.status])

chosen = []
for _, r in opts.iterrows():
    if x[(r['StockCode'], r['discount'])].value() == 1:
        chosen.append(r)
chosen_df = pd.DataFrame(chosen)

total_incremental_rev = chosen_df['incremental_revenue'].sum()
total_promo_cost = chosen_df['promo_cost'].sum()
baseline_total_rev = (data['current_price'] * data['current_qty']).sum()

print(f"\nBudget: {BUDGET}")
print(f"Total promo cost used: {total_promo_cost:.2f}")
print(f"Total incremental revenue: {total_incremental_rev:.2f}")
print(f"Baseline revenue (top 300, current month): {baseline_total_rev:.2f}")
print(f"Incremental revenue as % of baseline: {100*total_incremental_rev/baseline_total_rev:.2f}%")
print(f"\nDiscount distribution chosen:")
print(chosen_df['discount'].value_counts().sort_index())

chosen_df.to_csv('clean/optimizer_solution.csv', index=False)

# ---- baseline comparison: uniform 15% discount across all products ----
uniform_d = 0.15
uniform_rows = []
for _, r in data.iterrows():
    p0, q0, elast = r['current_price'], r['current_qty'], r['elasticity_mean']
    p_new = p0 * (1 - uniform_d)
    q_new = q0 * (p_new / p0) ** elast
    uniform_rows.append({
        'incremental_revenue': p_new*q_new - p0*q0,
        'promo_cost': q_new*(p0-p_new)
    })
udf = pd.DataFrame(uniform_rows)
print(f"\n--- Baseline: uniform 15% discount across all 300 products ---")
print(f"Total promo cost: {udf.promo_cost.sum():.2f}")
print(f"Total incremental revenue: {udf.incremental_revenue.sum():.2f}")
