"""Hierarchical Bayesian price-elasticity model (PyMC) plus a per-product OLS comparison.

Model: log(qty) = alpha_p + beta_p * centered log(price) + gamma * holiday, with alpha_p and beta_p
partially pooled across products (non-centered parameterization).

Input : clean/model_input_top300.csv (from prepare_data.py)
Output: clean/beta_draws.npz, clean/elasticities.csv, results/model_summary.json
"""
import json
import os

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

sub = pd.read_csv(os.path.join(ROOT, "clean", "model_input_top300.csv"), dtype={"StockCode": str})
products = sub["StockCode"].astype("category")
product_idx = products.cat.codes.values
product_names = products.cat.categories.tolist()
n_products = len(product_names)
print(f"N products: {n_products}, N obs: {len(sub)}")

with pm.Model() as hier_model:
    mu_beta = pm.Normal("mu_beta", mu=-1.5, sigma=2)
    sigma_beta = pm.HalfNormal("sigma_beta", sigma=2)
    mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=5)
    sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=5)
    alpha_offset = pm.Normal("alpha_offset", mu=0, sigma=1, shape=n_products)
    beta_offset = pm.Normal("beta_offset", mu=0, sigma=1, shape=n_products)
    alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * alpha_offset)
    beta = pm.Deterministic("beta", mu_beta + sigma_beta * beta_offset)
    gamma_holiday = pm.Normal("gamma_holiday", mu=0, sigma=1)
    sigma_y = pm.HalfNormal("sigma_y", sigma=1)
    mu = alpha[product_idx] + beta[product_idx] * sub["log_price_c"].values + gamma_holiday * sub["is_nov_dec"].values
    pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=sub["log_qty"].values)
    trace = pm.sample(1000, tune=1000, chains=4, cores=2, target_accept=0.9, random_seed=42, progressbar=False)


post = trace.posterior
def draws(name):
    return post[name].stack(s=("chain", "draw")).values
def interval(x):
    return [round(float(np.percentile(x, 3)), 3), round(float(np.percentile(x, 97)), 3)]  # equal-tailed 94%

rhat = az.rhat(trace)
max_rhat = max(float(np.asarray(rhat[v]).max()) for v in rhat.data_vars)
ess = az.ess(trace)
min_ess = min(float(np.asarray(ess[v]).min()) for v in ess.data_vars)
divergences = int(trace.sample_stats["diverging"].sum())

beta_draws = draws("beta")  # (products, draws)
np.savez_compressed(os.path.join(ROOT, "clean", "beta_draws.npz"), beta=beta_draws, products=np.array(product_names))
lo, hi = np.percentile(beta_draws, 3, axis=1), np.percentile(beta_draws, 97, axis=1)

# Per-product OLS on the same data, the "estimate each product on its own" alternative
ols = []
for code, g in sub.groupby("StockCode"):
    x = np.column_stack([np.ones(len(g)), g["log_price_c"], g["is_nov_dec"]])
    coef, *_ = np.linalg.lstsq(x, g["log_qty"].values, rcond=None)
    ols.append((code, coef[1], len(g)))
ols = pd.DataFrame(ols, columns=["StockCode", "ols_elasticity", "n_months"]).set_index("StockCode")

el = pd.DataFrame({
    "StockCode": product_names,
    "elasticity_mean": beta_draws.mean(axis=1),
    "elasticity_ci94_low": lo,
    "elasticity_ci94_high": hi,
    "p_elasticity_negative": (beta_draws < 0).mean(axis=1),
}).set_index("StockCode").join(ols)
el.to_csv(os.path.join(ROOT, "clean", "elasticities.csv"))

summary = {
    "n_products": n_products,
    "n_obs": int(len(sub)),
    "chains": 4, "draws_per_chain": 1000, "tune": 1000,
    "divergences": divergences,
    "max_rhat": round(max_rhat, 4),
    "min_ess": round(min_ess, 1),
    "mu_beta_mean": round(float(draws("mu_beta").mean()), 3),
    "mu_beta_ci94": interval(draws("mu_beta")),
    "sigma_beta_mean": round(float(draws("sigma_beta").mean()), 3),
    "gamma_holiday_mean": round(float(draws("gamma_holiday").mean()), 3),
    "gamma_holiday_ci94": interval(draws("gamma_holiday")),
    "ols_median_elasticity": round(float(el["ols_elasticity"].median()), 3),
    "ols_wrong_sign_share": round(float((el["ols_elasticity"] > 0).mean()), 4),
    "bayes_wrong_sign_share_posterior_mean": round(float((el["elasticity_mean"] > 0).mean()), 4),
    "bayes_products_ci94_below_zero": int((el["elasticity_ci94_high"] < 0).sum()),
}
with open(os.path.join(ROOT, "results", "model_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
