import pandas as pd
import numpy as np
import pymc as pm
import arviz as az

sub = pd.read_csv('clean/model_input_top300.csv')

products = sub['StockCode'].astype('category')
product_idx = products.cat.codes.values
n_products = len(products.cat.categories)
product_names = products.cat.categories.tolist()

log_price_c = sub['log_price_c'].values
log_qty = sub['log_qty'].values
is_nov_dec = sub['is_nov_dec'].values

print(f"N products: {n_products}, N obs: {len(sub)}")

with pm.Model() as hier_model:
    # Hyperpriors: population-level elasticity distribution
    mu_beta = pm.Normal('mu_beta', mu=-1.5, sigma=2)       # population mean elasticity (expect negative)
    sigma_beta = pm.HalfNormal('sigma_beta', sigma=2)       # between-product variance in elasticity

    mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=5)
    sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=5)

    # Non-centered parameterization for better sampling
    alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=n_products)
    beta_offset = pm.Normal('beta_offset', mu=0, sigma=1, shape=n_products)

    alpha = pm.Deterministic('alpha', mu_alpha + sigma_alpha * alpha_offset)
    beta = pm.Deterministic('beta', mu_beta + sigma_beta * beta_offset)  # per-product elasticity

    # Seasonal (holiday) effect, shared across products
    gamma_holiday = pm.Normal('gamma_holiday', mu=0, sigma=1)

    sigma_y = pm.HalfNormal('sigma_y', sigma=1)

    mu = alpha[product_idx] + beta[product_idx] * log_price_c + gamma_holiday * is_nov_dec

    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_y, observed=log_qty)

    trace = pm.sample(1000, tune=1000, chains=4, cores=4, target_accept=0.9, random_seed=42, progressbar=True)

import pickle
with open('clean/trace.pkl', 'wb') as f:
    pickle.dump(trace, f)
with open('clean/product_names.pkl', 'wb') as f:
    pickle.dump(product_names, f)
print("Sampling complete, trace and product names saved.")

summary = az.summary(trace, var_names=['mu_beta','sigma_beta','mu_alpha','sigma_alpha','gamma_holiday','sigma_y'])
print(summary)

# max r_hat check across all params
rhat = az.rhat(trace)
max_rhat = max([float(rhat[v].max()) for v in rhat.data_vars])
print(f"\nMax R-hat across all params: {max_rhat:.4f}")
