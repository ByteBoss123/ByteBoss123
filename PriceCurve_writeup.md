# PriceCurve: Hierarchical Bayesian Price Elasticity & Promo Optimization

## Problem
Retailers need to decide which products to discount and by how much when promo budget is
limited. Naive approaches either apply a uniform discount across the catalog or estimate
elasticity per product independently — both are weak with real transaction data, where
most SKUs have too few observations to estimate elasticity reliably on their own.

## Data
UK Online Retail transaction dataset (Dec 2010–Dec 2011), 541,909 raw transaction line
items. Cleaned to remove cancellations, non-positive quantity/price rows, and non-UK
transactions (to keep a single currency), then aggregated to product-month panels
(quantity-weighted average price per product per month). Final modeling set: the top 300
products by revenue (covering ~60% of total revenue), 3,667 product-month observations,
median 12 months of history per product.

## Method

### 1. Hierarchical Bayesian elasticity model (PyMC)
Modeled log(quantity) as a linear function of centered log(price) with a partial-pooling
(non-centered) hierarchical structure: each product's elasticity is drawn from a shared
population distribution (`mu_beta`, `sigma_beta`), letting products with few observations
borrow statistical strength from the broader catalog rather than producing noisy,
unstable individual estimates. A shared holiday-season term (Nov/Dec) captures common
seasonal demand lift.

- 4 chains, 1,000 tuning + 1,000 sampling draws, 0 divergences, max R-hat 1.008 (converged)
- Population mean elasticity: -3.71 (94% credible interval; consistent with the naive
  per-product median of -3.56)
- Holiday effect: +0.32 on log-quantity (positive, as expected)

**Why hierarchical pooling matters here (not just a modeling preference):** per-product
OLS regressions on the same data give the wrong sign (positive elasticity — quantity
rising with price) for **16.3%** of products, driven by small sample size (some products
have as few as 6-9 months of data). After hierarchical pooling, the wrong-sign rate drops
to **0.7%**, because noisy individual estimates are shrunk toward the population mean in
proportion to their uncertainty.

### 2. Mixed-integer promo optimizer (PuLP)
Given each product's posterior elasticity and current price/quantity baseline, the
optimizer chooses one discount level per product (0%, 10%, 20%, or 30%) to maximize total
projected incremental revenue, subject to a promo budget constraint (total $ given away
across all discounted units sold). Demand response at each discount level uses the same
constant-elasticity functional form as the Bayesian model, so the optimizer's projections
are internally consistent with the estimation step.

- At a $5,000 promo budget: the optimizer discounted 64 of 300 products by 10% (the
  remaining 236 stay at full price), generating **$17,710 in projected incremental
  revenue**.
- Compared against the best possible *uniform* discount at the same $5,000 budget (a 1%
  blanket discount, since a uniform discount spread across all 300 products exhausts
  budget quickly), which generates $5,622 in incremental revenue.
- **The targeted, elasticity-informed allocation delivers 215% more incremental revenue
  than the best uniform-discount alternative at equal spend**, by concentrating the
  budget on the most price-elastic products rather than spreading it evenly.

## Limitations (stated explicitly, not glossed over)
- Estimates are observational, not from a randomized price experiment — elasticity here is
  *estimated*, not *measured* under controlled conditions. Causal language should be
  qualified accordingly in any external framing.
- The constant-elasticity functional form is only reliable near the observed price range
  for each product; the optimizer's selected discounts (0% or 10% only) stayed within that
  safe range in this run, but larger hypothetical discounts should not be trusted without
  re-checking against actual price variation observed per product.
- This is UK e-commerce data (Olist-style online retail), not grocery/CPG or in-store
  retail — the modeling approach generalizes, but the specific elasticity values do not
  transfer directly to a different retail vertical.
- Single calendar year of data — cannot separate year-over-year trend from seasonal
  effects.

## Stack
Python, Pandas, NumPy, PyMC (hierarchical Bayesian modeling), ArviZ (posterior diagnostics),
PuLP (mixed-integer optimization), Matplotlib.
