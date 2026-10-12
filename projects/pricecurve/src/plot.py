"""Shrinkage plot: per-product OLS elasticity vs hierarchical posterior mean."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
el = pd.read_csv(os.path.join(ROOT, "clean", "elasticities.csv"))
fig, ax = plt.subplots(figsize=(6.5, 5))
sc = ax.scatter(el.ols_elasticity, el.elasticity_mean, c=el.n_months, cmap="viridis", s=18)
lim = [-20, 10]  # a few OLS estimates (min shown in title) fall outside; axis clipped for readability
n_out = int((el.ols_elasticity < lim[0]).sum())
ax.plot(lim, lim, color="grey", lw=1, ls="--", label="no shrinkage")
ax.axhline(0, color="black", lw=0.6); ax.axvline(0, color="black", lw=0.6)
ax.set_xlabel("Per-product OLS elasticity"); ax.set_ylabel("Hierarchical posterior mean elasticity")
ax.set_xlim(lim); ax.set_ylim(-12, 4)
ax.set_title(f"Partial pooling pulls noisy products toward the catalog mean\n({n_out} OLS estimates below -20 not shown)", fontsize=10)
fig.colorbar(sc, label="months of history"); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "results", "shrinkage.png"), dpi=150)
print("saved results/shrinkage.png")
