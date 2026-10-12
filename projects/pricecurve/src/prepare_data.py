"""Build the product-month modeling panel from the raw UCI Online Retail file.

Input : data/raw/online_retail.csv (fetched by data/download.sh, SHA-256 checked)
Output: clean/panel_all.csv          all UK product-months after cleaning
        clean/model_input_top300.csv top-300-revenue products, used by model_bayesian.py
        clean/prep_summary.json      row counts at every step
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "online_retail.csv")
OUT = os.path.join(ROOT, "clean")
TOP_N = 300
MIN_MONTHS = 6  # a product needs this many months of history to estimate a slope

os.makedirs(OUT, exist_ok=True)
summary = {}

df = pd.read_csv(RAW, dtype={"StockCode": str, "InvoiceNo": str})
summary["raw_rows"] = len(df)

df = df[~df["InvoiceNo"].str.startswith("C")]           # cancellations
summary["after_drop_cancellations"] = len(df)
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
summary["after_drop_nonpositive"] = len(df)
df = df[df["Country"] == "United Kingdom"]               # single currency, single market
summary["after_uk_only"] = len(df)
# non-product codes (postage, manual adjustments, bank charges, samples)
non_product = df["StockCode"].str.fullmatch(r"(POST|DOT|M|C2|BANK CHARGES|AMAZONFEE|S|D|PADS|CRUK|B|m)", case=False)
df = df[~non_product]
summary["after_drop_non_product_codes"] = len(df)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")
df["month"] = df["InvoiceDate"].dt.to_period("M")
# Dec 2011 has only 9 days of data; a partial month would look like a demand drop
df = df[df["month"] < pd.Period("2011-12", "M")]
summary["after_drop_partial_dec_2011"] = len(df)
df["revenue"] = df["Quantity"] * df["UnitPrice"]

panel = (df.groupby(["StockCode", "month"])
           .agg(qty=("Quantity", "sum"), revenue=("revenue", "sum"))
           .reset_index())
panel["price"] = panel["revenue"] / panel["qty"]          # quantity-weighted average price
panel.to_csv(os.path.join(OUT, "panel_all.csv"), index=False)
summary["product_months_all"] = len(panel)
summary["products_all"] = panel["StockCode"].nunique()

months = panel.groupby("StockCode")["month"].nunique()
eligible = months[months >= MIN_MONTHS].index
rev = panel[panel["StockCode"].isin(eligible)].groupby("StockCode")["revenue"].sum()
top = rev.sort_values(ascending=False).head(TOP_N).index
summary["top_n_revenue_share"] = round(float(rev.loc[top].sum() / panel["revenue"].sum()), 4)

sub = panel[panel["StockCode"].isin(top)].copy()
sub["log_qty"] = np.log(sub["qty"])
sub["log_price"] = np.log(sub["price"])
sub["log_price_c"] = sub["log_price"] - sub.groupby("StockCode")["log_price"].transform("mean")
sub["is_nov_dec"] = sub["month"].dt.month.isin([11, 12]).astype(int)
sub["month"] = sub["month"].astype(str)
sub.to_csv(os.path.join(OUT, "model_input_top300.csv"), index=False)

summary["model_products"] = int(sub["StockCode"].nunique())
summary["model_obs"] = len(sub)
summary["model_median_months_per_product"] = float(sub.groupby("StockCode").size().median())
summary["model_min_months_per_product"] = int(sub.groupby("StockCode").size().min())
with open(os.path.join(OUT, "prep_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
