"""
RingGuard feature engineering.

The one rule everything here respects: a feature is only allowed to use
transactions that happened on or before that account's own decision_day_offset
(set in data_generator.assign_decision_points). For accounts opened during
the observation window ("cold start"), that cutoff is just a few days after
opening -- so their behavioral features are naturally thin, exactly like a
real new account would be. For accounts already established before the
window started, the cutoff is the end of the window, so they get full
accumulated history. Without this cutoff, an evaluation would silently let
a "cold start" account's *future* 90 days of transactions leak into the
features used to score it on day five, which would make the cold-start
problem look solved when it hasn't actually been touched.

  - COLD_START_FEATURES: signals available the instant an account is opened
    -- KYC score and the device/graph signals, both of which come from the
    account-opening and device-fingerprint data, not transaction history.
  - BEHAVIORAL_FEATURES: signals that require accumulated transaction
    history (txn count, velocity, amount stats, rail diversity, counterparty
    diversity). Computed only from transactions up to each account's
    decision cutoff.
"""
import numpy as np
import pandas as pd

import data_generator as _dg

COLD_START_FEATURES = [
    "kyc_risk_score", "device_degree", "component_size",
]

BEHAVIORAL_FEATURES = [
    "txn_count", "total_amount", "max_amount", "amount_std",
    "active_days", "txn_per_active_day", "rail_diversity", "txn_graph_degree",
]

ALL_FEATURES = COLD_START_FEATURES + BEHAVIORAL_FEATURES


def _behavioral_agg(txns_subset):
    if len(txns_subset) == 0:
        return pd.DataFrame(columns=[
            "account_id", "txn_count", "total_amount", "max_amount",
            "amount_std", "active_days", "rail_diversity"
        ])
    return txns_subset.groupby("account_id").agg(
        txn_count=("transaction_id", "count"),
        total_amount=("amount", "sum"),
        max_amount=("amount", "max"),
        amount_std=("amount", "std"),
        active_days=("timestamp", lambda s: s.dt.date.nunique()),
        rail_diversity=("payment_rail", "nunique"),
    ).reset_index()


def build_account_table(accounts_df, graph_feats_df, txns_df):
    """Builds one row per account using only transactions visible as of that
    account's own decision_day_offset (no peeking into its future)."""
    txns = txns_df.copy()
    txns["day_offset"] = (txns["timestamp"] - _dg.START_DATE).dt.total_seconds() / 86400.0

    cutoffs = accounts_df[["account_id", "decision_day_offset"]]

    # Sender-side visibility: an account's own outgoing activity, filtered
    # by ITS OWN decision cutoff.
    sent = txns.merge(cutoffs, on="account_id", how="left")
    sent_visible = sent[sent["day_offset"] <= sent["decision_day_offset"]].copy()

    agg = _behavioral_agg(sent_visible)
    agg["amount_std"] = agg["amount_std"].fillna(0)
    agg["txn_per_active_day"] = agg["txn_count"] / agg["active_days"].replace(0, 1)

    # Counterparty/graph degree must respect EACH account's own cutoff in
    # both roles. Counting an account as a counterparty using the SENDER's
    # cutoff (rather than its own) would let a brand-new account pick up
    # graph-degree credit for transactions that happened after its own
    # decision time, just because an established counterparty's cutoff was
    # later -- a subtle but real leakage path.
    recv_cutoffs = cutoffs.rename(columns={"account_id": "counterparty_account_id"})
    recv = txns.merge(recv_cutoffs, on="counterparty_account_id", how="left")
    recv_visible = recv[recv["day_offset"] <= recv["decision_day_offset"]].copy()

    txn_degree = pd.concat([
        sent_visible[["account_id"]].rename(columns={"account_id": "acct"}),
        recv_visible[["counterparty_account_id"]].rename(columns={"counterparty_account_id": "acct"})
    ]).groupby("acct").size().rename("txn_graph_degree")

    table = accounts_df.merge(agg, on="account_id", how="left")
    table = table.merge(
        graph_feats_df[["account_id", "device_degree", "component_size", "component_id"]],
        on="account_id", how="left"
    )
    table = table.merge(
        txn_degree.rename_axis("account_id").reset_index(), on="account_id", how="left"
    )

    for col in ["txn_count", "total_amount", "max_amount", "amount_std",
                "active_days", "rail_diversity", "txn_per_active_day",
                "txn_graph_degree"]:
        table[col] = table[col].fillna(0)

    # The label (is this account ever fraudulent) is the evaluation target,
    # not a feature -- it's allowed to use the FULL transaction history
    # regardless of decision time, so cold-start accounts whose fraud event
    # happens just after the decision cutoff are still labeled correctly.
    full_label = txns_df.groupby("account_id")["is_fraud"].max().rename("label_any_fraud")
    table = table.merge(full_label.reset_index(), on="account_id", how="left")
    table["label_any_fraud"] = table["label_any_fraud"].fillna(0).astype(int)

    table["device_degree"] = table["device_degree"].fillna(1)
    table["component_size"] = table["component_size"].fillna(1)
    return table
