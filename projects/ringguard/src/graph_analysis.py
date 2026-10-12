"""
RingGuard graph analysis module.

Builds a bipartite account<->device graph (the same shape as a multi-tenant
"global consortium" device-intelligence dataset) and derives per-account
network features:

  - device_degree:        how many distinct accounts share this account's
                           primary device (a hallmark of synthetic identity
                           farms and mule networks).
  - component_size:        size of the connected component this account
                           belongs to once accounts and devices are linked.
  - counterparty_in_ring:  whether this account transacts with other
                           accounts inside the same connected component
                           (used to surface layering/kiting rings).
  - txn_graph_degree:      number of distinct counterparties transacted with.

These features feed both the cold-start scorer and the supervised baseline.
"""
import networkx as nx
import pandas as pd


def build_account_device_graph(devices_df):
    g = nx.Graph()
    for _, row in devices_df.iterrows():
        acct_node = ("acct", row["account_id"])
        dev_node = ("dev", row["device_id"])
        g.add_node(acct_node, kind="account")
        g.add_node(dev_node, kind="device")
        g.add_edge(acct_node, dev_node)
    return g


def add_transaction_edges(g, txns_df):
    """OPTIONAL / not used by the default pipeline.

    Layering counterparty edges onto the device graph sounds appealing for
    catching mule-ring layering, but in practice it collapses almost the
    entire legit population into one giant connected component -- a classic
    small-world / random-graph effect once you have thousands of accounts
    transacting with each other. That destroys the signal: every account
    ends up "in the same ring" as everyone else.

    Kept here for reference and as a base for a smarter version (e.g. only
    adding edges above a repetition/amount threshold, or restricting to
    short time-window cycles) but the pipeline relies on the device-sharing
    bipartite graph alone, which already isolates the injected fraud rings
    cleanly because legit accounts almost never share a device.
    """
    for _, row in txns_df.iterrows():
        src = ("acct", row["account_id"])
        dst = ("acct", row["counterparty_account_id"])
        if g.has_node(dst):
            g.add_edge(src, dst)
    return g


def compute_account_features(g, accounts_df, devices_df, txns_df):
    device_share_count = devices_df.groupby("device_id")["account_id"].nunique()
    devices_df = devices_df.merge(
        device_share_count.rename("device_degree"), on="device_id", how="left"
    )

    components = list(nx.connected_components(g))
    comp_size_by_account = {}
    comp_id_by_account = {}
    for i, comp in enumerate(components):
        size = len(comp)
        for node in comp:
            if node[0] == "acct":
                comp_size_by_account[node[1]] = size
                comp_id_by_account[node[1]] = i

    txn_degree = pd.concat([
        txns_df[["account_id"]].rename(columns={"account_id": "acct"}),
        txns_df[["counterparty_account_id"]].rename(columns={"counterparty_account_id": "acct"})
    ]).groupby("acct").size().rename("txn_graph_degree")

    feats = accounts_df[["account_id"]].copy()
    feats = feats.merge(
        devices_df[["account_id", "device_degree"]], on="account_id", how="left"
    )
    feats["component_size"] = feats["account_id"].map(comp_size_by_account).fillna(1)
    feats["component_id"] = feats["account_id"].map(comp_id_by_account)
    feats = feats.merge(
        txn_degree.rename_axis("account_id").reset_index(), on="account_id", how="left"
    )
    feats["txn_graph_degree"] = feats["txn_graph_degree"].fillna(0)
    feats["device_degree"] = feats["device_degree"].fillna(1)
    return feats


def flag_suspicious_components(feats_df, accounts_df, min_component_size=4,
                                min_device_degree=3):
    """Surface connected components that look like rings: many accounts
    sharing few devices, most of them opened recently. This is the
    'cold start' consortium signal -- it works even with zero individual
    transaction history because it is a property of the network, not the
    account."""
    merged = feats_df.merge(
        accounts_df[["account_id", "account_age_at_sim_start_days", "segment", "fraud_ring_id"]],
        on="account_id", how="left"
    )
    comp_summary = merged.groupby("component_id").agg(
        n_accounts=("account_id", "nunique"),
        max_device_degree=("device_degree", "max"),
        median_age_days=("account_age_at_sim_start_days", "median"),
        true_fraud_rings=("fraud_ring_id", lambda s: s.dropna().nunique()),
    ).reset_index()

    suspicious = comp_summary[
        (comp_summary.n_accounts >= min_component_size) &
        (comp_summary.max_device_degree >= min_device_degree)
    ].sort_values("n_accounts", ascending=False)

    return merged, comp_summary, suspicious
