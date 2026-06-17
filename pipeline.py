"""
RingGuard end-to-end pipeline.

Run with: python src/pipeline.py

Produces:
  results/account_scores.csv      -- every account with both risk scores
  results/suspicious_components.csv -- flagged device/transaction rings
  results/metrics_report.json     -- precision/recall/AUC/KS/FPR for both
                                      models, overall and on cold-start-only
                                      accounts
  results/sample_explanations.json -- human-readable explanations for the
                                      top flagged accounts
  results/roc_curves.png          -- ROC curves, cold-start vs supervised
  results/fraud_graph.png         -- visualization of a flagged ring
"""
import os
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_generator
import graph_analysis
import feature_engineering
import coldstart_scoring
import supervised_baseline
import evaluation
import llm_explainer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("1/6  Generating synthetic consortium dataset")
    print("=" * 60)
    accounts_df, devices_df, txns_df = data_generator.main(out_dir=DATA_DIR)

    print("\n" + "=" * 60)
    print("2/6  Building account/device/transaction graph")
    print("=" * 60)
    g = graph_analysis.build_account_device_graph(devices_df)
    graph_feats = graph_analysis.compute_account_features(g, accounts_df, devices_df, txns_df)
    merged, comp_summary, suspicious = graph_analysis.flag_suspicious_components(
        graph_feats, accounts_df
    )
    suspicious.to_csv(os.path.join(RESULTS_DIR, "suspicious_components.csv"), index=False)
    print(f"Flagged {len(suspicious)} suspicious connected components out of "
          f"{len(comp_summary)} total components.")
    true_positive_rings = (suspicious.true_fraud_rings > 0).sum()
    print(f"  -> {true_positive_rings} of those actually correspond to an injected fraud ring "
          f"({true_positive_rings}/{len(suspicious)} precision on ring-level detection).")

    print("\n" + "=" * 60)
    print("3/6  Feature engineering")
    print("=" * 60)
    table = feature_engineering.build_account_table(accounts_df, graph_feats, txns_df)
    print(f"Account table: {table.shape[0]} rows x {table.shape[1]} cols, "
          f"{table.is_cold_start.sum()} cold-start accounts "
          f"(opened during the observation window, scored within "
          f"{int(table.loc[table.is_cold_start==1,'account_age_days_at_decision'].iloc[0]) if table.is_cold_start.sum() else '?'} days of opening).")

    print("\n" + "=" * 60)
    print("4/6  Fitting models: cold-start (unsupervised), supervised")
    print("     (behavioral-only vs. consortium-enriched)")
    print("=" * 60)
    cs_model, cs_scaler, table = coldstart_scoring.fit_coldstart_model(table)
    sup_behavioral_model, sup_behavioral_test = supervised_baseline.fit_supervised_model(
        table, feature_list=feature_engineering.BEHAVIORAL_FEATURES
    )
    sup_full_model, sup_full_test = supervised_baseline.fit_supervised_model(
        table, feature_list=feature_engineering.ALL_FEATURES
    )
    print("All models fit successfully.")

    print("\n" + "=" * 60)
    print("5/6  Evaluating")
    print("=" * 60)
    metrics = {}

    metrics["coldstart_overall"] = evaluation.evaluate(
        table["label_any_fraud"].values, table["coldstart_risk_score"].values,
        label="Cold-start model (KYC + consortium/graph features), all accounts"
    )
    cold_only = table[table.is_cold_start == 1]
    metrics["coldstart_on_coldstart_accounts"] = evaluation.evaluate(
        cold_only["label_any_fraud"].values, cold_only["coldstart_risk_score"].values,
        label="Cold-start model, cold-start accounts only"
    )

    def _safe_eval(tbl, score_col, label):
        sub = tbl[tbl.is_cold_start == 1]
        if len(sub) > 5 and sub["label_any_fraud"].nunique() > 1:
            return evaluation.evaluate(sub["label_any_fraud"].values, sub[score_col].values, label=label)
        return {"label": label, "note": "Too few cold-start accounts with both classes "
                                          "in this test split to compute a stable metric."}

    metrics["supervised_behavioral_overall"] = evaluation.evaluate(
        sup_behavioral_test["label_any_fraud"].values, sup_behavioral_test["supervised_risk_score"].values,
        label="Supervised (behavioral features only), held-out test set"
    )
    metrics["supervised_behavioral_on_coldstart_accounts"] = _safe_eval(
        sup_behavioral_test, "supervised_risk_score",
        "Supervised (behavioral features only), cold-start accounts in test split"
    )

    metrics["supervised_full_overall"] = evaluation.evaluate(
        sup_full_test["label_any_fraud"].values, sup_full_test["supervised_risk_score"].values,
        label="Supervised (behavioral + KYC + consortium/graph features), held-out test set"
    )
    metrics["supervised_full_on_coldstart_accounts"] = _safe_eval(
        sup_full_test, "supervised_risk_score",
        "Supervised (full feature set), cold-start accounts in test split"
    )

    for k, v in metrics.items():
        print(f"\n[{k}]")
        print(json.dumps(v, indent=2))

    with open(os.path.join(RESULTS_DIR, "metrics_report.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    table.to_csv(os.path.join(RESULTS_DIR, "account_scores.csv"), index=False)

    print("\n" + "=" * 60)
    print("6/6  Generating human-in-the-loop explanations for top flags")
    print("=" * 60)
    explanations = llm_explainer.explain_top_flags(table, "coldstart_risk_score", top_n=8)
    with open(os.path.join(RESULTS_DIR, "sample_explanations.json"), "w") as f:
        json.dump(explanations, f, indent=2)
    source = explanations[0]["explanation_source"] if explanations else "none"
    print(f"Generated {len(explanations)} explanations (source: {source}). "
          f"Sample:\n  {explanations[0]['explanation']}" if explanations else "No flags to explain.")

    # ---- Charts -------------------------------------------------------
    plt.figure(figsize=(6, 5))
    for score_col, y_col, lbl, tbl in [
        ("coldstart_risk_score", "label_any_fraud", "Cold-start (Isolation Forest, KYC+graph)", table),
        ("supervised_risk_score", "label_any_fraud", "Supervised (behavioral-only)", sup_behavioral_test),
        ("supervised_risk_score", "label_any_fraud", "Supervised (full feature set)", sup_full_test),
    ]:
        fpr, tpr, _ = roc_curve(tbl[y_col], tbl[score_col])
        plt.plot(fpr, tpr, label=lbl)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("RingGuard: ROC by model, all accounts")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"), dpi=140)
    plt.close()

    # This is the chart that actually makes the project's point: restricted
    # to accounts opened during the observation window and scored on day
    # zero, before any transaction history exists.
    plt.figure(figsize=(6, 5))
    cold_table = table[table.is_cold_start == 1]
    cold_behavioral = sup_behavioral_test[sup_behavioral_test.is_cold_start == 1]
    cold_full = sup_full_test[sup_full_test.is_cold_start == 1]
    for score_col, lbl, tbl in [
        ("coldstart_risk_score", "Cold-start (Isolation Forest, KYC+graph)", cold_table),
        ("supervised_risk_score", "Supervised (behavioral-only)", cold_behavioral),
        ("supervised_risk_score", "Supervised (full feature set)", cold_full),
    ]:
        if tbl["label_any_fraud"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(tbl["label_any_fraud"], tbl[score_col])
        plt.plot(fpr, tpr, label=lbl)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("RingGuard: ROC on cold-start accounts only\n(scored day zero, zero transaction history)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curves_coldstart_only.png"), dpi=140)
    plt.close()

    if len(suspicious) > 0:
        top_comp_id = suspicious.iloc[0]["component_id"]
        sub_nodes = [n for n in g.nodes if (n[0] == "acct" and
                     merged.loc[merged.account_id == n[1], "component_id"].eq(top_comp_id).any())
                     or (n[0] == "dev")]
        comp_nodes = [c for c in nx.connected_components(g) if any(
            n[0] == "acct" and merged.loc[merged.account_id == n[1], "component_id"].eq(top_comp_id).any()
            for n in c)]
        if comp_nodes:
            sub = g.subgraph(comp_nodes[0])
            plt.figure(figsize=(7, 7))
            pos = nx.spring_layout(sub, seed=42)
            colors = ["#d62728" if n[0] == "acct" else "#1f77b4" for n in sub.nodes]
            nx.draw(sub, pos, node_color=colors, node_size=180, with_labels=False, edge_color="#999999")
            plt.title(f"Flagged ring (component {top_comp_id}) -- red=account, blue=device")
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, "fraud_graph.png"), dpi=140)
            plt.close()

    print(f"\nAll results written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
