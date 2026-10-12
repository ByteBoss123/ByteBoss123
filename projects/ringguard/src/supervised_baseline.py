"""
RingGuard supervised baseline.

A standard XGBoost classifier. Deliberately fit twice with different feature
sets, because that's the actual point this project is making:

  - BEHAVIORAL_FEATURES only: this is what a "traditional" supervised fraud
    model relies on -- transaction count, velocity, amount stats. It's
    exactly the kind of model that quietly fails on day-zero accounts,
    because a brand-new account has almost no behavioral signal yet
    (txn_count near 0, amount_std near 0, etc). This is the model the JD's
    "Cold Start" problem is actually describing.
  - ALL_FEATURES (behavioral + KYC + consortium/graph signals): once you
    add in features that exist the moment an account opens, the same
    model architecture recovers most of its cold-start performance. This
    is the comparison that justifies why "Mine the Global Consortium" and
    "Architect Cold Start Logic" are listed as separate responsibilities
    in the JD, rather than just "train a better classifier."
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from feature_engineering import ALL_FEATURES, BEHAVIORAL_FEATURES


def fit_supervised_model(table, feature_list=None, test_size=0.3, random_state=42):
    if feature_list is None:
        feature_list = BEHAVIORAL_FEATURES
    X = table[feature_list].fillna(0)
    y = table["label_any_fraud"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, table.index, test_size=test_size, random_state=random_state,
        stratify=y
    )

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=random_state,
        scale_pos_weight=max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum())),
    )
    model.fit(X_train, y_train)

    score_col = "supervised_risk_score"
    test_table = table.loc[idx_test].copy()
    test_table[score_col] = model.predict_proba(X_test)[:, 1]
    return model, test_table
