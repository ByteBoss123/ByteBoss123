"""
RingGuard cold-start scorer.

Trains an unsupervised Isolation Forest on features available the moment an
account is opened (KYC score, age, device-sharing degree, graph component
size, txn-graph degree). No fraud labels and no transaction history are
required, so this model can score a brand-new account on day zero -- solving
the "Cold Start" problem called out in the JD: new clients/accounts get
immediate protection before enough behavioral history accumulates to train
or apply a supervised model.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from feature_engineering import COLD_START_FEATURES


def fit_coldstart_model(table, contamination=0.05, random_state=42):
    X = table[COLD_START_FEATURES].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=300, contamination=contamination, random_state=random_state
    )
    model.fit(Xs)

    # Convert IsolationForest's raw score (higher = more normal) into a
    # 0-1 risk score where higher = riskier, which is what an analyst /
    # downstream decision engine expects.
    raw = model.score_samples(Xs)
    risk = (raw.max() - raw) / (raw.max() - raw.min())

    table = table.copy()
    table["coldstart_risk_score"] = risk
    return model, scaler, table
