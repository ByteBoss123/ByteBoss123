"""
RingGuard synthetic data generator.

Generates a fraud/AML-style dataset across multiple payment rails (ACH, Wire,
RTP, Check) with three injected fraud typologies that mirror the kind of
patterns a consortium fraud platform (DataVisor-style) needs to catch:

  1. SYNTHETIC IDENTITY  - clusters of brand-new accounts sharing a device
                            fingerprint, doing small "bust-out" test
                            transactions before a large cash-out.
  2. ACCOUNT TAKEOVER     - an established, well-behaved account suddenly
                            transacts from a brand-new device/location with a
                            large transfer to a never-seen-before counterparty.
  3. MULE / KITING RING   - a tight ring of accounts rapidly layering money
                            through each other (circular transfers, high
                            velocity, check-kiting style early withdrawal).

None of this is real financial data. It is fully synthetic and seeded for
reproducibility.
"""
import numpy as np
import pandas as pd
import uuid
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

PAYMENT_RAILS = ["ACH", "Wire", "RTP", "Check"]
START_DATE = datetime(2026, 1, 1)
SIM_DAYS = 90


def _new_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def generate_accounts(n_legit=3000, n_synthetic_id_rings=15, ring_size=6,
                       n_ato_targets=120, n_mule_rings=10, mule_ring_size=8,
                       n_new_legit=300):
    """Build the account population: established legit, synthetic-identity
    ring accounts, ATO targets (legit accounts compromised later), mule ring
    accounts, and a control group of genuinely new but legitimate accounts
    opened during the observation window (needed so the cold-start
    evaluation has true negatives, not just fraud rings, among "brand new"
    accounts)."""
    accounts = []

    # Established legit population -- already open well before the
    # observation window starts, exactly like a real customer base.
    for _ in range(n_legit):
        age_days = int(RNG.integers(30, 730))
        open_date = START_DATE - timedelta(days=age_days)
        accounts.append({
            "account_id": _new_id("acct"),
            "open_date": open_date,
            "kyc_risk_score": float(np.clip(RNG.normal(0.25, 0.12), 0, 1)),
            "segment": "legit",
            "fraud_ring_id": None,
        })

    # New but legitimate accounts, opened during the observation window --
    # the control group for the cold-start cohort. Without these, "newly
    # opened account" would be a perfect proxy for "fraud," which is an
    # unrealistically easy problem.
    for _ in range(n_new_legit):
        open_offset = int(RNG.integers(0, SIM_DAYS - 10))
        open_date = START_DATE + timedelta(days=open_offset)
        accounts.append({
            "account_id": _new_id("acct"),
            "open_date": open_date,
            "kyc_risk_score": float(np.clip(RNG.normal(0.22, 0.1), 0, 1)),
            "segment": "new_legit",
            "fraud_ring_id": None,
        })

    # Synthetic identity rings: many accounts opened within days of each
    # other, all very young at simulation start.
    for r in range(n_synthetic_id_rings):
        ring_id = f"synid_ring_{r}"
        base_open = START_DATE + timedelta(days=int(RNG.integers(0, SIM_DAYS - 10)))
        for _ in range(ring_size):
            open_date = base_open + timedelta(days=int(RNG.integers(0, 3)))
            accounts.append({
                "account_id": _new_id("acct"),
                "open_date": open_date,
                "kyc_risk_score": float(np.clip(RNG.normal(0.55, 0.1), 0, 1)),
                "segment": "synthetic_id",
                "fraud_ring_id": ring_id,
            })

    # ATO targets: long-standing, low-risk accounts that will be hijacked.
    for _ in range(n_ato_targets):
        age_days = int(RNG.integers(180, 1000))
        open_date = START_DATE - timedelta(days=age_days)
        accounts.append({
            "account_id": _new_id("acct"),
            "open_date": open_date,
            "kyc_risk_score": float(np.clip(RNG.normal(0.15, 0.08), 0, 1)),
            "segment": "ato_target",
            "fraud_ring_id": None,
        })

    # Mule / kiting rings.
    for r in range(n_mule_rings):
        ring_id = f"mule_ring_{r}"
        for _ in range(mule_ring_size):
            age_days = int(RNG.integers(10, 400))
            open_date = START_DATE - timedelta(days=age_days)
            accounts.append({
                "account_id": _new_id("acct"),
                "open_date": open_date,
                "kyc_risk_score": float(np.clip(RNG.normal(0.4, 0.15), 0, 1)),
                "segment": "mule_ring",
                "fraud_ring_id": ring_id,
            })

    df = pd.DataFrame(accounts)
    # Signed offset in days from simulation start; negative = opened before
    # the window we can observe, positive = opened during it. This is the
    # field every later temporal-leakage check is built on.
    df["open_day_offset"] = (df["open_date"] - START_DATE).dt.days
    df["account_age_at_sim_start_days"] = (-df["open_day_offset"]).clip(lower=0)
    return df


def assign_decision_points(accounts_df, cold_start_window_days=0):
    """Decide, per account, the calendar point at which it gets scored, and
    whether that makes it a 'cold start' case.

    - Accounts opened during the observation window (open_day_offset >= 0)
      are scored shortly after opening (open_day_offset + window). These are
      the cold-start cases -- by construction they cannot have more than
      `cold_start_window_days` of transaction history available.
    - Accounts already open before the window starts are scored at the end
      of the window, i.e. with their full accumulated history visible. These
      are the "mature" cases the traditional behavioral model is built for.
    """
    df = accounts_df.copy()
    is_cold = (df["open_day_offset"] >= 0).astype(int)
    df["is_cold_start"] = is_cold
    df["decision_day_offset"] = np.where(
        is_cold == 1,
        df["open_day_offset"] + cold_start_window_days,
        SIM_DAYS,
    )
    df["account_age_days_at_decision"] = df["decision_day_offset"] - df["open_day_offset"]
    return df


def generate_devices(accounts_df):
    """Assign devices. Legit accounts mostly get their own device. Synthetic
    identity and mule ring accounts deliberately SHARE a small pool of
    devices within their ring -- that shared-device signal is exactly what a
    consortium device-intelligence graph is built to surface."""
    device_rows = []
    account_device_map = {}

    for _, row in accounts_df.iterrows():
        if row["segment"] in ("synthetic_id", "mule_ring") and row["fraud_ring_id"]:
            # Reuse 1-2 devices per ring to create shared-device clusters.
            ring_key = row["fraud_ring_id"]
            if ring_key not in account_device_map.get("_ring_pool", {}):
                account_device_map.setdefault("_ring_pool", {})[ring_key] = [
                    _new_id("dev") for _ in range(2)
                ]
            device_id = RNG.choice(account_device_map["_ring_pool"][ring_key])
        else:
            device_id = _new_id("dev")

        account_device_map[row["account_id"]] = device_id
        device_rows.append({
            "account_id": row["account_id"],
            "device_id": device_id,
            "device_risk_seed": float(RNG.uniform(0, 1)),
        })

    return pd.DataFrame(device_rows)


def generate_transactions(accounts_df, devices_df):
    """Generate transaction-level events with embedded fraud typologies."""
    dev_map = devices_df.set_index("account_id")["device_id"].to_dict()
    acct_ids = accounts_df["account_id"].tolist()
    txns = []

    def add_txn(account_id, day_offset, amount, rail, counterparty,
                device_id=None, is_fraud=0, fraud_type="none"):
        ts = START_DATE + timedelta(days=day_offset, hours=float(RNG.uniform(0, 24)))
        txns.append({
            "transaction_id": _new_id("txn"),
            "account_id": account_id,
            "device_id": device_id or dev_map.get(account_id),
            "counterparty_account_id": counterparty,
            "timestamp": ts,
            "amount": round(float(amount), 2),
            "payment_rail": rail,
            "is_fraud": is_fraud,
            "fraud_type": fraud_type,
        })

    legit_ids = accounts_df.loc[accounts_df.segment == "legit", "account_id"].tolist()

    # --- Normal background activity for every account ---------------------
    for _, row in accounts_df.iterrows():
        n_txn = int(max(1, RNG.poisson(6)))
        earliest_day = max(0, int(row["open_day_offset"]))
        if earliest_day >= SIM_DAYS:
            continue
        for _ in range(n_txn):
            day = int(RNG.integers(earliest_day, SIM_DAYS))
            amount = float(np.clip(RNG.lognormal(mean=4.5, sigma=1.0), 5, 5000))
            counterparty = RNG.choice(legit_ids)
            rail = RNG.choice(PAYMENT_RAILS, p=[0.45, 0.15, 0.30, 0.10])
            add_txn(row["account_id"], day, amount, rail, counterparty)

    # --- Pattern 1: synthetic identity bust-out ----------------------------
    for ring_id, ring_df in accounts_df[accounts_df.segment == "synthetic_id"].groupby("fraud_ring_id"):
        ring_accts = ring_df.set_index("account_id")["open_day_offset"].to_dict()
        # small test transactions to build a thin credit history -- timed
        # relative to EACH account's own open day, never before it opened
        for acct, own_open_day in ring_accts.items():
            for _ in range(int(RNG.integers(2, 4))):
                day = own_open_day + int(RNG.integers(1, 5))
                amount = float(RNG.uniform(15, 80))
                rail = RNG.choice(["ACH", "RTP"])
                add_txn(acct, day, amount, rail, RNG.choice(legit_ids),
                        is_fraud=1, fraud_type="synthetic_id_testing")
        # the bust-out: large simultaneous cash-outs once "trust" is built,
        # timed after every member's testing period has run
        cashout_day = max(ring_accts.values()) + int(RNG.integers(6, 10))
        for acct in ring_accts:
            amount = float(RNG.uniform(2500, 9000))
            add_txn(acct, cashout_day, amount, "RTP", RNG.choice(list(ring_accts)),
                    is_fraud=1, fraud_type="synthetic_id_bustout")

    # --- Pattern 2: account takeover ---------------------------------------
    ato_targets = accounts_df[accounts_df.segment == "ato_target"]["account_id"].tolist()
    for acct in ato_targets:
        takeover_day = int(RNG.integers(20, SIM_DAYS - 5))
        new_device = _new_id("dev_compromised")
        new_counterparty = _new_id("acct_external_mule")
        amount = float(RNG.uniform(4000, 15000))
        add_txn(acct, takeover_day, amount, RNG.choice(["Wire", "RTP"]),
                new_counterparty, device_id=new_device,
                is_fraud=1, fraud_type="account_takeover")
        # frequently a second smaller drain follows within hours/a day
        if RNG.random() < 0.6:
            amount2 = float(RNG.uniform(1000, 4000))
            add_txn(acct, takeover_day + 1, amount2, "RTP", new_counterparty,
                    device_id=new_device, is_fraud=1, fraud_type="account_takeover")

    # --- Pattern 3: mule / kiting ring layering -----------------------------
    for ring_id, ring_df in accounts_df[accounts_df.segment == "mule_ring"].groupby("fraud_ring_id"):
        ring_accts = ring_df["account_id"].tolist()
        n_layers = int(RNG.integers(3, 6))
        layer_start = int(RNG.integers(0, SIM_DAYS - 10))
        principal = float(RNG.uniform(8000, 30000))
        for layer in range(n_layers):
            src = ring_accts[layer % len(ring_accts)]
            dst = ring_accts[(layer + 1) % len(ring_accts)]
            day = layer_start + layer  # rapid, day-over-day layering
            amount = principal * float(RNG.uniform(0.85, 0.98))
            add_txn(src, day, amount, RNG.choice(["ACH", "Wire", "Check"]), dst,
                    is_fraud=1, fraud_type="mule_layering")
            principal = amount

    df = pd.DataFrame(txns).sort_values("timestamp").reset_index(drop=True)
    return df


def main(out_dir=None):
    if out_dir is None:
        import os
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        os.makedirs(out_dir, exist_ok=True)
    accounts_df = generate_accounts()
    accounts_df = assign_decision_points(accounts_df)
    devices_df = generate_devices(accounts_df)
    txns_df = generate_transactions(accounts_df, devices_df)

    accounts_df.to_csv(f"{out_dir}/accounts.csv", index=False)
    devices_df.to_csv(f"{out_dir}/devices.csv", index=False)
    txns_df.to_csv(f"{out_dir}/transactions.csv", index=False)

    print(f"accounts:      {len(accounts_df):>6}")
    print(f"devices:       {len(devices_df):>6}")
    print(f"transactions:  {len(txns_df):>6}  (fraud={txns_df.is_fraud.sum()}, "
          f"rate={txns_df.is_fraud.mean():.3%})")
    return accounts_df, devices_df, txns_df


if __name__ == "__main__":
    main()
