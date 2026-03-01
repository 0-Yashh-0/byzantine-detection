"""
ml/preprocess.py
----------------
Loads the three raw "normal" CSVs, cleans them, fits a MinMaxScaler,
builds per-node sliding windows, and saves everything to dataset/processed/.

Output files
------------
dataset/processed/
    X_train.npy      float32 array  (N_train, SEQ_LEN, 4)
    X_val.npy        float32 array  (N_val,   SEQ_LEN, 4)
    feature_names.txt  one feature name per line
ml/
    scaler.pkl       fitted MinMaxScaler (used at inference time too)
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# ── Configuration ────────────────────────────────────────────────────────────
RAW_FILES = [
    "dataset/v2_baseline_clean.csv",   # idle baseline (mem ~22-25%)
    "dataset/CPUbaseline.csv",         # current cluster idle (mem ~29%)
    "dataset/CPULightLoad.csv",        # light stress  (~26% CPU)
    "dataset/CPUMediumLoad.csv",       # medium stress (~54% CPU)
]
OUT_DIR      = "dataset/processed"
SCALER_PATH  = "ml/scaler.pkl"

COLS         = ["timestamp", "instance", "cpu", "mem", "net_in", "net_out"]
FEATURES     = ["cpu", "mem", "net_in", "net_out"]
SEQ_LEN      = 20      # 20 × 5 s = 100 s window
VAL_SPLIT    = 0.20    # 20 % held-out for validation / threshold

# ── Step 1 : Load & validate ─────────────────────────────────────────────────
def load_all(paths):
    frames = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  [SKIP] Not found: {p}")
            continue
        df = pd.read_csv(p, header=None, names=COLS)

        # Drop rows with any NaN in feature columns
        before = len(df)
        df = df.dropna(subset=FEATURES)
        df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=FEATURES)
        after = len(df)

        if before != after:
            print(f"  [WARN] {p}: dropped {before - after} bad rows")

        frames.append(df)
        print(f"  Loaded  {p:45s}  rows={len(df):>5}  "
              f"nodes={len(df['instance'].unique())}")

    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    combined = combined.dropna(subset=["timestamp"])
    combined.sort_values(["instance", "timestamp"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined

# ── Step 2 : Per-node sliding windows ────────────────────────────────────────
def make_windows(df, scaler, seq_len):
    """Sliding windows within each node's own time-series (no cross-node bleed)."""
    windows = []
    node_counts = {}
    for node, grp in df.groupby("instance"):
        vals   = grp[FEATURES].values.astype(np.float32)
        scaled = scaler.transform(vals)
        n_win  = max(0, len(scaled) - seq_len)
        for i in range(n_win):
            windows.append(scaled[i : i + seq_len])
        node_counts[node] = n_win

    print("\n  Windows per node:")
    for node, cnt in sorted(node_counts.items()):
        print(f"    {node}: {cnt}")
    return np.array(windows, dtype=np.float32)

# ── Step 3 : Train / val split ───────────────────────────────────────────────
def split(X, val_frac):
    """Chronological (last val_frac fraction is validation)."""
    cut = int(len(X) * (1 - val_frac))
    return X[:cut], X[cut:]

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("STEP 1  Loading raw CSVs")
    print("=" * 60)
    df = load_all(RAW_FILES)
    print(f"\n  Combined: {len(df)} rows  |  "
          f"{df['instance'].nunique()} unique nodes")

    # ── Display quick stats ──────────────────────────────────────────────────
    print("\n  Per-feature statistics (raw values):")
    print(df[FEATURES].describe().round(2).to_string())

    print("\n" + "=" * 60)
    print("STEP 2  Fitting MinMaxScaler (FIXED domain ranges)")
    print("=" * 60)

    # Use fixed ranges so the scaler is immune to data drift.
    # CPU and memory are percentages [0, 100].
    # Network traffic: 0 to 100 KB/s covers normal + stress patterns.
    FIXED_MIN = np.array([0.0,  0.0,    0.0,    0.0   ], dtype=np.float32)
    FIXED_MAX = np.array([100.0, 100.0, 100000.0, 100000.0], dtype=np.float32)

    scaler = MinMaxScaler()
    # Trick: fit on a 2-row array containing the min and max rows
    scaler.fit(np.vstack([FIXED_MIN, FIXED_MAX]))

    print("  Feature ranges (fixed):")
    for feat, mn, mx in zip(FEATURES, scaler.data_min_, scaler.data_max_):
        print(f"    {feat:10s}  [{mn:.2f}  ..  {mx:.2f}]")

    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n  Scaler saved  ->  {SCALER_PATH}")

    print("\n" + "=" * 60)
    print("STEP 3  Creating per-node sliding windows")
    print("=" * 60)
    X = make_windows(df, scaler, SEQ_LEN)
    print(f"\n  Total windows : {len(X)}")
    print(f"  Window shape  : {X.shape[1:]}  (seq_len × features)")

    print("\n" + "=" * 60)
    print("STEP 4  Train / Validation split")
    print("=" * 60)
    X_train, X_val = split(X, VAL_SPLIT)
    print(f"  Train : {len(X_train)} windows")
    print(f"  Val   : {len(X_val)}   windows")

    print("\n" + "=" * 60)
    print("STEP 5  Saving processed arrays")
    print("=" * 60)
    os.makedirs(OUT_DIR, exist_ok=True)

    train_path = os.path.join(OUT_DIR, "X_train.npy")
    val_path   = os.path.join(OUT_DIR, "X_val.npy")
    feat_path  = os.path.join(OUT_DIR, "feature_names.txt")

    np.save(train_path, X_train)
    np.save(val_path,   X_val)
    with open(feat_path, "w") as f:
        f.write("\n".join(FEATURES))

    print(f"  X_train  →  {train_path}  ({X_train.nbytes / 1024:.1f} KB)")
    print(f"  X_val    →  {val_path}    ({X_val.nbytes / 1024:.1f} KB)")
    print(f"  Features →  {feat_path}")

    print("\n" + "=" * 60)
    print("✓  Preprocessing complete.  Run  ml/train_lstm.py  next.")
    print("=" * 60)

if __name__ == "__main__":
    main()
