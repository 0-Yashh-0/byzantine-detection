import numpy as np
import torch
import joblib
import os

MODEL_PATH     = "ml/lstm_model.pth"
SCALER_PATH    = "ml/scaler.pkl"
THRESHOLD_PATH = "ml/threshold.txt"
VAL_DATA_PATH  = "dataset/processed/X_val.npy"

# ── Load model (same architecture as train_lstm.py) ──────────────────────────
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.encoder_lstm  = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden2latent = nn.Linear(hidden_dim, latent_dim)
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm  = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer  = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        latent  = self.hidden2latent(h_n[-1])
        h_dec   = self.latent2hidden(latent).unsqueeze(1).repeat(1, x.shape[1], 1)
        dec_out, _ = self.decoder_lstm(h_dec)
        return self.output_layer(dec_out)

def get_errors(model, X_np):
    model.eval()
    with torch.no_grad():
        t     = torch.FloatTensor(X_np)
        recon = model(t)
        return torch.mean(torch.abs(recon - t), dim=(1, 2)).numpy()

def make_anomalous_windows(scaler, n=300, seq_len=20):
    """
    Creates synthetic Byzantine-like windows that differ from normal patterns.
    These represent nodes with abnormal behaviour the model has never seen.
    """
    rng     = np.random.default_rng(42)
    windows = []

    # Type 1: HIGH CPU + near-zero network (CPU-mining / isolated node)
    for _ in range(n // 3):
        w = np.zeros((seq_len, 4))
        w[:, 0] = rng.uniform(0.88, 1.0,  seq_len)  # cpu ≈ 90-100%
        w[:, 1] = rng.uniform(0.3,  0.5,  seq_len)  # mem normal
        w[:, 2] = rng.uniform(0.0,  0.02, seq_len)  # net_in ≈ 0 (isolated)
        w[:, 3] = rng.uniform(0.0,  0.02, seq_len)  # net_out ≈ 0
        windows.append(w)

    # Type 2: FLATLINED all features (crashed / zombie node)
    for _ in range(n // 3):
        flat = rng.uniform(0.0, 0.05, 4)            # stuck near zero
        w    = np.tile(flat, (seq_len, 1))
        w   += rng.normal(0, 0.005, w.shape)        # tiny noise
        windows.append(np.clip(w, 0, 1))

    # Type 3: RANDOM NOISE (corrupted / adversarial node)
    for _ in range(n // 3):
        w = rng.uniform(0.0, 1.0, (seq_len, 4))     # totally random
        windows.append(w)

    return np.array(windows, dtype=np.float32)

def metrics(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp)  if (tp + fp)  > 0 else 0.0
    recall    = tp / (tp + fn)  if (tp + fn)  > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn)  > 0 else 0.0
    return precision, recall, f1, fpr, tp, fp, tn, fn

def main():
    # ── Load artefacts ────────────────────────────────────────────────────────
    for p in [MODEL_PATH, SCALER_PATH, THRESHOLD_PATH, VAL_DATA_PATH]:
        if not os.path.exists(p):
            print(f"ERROR: Missing {p}  — run preprocess.py then train_lstm.py first.")
            return

    threshold = float(open(THRESHOLD_PATH).read().strip())
    scaler    = joblib.load(SCALER_PATH)

    model = LSTMAutoencoder(input_dim=4)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    print(f"Model loaded  |  Threshold = {threshold:.6f}\n")

    # ── 1. Normal validation data ─────────────────────────────────────────────
    X_val        = np.load(VAL_DATA_PATH)
    errors_norm  = get_errors(model, X_val)
    pred_norm    = (errors_norm > threshold).astype(int)  # should all be 0
    fpr_normal   = pred_norm.mean() * 100

    print("=" * 55)
    print("1. NORMAL DATA (validation set — should NOT be flagged)")
    print("=" * 55)
    print(f"   Windows tested     : {len(X_val)}")
    print(f"   MAE  mean          : {errors_norm.mean():.6f}")
    print(f"   MAE  std           : {errors_norm.std():.6f}")
    print(f"   MAE  max           : {errors_norm.max():.6f}")
    print(f"   Wrongly flagged    : {pred_norm.sum()} / {len(X_val)}")
    print(f"   False Positive Rate: {fpr_normal:.2f}%")

    # ── 2. Anomalous / Byzantine data ─────────────────────────────────────────
    X_anom      = make_anomalous_windows(scaler, n=300, seq_len=X_val.shape[1])
    errors_anom = get_errors(model, X_anom)
    pred_anom   = (errors_anom > threshold).astype(int)  # should all be 1
    detect_rate = pred_anom.mean() * 100

    print()
    print("=" * 55)
    print("2. SYNTHETIC ANOMALY DATA (should ALL be flagged)")
    print("=" * 55)
    print(f"   Windows tested     : {len(X_anom)}")
    print(f"   MAE  mean          : {errors_anom.mean():.6f}  (vs normal: {errors_norm.mean():.6f})")
    print(f"   Correctly flagged  : {pred_anom.sum()} / {len(X_anom)}")
    print(f"   Detection Rate     : {detect_rate:.2f}%")

    # ── 3. Combined precision / recall / F1 ───────────────────────────────────
    y_true = np.concatenate([np.zeros(len(X_val)),  np.ones(len(X_anom))])
    y_pred = np.concatenate([pred_norm,              pred_anom])
    precision, recall, f1, fpr, tp, fp, tn, fn = metrics(y_true, y_pred)

    print()
    print("=" * 55)
    print("3. OVERALL METRICS")
    print("=" * 55)
    print(f"   TP (anomaly flagged correctly) : {tp}")
    print(f"   FP (normal flagged wrongly)    : {fp}")
    print(f"   TN (normal passed correctly)   : {tn}")
    print(f"   FN (anomaly missed)            : {fn}")
    print()
    print(f"   Precision  : {precision*100:.1f}%   (of flagged, how many are real)")
    print(f"   Recall     : {recall*100:.1f}%   (of real anomalies, how many caught)")
    print(f"   F1 Score   : {f1*100:.1f}%   (harmonic mean of P and R)")
    print(f"   FP Rate    : {fpr*100:.2f}%  (normal windows wrongly flagged)")
    print()
    if f1 >= 0.90:
        print("   ✅ Excellent — model is production-ready")
    elif f1 >= 0.75:
        print("   ⚠️  Good but could improve — consider more training data")
    else:
        print("   ❌ Needs improvement — check threshold or add anomalous training data")

if __name__ == "__main__":
    main()
