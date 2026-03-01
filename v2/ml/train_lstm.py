"""
ml/train_lstm.py
----------------
Trains an LSTM Autoencoder on preprocessed "normal" windows.
Run ml/preprocess.py FIRST to generate dataset/processed/*.npy

Saves:
    ml/lstm_model.pth    — best checkpoint (lowest val loss)
    ml/threshold.txt     — anomaly decision boundary (mean+3σ on val set)
"""
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import joblib

# ── Configuration ─────────────────────────────────────────────────────────────
PROCESSED_DIR   = "dataset/processed"
MODEL_PATH      = "ml/lstm_model.pth"
SCALER_PATH     = "ml/scaler.pkl"
THRESHOLD_PATH  = "ml/threshold.txt"

SEQUENCE_LENGTH = 20
BATCH_SIZE      = 64
EPOCHS          = 300   # val loss still declining at 100 → more epochs needed
LEARNING_RATE   = 0.001
HIDDEN_SIZE     = 64
LATENT_DIM      = 16
NOISE_STD       = 0.01    # denoising augmentation applied to training input

# ── Model ─────────────────────────────────────────────────────────────────────
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

# ── Helpers ───────────────────────────────────────────────────────────────────
def add_noise(X, std):
    return (X + np.random.normal(0, std, X.shape)).astype(np.float32)

def make_loader(X_in, X_target, batch_size, shuffle):
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.FloatTensor(X_in), torch.FloatTensor(X_target))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def recon_error(model, X_np):
    """Mean-absolute-error per window, shape (N,)"""
    model.eval()
    with torch.no_grad():
        t = torch.FloatTensor(X_np)
        recon = model(t)
        return torch.mean(torch.abs(recon - t), dim=(1, 2)).numpy()

# ── Training ──────────────────────────────────────────────────────────────────
def train():
    # 1. Load preprocessed data
    train_path = os.path.join(PROCESSED_DIR, "X_train.npy")
    val_path   = os.path.join(PROCESSED_DIR, "X_val.npy")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("ERROR: Preprocessed data not found.")
        print("       Run   python ml/preprocess.py   first.")
        return

    X_train_clean = np.load(train_path)   # (N, SEQ, 4)
    X_val         = np.load(val_path)
    X_train_noisy = add_noise(X_train_clean, NOISE_STD)

    n_features = X_train_clean.shape[2]
    print(f"Loaded  train={len(X_train_clean)}  val={len(X_val)}  "
          f"features={n_features}  seq_len={X_train_clean.shape[1]}")
    print(f"Model   hidden={HIDDEN_SIZE}  latent={LATENT_DIM}")
    print(f"Train   epochs={EPOCHS}  lr={LEARNING_RATE}  batch={BATCH_SIZE}\n")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # 2. Data loaders  (noisy input → clean target = denoising autoencoder)
    train_loader = make_loader(X_train_noisy, X_train_clean, BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(X_val,         X_val,         BATCH_SIZE, shuffle=False)

    # 3. Model, loss, optimiser
    model     = LSTMAutoencoder(n_features, HIDDEN_SIZE, LATENT_DIM)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # ReduceLROnPlateau: only halves LR when val loss doesn't improve for
    # 'patience' epochs — adaptive, won't freeze learning on a fixed schedule
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )

    best_val = float("inf")

    # 4. Training loop
    for epoch in range(EPOCHS):
        # — train —
        model.train()
        t_loss = 0.0
        for x_in, x_tgt in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x_in), x_tgt)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)

        # — validate —
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for x_in, x_tgt in val_loader:
                v_loss += criterion(model(x_in), x_tgt).item()
        v_loss /= len(val_loader)

        scheduler.step(v_loss)   # ReduceLROnPlateau watches val loss

        # save best checkpoint
        is_best = v_loss < best_val
        if is_best:
            best_val = v_loss
            torch.save(model.state_dict(), MODEL_PATH)

        if (epoch + 1) % 10 == 0:
            marker = " ← best" if is_best else ""
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                  f"Train: {t_loss:.6f} | Val: {v_loss:.6f}{marker}")

    print(f"\nBest model saved → {MODEL_PATH}  (val_loss={best_val:.6f})")

    # 5. Compute anomaly threshold on val set using best model
    print("\nComputing anomaly threshold on validation set …")
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    errors = recon_error(model, X_val)

    mean_e, std_e = np.mean(errors), np.std(errors)
    # Use 4σ with a floor — fixed 0-100 scaling makes val errors very small,
    # but live data has natural drift (network jitter, memory GC, etc.)
    # Real anomalies produce loss >> 0.1 (evaluate.py confirms), so 0.05 is safe.
    threshold     = max(mean_e + 4 * std_e, 0.05)

    print(f"  Val MAE  mean={mean_e:.6f}  std={std_e:.6f}")
    print(f"  Threshold (max of mean+4s, 0.02) = {threshold:.6f}")

    with open(THRESHOLD_PATH, "w") as f:
        f.write(str(threshold))
    print(f"  Threshold saved → {THRESHOLD_PATH}")
    print("\nDone ✓  All artefacts ready for deployment.")

if __name__ == "__main__":
    train()
