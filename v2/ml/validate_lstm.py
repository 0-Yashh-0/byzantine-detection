import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# Configuration
MODEL_PATH = "ml/lstm_model.pth"
SCALER_PATH = "ml/scaler.pkl"
THRESHOLD_PATH = "ml/threshold.txt"
SEQUENCE_LENGTH = 10
HIDDEN_SIZE = 32
LATENT_DIM = 8

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, latent_dim=8):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden2latent = nn.Linear(hidden_dim, latent_dim)
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        _, (hidden, _) = self.encoder(x)
        latent = self.hidden2latent(hidden[-1])
        hidden_decoded = self.latent2hidden(latent)
        hidden_repeated = hidden_decoded.unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(hidden_repeated)
        output = self.output_layer(decoded)
        return output

def validate():
    print("--- LSTM Validation Mode ---")
    
    # Load Artifacts
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Model or Scaler not found!")
        return

    scaler = joblib.load(SCALER_PATH)
    
    with open(THRESHOLD_PATH, "r") as f:
        THRESHOLD = float(f.read().strip())
    print(f"Loaded Threshold: {THRESHOLD:.4f}")
    
    # 1. Generate NORMAL Sequence (Idle)
    # CPU~0, Mem~18, Net~0
    normal_data = np.zeros((SEQUENCE_LENGTH, 4))
    normal_data[:, 1] = 18.0 # Memory
    # Add small noise
    normal_data += np.random.normal(0, 0.1, normal_data.shape)
    
    # 2. Generate ANOMALY Sequence (Slow Memory Leak)
    # Memory increases from 18 to 28 over 10 steps
    leak_data = np.zeros((SEQUENCE_LENGTH, 4))
    leak_data[:, 1] = np.linspace(18, 50, SEQUENCE_LENGTH) # RAM spiking
    
    # Scale Data
    normal_scaled = scaler.transform(normal_data)
    leak_scaled = scaler.transform(leak_data)
    
    # Convert to Tensor (Add batch dim)
    normal_tensor = torch.FloatTensor(normal_scaled).unsqueeze(0)
    leak_tensor = torch.FloatTensor(leak_scaled).unsqueeze(0)
    
    # Load Model
    model = LSTMAutoencoder(input_dim=4, hidden_dim=HIDDEN_SIZE, latent_dim=LATENT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    # Prediction
    with torch.no_grad():
        # Normal
        rec_normal = model(normal_tensor)
        loss_normal = torch.mean(torch.abs(rec_normal - normal_tensor)).item()
        
        # Leak
        rec_leak = model(leak_tensor)
        loss_leak = torch.mean(torch.abs(rec_leak - leak_tensor)).item()
        
    print("\nTest Results:")
    print(f"[Normal Sequence] Reconstruction Error: {loss_normal:.4f}")
    if loss_normal > THRESHOLD:
        print("  -> FALSE POSITIVE! (Flagged Normal as Anomaly)")
    else:
        print("  -> Correctly classified as Normal.")
        
    print(f"\n[Slow Leak Sequence] Reconstruction Error: {loss_leak:.4f}")
    if loss_leak > THRESHOLD:
        print("  -> SUCCESS! Anomaly Detected.")
    else:
        print("  -> FAILURE! Leak went undetected.")

if __name__ == "__main__":
    validate()
