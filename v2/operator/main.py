import kopf
import kubernetes
import torch
import torch.nn as nn
import joblib
import numpy as np
import os
import requests
import time
from kubernetes import client, config

# --- MONITORING CONFIG ---
# In-Cluster: http://prometheus-service.default.svc:9090
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus-service.default.svc:9090")
#MODEL_PATH = os.getenv("MODEL_PATH", "v2/ml/lstm_model.pth")
# SCALER_PATH = os.getenv("SCALER_PATH", "v2/ml/scaler.pkl")
# THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "v2/ml/threshold.txt")
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(_PROJECT_ROOT, "v2", "ml", "lstm_model.pth"))
SCALER_PATH = os.getenv("SCALER_PATH", os.path.join(_PROJECT_ROOT, "v2", "ml", "scaler.pkl"))
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", os.path.join(_PROJECT_ROOT, "v2", "ml", "threshold.txt"))

# --- MODEL DEFINITION ---
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

# --- TRUST TRACKER ---
class TrustTracker:
    def __init__(self):
        self.scores = {}  # {node_name: float}
        self.status = {}  # {node_name: str}

    def get_score(self, node):
        return self.scores.get(node, 100.0)

    def update(self, node, is_anomaly):
        current = self.get_score(node)
        if is_anomaly:
            new_score = max(0.0, current - 5.0) # Decay
        else:
            new_score = min(100.0, current + 3.0) # Reward
        
        self.scores[node] = new_score
        return new_score

# --- GLOBAL STATE ---
MODEL = None
SCALER = None
THRESHOLD = 0.5
TRUST_SYSTEM = TrustTracker()
SEQUENCE_LENGTH = 20
V1_API = None
HTTP_SESSION = requests.Session()
# Rolling buffer: {node_name: deque of [cpu, mem, net_in, net_out] rows}
from collections import deque
NODE_BUFFERS: dict = {}

def get_node_ip(node_name):
    """Get the InternalIP of a node from Kubernetes API."""
    try:
        global V1_API
        if V1_API is None: return None
        node = V1_API.read_node(node_name)
        for addr in node.status.addresses:
            if addr.type == "InternalIP":
                return addr.address
    except Exception:
        pass
    return None

def fetch_prometheus_data(node_name: str):
    """
    Fetch ONE instant snapshot for this node and append to its rolling buffer.
    Returns np.array shape (SEQUENCE_LENGTH, 4) once buffer is full, else None.
    Uses instant queries (same as collect_data.py) to match training data format.
    Feature order: [cpu, mem, net_in, net_out]
    """
    node_ip = get_node_ip(node_name)
    if node_ip is None:
        return None

    instant_queries = {
        "cpu":     '100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[30s])) * 100)',
        "mem":     '100 * (1 - ((node_memory_MemFree_bytes + node_memory_Buffers_bytes + node_memory_Cached_bytes) / node_memory_MemTotal_bytes))',
        "net_in":  'sum by (instance) (rate(node_network_receive_bytes_total[30s]))',
        "net_out": 'sum by (instance) (rate(node_network_transmit_bytes_total[30s]))',
    }

    def instant_query(q):
        resp = HTTP_SESSION.get(f"{PROMETHEUS_URL}/api/v1/query",
                                params={"query": q}, timeout=3.0)
        for item in resp.json()["data"]["result"]:
            inst = item["metric"].get("instance", "")
            if inst.startswith(node_ip + ":"):
                return float(item["value"][1])
        return None

    try:
        cpu     = instant_query(instant_queries["cpu"])
        mem     = instant_query(instant_queries["mem"])
        net_in  = instant_query(instant_queries["net_in"])
        net_out = instant_query(instant_queries["net_out"])

        if any(v is None for v in [cpu, mem, net_in, net_out]):
            return None

        # Append fresh measurement to rolling buffer
        if node_name not in NODE_BUFFERS:
            NODE_BUFFERS[node_name] = deque(maxlen=SEQUENCE_LENGTH)
        NODE_BUFFERS[node_name].append([cpu, mem, net_in, net_out])

        if len(NODE_BUFFERS[node_name]) < SEQUENCE_LENGTH:
            return None  # not enough history yet — wait

        return np.array(list(NODE_BUFFERS[node_name]))

    except Exception:
        return None


# --- KOPF HANDLERS ---
@kopf.on.startup()
def configure(settings: kopf.OperatorSettings, **_):
    global MODEL, SCALER, THRESHOLD, V1_API
    print("--- 🛡️ Starting Byzantine Defense Operator v2 ---", flush=True)
    try:
        config.load_incluster_config()
    except:
        config.load_kube_config()
        
    V1_API = client.CoreV1Api()
        
    try:
        if os.path.exists(MODEL_PATH):
            MODEL = LSTMAutoencoder(input_dim=4, hidden_dim=64, latent_dim=16)
            MODEL.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
            MODEL.eval()
        
        if os.path.exists(SCALER_PATH):
            SCALER = joblib.load(SCALER_PATH)

        if os.path.exists(THRESHOLD_PATH):
            with open(THRESHOLD_PATH, 'r') as f:
                THRESHOLD = float(f.read().strip())
        
        print(f"✅ Loaded Brain (Threshold: {THRESHOLD:.4f})", flush=True)
    except Exception as e:
        print(f"❌ Artifact Config Error: {e}", flush=True)

@kopf.timer('security.example.com', 'v1', 'byzantinepolicies', interval=10.0)
def reconcile(spec, name, logger, **kwargs):
    if MODEL is None or SCALER is None or V1_API is None:
        return  # Silently wait — startup not done yet

    limit_score = 40.0

    for node in V1_API.list_node().items:
        n_name = node.metadata.name
        if "control-plane" in n_name:
            continue

        raw_seq = fetch_prometheus_data(n_name)
        if raw_seq is None:
            continue

        if SCALER is None or MODEL is None:
            continue

        seq_scaled = SCALER.transform(raw_seq)
        seq_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0)  # (1, 10, 4)

        with torch.no_grad():
            reconstruction = MODEL(seq_tensor)
            loss = torch.mean(torch.abs(reconstruction - seq_tensor)).item()

        is_anomaly = loss > THRESHOLD
        score = TRUST_SYSTEM.update(n_name, is_anomaly)

        # ── Only print the per-node result ──────────────────────────────────
        status = "⚠️  ANOMALY" if is_anomaly else "✅ normal"
        log_line = f"[{n_name}] Loss={loss:.4f} | Trust={score:5.1f} | {status}"
        print(log_line, flush=True)

        # Log metrics to file for evaluation checking
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "operator_metrics.log")
        safe_line = log_line.encode("ascii", errors="replace").decode("ascii")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {log_line}\n")

        if score < limit_score:
            cordon_node(n_name, logger)
        elif spec.get('autoRemediate', True) and score > 60:
            uncordon_node(n_name, logger)

def cordon_node(node_name, logger):
    global V1_API
    node = V1_API.read_node(node_name)
    if not node.spec.unschedulable:
        V1_API.patch_node(node_name, {"spec": {"unschedulable": True}})
        print(f"🚫 CORDONED  {node_name}  (trust too low)", flush=True)

def uncordon_node(node_name, logger):
    global V1_API
    node = V1_API.read_node(node_name)
    if node.spec.unschedulable:
        V1_API.patch_node(node_name, {"spec": {"unschedulable": False}})
        print(f"✅ UNCORDONED {node_name}  (trust recovered)", flush=True)
