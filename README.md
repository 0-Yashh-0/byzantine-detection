# 🛡️ Byzantine Fault Detection in Kubernetes Edge Clusters

> **An LSTM Autoencoder-based anomaly detection system that identifies and isolates Byzantine-faulty nodes in Kubernetes clusters in real-time.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-1.28+-326CE5.svg)](https://kubernetes.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org)

---

## 📖 Overview

In distributed edge computing environments, **Byzantine faults** — where nodes exhibit arbitrary, unpredictable, or malicious behavior — pose a critical threat to system reliability. Traditional threshold-based monitoring fails to detect these sophisticated anomalies because Byzantine nodes may not simply "crash" but instead exhibit subtle deviations in resource usage patterns.

This project implements a **Kubernetes Operator** that leverages an **LSTM Autoencoder** (deep learning) to learn the normal behavioral patterns of cluster nodes and automatically detect anomalies in real-time. When a node is identified as Byzantine, the operator **cordons** it (prevents new workloads from being scheduled) and **uncordons** it automatically once behavior returns to normal.

### Key Features

- **Deep Learning-based Detection** — LSTM Autoencoder learns temporal patterns across CPU, memory, and network metrics
- **Trust Score System** — Reputation-based scoring with configurable decay/reward rates
- **Automatic Remediation** — Autonomous cordoning/uncordoning of faulty nodes
- **Real-time Monitoring** — 10-second reconciliation loop via Prometheus metrics
- **Zero False Positives** — 0% FP rate on validation data, 98.3% anomaly detection rate
- **Production-ready Operator** — Built with [Kopf](https://kopf.readthedocs.io/) framework for Kubernetes

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                       │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │ Worker 1 │   │ Worker 2 │   │ Worker 3 │   │ Worker N │  │
│  │          │   │          │   │          │...│          │  │
│  │ node-    │   │ node-    │   │ node-    │   │ node-    │  │
│  │ exporter │   │ exporter │   │ exporter │   │ exporter │  │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘  │
│       │              │              │              │        │
│       └──────────────┴──────┬───────┴──────────────┘        │
│                             │                               │
│                    ┌────────▼────────┐                      │
│                    │   Prometheus    │                      │
│                    │  (Metrics DB)   │                      │
│                    └────────┬────────┘                      │
│                             │                               │
│               ┌─────────────▼─────────────┐                 │
│               │  Byzantine Operator       │                 │
│               │  ┌─────────────────────┐  │                 │
│               │  │  LSTM Autoencoder   │  │                 │
│               │  │  (Anomaly Detector) │  │                 │
│               │  └─────────┬───────────┘  │                 │
│               │  ┌─────────▼───────────┐  │                 │
│               │  │  Trust Tracker      │  │                 │
│               │  │  (Score Management) │  │                 │
│               │  └─────────┬───────────┘  │                 │
│               │            │ cordon/      │                 │
│               │            │ uncordon     │                 │
│               └────────────┼──────────────┘                 │
│                            │                                │
│                   Kubernetes API Server                     │
└─────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Data Collection** — Prometheus scrapes CPU, memory, network I/O metrics from `node-exporter` on each worker node every 15 seconds.
2. **Feature Extraction** — The operator queries Prometheus for 4 features: `cpu_usage`, `memory_usage`, `network_rx`, `network_tx`.
3. **Anomaly Detection** — A sliding window of 20 measurements is fed to the LSTM Autoencoder. The model attempts to reconstruct the input; high reconstruction error (MAE > threshold) indicates anomalous behavior.
4. **Trust Management** — Each node has a trust score (0–100). Anomalies decrease trust by 5; normal behavior increases it by 3.
5. **Enforcement** — Nodes with trust < 40 are **cordoned** (isolated). Nodes recovering above 60 are **uncordoned** (restored).

---

## 📂 Project Structure

```
v2/
├── ml/                          # Machine Learning Pipeline
│   ├── collect_data.py          # Baseline data collection from live cluster
│   ├── preprocess.py            # Data preprocessing & MinMaxScaler fitting
│   ├── train_lstm.py            # LSTM Autoencoder training & threshold calc
│   ├── evaluate.py              # Model evaluation with synthetic anomalies
│   ├── validate_lstm.py         # Additional validation utilities
│   ├── lstm_model.pth           # Trained model weights
│   ├── scaler.pkl               # Fitted MinMaxScaler (fixed domain ranges)
│   └── threshold.txt            # Anomaly detection threshold
│
├── operator/                    # Kubernetes Operator
│   ├── main.py                  # Core operator logic (Kopf framework)
│   ├── crd.yaml                 # Custom Resource Definition
│   ├── policy-sample.yaml       # ByzantinePolicy custom resource
│   ├── rbac.yaml                # RBAC permissions
│   ├── deployment.yaml          # Operator deployment manifest
│   └── Dockerfile               # Container image for in-cluster deployment
│
├── observability/               # Monitoring Stack
│   ├── prometheus-deployment.yaml
│   ├── prometheus-config.yaml
│   ├── node-exporter.yaml
│   └── node-exporter-service.yaml
│
├── simulation/                  # Attack Simulation
│   ├── stress-worker4.yaml      # CPU/memory stress pod for testing
│   └── stress-worker2.yaml      # Stress pod for second node
│
├── dataset/                     # Training & Evaluation Data
│   ├── v2_baseline_clean.csv    # Idle baseline data
│   ├── CPUbaseline.csv          # Current cluster baseline
│   ├── CPULightLoad.csv         # Light stress test data (~26% CPU)
│   ├── CPUMediumLoad.csv        # Medium stress test data (~54% CPU)
│   └── processed/               # Preprocessed numpy arrays
│
├── helping hands/               # Documentation & Guides
│   ├── HOW_TO_RUN.txt           # Full setup guide
│   ├── HOW_TO_ADD_LOAD.txt      # Load testing guide
│   ├── HOW_TO_TEST_OPERATOR.txt # Operator testing guide
│   └── FINAL_EVALUATION_GUIDE.txt  # Step-by-step eval guide
│
├── kind-multinode.yaml          # Kind cluster config (1 CP + 5 workers)
└── requirements.txt             # Python dependencies
```

---

## 🧠 ML Model

### LSTM Autoencoder Architecture

| Component         | Details                                                    |
|-------------------|------------------------------------------------------------|
| **Type**          | Sequence-to-Sequence Autoencoder                           |
| **Encoder**       | LSTM (input=4, hidden=64) → Linear (64→16)                 |
| **Decoder**       | Linear (16→64) → LSTM (64, 64) → Linear (64→4)             |
| **Input**         | Sliding window of 20 timesteps × 4 features                |
| **Loss**          | Mean Absolute Error (MAE)                                  |
| **Optimizer**     | Adam (lr=0.001) with ReduceLROnPlateau                     |
| **Training**      | 300 epochs, batch size 64, early stopping via LR scheduler |

### Features (4-dimensional input)

| Feature              | PromQL Query                                                       | Domain    →  Scaled  |
|----------------------|--------------------------------------------------------------------|----------------------|
| CPU Usage (%)        | `100 - avg(irate(node_cpu_seconds_total{mode="idle"}[30s])) * 100` | [0, 100]  → [0, 1]   |
| Memory Usage (%)     | `100 * (1 - (MemFree + Buffers + Cached) / MemTotal)`              | [0, 100]  → [0, 1]   |
| Network RX (bytes/s) | `sum(rate(node_network_receive_bytes_total[30s]))`                 | [0, 100K] → [0, 1]   |
| Network TX (bytes/s) | `sum(rate(node_network_transmit_bytes_total[30s]))`                | [0, 100K] → [0, 1]   |

### Normalization Strategy

The scaler uses **fixed domain ranges** rather than auto-fit from training data. This prevents scaler drift — a common production failure where the scaler is fitted on a narrow data range and fails when live values shift slightly.

### Model Performance

| Metric                     |              Value            |
|----------------------------|-------------------------------|
| Validation MAE             | 0.003269                      |
| Threshold                  | 0.05                          |
| False Positive Rate        | **0.00%** (0/1748 windows)    |
| Anomaly Detection Rate     | **98.33%** (295/300 windows)  |
| Normal node loss           | ~0.02                         |
| Anomalous node loss        | > 0.08                        |

---

## ⚙️ Trust Score System

The operator maintains a per-node trust score that provides temporal smoothing and prevents single-reading false alarms.

| Parameter                | Value    | Description                             |
|--------------------------|----------|-----------------------------------------|
| Initial Trust            | 100.0    | All nodes start fully trusted           |
| Anomaly Penalty          | -5.0     | Trust decreases per anomalous reading   |
| Normal Reward            | +3.0     | Trust increases per normal reading      |
| Cordon Threshold         | < 40.0   | Node is cordoned (isolated)             |
| Uncordon Threshold       | > 60.0   | Node is uncordoned (restored)           |
| Score Range              | [0, 100] | Clamped to valid range                  |

**Design rationale**: Asymmetric decay/reward means the system is **quick to suspect** (100→40 in ~2 min) but **cautious to forgive** (0→60 in ~3.3 min), preventing premature restoration of a potentially compromised node.

---

## 🛠️ Tech Stack

| Component                   | Technology                              |
|-----------------------------|-----------------------------------------|
| **Container Orchestration** | Kubernetes (via Kind)                   |
| **Operator Framework**      | Kopf (Python)                           |
| **Deep Learning**           | PyTorch (LSTM Autoencoder)              |
| **Metrics Collection**      | Prometheus + node-exporter              |
| **Data Processing**         | NumPy, Pandas, scikit-learn             |
| **Cluster Management**      | Kind (Kubernetes in Docker)             |
| **Platform**                | WSL2 on Windows / Linux                 |

---

## 📝 References

- Lamport, L., Shostak, R., & Pease, M. (1982). *The Byzantine Generals Problem*. ACM Transactions on Programming Languages and Systems.
- Malhotra, P., et al. (2016). *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection*. ICML Workshop.
- Kopf Framework: https://kopf.readthedocs.io/
- Prometheus: https://prometheus.io/

---