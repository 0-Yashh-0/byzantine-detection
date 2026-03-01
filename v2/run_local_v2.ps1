$env:PROMETHEUS_URL="http://localhost:9090"
$env:MODEL_PATH="ml/lstm_model.pth"
$env:SCALER_PATH="ml/scaler.pkl"
$env:THRESHOLD_PATH="ml/threshold.txt"

Write-Host "Starting Byzantine Operator Locally..."
Write-Host "Ensure you have port-forwarded Prometheus: kubectl port-forward svc/prometheus-service 9090:9090"

# Install requirements if needed (optional check)
# pip install -r requirements.txt

# Run Kopf using venv
# Run Kopf using venv
$env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python -m kopf run operator/main.py --verbose
