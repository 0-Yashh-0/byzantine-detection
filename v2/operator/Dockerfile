FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY v2/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Operator Code
COPY v2/operator/main.py .

# Copy ML Artifacts (Structure must match script expectations)
# Script expects: v2/ml/lstm_model.pth
RUN mkdir -p v2/ml
COPY v2/ml/lstm_model.pth v2/ml/
COPY v2/ml/scaler.pkl v2/ml/
COPY v2/ml/threshold.txt v2/ml/

# Debug: Verify files exist during build
RUN ls -R /app/v2/ml

# Environment Variables
ENV MODEL_PATH=v2/ml/lstm_model.pth
ENV SCALER_PATH=v2/ml/scaler.pkl
ENV THRESHOLD_PATH=v2/ml/threshold.txt
# Disable buffering for logs
ENV PYTHONUNBUFFERED=1

# Run Kopf
CMD ["kopf", "run", "/app/main.py", "--log-format=plain", "--loglevel=warning"]
