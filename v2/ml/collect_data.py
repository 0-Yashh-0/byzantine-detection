import time
import csv
import os
import requests
import pandas as pd
from datetime import datetime

# Configuration
PROMETHEUS_URL = "http://localhost:9090"
OUTPUT_FILE = "dataset/CPUbaseline.csv" 
DURATION_SECONDS = 3600 # time for testing
INTERVAL_SECONDS = 5

# PromQL Queries
QUERIES = {
    "cpu": '100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[30s])) * 100)',
    "mem": '100 * (1 - ((node_memory_MemFree_bytes + node_memory_Buffers_bytes + node_memory_Cached_bytes) / node_memory_MemTotal_bytes))',
    "net_in": 'sum by (instance) (rate(node_network_receive_bytes_total[30s]))',
    "net_out": 'sum by (instance) (rate(node_network_transmit_bytes_total[30s]))'
}

def fetch_metric(query):
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': query})
        data = response.json()
        if data['status'] == 'success' and len(data['data']['result']) > 0:
            # Return dict {instance: value}
            results = {}
            for res in data['data']['result']:
                # Instance might be "10.244.1.3:9100" or similar
                # We want to map it to node name if possible, or just keep unique ID
                instance = res['metric'].get('instance', 'unknown')
                value = float(res['value'][1])
                results[instance] = value
            return results
    except Exception as e:
        print(f"Error fetching {query}: {e}")
    return {}

def collect():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Initialize CSV if not exists
    file_exists = os.path.exists(OUTPUT_FILE)
    
    print(f"Starting data collection for {DURATION_SECONDS} seconds...")
    print(f"Saving to {OUTPUT_FILE}")
    
    start_time = time.time()
    
    with open(OUTPUT_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "instance", "cpu", "mem", "net_in", "net_out"])
            
        while time.time() - start_time < DURATION_SECONDS:
            step_start = time.time()
            
            # Fetch all metrics
            cpu_data = fetch_metric(QUERIES['cpu'])
            mem_data = fetch_metric(QUERIES['mem'])
            net_in_data = fetch_metric(QUERIES['net_in'])
            net_out_data = fetch_metric(QUERIES['net_out'])
            
            # Aggregate by instance (intersection of all instances found)
            instances = set(cpu_data.keys()) | set(mem_data.keys())
            
            timestamp = datetime.now().isoformat()
            
            for inst in instances:
                row = [
                    timestamp,
                    inst,
                    cpu_data.get(inst, 0.0),
                    mem_data.get(inst, 0.0),
                    net_in_data.get(inst, 0.0),
                    net_out_data.get(inst, 0.0)
                ]
                writer.writerow(row)
                print(f"Logged {inst}: CPU={row[2]:.1f}% Mem={row[3]:.1f}%")
            
            f.flush()  # Force write to disk every tick
            
            # Wait for next interval
            elapsed = time.time() - step_start
            sleep_time = max(0.0, INTERVAL_SECONDS - elapsed)
            time.sleep(sleep_time)

if __name__ == "__main__":
    collect()
