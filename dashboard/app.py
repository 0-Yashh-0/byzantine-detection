import streamlit as st
import json
import time
import pandas as pd
import os

st.set_page_config(
    page_title="Byzantine Architecture Sentinel",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Byzantine Architecture Sentinel")
st.markdown("### Real-time Kubernetes Edge Cluster Defense System")

STATE_FILE = "dashboard/state.json"

def load_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

# Auto-refresh mechanism
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

# Layout
col1, col2 = st.columns([2, 1])

state = load_state()

with col1:
    st.subheader("Cluster Topology & Health")
    
    if not state:
        st.warning("Waiting for controller data...")
    else:
        # Dynamic columns for nodes
        cols = st.columns(len(state))
        
        for idx, (node_name, metrics) in enumerate(state.items()):
            status = metrics.get('status', 'Unknown')
            cpu = metrics.get('cpu', 0)
            score = metrics.get('score', 100) # Default to 100 if missing
            
            with cols[idx]:
                if status == "Banned":
                     st.error(f"**{node_name}**\n\n⛔ BANNED")
                elif status == "Probation":
                     st.warning(f"**{node_name}**\n\n⚠️ PROBATION")
                else:
                     st.success(f"**{node_name}**\n\n✅ HEALTHY")
                
                st.metric("CPU Usage", f"{cpu:.1f}%")
                st.metric("Trust Score", f"{score:.1f}")
                st.progress(min(100, max(0, int(score))))

with col2:
    st.subheader("Live Anomalies")
    if state:
        bad_nodes = [n for n, m in state.items() if m['status'] in ['Banned', 'Probation']]
        if bad_nodes:
            for node in bad_nodes:
                status = state[node]['status']
                if status == "Banned":
                    st.error(f"⛔ **{node}** is BANNED (Cordoned/Drained).")
                else:
                    st.warning(f"⚠️ **{node}** is on PROBATION (Tainted).")
        else:
            st.info("All nodes are trusted.")
            
    st.markdown("---")
    st.caption("System Status: **Active**")
    st.caption("Policy: **Trust-Based Enforcement**")

# Charting (Mock history from state for now, ideally strictly from CSV)
st.subheader("Live Telemetry")
if state:
    data = []
    for node, metrics in state.items():
        data.append({"Node": node, "CPU": metrics.get('cpu', 0)})
    
    df = pd.DataFrame(data)
    st.bar_chart(df.set_index("Node"))

# Auto-rerun
time.sleep(1)
st.rerun()
