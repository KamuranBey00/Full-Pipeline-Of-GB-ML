"""
ui/streamlit_app.py
===================
Phase 9 — a thin web UI over the deployed API.

The UI holds NO model code: it only sends raw sensor readings to the FastAPI
service and renders the answer. One model, one place (the API) — the UI can be
redeployed or replaced without touching the ML side.

Run locally:  streamlit run ui/streamlit_app.py
Point it at a local API:  API_URL=http://127.0.0.1:8000 streamlit run ui/streamlit_app.py
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "https://predictive-maintenance-api-wg6l.onrender.com")
TIMEOUT = 90   # Render free tier sleeps when idle; the first call wakes it (~1 min)

st.set_page_config(page_title="Predictive Maintenance", page_icon="🛠️")
st.title("🛠️ Machine Failure Prediction")
st.caption("Enter live sensor readings from a milling machine. A Gradient Boosting "
           "model (AI4I 2020 dataset) estimates the probability of failure.")

# Slider bounds mirror the API's validation ranges (api/schemas.py);
# defaults are a typical healthy machine.
with st.sidebar:
    st.header("Sensor readings")
    reading = {
        "type": st.selectbox("Product quality (Type)", ["L", "M", "H"],
                             help="L = low, M = medium, H = high quality variant"),
        "air_temperature": st.slider("Air temperature [K]", 290.0, 310.0, 298.1, 0.1),
        "process_temperature": st.slider("Process temperature [K]", 300.0, 320.0, 308.6, 0.1),
        "rotational_speed": st.slider("Rotational speed [rpm]", 1000, 3500, 1551, 1),
        "torque": st.slider("Torque [Nm]", 0.0, 100.0, 42.8, 0.1),
        "tool_wear": st.slider("Tool wear [min]", 0, 300, 108, 1),
    }
    predict = st.button("Predict", type="primary", use_container_width=True)

if predict:
    with st.spinner("Scoring… (first request may take ~1 min while the API wakes up)"):
        try:
            r = requests.post(f"{API_URL}/predict", json=reading, timeout=TIMEOUT)
            r.raise_for_status()
            result = r.json()
        except requests.RequestException as e:
            st.error(f"API request failed: {e}")
            st.stop()

    prob = result["failure_probability"]
    risk = result["risk_level"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Failure probability", f"{prob:.1%}")
    col2.metric("Risk level", risk.upper())
    col3.metric("Decision threshold", f"{result['threshold_used']:.1%}")

    if result["failure_predicted"]:
        st.error("⚠️ Failure predicted — schedule maintenance before the next run.")
    elif risk == "medium":
        st.warning("Elevated risk — keep this machine under observation.")
    else:
        st.success("✅ No failure expected.")
    st.progress(min(prob, 1.0))

    st.caption(f"The threshold is tuned low ({result['threshold_used']:.1%}) on purpose: "
               "a missed failure costs far more than an extra inspection, so the model "
               "is set to catch ≥85% of real failures.")
else:
    st.info("Set the sensor values in the sidebar and press **Predict**.  \n"
            "Try a risky machine: torque **65 Nm**, tool wear **220 min**, speed **1300 rpm**.")

st.divider()
st.caption(f"API: [{API_URL}/docs]({API_URL}/docs)")
