import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from catboost import Pool

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Dangerous Goods Incident Prediction",
    page_icon="⚠️",
    layout="wide"
)

# =====================================================
# LOAD DATA & MODEL
# =====================================================
@st.cache_data
def load_reference_data():
    # Used to populate dropdowns from the test set
    return pd.read_csv("dg_incident_test.csv")

@st.cache_resource
def load_model():
    # Loads the CatBoostRegressor model
    return joblib.load("dg_incident_model.pkl")

try:
    reference_df = load_reference_data()
    model = load_model()
except Exception as e:
    st.error(f"Error loading required files: {e}")
    st.stop()

# =====================================================
# UI HEADER
# =====================================================
st.title("⚠️ Dangerous Goods Incident Prediction")
st.markdown("Calculate shipment risk magnitude using operational and cargo parameters.")
st.divider()

# =====================================================
# INPUT FORM
# =====================================================
with st.form("prediction_form"):
    st.subheader("📦 Shipment Parameters")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        shc_code = st.selectbox("SHC Code", sorted(reference_df['shc_code'].unique()))
        origin_destination = st.selectbox("Origin-Destination", sorted(reference_df['origin_destination'].unique()))
        dg_class = st.selectbox("DG Class", sorted(reference_df['dg_class'].astype(float).unique()))
        packaging_type = st.selectbox("Packaging Type", sorted(reference_df['packaging_type'].unique()))
        weather_condition = st.selectbox("Weather Condition", sorted(reference_df['weather_condition'].unique()))

    with col2:
        cargo_weight_kg = st.number_input("Cargo Weight (kg)", min_value=0.0, value=15000.0)
        temperature_celsius = st.number_input("Temperature (°C)", value=25.0)
        humidity_percentage = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
        safety_staff_count = st.number_input("Safety Staff Count", min_value=0, value=10)
        doc_audit_result = st.selectbox("Document Audit", [0, 1], format_func=lambda x: "Pass" if x == 1 else "Fail")

    with col3:
        handling_error_count = st.number_input("Handling Error Count", min_value=0, value=0)
        previous_incident_count = st.number_input("Previous Incident Count", min_value=0, value=0)
        

    submit = st.form_submit_button("🚀 Run Risk Assessment")

# =====================================================
# PREDICTION LOGIC
# =====================================================
if submit:
    # 1. Automatic Temporal Feature Extraction (REQUIRED BY MODEL)
    now = datetime.now()
    shipment_hour = now.hour
    shipment_day_of_week = now.weekday() # 0=Mon, 6=Sun
    shipment_month = now.month

    # 2. Replicate Feature Engineering Logic
    is_critical_mishandling = 1 if handling_error_count > 7 else 0
    climate_shock_risk = 1 if (origin_destination == 'OSL-JED' and temperature_celsius > 30) else 0
    is_untrusted_shipper = 1 if previous_incident_count > 5 else 0

    gas_handling_risk = 1 if (dg_class == 2.1 and handling_error_count > 5) else 0
    thermal_expansion_risk = 1 if (origin_destination == 'OSL-JED' and temperature_celsius > 35 and dg_class == 3.0) else 0
    shipper_hazard_combo = 1 if (previous_incident_count > 5 and shc_code == 'CAO') else 0

    # 3. Construct Dataframe with ALL 21 features in EXACT order
    input_dict = {
        'shc_code': [str(shc_code)],
        'origin_destination': [str(origin_destination)],
        'dg_class': [float(dg_class)],
        'packaging_type': [str(packaging_type)],
        'handling_error_count': [int(handling_error_count)],
        'previous_incident_count': [int(previous_incident_count)],
        'cargo_weight_kg': [float(cargo_weight_kg)],
        'temperature_celsius': [float(temperature_celsius)],
        'humidity_percentage': [float(humidity_percentage)],
        'weather_condition': [str(weather_condition)],
        'safety_staff_count': [int(safety_staff_count)],
        'doc_audit_result': [int(doc_audit_result)],
        'shipment_hour': [int(shipment_hour)],
        'shipment_day_of_week': [int(shipment_day_of_week)],
        'shipment_month': [int(shipment_month)],
        'is_critical_mishandling': [int(is_critical_mishandling)],
        'climate_shock_risk': [int(climate_shock_risk)],
        'is_untrusted_shipper': [int(is_untrusted_shipper)],
        'gas_handling_risk': [int(gas_handling_risk)],
        'thermal_expansion_risk': [int(thermal_expansion_risk)],
        'shipper_hazard_combo': [int(shipper_hazard_combo)]
    }
    
    input_data = pd.DataFrame(input_dict)

    try:
        # Predict Risk Score using Regressor
        risk_score = model.predict(input_data)[0]
        
        # Clip value for progress bar safety
        display_score = max(0.0, min(float(risk_score), 1.0))

        st.divider()
        st.subheader("🎯 Prediction Result")

        m1, m2 = st.columns([1, 2])
        
        with m1:
            if risk_score >= 0.7:
                st.error("🚨 **HIGH RISK**")
            elif risk_score >= 0.4:
                st.warning("⚠️ **MEDIUM RISK**")
            else:
                st.success("✅ **LOW RISK**")
            
            st.metric("Predicted Risk Score", f"{risk_score:.2%}")

        with m2:
            st.write("Risk Magnitude")
            st.progress(display_score)
            
        # UI Reason Codes
        st.write("### 🔍 Risk Reason Codes")
        reasons = []
        if handling_error_count > 7: reasons.append("CRITICAL MISHANDLING: High error count detected.")
        if dg_class in [2.1, 3.0, 4.0]: reasons.append(f"VOLATILE COMMODITY: Class {dg_class} high hazard.")
        if climate_shock_risk: reasons.append("CLIMATE SHOCK: Route/Temperature expansion risk.")
        if is_untrusted_shipper: reasons.append("SHIPPER HISTORY: Previous safety violations.")
        if gas_handling_risk: reasons.append("COMPOUND RISK: Flammable gas mishandling.")
        
        if reasons:
            for r in reasons:
                st.info(r)
        else:
            st.write("No critical risk combinations triggered.")

        with st.expander("View Full Model Input (21 Features)"):
            st.dataframe(input_data)

    except Exception as e:
        st.error(f"**Prediction Error:** {e}")
        st.info("Note: The model expects 21 features. Check if 'shipment_hour', 'shipment_day_of_week', and 'shipment_month' were dropped in training.")

st.divider()
st.caption("DG Safety Prediction System | Powered by CatBoostRegressor")