import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import plotly.express as px
import plotly.graph_objects as go

from catboost import Pool

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="DG Incident Prediction",
    layout="wide"
)

# =====================================================
# LOAD DATA AND MODEL
# =====================================================

@st.cache_data
def load_reference_data():
    return pd.read_csv("dg_incident_test.csv")


@st.cache_resource
def load_model():
    return joblib.load("dg_incident_model.pkl")


try:
    reference_df = load_reference_data()
    model = load_model()

except Exception as e:
    st.error(f"Error loading required files: {e}")
    st.stop()

# =====================================================
# HEADER
# =====================================================

st.title("Dangerous Goods Incident Prediction")

st.markdown(
    """
    AI-Powered Cargo Safety Risk Assessment System
    """
)

st.divider()

# =====================================================
# INPUT FORM
# =====================================================

with st.form("prediction_form"):

    st.subheader("Shipment and Operational Inputs")

    col1, col2, col3 = st.columns(3)

    # =================================================
    # COLUMN 1
    # =================================================

    with col1:

        st.markdown("### Cargo and Route")

        shc_code = st.selectbox(
            "SHC Code",
            sorted(reference_df["shc_code"].astype(str).unique())
        )

        origin_destination = st.selectbox(
            "Route",
            sorted(reference_df["origin_destination"].astype(str).unique())
        )

        dg_class = st.selectbox(
            "DG Class",
            sorted(reference_df["dg_class"].astype(float).unique())
        )

        packaging_type = st.selectbox(
            "Packaging Type",
            sorted(reference_df["packaging_type"].astype(str).unique())
        )

        weather_condition = st.selectbox(
            "Weather Condition",
            sorted(reference_df["weather_condition"].astype(str).unique())
        )

    # =================================================
    # COLUMN 2
    # =================================================

    with col2:

        st.markdown("### Environmental and Time")

        cargo_weight_kg = st.number_input(
            "Cargo Weight (kg)",
            min_value=0.0,
            value=15000.0
        )

        temperature_celsius = st.number_input(
            "Temperature (°C)",
            value=25.0
        )

        humidity_percentage = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0
        )

        shipment_hour = st.slider(
            "Shipment Hour",
            0,
            23,
            12
        )

        shipment_day_of_week = st.selectbox(
            "Shipment Day",
            options=[0, 1, 2, 3, 4, 5, 6],
            format_func=lambda x:
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x]
        )

        shipment_month = st.slider(
            "Shipment Month",
            1,
            12,
            5
        )

    # =================================================
    # COLUMN 3
    # =================================================

    with col3:

        st.markdown("### Compliance and History")

        handling_error_count = st.number_input(
            "Handling Error Count",
            min_value=0,
            value=0
        )

        previous_incident_count = st.number_input(
            "Previous Incident Count",
            min_value=0,
            value=0
        )

        safety_staff_count = st.number_input(
            "Safety Staff Count",
            min_value=0,
            value=10
        )

        doc_audit_result = st.radio(
            "Documentation Audit",
            options=[1, 0],
            format_func=lambda x:
            "Pass" if x == 1 else "Fail"
        )

    submit = st.form_submit_button(
        "Run Risk Assessment",
        use_container_width=True
    )

# =====================================================
# PREDICTION SECTION
# =====================================================

if submit:

    # =================================================
    # FEATURE ENGINEERING
    # =================================================

    is_critical_mishandling = 1 if (
        int(handling_error_count) > 7
    ) else 0

    is_untrusted_shipper = 1 if (
        int(previous_incident_count) > 5
    ) else 0

    climate_shock_risk = 1 if (
        str(origin_destination) == "OSL-JED"
        and float(temperature_celsius) > 30
    ) else 0

    gas_handling_risk = 1 if (
        round(float(dg_class), 1) == 2.1
        and int(handling_error_count) > 5
    ) else 0

    thermal_expansion_risk = 1 if (
        str(origin_destination) == "OSL-JED"
        and float(temperature_celsius) > 35
        and round(float(dg_class), 1) == 3.0
    ) else 0

    shipper_hazard_combo = 1 if (
        int(previous_incident_count) > 5
        and str(shc_code) == "CAO"
    ) else 0

    # =================================================
    # INPUT DATAFRAME
    # =================================================

    input_dict = {

        "shc_code": [str(shc_code)],

        "origin_destination": [str(origin_destination)],

        "dg_class": [float(dg_class)],

        "packaging_type": [str(packaging_type)],

        "handling_error_count": [int(handling_error_count)],

        "previous_incident_count": [
            int(previous_incident_count)
        ],

        "cargo_weight_kg": [float(cargo_weight_kg)],

        "temperature_celsius": [
            float(temperature_celsius)
        ],

        "humidity_percentage": [
            float(humidity_percentage)
        ],

        "weather_condition": [
            str(weather_condition)
        ],

        "safety_staff_count": [
            int(safety_staff_count)
        ],

        "doc_audit_result": [
            int(doc_audit_result)
        ],

        "shipment_hour": [
            int(shipment_hour)
        ],

        "shipment_day_of_week": [
            int(shipment_day_of_week)
        ],

        "shipment_month": [
            int(shipment_month)
        ],

        "is_critical_mishandling": [
            int(is_critical_mishandling)
        ],

        "is_untrusted_shipper": [
            int(is_untrusted_shipper)
        ],

        "climate_shock_risk": [
            int(climate_shock_risk)
        ],

        "gas_handling_risk": [
            int(gas_handling_risk)
        ],

        "thermal_expansion_risk": [
            int(thermal_expansion_risk)
        ],

        "shipper_hazard_combo": [
            int(shipper_hazard_combo)
        ],
    }

    input_data = pd.DataFrame(input_dict)

    # =================================================
    # CATBOOST POOL
    # =================================================

    prediction_pool = Pool(
        data=input_data,
        cat_features=[
            "shc_code",
            "origin_destination",
            "packaging_type",
            "weather_condition"
        ]
    )

    # =================================================
    # PREDICTION
    # =================================================

    try:

        raw_prediction = model.predict(
            prediction_pool
        )[0]

        risk_score = float(
            np.clip(raw_prediction, 0.0, 1.0)
        )

        # =================================================
        # RISK LEVEL
        # =================================================

        if risk_score >= 0.75:
            risk_level = "HIGH"

        elif risk_score >= 0.45:
            risk_level = "MEDIUM"

        else:
            risk_level = "LOW"

        # =================================================
        # METRICS
        # =================================================

        st.divider()

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric(
                "Risk Score",
                f"{risk_score:.1%}"
            )

        with metric2:
            st.metric(
                "Risk Level",
                risk_level
            )

        with metric3:
            st.metric(
                "DG Class",
                dg_class
            )

        # =================================================
        # RISK GAUGE
        # =================================================

        st.subheader("Overall Risk Magnitude")

        gauge_fig = go.Figure(
            go.Indicator(

                mode="gauge+number",

                value=risk_score * 100,

                number={
                    'suffix': "%"
                },

                gauge={

                    'axis': {
                        'range': [0, 100]
                    },

                    'bar': {
                        'thickness': 0.3
                    },

                    'steps': [

                        {
                            'range': [0, 45],
                            'color': "#A8E6A3"
                        },

                        {
                            'range': [45, 75],
                            'color': "#FFD580"
                        },

                        {
                            'range': [75, 100],
                            'color': "#FF8A80"
                        }
                    ]
                }
            )
        )

        gauge_fig.update_layout(

            height=350,

            margin=dict(
                l=20,
                r=20,
                t=40,
                b=20
            )
        )

        st.plotly_chart(
            gauge_fig,
            use_container_width=True
        )

        # =================================================
        # FEATURE CONTRIBUTION ANALYSIS
        # =================================================

        st.divider()

        st.subheader(
            "Feature Contribution Analysis"
        )

        # SHAP EXPLAINER

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(
            prediction_pool
        )

        # =================================================
        # CREATE CONTRIBUTION DATAFRAME
        # =================================================

        contribution_df = pd.DataFrame({

            "Feature": input_data.columns,

            "Contribution": np.abs(shap_values[0])
        })

        # =================================================
        # CONVERT TO PERCENTAGE
        # =================================================

        contribution_df["ContributionPercent"] = (
            contribution_df["Contribution"] * 100
        )

        # =================================================
        # REMOVE VERY SMALL CONTRIBUTIONS
        # ONLY KEEP > 0.5%
        # =================================================

        contribution_df = contribution_df[
            contribution_df["ContributionPercent"] > 0.5
        ]

        # =================================================
        # SORT
        # =================================================

        contribution_df = contribution_df.sort_values(
            by="ContributionPercent",
            ascending=True
        )

        # =================================================
        # SHOW GRAPH ONLY IF FEATURES EXIST
        # =================================================

        if len(contribution_df) > 0:

            contribution_fig = px.bar(

                contribution_df,

                x="ContributionPercent",

                y="Feature",

                orientation="h",

                height=600,

                title="Feature Influence on Current Prediction",

                text="ContributionPercent"
            )

            contribution_fig.update_traces(

                texttemplate='%{text:.2f}%',

                textposition="outside"
            )

            contribution_fig.update_layout(

                title_font_size=22,

                font_size=14,

                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                ),

                xaxis_title="Contribution (%)",

                yaxis_title="",

                template="simple_white"
            )

            st.plotly_chart(
                contribution_fig,
                use_container_width=True
            )

            # =================================================
            # DETAILED TABLE
            # =================================================

            st.subheader(
                "Detailed Feature Influence"
            )

            display_df = contribution_df.sort_values(
                by="ContributionPercent",
                ascending=False
            )[
                ["Feature", "ContributionPercent"]
            ]

            display_df["ContributionPercent"] = (
                display_df["ContributionPercent"]
                .round(2)
                .astype(str) + "%"
            )

            display_df.columns = [
                "Feature",
                "Contribution"
            ]

            st.dataframe(
                display_df,
                use_container_width=True
            )

        else:

            st.info(
                "No significant feature contributions detected."
            )

        # =================================================
        # OPERATIONAL AUDIT LOGS
        # =================================================

        st.divider()

        st.subheader(
            "Operational Risk Audit"
        )

        reasons = []

        if is_critical_mishandling:

            reasons.append(
                "Critical mishandling threshold exceeded."
            )

        if gas_handling_risk:

            reasons.append(
                "Gas handling operational instability detected."
            )

        if thermal_expansion_risk:

            reasons.append(
                "Thermal expansion hazard detected."
            )

        if shipper_hazard_combo:

            reasons.append(
                "High-risk shipper and CAO combination detected."
            )

        if climate_shock_risk:

            reasons.append(
                "Climate shock route-temperature condition detected."
            )

        if is_untrusted_shipper:

            reasons.append(
                "Historical shipper incidents exceed threshold."
            )

        if len(reasons) == 0:

            st.success(
                "No major operational risk indicators detected."
            )

        else:

            for r in reasons:
                st.warning(r)

    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Cargo Safety Analytics Dashboard | CatBoostRegressor"
)
