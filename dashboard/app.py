import sys
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="Oil & Gas AI",
    page_icon="🛢️",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🛢️ Oil & Gas AI")
st.subheader(
    "Industrial AI Data Intelligence & Predictive Maintenance Platform"
)

st.markdown(
    """
This platform demonstrates an AI-based industrial monitoring system
using synthetic oil & gas sensor data.

**Technologies:** Python · PostgreSQL · SQL · Machine Learning · SHAP ·
FastAPI · Streamlit · Docker
"""
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    csv_path = (
        BASE_DIR
        / "data"
        / "raw"
        / "industrial_sensor_data.csv"
    )

    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

    # -----------------------------------------------------
    # WELL ID
    # -----------------------------------------------------

    if "well_id" not in df.columns:

        possible_columns = [
            "well",
            "well_name",
            "wellid"
        ]

        found_column = None

        for column in possible_columns:

            if column in df.columns:
                found_column = column
                break

        if found_column:

            df["well_id"] = df[found_column].astype(str)

        else:

            df["well_id"] = [
                f"WELL-{(i % 10) + 1:03d}"
                for i in range(len(df))
            ]

    # -----------------------------------------------------
    # EQUIPMENT
    # -----------------------------------------------------

    if "equipment" not in df.columns:

        possible_equipment_columns = [
            "equipment_id",
            "equipment_name",
            "pump_id",
            "pump_name",
            "machine_id",
            "machine"
        ]

        found_equipment = None

        for column in possible_equipment_columns:

            if column in df.columns:
                found_equipment = column
                break

        if found_equipment:

            df["equipment"] = (
                df[found_equipment]
                .astype(str)
            )

        else:

            df["equipment"] = (
                "PUMP-"
                + df["well_id"]
                .astype(str)
                .str.replace(
                    "WELL-",
                    "",
                    regex=False
                )
            )

    # -----------------------------------------------------
    # FAILURE
    # -----------------------------------------------------

    if "failure" not in df.columns:

        possible_failure_columns = [
            "failure_flag",
            "failure_status",
            "target",
            "label"
        ]

        found_failure = None

        for column in possible_failure_columns:

            if column in df.columns:
                found_failure = column
                break

        if found_failure:

            df["failure"] = pd.to_numeric(
                df[found_failure],
                errors="coerce"
            ).fillna(0)

        else:

            df["failure"] = 0

    return df


df = load_data()


# =========================================================
# LOAD ML MODEL
# =========================================================

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "failure_model.pkl"
)

SCALER_PATH = (
    BASE_DIR
    / "models"
    / "scaler.pkl"
)


try:

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    MODEL_LOADED = True

except Exception as e:

    model = None
    scaler = None

    MODEL_LOADED = False

    st.error(
        f"Could not load ML model: {e}"
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ System")

st.sidebar.success(
    "AI System Online"
)

st.sidebar.metric(
    "Total Records",
    f"{len(df):,}"
)

st.sidebar.metric(
    "Wells",
    df["well_id"].nunique()
)

st.sidebar.metric(
    "Failures",
    int(df["failure"].sum())
)

st.sidebar.metric(
    "Failure Rate",
    f"{df['failure'].mean() * 100:.2f}%"
)


# =========================================================
# SYSTEM OVERVIEW
# =========================================================

st.header("📊 System Overview")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Sensor Records",
        f"{len(df):,}"
    )


with col2:

    st.metric(
        "Active Wells",
        df["well_id"].nunique()
    )


with col3:

    st.metric(
        "Failure Events",
        int(df["failure"].sum())
    )


with col4:

    st.metric(
        "Failure Rate",
        f"{df['failure'].mean() * 100:.2f}%"
    )


# =========================================================
# WELL RISK MONITORING
# =========================================================

st.header("🚨 Well Risk Monitoring")


well_risk = (
    df
    .groupby("well_id")
    .agg(
        total_readings=("failure", "count"),
        failure_count=("failure", "sum"),
        avg_temperature=("temperature_c", "mean"),
        avg_vibration=("vibration_mm_s", "mean"),
        avg_production=(
            "production_rate_bbl_day",
            "mean"
        )
    )
    .reset_index()
)


well_risk["failure_rate"] = (
    well_risk["failure_count"]
    / well_risk["total_readings"]
    * 100
)


def classify_risk(rate):

    if rate >= 2.5:
        return "HIGH"

    elif rate >= 1.5:
        return "MEDIUM"

    else:
        return "LOW"


well_risk["risk_level"] = (
    well_risk["failure_rate"]
    .apply(classify_risk)
)


well_risk = well_risk.sort_values(
    "failure_rate",
    ascending=False
)


display_df = well_risk.copy()

display_df["failure_rate"] = (
    display_df["failure_rate"]
    .round(2)
)

display_df["avg_temperature"] = (
    display_df["avg_temperature"]
    .round(2)
)

display_df["avg_vibration"] = (
    display_df["avg_vibration"]
    .round(2)
)

display_df["avg_production"] = (
    display_df["avg_production"]
    .round(2)
)


st.dataframe(
    display_df,
    width="stretch",
    hide_index=True
)


# =========================================================
# SELECTED WELL
# =========================================================

st.header("🔎 Selected Well Analysis")


selected_well = st.selectbox(
    "Select a well",
    sorted(df["well_id"].unique())
)


well_data = df[
    df["well_id"] == selected_well
].copy()


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Readings",
        len(well_data)
    )


with col2:

    st.metric(
        "Failures",
        int(well_data["failure"].sum())
    )


with col3:

    st.metric(
        "Avg Temperature",
        f"{well_data['temperature_c'].mean():.2f} °C"
    )


with col4:

    st.metric(
        "Avg Vibration",
        f"{well_data['vibration_mm_s'].mean():.2f} mm/s"
    )


# =========================================================
# LIVE FAILURE PREDICTION
# =========================================================

st.header("🤖 Live Equipment Failure Prediction")


st.write(
    "Enter industrial sensor values and let the AI model "
    "estimate failure risk."
)


col1, col2 = st.columns(2)


with col1:

    pressure = st.number_input(
        "Pressure (bar)",
        min_value=0.0,
        max_value=500.0,
        value=150.0
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=200.0,
        value=75.0
    )

    vibration = st.number_input(
        "Vibration (mm/s)",
        min_value=0.0,
        max_value=30.0,
        value=3.5
    )


with col2:

    flow_rate = st.number_input(
        "Flow Rate (bbl/day)",
        min_value=0.0,
        max_value=5000.0,
        value=1200.0
    )

    production_rate = st.number_input(
        "Production Rate (bbl/day)",
        min_value=0.0,
        max_value=5000.0,
        value=1170.0
    )

    pump_speed = st.number_input(
        "Pump Speed (RPM)",
        min_value=0.0,
        max_value=5000.0,
        value=1450.0
    )


if st.button(
    "🔮 Predict Failure Risk",
    width="stretch"
):

    if not MODEL_LOADED:

        st.error(
            "ML model is not available."
        )

    else:

        features = np.array([[
            pressure,
            temperature,
            vibration,
            flow_rate,
            production_rate,
            pump_speed
        ]])

        features_scaled = (
            scaler.transform(features)
        )

        probability = model.predict_proba(
            features_scaled
        )[0][1]

        probability_percent = (
            probability * 100
        )


        if probability >= 0.80:

            risk_level = "HIGH"

        elif probability >= 0.40:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"


        st.subheader(
            "AI Prediction Result"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Failure Probability",
                f"{probability_percent:.2f}%"
            )


        with col2:

            st.metric(
                "Risk Level",
                risk_level
            )


        if risk_level == "HIGH":

            st.error(
                "🚨 HIGH RISK — The AI model detects a high "
                "failure probability."
            )

        elif risk_level == "MEDIUM":

            st.warning(
                "⚠️ MEDIUM RISK — Equipment should be monitored."
            )

        else:

            st.success(
                "✅ LOW RISK — No significant failure risk detected."
            )


# =========================================================
# SHAP EXPLANATION
# =========================================================

st.header("🧠 AI Explainability — SHAP")


st.write(
    """
SHAP helps explain which sensor features have the strongest
influence on the machine-learning model.
"""
)


shap_summary_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "shap_summary.png"
)


shap_importance_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "shap_feature_importance.png"
)


col1, col2 = st.columns(2)


with col1:

    if shap_summary_path.exists():

        st.image(
            str(shap_summary_path),
            caption="SHAP Summary"
        )

    else:

        st.warning(
            "SHAP summary image not found."
        )


with col2:

    if shap_importance_path.exists():

        st.image(
            str(shap_importance_path),
            caption="SHAP Feature Importance"
        )

    else:

        st.warning(
            "SHAP feature importance image not found."
        )


# =========================================================
# SENSOR OVERVIEW
# =========================================================

st.header("📡 Sensor Overview")


sensor_columns = [
    "pressure_bar",
    "temperature_c",
    "vibration_mm_s",
    "flow_rate_bbl_day",
    "production_rate_bbl_day",
    "pump_speed_rpm"
]


sensor_available = [
    column
    for column in sensor_columns
    if column in df.columns
]


sensor_stats = (
    df[sensor_available]
    .describe()
    .T
)


st.dataframe(
    sensor_stats.round(2),
    width="stretch"
)


# =========================================================
# SENSOR DATA TABLE
# =========================================================

st.header("📋 Industrial Sensor Data")


show_rows = st.slider(
    "Number of rows to display",
    min_value=10,
    max_value=500,
    value=100,
    step=10
)


st.dataframe(
    df.head(show_rows),
    width="stretch",
    height=400
)


# =========================================================
# EQUIPMENT HEALTH MONITORING
# =========================================================

st.header("🏭 Equipment Health Monitoring")


health_image_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "equipment_health_monitoring.png"
)


if health_image_path.exists():

    st.image(
        str(health_image_path),
        caption="Equipment Health Monitoring"
    )

else:

    st.warning(
        "Equipment health monitoring image not found."
    )


# =========================================================
# TIME-SERIES MONITORING
# =========================================================

st.header("📈 Time-Series Monitoring")


if "timestamp" in well_data.columns:

    well_data = well_data.sort_values(
        "timestamp"
    )


    # -----------------------------------------------------
    # AI FAILURE PROBABILITY OVER TIME
    # -----------------------------------------------------

    if MODEL_LOADED:

        prediction_features = well_data[
            [
                "pressure_bar",
                "temperature_c",
                "vibration_mm_s",
                "flow_rate_bbl_day",
                "production_rate_bbl_day",
                "pump_speed_rpm"
            ]
        ].values


        prediction_scaled = (
            scaler.transform(
                prediction_features
            )
        )


        probabilities = model.predict_proba(
            prediction_scaled
        )[:, 1]


        well_data[
            "failure_probability"
        ] = probabilities * 100


    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🌡️ Temperature",
            "📳 Vibration",
            "💨 Pressure",
            "🛢️ Production"
        ]
    )


    with tab1:

        st.line_chart(
            well_data.set_index(
                "timestamp"
            )["temperature_c"]
        )


    with tab2:

        st.line_chart(
            well_data.set_index(
                "timestamp"
            )["vibration_mm_s"]
        )


    with tab3:

        st.line_chart(
            well_data.set_index(
                "timestamp"
            )["pressure_bar"]
        )


    with tab4:

        st.line_chart(
            well_data.set_index(
                "timestamp"
            )["production_rate_bbl_day"]
        )


    # -----------------------------------------------------
    # AI FAILURE PROBABILITY
    # -----------------------------------------------------

    if (
        "failure_probability"
        in well_data.columns
    ):

        st.subheader(
            "🤖 AI Failure Probability Over Time"
        )


        st.line_chart(
            well_data.set_index(
                "timestamp"
            )["failure_probability"]
        )


        st.caption(
            "Risk thresholds: LOW < 40%, "
            "MEDIUM = 40–80%, HIGH ≥ 80%"
        )


        # -------------------------------------------------
        # STATISTICS
        # -------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Max Temperature",
                f"{well_data['temperature_c'].max():.2f} °C"
            )


        with col2:

            st.metric(
                "Max Vibration",
                f"{well_data['vibration_mm_s'].max():.2f} mm/s"
            )


        with col3:

            st.metric(
                "Max AI Risk",
                f"{well_data['failure_probability'].max():.2f}%"
            )


        with col4:

            st.metric(
                "Avg AI Risk",
                f"{well_data['failure_probability'].mean():.2f}%"
            )


        # -------------------------------------------------
        # HIGH RISK EVENTS
        # -------------------------------------------------

        high_risk_events = well_data[
            well_data[
                "failure_probability"
            ] >= 80
        ]


        st.subheader(
            "🚨 High-Risk Events"
        )


        if len(high_risk_events) > 0:

            display_events = high_risk_events[
                [
                    "timestamp",
                    "temperature_c",
                    "vibration_mm_s",
                    "pressure_bar",
                    "production_rate_bbl_day",
                    "failure_probability"
                ]
            ].copy()


            display_events[
                "failure_probability"
            ] = display_events[
                "failure_probability"
            ].round(2)


            st.dataframe(
                display_events,
                width="stretch",
                hide_index=True
            )

        else:

            st.success(
                "No high-risk events detected."
            )


else:

    st.warning(
        "Timestamp column is not available."
    )


# =========================================================
# POSTGRESQL INDUSTRIAL INTELLIGENCE
# =========================================================

st.header("🗄️ PostgreSQL Industrial Intelligence")


st.write(
    """
The PostgreSQL layer stores industrial sensor data and
provides SQL-based well intelligence analytics.
"""
)


if st.button(
    "🔄 Load Well Intelligence from PostgreSQL",
    width="stretch"
):

    try:

        from database.db_connection import engine
        from database.queries import (
            WELL_INTELLIGENCE_QUERY
        )


        sql_df = pd.read_sql(
            WELL_INTELLIGENCE_QUERY,
            engine
        )


        st.success(
            "PostgreSQL query executed successfully."
        )


        st.dataframe(
            sql_df,
            width="stretch",
            hide_index=True
        )


    except Exception as e:

        st.error(
            "PostgreSQL query failed."
        )

        st.code(
            str(e)
        )


# =========================================================
# FASTAPI AI PREDICTION API
# =========================================================

st.header("🚀 FastAPI AI Prediction API")


st.write(
    """
The dashboard can send sensor values to the FastAPI service,
which runs the trained ML model and returns the predicted risk.
"""
)


api_col1, api_col2 = st.columns(2)


with api_col1:

    api_pressure = st.number_input(
        "API Pressure (bar)",
        min_value=0.0,
        max_value=500.0,
        value=170.0,
        key="api_pressure"
    )


    api_temperature = st.number_input(
        "API Temperature (°C)",
        min_value=0.0,
        max_value=200.0,
        value=96.0,
        key="api_temperature"
    )


    api_vibration = st.number_input(
        "API Vibration (mm/s)",
        min_value=0.0,
        max_value=30.0,
        value=9.4,
        key="api_vibration"
    )


with api_col2:

    api_flow_rate = st.number_input(
        "API Flow Rate (bbl/day)",
        min_value=0.0,
        max_value=5000.0,
        value=1080.0,
        key="api_flow_rate"
    )


    api_production_rate = st.number_input(
        "API Production Rate (bbl/day)",
        min_value=0.0,
        max_value=5000.0,
        value=985.0,
        key="api_production_rate"
    )


    api_pump_speed = st.number_input(
        "API Pump Speed (RPM)",
        min_value=0.0,
        max_value=5000.0,
        value=1450.0,
        key="api_pump_speed"
    )


# =========================================================
# FASTAPI REQUEST
# =========================================================

if st.button(
    "🤖 Predict Failure Risk",
    width="stretch"
):

    payload = {

        "pressure_bar":
            api_pressure,

        "temperature_c":
            api_temperature,

        "vibration_mm_s":
            api_vibration,

        "flow_rate_bbl_day":
            api_flow_rate,

        "production_rate_bbl_day":
            api_production_rate,

        "pump_speed_rpm":
            api_pump_speed
    }


    # -----------------------------------------------------
    # API URL
    # -----------------------------------------------------

    API_URL = os.getenv(
        "API_URL",
        "http://127.0.0.1:8000"
    )


    API_ENDPOINT = (
        f"{API_URL}/predict"
    )


    # -----------------------------------------------------
    # SEND REQUEST
    # -----------------------------------------------------

    try:

        response = requests.post(
            API_ENDPOINT,
            json=payload,
            timeout=10
        )


        if response.status_code == 200:

            result = response.json()


            st.success(
                "FastAPI prediction successful."
            )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Failure Probability",
                    f"{result['failure_probability_percent']:.2f}%"
                )


            with col2:

                st.metric(
                    "Risk Level",
                    result["risk_level"]
                )


            if result["risk_level"] == "HIGH":

                st.error(
                    "🚨 HIGH RISK detected by FastAPI."
                )


            elif result["risk_level"] == "MEDIUM":

                st.warning(
                    "⚠️ MEDIUM RISK detected by FastAPI."
                )


            else:

                st.success(
                    "✅ LOW RISK detected by FastAPI."
                )


            with st.expander(
                "View API Response"
            ):

                st.json(result)


        else:

            st.error(
                f"FastAPI returned HTTP "
                f"{response.status_code}"
            )


            st.code(
                response.text
            )


    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to FastAPI."
        )


        st.info(
            f"API endpoint: {API_ENDPOINT}"
        )


    except requests.exceptions.Timeout:

        st.error(
            "❌ FastAPI request timed out."
        )


    except Exception as e:

        st.error(
            "❌ Unexpected API error."
        )


        st.code(
            str(e)
        )


# =========================================================
# DATASET INFORMATION
# =========================================================

st.header("📚 Dataset Information")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Rows",
        f"{len(df):,}"
    )


with col2:

    st.metric(
        "Columns",
        len(df.columns)
    )


with col3:

    st.metric(
        "Wells",
        df["well_id"].nunique()
    )


st.write(
    "Dataset columns:"
)

st.write(
    list(df.columns)
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Oil & Gas AI — Industrial AI Data Intelligence Platform"
)

st.caption(
    "⚠️ This project uses synthetic industrial sensor data. "
    "The model results and risk thresholds are for demonstration "
    "and portfolio purposes and are not real field safety standards."
)