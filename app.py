import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Smart Healthcare Analytics System",
    page_icon="🏥",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data/patient_data.csv")

# =========================
# LOAD MODEL
# =========================

model = joblib.load(
    "models/disease_prediction_model.pkl"
)

gender_encoder = joblib.load(
    "models/gender_encoder.pkl"
)

disease_encoder = joblib.load(
    "models/disease_encoder.pkl"
)

# =========================
# TITLE
# =========================

st.title("🏥 Smart Healthcare Analytics System")

st.markdown(
    """
    AI-powered healthcare analytics dashboard for patient monitoring,
    disease analysis, and disease prediction using Machine Learning.
    """
)

st.write("---")

# =========================
# KPI METRICS
# =========================

total_patients = len(df)

avg_age = round(
    df["Age"].mean(),
    1
)

avg_stay = round(
    df["Hospital_Stay_Days"].mean(),
    1
)

most_common_disease = df["Disease"].mode()[0]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Patients",
    total_patients
)

col2.metric(
    "Average Age",
    avg_age
)

col3.metric(
    "Average Stay Days",
    avg_stay
)

col4.metric(
    "Most Common Disease",
    most_common_disease
)

st.write("---")

# =========================
# DISEASE DISTRIBUTION
# =========================

st.subheader("📊 Disease Distribution")

fig1 = px.histogram(
    df,
    x="Disease",
    color="Disease",
    title="Disease Distribution"
)

fig1.update_layout(
    height=500,
    title_x=0.3,
    margin=dict(
        l=40,
        r=40,
        t=70,
        b=40
    )
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# =========================
# AGE DISTRIBUTION
# =========================

st.write("")

st.subheader("👥 Patient Age Distribution")

fig2 = px.histogram(
    df,
    x="Age",
    nbins=20,
    marginal="box",
    opacity=0.85,
    title="Patient Age Distribution"
)

fig2.update_layout(
    height=500,
    bargap=0.08,
    title_x=0.35,
    xaxis_title="Age",
    yaxis_title="Number of Patients",
    margin=dict(
        l=40,
        r=40,
        t=70,
        b=40
    )
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# =========================
# BLOOD PRESSURE ANALYSIS
# =========================

st.write("")

st.subheader("❤️ Blood Pressure Analysis")

sample_df = df.sample(2500)

fig3 = px.scatter(
    sample_df,
    x="Age",
    y="Blood_Pressure",
    color="Disease",
    opacity=0.65,
    title="Age vs Blood Pressure by Disease",
    hover_data=[
        "Disease",
        "Gender"
    ]
)

fig3.update_traces(
    marker=dict(size=6)
)

fig3.update_layout(
    height=650,
    title_x=0.3,
    xaxis_title="Patient Age",
    yaxis_title="Blood Pressure",
    legend_title="Disease",
    margin=dict(
        l=40,
        r=40,
        t=70,
        b=40
    )
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# =========================
# HOSPITAL STAY ANALYSIS
# =========================

st.write("")

st.subheader("🏥 Hospital Stay Analysis")

fig4 = px.box(
    df,
    x="Disease",
    y="Hospital_Stay_Days",
    color="Disease",
    title="Hospital Stay Duration by Disease"
)

fig4.update_layout(
    height=600,
    title_x=0.3,
    margin=dict(
        l=40,
        r=40,
        t=70,
        b=40
    )
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

# =========================
# GENDER ANALYSIS
# =========================

st.write("")

st.subheader("🧑 Gender Distribution")

fig5 = px.pie(
    df,
    names="Gender",
    title="Gender Distribution",
    hole=0.4
)

fig5.update_layout(
    height=500,
    title_x=0.35
)

st.plotly_chart(
    fig5,
    use_container_width=True
)

# =========================
# AI DISEASE PREDICTION
# =========================

st.write("---")

st.header("🧠 AI Disease Prediction")

st.markdown(
    """
    Enter patient vitals to predict possible disease using
    Machine Learning.
    """
)

# =========================
# USER INPUTS
# =========================

col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Age",
        18,
        90,
        35
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    heart_rate = st.slider(
        "Heart Rate",
        60,
        140,
        90
    )

with col2:

    blood_pressure = st.slider(
        "Blood Pressure",
        90,
        180,
        120
    )

    temperature = st.slider(
        "Temperature",
        97.0,
        104.0,
        98.6
    )

# =========================
# ENCODE INPUTS
# =========================

gender_encoded = gender_encoder.transform(
    [gender]
)[0]

# =========================
# PREDICTION
# =========================

if st.button("Predict Disease"):

    input_data = np.array([
        [
            age,
            gender_encoded,
            heart_rate,
            blood_pressure,
            temperature
        ]
    ])

    prediction = model.predict(
        input_data
    )

    predicted_disease = disease_encoder.inverse_transform(
        prediction
    )[0]

    st.success(
        f"Predicted Disease: {predicted_disease}"
    )

    # =========================
    # RISK MESSAGE
    # =========================

    if predicted_disease == "Healthy":

        st.info(
            "Patient appears healthy."
        )

    elif predicted_disease == "Flu":

        st.warning(
            "Mild infection symptoms detected."
        )

    else:

        st.error(
            "Potential high-risk condition detected."
        )

# =========================
# FEATURE IMPORTANCE
# =========================

st.write("---")

st.header("📌 Model Feature Importance")

st.markdown(
    """
    Feature importance analysis showing which patient
    vitals most influence disease prediction.
    """
)

feature_names = [
    "Age",
    "Gender",
    "Heart_Rate",
    "Blood_Pressure",
    "Temperature"
]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

fig_importance = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Importance Analysis",
    text_auto=".2f"
)

fig_importance.update_layout(
    height=500,
    title_x=0.3,
    xaxis_title="Importance Score",
    yaxis_title="Feature",
    margin=dict(
        l=40,
        r=40,
        t=70,
        b=40
    )
)

st.plotly_chart(
    fig_importance,
    use_container_width=True
)

# =========================
# FOOTER
# =========================

st.write("---")

st.caption(
    "Built by Soham Kadam "
)