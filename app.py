import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="CO₂ Emission Predictor",
    page_icon="🌍",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = joblib.load("polynomialRegModel.pkl")

# Same transformation used during training
poly = PolynomialFeatures(degree=2)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(34,197,94,0.10), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(6,182,212,0.10), transparent 30%),
        #07110d;
    color: #f4f7f5;
}

/* Hide Streamlit elements */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* Main container */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Hero section */
.hero {
    padding: 35px 40px;
    border-radius: 24px;
    background: linear-gradient(
        135deg,
        rgba(18, 48, 36, 0.95),
        rgba(7, 24, 18, 0.95)
    );
    border: 1px solid rgba(74, 222, 128, 0.18);
    box-shadow: 0 20px 60px rgba(0,0,0,0.30);
    margin-bottom: 30px;
}

.hero-tag {
    color: #4ade80;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 10px;
}

.hero-title {
    font-size: 46px;
    line-height: 1.1;
    font-weight: 800;
    margin: 0;
    color: #ffffff;
}

.hero-subtitle {
    margin-top: 12px;
    color: #a7b8ae;
    font-size: 16px;
}

/* Section title */
.section-title {
    font-size: 22px;
    font-weight: 700;
    margin: 25px 0 15px 0;
    color: #ffffff;
}

/* Input cards */
.input-card {
    background: rgba(16, 31, 24, 0.90);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 22px;
    height: 100%;
    box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}

.input-icon {
    font-size: 28px;
    margin-bottom: 8px;
}

.input-title {
    font-size: 15px;
    font-weight: 700;
    color: #e8f3ec;
}

.input-description {
    font-size: 12px;
    color: #8fa49a;
    margin-bottom: 12px;
}

/* Streamlit number input */
div[data-testid="stNumberInput"] input {
    background-color: #0b1b13 !important;
    color: white !important;
    border: 1px solid #294638 !important;
    border-radius: 10px !important;
}

/* Predict button */
.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 14px;
    border: none;
    background: linear-gradient(90deg, #22c55e, #14b8a6);
    color: #04110a;
    font-size: 16px;
    font-weight: 800;
    letter-spacing: 0.5px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(34,197,94,0.25);
}

/* Result card */
.result-card {
    margin-top: 30px;
    padding: 32px;
    border-radius: 22px;
    background: linear-gradient(
        135deg,
        rgba(20, 83, 45, 0.95),
        rgba(9, 46, 35, 0.95)
    );
    border: 1px solid rgba(74,222,128,0.25);
    text-align: center;
    box-shadow: 0 15px 45px rgba(0,0,0,0.25);
}

.result-label {
    color: #9ee6b4;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
}

.result-value {
    color: #ffffff;
    font-size: 48px;
    font-weight: 800;
    margin: 8px 0;
}

.result-unit {
    color: #b9d9c4;
    font-size: 14px;
}

/* Info cards */
.info-card {
    background: rgba(13, 27, 20, 0.90);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 22px;
    margin-top: 20px;
}

.info-title {
    font-size: 14px;
    color: #7ee2a0;
    font-weight: 700;
    letter-spacing: 1px;
}

.info-value {
    font-size: 20px;
    font-weight: 700;
    color: white;
    margin-top: 5px;
}

.footer {
    text-align: center;
    color: #64756c;
    font-size: 12px;
    margin-top: 45px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HERO
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-tag">CLIMATE • AI • PREDICTION</div>
    <div class="hero-title">CO₂ Emission Predictor</div>
    <div class="hero-subtitle">
        Estimate carbon emissions using an intelligent Polynomial Regression model.
    </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------
st.markdown(
    '<div class="section-title">Environmental & Economic Parameters</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="input-card">
        <div class="input-icon">⚡</div>
        <div class="input-title">Energy Consumption</div>
        <div class="input-description">
            Enter the energy consumption value.
        </div>
    </div>
    """, unsafe_allow_html=True)

    energy = st.number_input(
        "Energy Consumption",
        min_value=0.0,
        value=500.0,
        step=10.0,
        label_visibility="collapsed"
    )


with col2:
    st.markdown("""
    <div class="input-card">
        <div class="input-icon">♻️</div>
        <div class="input-title">Renewable Percentage</div>
        <div class="input-description">
            Percentage of energy generated from renewable sources.
        </div>
    </div>
    """, unsafe_allow_html=True)

    renewable = st.number_input(
        "Renewable Percentage",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=1.0,
        label_visibility="collapsed"
    )


with col3:
    st.markdown("""
    <div class="input-card">
        <div class="input-icon">📈</div>
        <div class="input-title">GDP</div>
        <div class="input-description">
            Enter the GDP value used by the model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    gdp = st.number_input(
        "GDP",
        min_value=0.0,
        value=25000.0,
        step=500.0,
        label_visibility="collapsed"
    )


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

predict = st.button("🌍  PREDICT CO₂ EMISSIONS")


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if predict:

    input_data = pd.DataFrame(
        [[energy, renewable, gdp]],
        columns=[
            "Energy_Consumption",
            "Renewable_Percentage",
            "GDP"
        ]
    )

    # Apply the same polynomial transformation
    input_poly = poly.fit_transform(input_data)

    prediction = model.predict(input_poly)[0]

    # Result
    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">PREDICTED CO₂ EMISSIONS</div>
        <div class="result-value">{prediction:,.2f}</div>
        <div class="result-unit">Estimated emission units</div>
    </div>
    """, unsafe_allow_html=True)

    # Model information
    info1, info2, info3 = st.columns(3)

    with info1:
        st.markdown("""
        <div class="info-card">
            <div class="info-title">MODEL</div>
            <div class="info-value">Polynomial Regression</div>
        </div>
        """, unsafe_allow_html=True)

    with info2:
        st.markdown("""
        <div class="info-card">
            <div class="info-title">POLYNOMIAL DEGREE</div>
            <div class="info-value">2</div>
        </div>
        """, unsafe_allow_html=True)

    with info3:
        st.markdown("""
        <div class="info-card">
            <div class="info-title">INPUT VARIABLES</div>
            <div class="info-value">3 Parameters</div>
        </div>
        """, unsafe_allow_html=True)


# --------------------------------------------------
# MODEL DETAILS
# --------------------------------------------------
with st.expander("ℹ️ About this model"):

    st.write(
        """
        This application uses a Polynomial Regression model to estimate
        CO₂ emissions based on energy consumption, renewable energy
        percentage, and GDP.
        """
    )

    st.write("**Features used:**")
    st.write(
        "• Energy Consumption  \n"
        "• Renewable Percentage  \n"
        "• GDP"
    )

    st.write("**Target:** CO₂ Emissions")

    st.write("**Algorithm:** Polynomial Regression (Degree 2)")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<div class="footer">
    CO₂ Emission Prediction • Machine Learning Application
</div>
""", unsafe_allow_html=True)
