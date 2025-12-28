import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📉",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL ARTIFACTS
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/churn_prediction_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, feature_columns, scaler

model, feature_columns, scaler = load_artifacts()

# --------------------------------------------------
# REASON GENERATOR (EXPLAINABILITY LAYER)
# --------------------------------------------------
def generate_reasons_with_severity(
    tenure, monthly_charges, contract, internet_service, paperless_billing
):
    reasons = []

    if tenure < 12:
        reasons.append(("Low tenure", "High", "Customer has a short relationship with the company."))
    elif tenure < 24:
        reasons.append(("Moderate tenure", "Medium", "Customer loyalty is not fully established."))

    if contract == "Month-to-month":
        reasons.append(("Flexible contract", "High", "Month-to-month contracts are associated with higher churn risk."))
    else:
        reasons.append(("Long-term contract", "Low", "Long-term contracts reduce churn risk."))

    if monthly_charges > 70:
        reasons.append(("High monthly charges", "High", "Customer may be sensitive to pricing."))
    elif monthly_charges > 50:
        reasons.append(("Moderate monthly charges", "Medium", "Pricing level can influence churn."))

    if internet_service == "Fiber optic":
        reasons.append(("Fiber optic service", "Medium", "Higher service expectations may increase churn."))

    if paperless_billing == "Yes":
        reasons.append(("Paperless billing", "Low", "Digital billing users are often more flexible to switch."))

    return reasons

# --------------------------------------------------
# CUSTOM CSS (ADVANCED UI)
# --------------------------------------------------
st.markdown("""
<style>
.main { background-color: #f8fafc; }

.title-text {
    font-size: 42px;
    font-weight: 800;
    color: #FFFFFF;
}

.subtitle-text {
    font-size: 18px;
    color: #475569;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

.metric-card {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    color: white;
    padding: 25px;
    border-radius: 16px;
    text-align: center;
}

.metric-value {
    font-size: 36px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown('<div class="title-text">Customer Churn Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">'
    'AI-powered churn prediction with business-focused explanations.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------
left, right = st.columns([1.2, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Customer Profile")

    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", value=70.0)
    total_charges = st.number_input("Total Charges", value=800.0)

    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# BUILD INPUT DATA
# --------------------------------------------------
input_data = pd.DataFrame(0, index=[0], columns=feature_columns)

input_data["tenure"] = tenure
input_data["MonthlyCharges"] = monthly_charges
input_data["TotalCharges"] = total_charges

# Apply scaling
input_data[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(
    input_data[["tenure", "MonthlyCharges", "TotalCharges"]]
)

# Encode categoricals safely
if contract != "Month-to-month" and f"Contract_{contract}" in input_data.columns:
    input_data[f"Contract_{contract}"] = 1

if internet_service != "DSL" and f"InternetService_{internet_service}" in input_data.columns:
    input_data[f"InternetService_{internet_service}"] = 1

if paperless_billing == "Yes" and "PaperlessBilling" in input_data.columns:
    input_data["PaperlessBilling"] = 1

# --------------------------------------------------
# OUTPUT SECTION
# --------------------------------------------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Output")

    if st.button("🚀 Analyze Churn Risk", use_container_width=True):
        churn_prob = model.predict_proba(input_data)[0][1]

        # Metric
        st.markdown(
            f"""
            <div class="metric-card">
                <div>Churn Probability</div>
                <div class="metric-value">{churn_prob:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Generate reasons
        reasons = generate_reasons_with_severity(
            tenure, monthly_charges, contract, internet_service, paperless_billing
        )

        # ---------------- Top 3 Drivers ----------------
        st.write("### 🏷️ Top 3 Risk Drivers")
        top_drivers = reasons[:3]
        cols = st.columns(len(top_drivers))

        for i, (title, severity, _) in enumerate(top_drivers):
            with cols[i]:
                st.markdown(
                    f"""
                    <div style="
                        padding: 12px;
                        border-radius: 12px;
                        background-color: #000000;
                        text-align: center;
                        font-weight: 600;
                    ">
                        {title}<br/>
                        <span style="font-size: 13px; color: #FFFFFF;">
                            Severity: {severity}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ---------------- Severity Indicator ----------------
        severities = [s for _, s, _ in reasons]
        st.write("### ⚠️ Explanation Severity")

        if "High" in severities:
            st.error("High — multiple strong churn signals detected.")
        elif "Medium" in severities:
            st.warning("Medium — some churn indicators present.")
        else:
            st.success("Low — limited churn indicators.")

        # ---------------- Expandable WHY Panel ----------------
        with st.expander("🔍 Why is this customer predicted this way?"):
            for title, severity, explanation in reasons:
                st.markdown(
                    f"""
                    **{title}**  
                    *Severity:* {severity}  
                    {explanation}
                    ---
                    """
                )

    st.markdown('</div>', unsafe_allow_html=True)
