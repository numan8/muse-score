import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("muse_score_model.pkl")

st.set_page_config(page_title="Muse Score Predictor", layout="centered")
st.title("🎯 Muse Score Predictor")

st.subheader("🔧 Enter Your Information")

agi = st.slider("AGI (Adjusted Gross Income)", 15000, 150000, 50000, step=5000)
effective_tax_rate = st.slider("Effective Tax Rate (%)", 5.0, 35.0, 15.0) / 100
refund = st.slider("Expected Refund ($)", 0, 10000, 2000, step=500)
coli_index = st.selectbox("COLI Index (Cost of Living)", [0.85, 0.95, 1.0, 1.05, 1.15, 1.25])
deduction_strategy = st.selectbox("Deduction Strategy", ["Standard", "Itemized"])
filing_status = st.selectbox("Filing Status", ["Single", "Married Joint", "Married Separate", "Head of Household"])
dependents = st.slider("Number of Dependents", 0, 5, 1)
boost_flag = st.selectbox("Boost Activated", [0, 1])

tax_paid = agi * effective_tax_rate
net_income = agi - tax_paid + refund
refund_ratio = refund / agi
liquidity_margin = net_income / (coli_index * agi)

agi_bucket = pd.cut([agi], bins=[0,30000,60000,90000,120000,150000],
                    labels=["Low","Lower-Mid","Mid","Upper-Mid","High"])[0]

input_data = pd.DataFrame([{
    "AGI": agi,
    "Effective_Tax_Rate": effective_tax_rate,
    "Tax_Paid": tax_paid,
    "Refund": refund,
    "Net_Income": net_income,
    "COLI_Index": coli_index,
    "Deduction_Strategy": deduction_strategy,
    "Filing_Status": filing_status,
    "Dependents": dependents,
    "Boost_Flag": boost_flag,
    "Refund_Ratio": refund_ratio,
    "Liquidity_Margin": liquidity_margin,
    "AGI_Bucket": agi_bucket
}])

predicted_score = model.predict(input_data)[0]

st.subheader("📈 Prediction Result")
st.markdown(f"### 🧠 Your Predicted Muse Score: `{round(predicted_score, 2)}`")
st.subheader("🔍 Input Summary")
st.dataframe(input_data)
