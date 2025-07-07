import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load dataset from local file
df = pd.read_csv("simulated_muse_score_dataset.csv")

# Feature engineering
df['Refund_Ratio'] = df['Refund'] / df['AGI']
df['Liquidity_Margin'] = df['Net_Income'] / (df['COLI_Index'] * df['AGI'])
df['AGI_Bucket'] = pd.cut(df['AGI'], bins=[0,30000,60000,90000,120000,150000], labels=["Low","Lower-Mid","Mid","Upper-Mid","High"])

# Define features and target
features = ['AGI', 'Effective_Tax_Rate', 'Tax_Paid', 'Refund', 'Net_Income',
            'COLI_Index', 'Deduction_Strategy', 'Filing_Status', 'Dependents',
            'Boost_Flag', 'Refund_Ratio', 'Liquidity_Margin', 'AGI_Bucket']
target = 'Muse_Score'

X = df[features]
y = df[target]

# Preprocessing
numeric = X.select_dtypes(include=np.number).columns.tolist()
categorical = ['Deduction_Strategy', 'Filing_Status', 'AGI_Bucket']
numeric = list(set(numeric) - set(categorical))

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(drop='first'), categorical)
])

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
pipeline.fit(X, y)

# ---- Streamlit UI ----
st.title("🎯 Muse Score Predictor")

agi = st.slider("AGI", 15000, 150000, 50000, step=5000)
effective_tax_rate = st.slider("Effective Tax Rate (%)", 5.0, 35.0, 15.0) / 100
refund = st.slider("Expected Refund", 0, 10000, 2000, step=500)
coli_index = st.selectbox("COLI Index", [0.85, 0.95, 1.0, 1.05, 1.15, 1.25])
deduction_strategy = st.selectbox("Deduction Strategy", ["Standard", "Itemized"])
filing_status = st.selectbox("Filing Status", ["Single", "Married Joint", "Married Separate", "Head of Household"])
dependents = st.slider("Dependents", 0, 5, 1)
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

# Predict
predicted_score = pipeline.predict(input_data)[0]

st.subheader("📈 Prediction")
st.markdown(f"### 🧠 Muse Score: `{round(predicted_score, 2)}`")

st.subheader("📋 Input Summary")
st.dataframe(input_data)
