# 🎯 Muse Score™ Predictor

The **Muse Score™** is a modern, personalized financial health metric that evaluates a user's tax efficiency, real purchasing power, and behavioral optimization. This Streamlit app allows users to simulate their Muse Score based on Adjusted Gross Income (AGI), tax behavior, and geographic cost of living (COLA).

---


## 🧠 What Is Muse Score?

**Muse Score™** (short for *Measuring User Spending Efficiency*) is a data-driven score that helps users and employers evaluate:
- 📊 **IRS tax behavior** (SOI norms: AGI bands, refund ratios, etc.)
- 🌍 **Cost of Living Adjustments (COLA)** from the U.S. Bureau of Labor Statistics
- 🔍 **Behavioral optimization** like deduction strategy and refund tuning

Scores range from **400 to 850**, similar to credit scores.

| Band         | Score Range | Interpretation                              |
|--------------|-------------|----------------------------------------------|
| 🟢 Excellent  | 750–850     | Very optimized tax + financial strategy      |
| 🟡 Strong     | 650–749     | Mostly optimized, minor inefficiencies       |
| 🟠 Moderate   | 550–649     | COLA stress or tax inefficiencies present    |
| 🔴 Vulnerable | 400–549     | Financial or tax planning risk areas exist   |

---

## ⚙️ App Features

- Dynamic inputs for AGI, refund, tax rate, COLA, dependents, filing status
- Real-time Muse Score prediction using a Random Forest Regressor
- Calculates refund ratio, liquidity margin, and COLA-adjusted power
- Simulated data for testing financial benchmarking

---

## 🛠 Tech Stack

- **Python** & **Streamlit** (for frontend and logic)
- **Scikit-learn** (Random Forest model pipeline)
- **pandas & numpy** (data handling)
- **Google Colab** (for initial simulation & model training)

---

