import streamlit as st
import numpy as np
import joblib
import plotly.express as px
import pandas as pd

# 🔥 LOAD MODEL
kmeans = joblib.load("kmeans.pkl")
columns = joblib.load("columns.pkl")


# 🎯 CLUSTER LABELS
cluster_map = {
    0: "🟡 Balanced Customer",
    1: "🔴 Risky Spender",
    2: "🟢 Premium Saver"
}

st.set_page_config(page_title="Customer Revenue Intelligence", layout="centered")

# 🎨 HEADER
st.title("💰 Customer Revenue Intelligence System")
st.markdown("AI-powered system for customer segmentation, revenue analysis & business decision-making")

st.divider()

# 🧾 INPUT SECTION
st.subheader("📥 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    income = st.number_input("Income", 1000, 200000, 50000)

with col2:
    freq = st.slider("Purchase Frequency", 0.0, 1.0, 0.5)
    spending = st.number_input("Spending", 0, 50000, 10000)

# 🔥 FEATURE ENGINEERING
ratio = spending / income if income != 0 else 0

if st.button("🚀 Analyze Customer"):

    X = np.array([[age, income, freq, spending, ratio]])

    cluster = kmeans.predict(X)[0]
    label = cluster_map[cluster]

    score = (1 - ratio) * 100

    st.divider()
    st.subheader("📊 Customer Intelligence Report")

    # 🎯 METRICS
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Customer Type", label)

    with col2:
        st.metric("Financial Score", f"{score:.2f}")

    # 🔥 RISK LEVEL
    if score < 40:
        risk = "High Risk 🔴"
        st.error(risk)
    elif score < 70:
        risk = "Moderate Risk 🟡"
        st.warning(risk)
    else:
        risk = "Low Risk 🟢"
        st.success(risk)

    # 💰 REVENUE POTENTIAL
    revenue_score = spending * 0.3

    # 🎯 ROI CALCULATION
    marketing_cost = 1000
    roi = (spending - marketing_cost) / marketing_cost

    st.subheader("💼 Business Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Revenue Potential", f"{revenue_score:.0f}")

    with col2:
        st.metric("ROI", f"{roi*100:.1f}%")

    with col3:
        st.metric("Spending Ratio", f"{ratio:.2f}")

    # 🧠 DECISION ENGINE
    st.subheader("🎯 Business Recommendation")

    if cluster == 1:
        action = "⚠️ Reduce risk: Monitor spending & limit credit exposure"
        st.error(action)
    elif cluster == 2:
        action = "💎 Offer premium products & upsell services"
        st.success(action)
    else:
        action = "👍 Maintain engagement with regular offers"
        st.info(action)

    # 🧠 AI INSIGHT
    st.subheader("🧠 AI Insight")

    if ratio > 0.25:
        st.error("Customer is overspending relative to income — potential financial instability.")
    elif ratio < 0.15:
        st.success("Customer demonstrates strong financial discipline.")
    else:
        st.info("Customer shows balanced spending behavior.")

    # 📊 VISUALIZATION
    st.subheader("📈 Market Position")

    sample_data = pd.DataFrame({
        "income": np.random.randint(20000, 100000, 100),
        "spending": np.random.randint(2000, 20000, 100),
        "cluster": np.random.choice([0,1,2], 100)
    })

    fig = px.scatter(
        sample_data,
        x="income",
        y="spending",
        color="cluster",
        title="Customer Segmentation Map"
    )

    fig.add_scatter(
        x=[income],
        y=[spending],
        mode="markers",
        marker=dict(size=12, color="black"),
        name="You"
    )

    st.plotly_chart(fig)

    # 💡 COMPARISON
    st.subheader("📊 Market Comparison")

    avg_spending = sample_data["spending"].mean()

    if spending > avg_spending:
        st.write("⚠️ Customer spends more than average users in this segment.")
    else:
        st.write("✅ Customer spends less than average users.")

    
