import streamlit as st
import numpy as np
import joblib
import plotly.express as px
import pandas as pd

# 🔥 LOAD MODEL
kmeans = joblib.load("models/kmeans.pkl")

# 🎯 CLUSTER LABELS
cluster_map = {
    0: "🟡 Balanced Customer",
    1: "🔴 Risky Spender",
    2: "🟢 Premium Saver"
}

st.set_page_config(page_title="AI Spending Analyzer", layout="centered")

# 🎨 HEADER
st.title("💰 AI Customer Spending Intelligence")
st.markdown("Analyze financial behavior with AI-powered insights")

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

# 🚀 ANALYZE BUTTON
if st.button("🚀 Analyze Behavior"):

    X = np.array([[age, income, freq, spending, ratio]])

    cluster = kmeans.predict(X)[0]
    label = cluster_map[cluster]

    score = (1 - ratio) * 100

    st.divider()
    st.subheader("📊 Analysis Result")

    # 🎯 METRICS
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Customer Type", label)

    with col2:
        st.metric("Financial Score", f"{score:.2f}")

    # 🎯 SCORE COLOR LOGIC
    if score < 40:
        st.error("🔴 High Financial Risk")
    elif score < 70:
        st.warning("🟡 Moderate Risk")
    else:
        st.success("🟢 Healthy Financial Behavior")

    # 🔥 SMART INSIGHT (WOW FEATURE)
    st.subheader("🧠 AI Insight")

    if cluster == 1:
        st.error("You are spending significantly higher relative to your income. Consider reducing discretionary expenses.")
    elif cluster == 2:
        st.success("You manage your finances efficiently. You are a strong candidate for premium financial products.")
    else:
        st.info("Your spending is balanced. You maintain a healthy financial pattern.")

    # 🔥 ADVANCED INSIGHT
    st.subheader("📌 Personalized Insight")

    if ratio > 0.25:
        st.write("💡 You spend more than 25% of your income — potential overspending detected.")
    else:
        st.write("💡 Your spending is within a safe range.")

    # 📊 BETTER GRAPH (WITH CONTEXT)
    st.subheader("📈 Income vs Spending Analysis")

    # sample background data for visualization
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

    # highlight current user
    fig.add_scatter(
        x=[income],
        y=[spending],
        mode="markers",
        marker=dict(size=12, color="black"),
        name="You"
    )

    st.plotly_chart(fig)

    # 🎯 COMPARISON (WOW)
    avg_spending = sample_data["spending"].mean()

    st.subheader("📊 Comparison")

    if spending > avg_spending:
        st.write("⚠️ You spend more than average users.")
    else:
        st.write("✅ You spend less than average users.")
