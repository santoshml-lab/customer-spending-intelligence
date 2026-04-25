
import streamlit as st
import numpy as np
import joblib
import plotly.express as px

# 🔥 LOAD MODEL + FEATURES
kmeans = joblib.load("kmeans.pkl")
features = joblib.load("columns.pkl")

# 🔥 CLUSTER LABEL MAP
cluster_map = {
    0: "Balanced Customer",
    1: "Risky Spender",
    2: "Premium Saver"
}

st.set_page_config(page_title="AI Spending Analyzer", layout="centered")

st.title("💰 AI Customer Spending Behavior Analyzer")

st.write("Enter customer details to analyze spending behavior")

# 🧾 INPUTS
age = st.number_input("Age", 18, 100, 30)
income = st.number_input("Income", 1000, 200000, 50000)
freq = st.slider("Purchase Frequency", 0.0, 1.0, 0.5)
spending = st.number_input("Spending", 0, 50000, 10000)

# 🔥 FEATURE ENGINEERING
ratio = spending / income if income != 0 else 0

# 🚀 BUTTON
if st.button("Analyze"):

    X = np.array([[age, income, freq, spending, ratio]])

    # 🔥 PREDICT CLUSTER
    cluster = kmeans.predict(X)[0]
    label = cluster_map[cluster]

    # 🔥 SCORE
    score = (1 - ratio) * 100

    st.subheader("📊 Result")

    # RESULT
    st.write(f"### Customer Type: {label}")
    st.metric("Financial Score", f"{score:.2f}")

    # 🔥 INSIGHTS
    if cluster == 1:
        st.error("⚠️ You are overspending compared to your income.")
    elif cluster == 2:
        st.success("💎 You manage your finances very well.")
    else:
        st.info("👍 You have balanced spending habits.")

    # 📊 GRAPH
    fig = px.scatter(
        x=[income],
        y=[spending],
        labels={'x': 'Income', 'y': 'Spending'},
        title="Income vs Spending"
    )

    st.plotly_chart(fig)
