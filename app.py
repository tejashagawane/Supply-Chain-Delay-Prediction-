import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Supply Chain Risk Dashboard",
    layout="wide",
    page_icon="🚚"
)

# =========================
# CUSTOM CSS (PREMIUM LOOK)
# =========================
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: 700;
}
.metric-card {
    background-color: #f5f7fa;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD FILES
# =========================
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")
base_input = joblib.load("base_input.pkl")

# =========================
# HEADER
# =========================
st.markdown('<p class="big-title">🚚 Supply Chain Delay Prediction</p>', unsafe_allow_html=True)
st.caption("Predict shipment delays using machine learning & optimize logistics decisions")

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("📥 Input Parameters")

shipping_mode = st.sidebar.selectbox("Shipping Mode", ["Standard", "Express"])
order_region = st.sidebar.selectbox("Order Region", ["West", "East", "South", "North"])
market = st.sidebar.selectbox("Market", ["Africa", "Asia", "Europe", "US"])

quantity = st.sidebar.number_input("Order Quantity", 1, 100, 10)
scheduled_days = st.sidebar.slider("Scheduled Days", 1, 10, 5)

# =========================
# BUILD INPUT
# =========================
input_dict = base_input.copy()

input_dict['Order Item Quantity'] = quantity
input_dict['Days for shipment (scheduled)'] = scheduled_days
input_dict['shipping_pressure'] = scheduled_days / (quantity + 1)
input_dict['mode_risk'] = 1 if shipping_mode == "Standard" else 0

# reset categorical
for col in input_dict:
    if col.startswith("Order Region_"):
        input_dict[col] = 0
    if col.startswith("Market_"):
        input_dict[col] = 0

input_dict[f"Order Region_{order_region}"] = 1
input_dict[f"Market_{market}"] = 1

input_df = pd.DataFrame([input_dict])[features]

# =========================
# PREDICTION
# =========================
if st.button("🚀 Predict Risk"):
    prob = model.predict_proba(input_df)[0][1]
    st.session_state["prob"] = prob

# =========================
# RESULTS
# =========================
if "prob" in st.session_state:
    prob = st.session_state["prob"]

    if prob < 0.35:
        risk = "🟢 Low Risk"
    elif prob < 0.65:
        risk = "🟡 Medium Risk"
    else:
        risk = "🔴 High Risk"

    st.markdown("## 📊 Prediction Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Level", risk)
    col2.metric("Delay Probability", f"{prob:.4f}")
    col3.metric("Shipping Pressure", f"{input_dict['shipping_pressure']:.2f}")

    # =========================
    # RISK GAUGE
    # =========================
    st.markdown("### 📊 Risk Gauge")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={'text': "Delay Risk"},
        gauge={
            'axis': {'range': [0, 1]},
            'steps': [
                {'range': [0, 0.3], 'color': "green"},
                {'range': [0.3, 0.6], 'color': "orange"},
                {'range': [0.6, 1], 'color': "red"},
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # FEATURE IMPORTANCE
    # =========================
    st.markdown("### 📈 Feature Importance")

    try:
        if hasattr(model, "calibrated_classifiers_"):
            base_model = model.calibrated_classifiers_[0].estimator
        else:
            base_model = model

        importances = base_model.feature_importances_

        imp_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(10)

        fig_imp = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Influential Features"
        )

        st.plotly_chart(fig_imp, use_container_width=True)

    except:
        st.warning("Feature importance not available")

    # =========================
    # WHAT-IF ANALYSIS
    # =========================
    st.markdown("### 📉 What-If Analysis (Quantity Impact)")

    quantities = list(range(1, 100, 5))
    probs_list = []

    for q in quantities:
        temp = input_dict.copy()
        temp['Order Item Quantity'] = q
        temp['shipping_pressure'] = scheduled_days / (q + 1)

        temp_df = pd.DataFrame([temp])[features]
        p = model.predict_proba(temp_df)[0][1]
        probs_list.append(p)

    chart_df = pd.DataFrame({
        "Quantity": quantities,
        "Risk": probs_list
    })

    fig_line = px.line(
        chart_df,
        x="Quantity",
        y="Risk",
        markers=True,
        title="Risk vs Order Quantity"
    )

    st.plotly_chart(fig_line, use_container_width=True)

# =========================
# INSIGHTS
# =========================
st.markdown("---")
st.markdown("### 📊 Business Insights")

st.markdown("""
- 🚚 Standard shipping increases delay risk  
- 📦 High order quantity increases complexity  
- ⏱️ Lower scheduled days increase pressure  
- 🌍 Region & market influence delivery  

👉 Use this dashboard to proactively manage logistics risk
""")
st.markdown("### 📌 Model Limitations")
st.markdown("""
- Predictions are probabilistic, not guaranteed  
- Some features use baseline values  
- Model may miss certain delayed shipments  
""")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Built by Tejas Hagwane | Data Analytics Project")