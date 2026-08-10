"""
M5 Demand Forecasting + Inventory Optimization Dashboard
-----------------------------------------------------------
Streamlit app to showcase forecasting results (ARIMA / LightGBM / LSTM)
and inventory recommendations (Safety Stock, Reorder Point, EOQ)
built on the Walmart M5 dataset.

Run locally:  streamlit run app.py
Deploy:       push this + data/ folder to GitHub, connect on
              https://share.streamlit.io (Streamlit Community Cloud)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

# -----------------------------------------------------------------
# Page config
# -----------------------------------------------------------------
st.set_page_config(
    page_title="M5 Demand Forecasting & Inventory Optimization",
    page_icon="📦",
    layout="wide"
)

# -----------------------------------------------------------------
# Data loading (cached so it only loads once)
# -----------------------------------------------------------------
DATA_DIR = "data/"  # saari CSV/parquet files isi folder me rakhni hain

@st.cache_data
def load_data():
    lgb_preds = pd.read_parquet(DATA_DIR + "lgb_test_predictions.parquet")
    inventory = pd.read_csv(DATA_DIR + "inventory_recommendations.csv")
    comparison = pd.read_csv(DATA_DIR + "model_comparison.csv")

    try:
        arima_results = pd.read_csv(DATA_DIR + "arima_results.csv", parse_dates=["date"])
    except FileNotFoundError:
        arima_results = None

    lgb_preds["date"] = pd.to_datetime(lgb_preds["date"])
    return lgb_preds, inventory, comparison, arima_results


try:
    lgb_preds, inventory, comparison, arima_results = load_data()
    data_loaded = True
except FileNotFoundError as e:
    data_loaded = False
    missing_file_error = str(e)

# -----------------------------------------------------------------
# Header
# -----------------------------------------------------------------
st.title("📦 Demand Forecasting + Inventory Optimization")
st.markdown(
    "**Walmart M5 Dataset** — 30,490 item-store series, 3 forecasting models "
    "(ARIMA/SARIMA, LightGBM, LSTM) compared, with inventory recommendations "
    "(Safety Stock, Reorder Point, EOQ) generated per item."
)

if not data_loaded:
    st.error(
        f"Data files nahi mile. `data/` folder me ye files honi chahiye:\n\n"
        f"- lgb_test_predictions.parquet\n- inventory_recommendations.csv\n"
        f"- model_comparison.csv\n- arima_results.csv (optional)\n\n"
        f"Error: {missing_file_error}"
    )
    st.stop()

st.markdown("---")

# -----------------------------------------------------------------
# Sidebar - Item/Store selector
# -----------------------------------------------------------------
st.sidebar.header("🔍 Select Item & Store")

state_list = sorted(lgb_preds["state_id"].unique())
selected_state = st.sidebar.selectbox("State", state_list)

store_list = sorted(lgb_preds.loc[lgb_preds["state_id"] == selected_state, "store_id"].unique())
selected_store = st.sidebar.selectbox("Store", store_list)

cat_list = sorted(lgb_preds.loc[lgb_preds["store_id"] == selected_store, "cat_id"].unique())
selected_cat = st.sidebar.selectbox("Category", cat_list)

item_list = sorted(
    lgb_preds.loc[
        (lgb_preds["store_id"] == selected_store) & (lgb_preds["cat_id"] == selected_cat),
        "item_id"
    ].unique()
)
selected_item = st.sidebar.selectbox("Item", item_list)

item_row = lgb_preds[
    (lgb_preds["store_id"] == selected_store) & (lgb_preds["item_id"] == selected_item)
].sort_values("date")

selected_id = item_row["id"].iloc[0] if len(item_row) > 0 else None

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: M5 Forecasting - Accuracy (Kaggle). "
    "Forecast model: LightGBM (item-store level). "
    "ARIMA & LSTM shown at aggregate level for baseline comparison."
)

# -----------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Item Forecast", "⚖️ Model Comparison", "🏭 Inventory Recommendation", "🎛️ What-If Simulator"]
)

# ===================================================================
# TAB 1 - Item-level forecast
# ===================================================================
with tab1:
    st.subheader(f"Forecast: {selected_item} @ {selected_store}")

    if len(item_row) == 0:
        st.warning("Is combination ke liye test-period data nahi mila.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=item_row["date"], y=item_row["sales"],
            mode="lines+markers", name="Actual Sales", line=dict(color="#1f77b4")
        ))
        fig.add_trace(go.Scatter(
            x=item_row["date"], y=item_row["pred_lgb"],
            mode="lines+markers", name="LightGBM Forecast", line=dict(color="#ff7f0e", dash="dash")
        ))
        fig.update_layout(
            xaxis_title="Date", yaxis_title="Units Sold",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        item_rmse = np.sqrt(np.mean((item_row["sales"] - item_row["pred_lgb"]) ** 2))
        item_avg_sales = item_row["sales"].mean()
        item_avg_pred = item_row["pred_lgb"].mean()

        col1.metric("Avg Actual Sales / day", f"{item_avg_sales:.1f} units")
        col2.metric("Avg Forecasted Sales / day", f"{item_avg_pred:.1f} units")
        col3.metric("RMSE (this item)", f"{item_rmse:.2f}")

# ===================================================================
# TAB 2 - Model comparison
# ===================================================================
with tab2:
    st.subheader("Model Comparison (CA-FOODS Aggregate, 28-day RMSE)")
    st.caption(
        "ARIMA aur LSTM aggregate (state+category) level pe train hue hain; "
        "LightGBM item-store granular level pe. Fair comparison ke liye LightGBM "
        "ko bhi isi aggregate level pe roll-up kiya gaya hai."
    )

    fig_comp = px.bar(
        comparison, x="Model", y="RMSE", color="Model",
        text_auto=".2f", title="28-day RMSE by Model (lower is better)"
    )
    fig_comp.update_layout(showlegend=False, height=420)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.dataframe(comparison, use_container_width=True)

    if arima_results is not None:
        st.markdown("#### ARIMA Forecast vs Actual (CA-FOODS Aggregate)")
        fig_arima = go.Figure()
        fig_arima.add_trace(go.Scatter(x=arima_results["date"], y=arima_results["actual"],
                                        mode="lines+markers", name="Actual"))
        fig_arima.add_trace(go.Scatter(x=arima_results["date"], y=arima_results["arima_pred"],
                                        mode="lines+markers", name="ARIMA Forecast"))
        fig_arima.update_layout(height=380, xaxis_title="Date", yaxis_title="Units Sold")
        st.plotly_chart(fig_arima, use_container_width=True)

# ===================================================================
# TAB 3 - Inventory recommendation
# ===================================================================
with tab3:
    st.subheader(f"Inventory Recommendation: {selected_item} @ {selected_store}")

    inv_row = inventory[inventory["id"] == selected_id] if selected_id else pd.DataFrame()

    if len(inv_row) == 0:
        st.warning("Is item ke liye inventory data nahi mila.")
    else:
        row = inv_row.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Daily Demand", f"{row['avg_daily_demand']:.1f} units")
        col2.metric("Safety Stock", f"{row['safety_stock']:.1f} units")
        col3.metric("Reorder Point", f"{row['reorder_point']:.1f} units")
        col4.metric("EOQ (Order Qty)", f"{row['eoq']:.0f} units")

        st.info(
            f"**Business reading:** Jab stock **{row['reorder_point']:.0f} units** tak gir jaaye, "
            f"tab naya order place karo. Har baar **~{row['eoq']:.0f} units** order karna cost-efficient hai "
            f"(order cost + holding cost balance karke). **{row['safety_stock']:.0f} units** ka buffer "
            f"demand-uncertainty cover karta hai (95% service level assume kiya gaya hai)."
        )

    st.markdown("---")
    st.markdown("#### All-Items Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Items Analyzed", f"{len(inventory):,}")
    col2.metric("Avg Safety Stock (all items)", f"{inventory['safety_stock'].mean():.2f} units")
    col3.metric("Avg EOQ (all items)", f"{inventory['eoq'].mean():.2f} units")

    with st.expander("Top 15 items by Reorder Point"):
        st.dataframe(
            inventory.sort_values("reorder_point", ascending=False)
            .head(15)[["id", "avg_daily_demand", "safety_stock", "reorder_point", "eoq"]],
            use_container_width=True
        )

# ===================================================================
# TAB 4 - What-if simulator
# ===================================================================
with tab4:
    st.subheader("What-If: Service Level Simulator")
    st.caption(
        "Service level badhane se stockout risk kam hota hai, lekin safety stock (aur holding cost) badh jaata hai. "
        "Neeche slider se dekho kaise trade-off badalta hai is item ke liye."
    )

    if len(inv_row) == 0:
        st.warning("Pehle Tab 3 me ek valid item select karo.")
    else:
        row = inv_row.iloc[0]
        lead_time = st.slider("Lead Time (days)", 1, 14, 3)
        service_level = st.slider("Target Service Level (%)", 80, 99, 95) / 100

        z = norm.ppf(service_level)
        new_safety_stock = z * row["demand_std"] if "demand_std" in inventory.columns else np.nan
        if pd.isna(new_safety_stock):
            demand_std_est = row["safety_stock"] / norm.ppf(0.95) / np.sqrt(3)  # reverse-engineer approx
            new_safety_stock = z * demand_std_est * np.sqrt(lead_time)
        else:
            new_safety_stock = z * new_safety_stock * np.sqrt(lead_time)

        new_reorder_point = (row["avg_daily_demand"] * lead_time) + new_safety_stock

        col1, col2 = st.columns(2)
        col1.metric("New Safety Stock", f"{new_safety_stock:.1f} units",
                     delta=f"{new_safety_stock - row['safety_stock']:.1f}")
        col2.metric("New Reorder Point", f"{new_reorder_point:.1f} units",
                     delta=f"{new_reorder_point - row['reorder_point']:.1f}")

        st.caption(
            "Z-score higher service level ke saath badhta hai — matlab zyada buffer stock chahiye "
            "stockout avoid karne ke liye. Business trade-off: high service level = kam stockouts, "
            "lekin zyada holding cost."
        )

st.markdown("---")
st.caption(
    "Built as a portfolio project | Data: M5 Forecasting - Accuracy (Kaggle) | "
    "Models: ARIMA/SARIMA, LightGBM, LSTM | Inventory formulas: Safety Stock, ROP, EOQ"
)
