"""
Phase 7 — Streamlit fraud monitoring dashboard.
Pulls live data from PostgreSQL prediction log.

Run: streamlit run dashboard/app.py
Env: POSTGRES_URL, API_URL
"""
import os
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

POSTGRES_URL = os.environ.get("POSTGRES_URL", "postgresql://fmcc:fmcc@localhost:5432/fmcc")
API_URL      = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="FMCC Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DB connection (cached) ────────────────────────────────────────────────────
@st.cache_resource
def get_engine():
    return create_engine(POSTGRES_URL)


@st.cache_data(ttl=300)
def load_predictions(days: int = 14) -> pd.DataFrame:
    engine = get_engine()
    since = date.today() - timedelta(days=days)
    q = f"""
        SELECT date, msisdn, fraud_probability, is_fraud, risk_tier, model_version, predicted_at
        FROM prediction_log
        WHERE date >= '{since}'
        ORDER BY date DESC, fraud_probability DESC
    """
    return pd.read_sql(q, engine)


@st.cache_data(ttl=300)
def load_drift_reports(days: int = 30) -> pd.DataFrame:
    engine = get_engine()
    since = date.today() - timedelta(days=days)
    q = f"""
        SELECT report_date, feature, drift_score, drift_detected
        FROM drift_report
        WHERE report_date >= '{since}'
        ORDER BY report_date DESC
    """
    return pd.read_sql(q, engine)


def check_api_health():
    try:
        import requests
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json()
    except Exception:
        return {"status": "unreachable", "model_loaded": False, "model_version": "—"}


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("FMCC Fraud Monitor")
st.sidebar.markdown("---")
window = st.sidebar.slider("Lookback (days)", 1, 30, 14)
risk_filter = st.sidebar.multiselect("Risk tiers", ["LOW", "MEDIUM", "HIGH"], default=["HIGH", "MEDIUM"])
st.sidebar.markdown("---")

health = check_api_health()
status_color = "🟢" if health["status"] == "ok" else "🔴"
st.sidebar.markdown(f"**API Status:** {status_color} {health['status']}")
st.sidebar.markdown(f"**Model:** `{health.get('model_version','—')}`")
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_predictions(window)
drift_df = load_drift_reports(window)

if df.empty:
    st.warning("No prediction data yet. Run the scoring pipeline first.")
    st.stop()

df_filtered = df[df["risk_tier"].isin(risk_filter)] if risk_filter else df
df["date"] = pd.to_datetime(df["date"])

# ── Header KPIs ───────────────────────────────────────────────────────────────
st.title("Telecom Fraud Detection — Live Dashboard")
st.caption(f"Data from last {window} days · {len(df):,} scored MSISDNs")
st.markdown("---")

c1, c2, c3, c4, c5 = st.columns(5)
total_scored  = len(df)
total_fraud   = df["is_fraud"].sum()
fraud_rate    = total_fraud / total_scored * 100 if total_scored else 0
avg_conf      = df[df["is_fraud"]]["fraud_probability"].mean() if total_fraud else 0
drift_days    = drift_df[drift_df["drift_detected"]]["report_date"].nunique() if not drift_df.empty else 0

c1.metric("MSISDNs Scored",  f"{total_scored:,}")
c2.metric("Fraud Flagged",   f"{total_fraud:,}")
c3.metric("Fraud Rate",      f"{fraud_rate:.2f}%")
c4.metric("Avg Fraud Conf.", f"{avg_conf:.2%}" if avg_conf else "—")
c5.metric("Drift Days",      str(drift_days), delta=f"{drift_days} alerts", delta_color="inverse")

st.markdown("---")

# ── Row 1: Daily fraud trend + Risk distribution ──────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Daily Fraud Rate")
    daily = (
        df.groupby("date")
        .agg(scored=("msisdn", "count"), fraud=("is_fraud", "sum"))
        .reset_index()
    )
    daily["rate"] = daily["fraud"] / daily["scored"] * 100

    fig = go.Figure()
    fig.add_bar(x=daily["date"], y=daily["scored"], name="Scored", marker_color="#1f77b4", opacity=0.4)
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["rate"], name="Fraud %",
        yaxis="y2", mode="lines+markers", line=dict(color="#d62728", width=2)
    ))
    fig.update_layout(
        yaxis=dict(title="MSISDNs"),
        yaxis2=dict(title="Fraud Rate %", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        height=350, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Risk Tier Breakdown")
    tier_counts = df["risk_tier"].value_counts().reset_index()
    tier_counts.columns = ["tier", "count"]
    colors = {"HIGH": "#d62728", "MEDIUM": "#ff7f0e", "LOW": "#2ca02c"}
    fig2 = px.pie(
        tier_counts, values="count", names="tier",
        color="tier", color_discrete_map=colors,
        hole=0.45,
    )
    fig2.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Confidence distribution + Top flagged MSISDNs ─────────────────────
col3, col4 = st.columns([1, 1])

with col3:
    st.subheader("Fraud Probability Distribution")
    fig3 = px.histogram(
        df, x="fraud_probability", color="is_fraud",
        barmode="overlay", nbins=40, opacity=0.75,
        color_discrete_map={True: "#d62728", False: "#1f77b4"},
        labels={"is_fraud": "Flagged as Fraud"},
    )
    fig3.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Top 10 Highest-Risk MSISDNs")
    top10 = (
        df[df["is_fraud"]]
        .sort_values("fraud_probability", ascending=False)
        .head(10)[["msisdn", "date", "fraud_probability", "risk_tier"]]
    )
    top10["fraud_probability"] = top10["fraud_probability"].map("{:.2%}".format)
    st.dataframe(top10, use_container_width=True, height=310)

# ── Row 3: Feature drift alerts ───────────────────────────────────────────────
st.markdown("---")
st.subheader("Feature Drift Monitoring (Evidently AI)")

if drift_df.empty:
    st.info("No drift reports yet. Run monitoring/drift_monitor.py to generate them.")
else:
    drift_pivot = (
        drift_df.groupby(["report_date", "feature"])["drift_detected"]
        .any()
        .reset_index()
    )
    # Heatmap: features × dates
    pivot = drift_pivot.pivot(index="feature", columns="report_date", values="drift_detected").fillna(False)
    fig4 = px.imshow(
        pivot.astype(int),
        color_continuous_scale=["#2ca02c", "#d62728"],
        aspect="auto",
        labels=dict(color="Drift"),
    )
    fig4.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        most_drifted = drift_df[drift_df["drift_detected"]]["feature"].value_counts().head(8)
        if not most_drifted.empty:
            st.markdown("**Most frequently drifted features**")
            st.bar_chart(most_drifted)
    with col6:
        recent_alerts = drift_df[drift_df["drift_detected"]].sort_values("report_date", ascending=False).head(20)
        st.markdown("**Recent drift alerts**")
        st.dataframe(recent_alerts[["report_date", "feature", "drift_score"]], use_container_width=True)

# ── Filtered table ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"Prediction Log — {', '.join(risk_filter) if risk_filter else 'All'} Risk")
st.dataframe(
    df_filtered[["date", "msisdn", "fraud_probability", "risk_tier", "model_version"]]
    .sort_values(["date", "fraud_probability"], ascending=[False, False])
    .head(500),
    use_container_width=True,
)
