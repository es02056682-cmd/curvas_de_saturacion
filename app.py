import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# ======================================
# CONFIG
# ======================================

DAYS_IN_MONTH = 30
TARGET_CPL = 40

st.set_page_config(page_title="Lead Scaling Simulator", layout="wide")

st.title("📈 Lead Scaling Simulator")

# ======================================
# LOAD DATA
# ======================================

@st.cache_data
def load_data():
    df = pd.read_csv("Raw_data_curvas.csv")
    df = df.rename(columns={
        "SupraCanal": "Canal",
        "Inversion_Fee": "Spend",
        "Leads Brutos": "Leads"
    })
    df["Leads"] = df["Leads"].fillna(0)
    df = df[df["Spend"] > 0]
    return df

df = load_data()

# ======================================
# MODEL
# ======================================

def power_model(x, a, b):
    return a * (x ** b)

def monthly_leads(spend_monthly, a, b):
    return DAYS_IN_MONTH * a * ((spend_monthly / DAYS_IN_MONTH) ** b)

def marginal_cpl(spend_daily, a, b):
    marginal_leads = a * b * (spend_daily ** (b - 1))
    return 1 / marginal_leads

# Fit por canal
results = {}

for canal in df["Canal"].unique():
    data = df[df["Canal"] == canal]
    x = data["Spend"].values
    y = data["Leads"].values
    if len(x) > 10:
        params, _ = curve_fit(power_model, x, y, maxfev=20000)
        results[canal] = {"a": params[0], "b": params[1]}

params_df = pd.DataFrame(results).T

# ======================================
# UI
# ======================================

canal = st.selectbox("Selecciona canal", params_df.index)

extra_budget = st.slider(
    "Presupuesto extra mensual (€)",
    min_value=0,
    max_value=30000,
    step=1000,
    value=5000
)

a = params_df.loc[canal, "a"]
b = params_df.loc[canal, "b"]

current_daily_spend = df[df["Canal"] == canal]["Spend"].mean()
current_monthly_spend = current_daily_spend * DAYS_IN_MONTH

new_monthly_spend = current_monthly_spend + extra_budget

current_leads = monthly_leads(current_monthly_spend, a, b)
new_leads = monthly_leads(new_monthly_spend, a, b)

incremental_leads = new_leads - current_leads
new_cpl = new_monthly_spend / new_leads

new_daily_spend = new_monthly_spend / DAYS_IN_MONTH
mcpl = marginal_cpl(new_daily_spend, a, b)

# ======================================
# EXECUTIVE METRICS
# ======================================

col1, col2, col3 = st.columns(3)

col1.metric("Leads Incrementales", f"{incremental_leads:,.0f}")
col2.metric("Nuevo CPL (€)", f"{new_cpl:.2f}")
col3.metric("CPL Marginal (€)", f"{mcpl:.2f}")

if mcpl <= TARGET_CPL:
    st.success("🟢 Escalable")
elif mcpl <= TARGET_CPL * 1.15:
    st.warning("🟡 Zona de riesgo")
else:
    st.error("🔴 Saturado")

# ======================================
# CURVA VISUAL
# ======================================

spend_range = np.linspace(0, current_monthly_spend * 1.6, 200)
leads_range = monthly_leads(spend_range, a, b)

st.line_chart(pd.DataFrame({
    "Spend": spend_range,
    "Leads": leads_range
}).set_index("Spend"))
