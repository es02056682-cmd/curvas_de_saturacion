import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Performance Scaling Dashboard",
    layout="wide"
)

# =====================================================
# BRAND STYLING (Movistar Prosegur Inspired)
# =====================================================

st.markdown("""
<style>
    body {
        background-color: #ffffff;
    }
    h1, h2, h3 {
        color: #003399;
    }
    .section-title {
        font-size: 22px;
        font-weight: 600;
        color: #003399;
        margin-top: 20px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# GLOBAL CONFIG
# =====================================================

DAYS_IN_MONTH = 30
TARGET_CPL = 40
TARGET_CPV = 120

# =====================================================
# CONVERSION RATE DICTIONARY
# =====================================================

CR_VENTA = {
    "Display": 0.0473,
    "Paid_Social": 0.0368,
    "Pmax": 0.0714,
    "SEM_Competencia": 0.0434,
    "SEM_Generico": 0.1399,
    "SEM_Marca_Derivada": 0.0456,
    "SEM_Marca_Pura": 0.0912,
    "Terceros": 0.0502
}

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    df = pd.read_csv("Raw_data_curvas.csv")
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "SupraCanal": "Canal",
        "Inversion_Fee": "Spend",
        "Leads Brutos": "Leads"
    })

    required_cols = ["Canal", "Spend", "Leads"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Falta la columna obligatoria: {col}")
            st.stop()

    df["Leads"] = df["Leads"].fillna(0)
    df = df[df["Spend"] > 0]

    return df

df = load_data()

# =====================================================
# MODEL FUNCTIONS
# =====================================================

def power_model(x, a, b):
    return a * (x ** b)

def monthly_leads(spend_monthly, a, b):
    return DAYS_IN_MONTH * a * ((spend_monthly / DAYS_IN_MONTH) ** b)

def marginal_cpl(spend_daily, a, b):
    return 1 / (a * b * (spend_daily ** (b - 1)))

# =====================================================
# FIT MODEL
# =====================================================

results = {}

for canal in df["Canal"].unique():
    data = df[df["Canal"] == canal]
    x = data["Spend"].values
    y = data["Leads"].values

    if len(x) > 10:
        params, _ = curve_fit(power_model, x, y, maxfev=20000)
        results[canal] = {"a": params[0], "b": params[1]}

params_df = pd.DataFrame(results).T

# =====================================================
# HEADER
# =====================================================

st.markdown("<h1 style='text-align:center;'>📊 Performance Scaling Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# =====================================================
# GLOBAL CURVES (ALL CHANNELS)
# =====================================================

st.markdown('<div class="section-title">Visión General – Curvas por Canal</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

fig_leads = go.Figure()
fig_sales = go.Figure()

spend_range = np.linspace(0, df["Spend"].max()*DAYS_IN_MONTH*1.3, 200)

for canal in params_df.index:
    a = params_df.loc[canal, "a"]
    b = params_df.loc[canal, "b"]
    cr = CR_VENTA.get(canal, 0.05)

    leads_curve = monthly_leads(spend_range, a, b)
    sales_curve = leads_curve * cr

    fig_leads.add_trace(go.Scatter(
        x=spend_range,
        y=leads_curve,
        mode="lines",
        name=canal
    ))

    fig_sales.add_trace(go.Scatter(
        x=spend_range,
        y=sales_curve,
        mode="lines",
        name=canal
    ))

fig_leads.update_layout(
    title="Curvas de Leads",
    xaxis_title="Spend mensual (€)",
    yaxis_title="Leads estimados",
    template="simple_white"
)

fig_sales.update_layout(
    title="Curvas de Ventas",
    xaxis_title="Spend mensual (€)",
    yaxis_title="Ventas estimadas",
    template="simple_white"
)

col1.plotly_chart(fig_leads, use_container_width=True)
col2.plotly_chart(fig_sales, use_container_width=True)

st.markdown("---")

# =====================================================
# CHANNEL SIMULATOR
# =====================================================

st.markdown('<div class="section-title">Simulador por Canal</div>', unsafe_allow_html=True)

colA, colB = st.columns([2,2])

with colA:
    canal = st.selectbox("Selecciona canal", params_df.index)

with colB:
    extra_budget = st.slider(
        "Presupuesto extra mensual (€)",
        0, 50000, 5000, step=1000
    )

# =====================================================
# CALCULATIONS
# =====================================================

a = params_df.loc[canal, "a"]
b = params_df.loc[canal, "b"]
cr = CR_VENTA.get(canal, 0.05)

current_daily_spend = df[df["Canal"] == canal]["Spend"].mean()
current_monthly_spend = current_daily_spend * DAYS_IN_MONTH
new_monthly_spend = current_monthly_spend + extra_budget

current_leads = monthly_leads(current_monthly_spend, a, b)
new_leads = monthly_leads(new_monthly_spend, a, b)

incremental_leads = new_leads - current_leads
new_cpl = new_monthly_spend / new_leads

ventas_actuales = current_leads * cr
ventas_nuevas = new_leads * cr
incremental_ventas = ventas_nuevas - ventas_actuales
new_cpv = new_monthly_spend / ventas_nuevas

new_daily_spend = new_monthly_spend / DAYS_IN_MONTH
mcpl = marginal_cpl(new_daily_spend, a, b)

# =====================================================
# METRICS
# =====================================================

col1, col2, col3 = st.columns(3)
col1.metric("Leads Incrementales", f"{incremental_leads:,.0f}")
col2.metric("Nuevo CPL (€)", f"{new_cpl:.2f}")
col3.metric("Elasticidad (b)", f"{b:.2f}")

col4, col5, col6 = st.columns(3)
col4.metric("Ventas Incrementales", f"{incremental_ventas:,.0f}")
col5.metric("Nuevo CPV (€)", f"{new_cpv:.2f}")
col6.metric("CPL Marginal (€)", f"{mcpl:.2f}")

# =====================================================
# SATURATION STATUS (Marginal-Based)
# =====================================================

if mcpl > TARGET_CPL:
    st.error("🔴 Canal saturado en captación (CPL marginal > target)")
elif new_cpl > TARGET_CPL:
    st.warning("🟡 CPL medio por encima del objetivo")
else:
    st.success("🟢 Canal escalable")

if new_cpv > TARGET_CPV:
    st.warning("⚠️ CPV por encima del objetivo")
