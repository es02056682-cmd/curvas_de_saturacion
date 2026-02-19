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
# BRAND STYLE
# =====================================================

st.markdown("""
<style>
body {
    background-color: #0E1117;
}
h1 {
    text-align: center;
    color: white;
}
h2 {
    color: white;
    font-weight: 800;
}
.section-title {
    font-size: 22px;
    font-weight: 800;
    color: white;
    margin-top: 20px;
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
# CONVERSION RATES
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
# CHANNEL COLORS
# =====================================================

CHANNEL_COLORS = {
    "SEM_Marca_Pura": "#6A0DAD",
    "SEM_Marca_Derivada": "#B57EDC",
    "Terceros": "#2E7D32",
    "Pmax": "#1C1C1C",
    "Paid_Social": "#F57C00",
    "SEM_Generico": "#0288D1",
    "SEM_Competencia": "#E91E63",
    "Display": "#7CB342"
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

    df["Leads"] = df["Leads"].fillna(0)
    df = df[df["Spend"] > 0]

    return df

df = load_data()

# =====================================================
# MODEL
# =====================================================

def power_model(x, a, b):
    return a * (x ** b)

def monthly_leads(spend_monthly, a, b):
    return DAYS_IN_MONTH * a * ((spend_monthly / DAYS_IN_MONTH) ** b)

def marginal_cpl(spend_daily, a, b):
    return 1 / (a * b * (spend_daily ** (b - 1)))

# Fit
# Fit con filtro de spend mínimo (anti-outliers)
results = {}

MIN_SPEND_THRESHOLD = 100  # puedes ajustar 50, 100, 200 según tu caso

for canal in df["Canal"].unique():

    data = df[df["Canal"] == canal]

    # 🔹 Eliminamos días con spend demasiado bajo
    data = data[data["Spend"] > MIN_SPEND_THRESHOLD]

    # Solo ajustamos si quedan suficientes datos
    if len(data) > 10:

        params, _ = curve_fit(
            power_model,
            data["Spend"],
            data["Leads"],
            maxfev=20000
        )

        results[canal] = {
            "a": params[0],
            "b": params[1]
        }


params_df = pd.DataFrame(results).T

# =====================================================
# HEADER
# =====================================================

st.markdown("<h1>📊 Performance Scaling Dashboard</h1>", unsafe_allow_html=True)

# =====================================================
# GLOBAL CPL & CPV CURVES
# =====================================================

st.markdown('<div class="section-title">Visión General – CPL y CPV por Canal</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

spend_range = np.linspace(100, df["Spend"].max()*DAYS_IN_MONTH*1.3, 200)

fig_cpl = go.Figure()
fig_cpv = go.Figure()

for canal in params_df.index:
    a = params_df.loc[canal, "a"]
    b = params_df.loc[canal, "b"]
    cr = CR_VENTA.get(canal, 0.05)

    leads_curve = monthly_leads(spend_range, a, b)
    ventas_curve = leads_curve * cr

    cpl_curve = spend_range / leads_curve
    cpv_curve = spend_range / ventas_curve

    fig_cpl.add_trace(go.Scatter(
        x=spend_range,
        y=cpl_curve,
        mode="lines",
        name=canal,
        line=dict(color=CHANNEL_COLORS.get(canal, "#999999"), width=3)
    ))

    fig_cpv.add_trace(go.Scatter(
        x=spend_range,
        y=cpv_curve,
        mode="lines",
        name=canal,
        line=dict(color=CHANNEL_COLORS.get(canal, "#999999"), width=3)
    ))

fig_cpl.update_layout(
    title="CPL por Canal",
    template="plotly_dark",
    xaxis_title="Spend mensual (€)",
    yaxis_title="CPL (€)"
)

fig_cpv.update_layout(
    title="CPV por Canal",
    template="plotly_dark",
    xaxis_title="Spend mensual (€)",
    yaxis_title="CPV (€)"
)

col1.plotly_chart(fig_cpl, use_container_width=True)
col2.plotly_chart(fig_cpv, use_container_width=True)

# =====================================================
# CHANNEL SIMULATOR
# =====================================================

st.markdown('<div class="section-title">Simulador por Canal</div>', unsafe_allow_html=True)

colA, colB = st.columns(2)

with colA:
    canal = st.selectbox("Selecciona canal", params_df.index)

with colB:
    extra_budget = st.slider("Presupuesto extra mensual (€)", 0, 50000, 5000, step=1000)

a = params_df.loc[canal, "a"]
b = params_df.loc[canal, "b"]
cr = CR_VENTA.get(canal, 0.05)

current_daily_spend = df[df["Canal"] == canal]["Spend"].mean()
current_monthly_spend = current_daily_spend * DAYS_IN_MONTH
new_monthly_spend = current_monthly_spend + extra_budget

current_leads = monthly_leads(current_monthly_spend, a, b)
new_leads = monthly_leads(new_monthly_spend, a, b)

ventas_actuales = current_leads * cr
ventas_nuevas = new_leads * cr

incremental_leads = new_leads - current_leads
incremental_ventas = ventas_nuevas - ventas_actuales

new_cpl = new_monthly_spend / new_leads
new_cpv = new_monthly_spend / ventas_nuevas

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Leads Incrementales", f"{incremental_leads:,.0f}")
col2.metric("Ventas Incrementales", f"{incremental_ventas:,.0f}")
col3.metric("Elasticidad (b)", f"{b:.2f}")

# =====================================================
# SINGLE CHANNEL CURVE
# =====================================================

st.markdown('<div class="section-title">Curva del Canal Seleccionado</div>', unsafe_allow_html=True)

single_spend = np.linspace(0, current_monthly_spend*1.6, 200)
single_leads = monthly_leads(single_spend, a, b)
single_sales = single_leads * cr

fig_single = go.Figure()

fig_single.add_trace(go.Scatter(
    x=single_spend,
    y=single_leads,
    mode="lines",
    name="Leads",
    line=dict(color="#0288D1", width=4)
))

fig_single.add_trace(go.Scatter(
    x=single_spend,
    y=single_sales,
    mode="lines",
    name="Ventas",
    line=dict(color="#F57C00", width=4)
))

fig_single.update_layout(
    template="plotly_dark",
    xaxis_title="Spend mensual (€)",
    yaxis_title="Volumen"
)

st.plotly_chart(fig_single, use_container_width=True)

