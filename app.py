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
body { background-color: #0E1117; }
h1 { text-align: center; color: white; }
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
# PUSH / PULL DEFINITION
# =====================================================

PULL_CHANNELS = ["SEM_Marca_Pura", "SEM_Marca_Derivada"]

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

# =====================================================
# FIT ROBUSTO
# =====================================================

results = {}
MIN_SPEND_THRESHOLD = 100

for canal in df["Canal"].unique():

    data = df[df["Canal"] == canal]
    data = data[data["Spend"] > MIN_SPEND_THRESHOLD]

    if len(data) > 10:

        params, _ = curve_fit(
            power_model,
            data["Spend"],
            data["Leads"],
            bounds=([0, 0], [np.inf, 1]),
            maxfev=20000
        )

        results[canal] = {"a": params[0], "b": params[1]}

params_df = pd.DataFrame(results).T

# Añadir tipo Push/Pull
params_df["Tipo"] = [
    "Pull" if canal in PULL_CHANNELS else "Push"
    for canal in params_df.index
]

# Inversión promedio mensual actual
avg_spend_df = (
    df.groupby("Canal")["Spend"]
    .mean()
    .reset_index()
)

avg_spend_df["Spend_Mensual_Promedio"] = avg_spend_df["Spend"] * DAYS_IN_MONTH

params_df = params_df.merge(
    avg_spend_df[["Canal", "Spend_Mensual_Promedio"]],
    left_index=True,
    right_on="Canal",
    how="left"
).set_index("Canal")

# =====================================================
# HEADER
# =====================================================

st.markdown("<h1>📊 Performance Scaling Dashboard</h1>", unsafe_allow_html=True)

# =====================================================
# GLOBAL CPL & CPV CURVES
# =====================================================

st.markdown('<div class="section-title">Visión General – CPL y CPV por Canal</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

avg_monthly_spend = df["Spend"].mean() * DAYS_IN_MONTH
min_real_spend = avg_monthly_spend * 0.3

spend_range = np.linspace(
    min_real_spend,
    df["Spend"].max() * DAYS_IN_MONTH * 1.3,
    200
)

fig_cpl = go.Figure()
fig_cpv = go.Figure()

for canal in params_df.index:

    a = params_df.loc[canal, "a"]
    b = params_df.loc[canal, "b"]
    cr = CR_VENTA.get(canal, 0.05)

    leads_curve = monthly_leads(spend_range, a, b)
    ventas_curve = leads_curve * cr

    fig_cpl.add_trace(go.Scatter(
        x=spend_range,
        y=spend_range / leads_curve,
        mode="lines",
        name=canal
    ))

    fig_cpv.add_trace(go.Scatter(
        x=spend_range,
        y=spend_range / ventas_curve,
        mode="lines",
        name=canal
    ))

fig_cpl.update_layout(template="plotly_dark", title="CPL por Canal")
fig_cpv.update_layout(template="plotly_dark", title="CPV por Canal")

col1.plotly_chart(fig_cpl, use_container_width=True)
col2.plotly_chart(fig_cpv, use_container_width=True)

# =====================================================
# SIMULADOR INDIVIDUAL
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

current_monthly_spend = params_df.loc[canal, "Spend_Mensual_Promedio"]
new_monthly_spend = current_monthly_spend + extra_budget

current_leads = monthly_leads(current_monthly_spend, a, b)
new_leads = monthly_leads(new_monthly_spend, a, b)

ventas_actuales = current_leads * cr
ventas_nuevas = new_leads * cr

col1, col2, col3 = st.columns(3)
col1.metric("Inversión mensual promedio actual", f"{current_monthly_spend:,.0f} €")
col2.metric("Leads Incrementales", f"{(new_leads-current_leads):,.0f}")
col3.metric("Ventas Incrementales", f"{(ventas_nuevas-ventas_actuales):,.0f}")

# =====================================================
# CURVA INDIVIDUAL
# =====================================================

st.markdown('<div class="section-title">Curva del Canal Seleccionado</div>', unsafe_allow_html=True)

single_spend = np.linspace(0, current_monthly_spend * 1.6, 200)

fig_single = go.Figure()

fig_single.add_trace(go.Scatter(
    x=single_spend,
    y=monthly_leads(single_spend, a, b),
    mode="lines",
    name="Leads"
))

fig_single.add_trace(go.Scatter(
    x=single_spend,
    y=monthly_leads(single_spend, a, b) * cr,
    mode="lines",
    name="Ventas"
))

fig_single.update_layout(template="plotly_dark")
st.plotly_chart(fig_single, use_container_width=True)

# =====================================================
# OPTIMIZADOR AUTOMÁTICO PUSH
# =====================================================

st.markdown('<div class="section-title">Optimización Automática Presupuesto Push</div>', unsafe_allow_html=True)

extra_total = st.slider(
    "Presupuesto total a distribuir en Push (€)",
    0, 100000, 20000, step=5000
)

push_channels = params_df[params_df["Tipo"] == "Push"].index

allocation = {}
efficiency_scores = []

for canal in push_channels:

    a = params_df.loc[canal, "a"]
    b = params_df.loc[canal, "b"]

    current_daily_spend = df[df["Canal"] == canal]["Spend"].mean()
    mcpl = marginal_cpl(current_daily_spend, a, b)

    efficiency_scores.append((canal, mcpl))

efficiency_scores.sort(key=lambda x: x[1])

total_inverse = sum(1 / score[1] for score in efficiency_scores)

for canal, mcpl in efficiency_scores:
    weight = (1 / mcpl) / total_inverse
    allocation[canal] = weight * extra_total

allocation_df = pd.DataFrame.from_dict(
    allocation,
    orient="index",
    columns=["Asignación Recomendada (€)"]
)

st.dataframe(allocation_df.style.format("{:,.0f}"))
