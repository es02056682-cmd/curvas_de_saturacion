import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go

# =====================================================
# CONFIGURACIÓN
# =====================================================

st.set_page_config(
    page_title="Performance Scaling Dashboard",
    layout="wide"
)

st.markdown("""
<style>
body { background-color: #0E1117; }
h1 { text-align: center; color: white; }
.section-title {
    font-size: 22px;
    font-weight: 800;
    color: white;
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)

DAYS_IN_MONTH = 30
TARGET_CPV = 120

PULL_CHANNELS = ["SEM_Marca_Pura", "SEM_Marca_Derivada"]

# Canales que gestionáis internamente (prioritarios)
PRIORITY_CHANNELS = [
    "SEM_Generico",
    "Paid_Social",
    "Pmax",
    "Display",
    "SEM_Competencia"
]

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
# CARGA DE DATOS
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
# MODELO
# =====================================================

def power_model(x, a, b):
    return a * (x ** b)

def monthly_leads(spend_monthly, a, b):
    return DAYS_IN_MONTH * a * ((spend_monthly / DAYS_IN_MONTH) ** b)

def marginal_cpl(spend_daily, a, b):
    return 1 / (a * b * (spend_daily ** (b - 1)))

# =====================================================
# AJUSTE
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

params_df["Tipo"] = [
    "Pull" if canal in PULL_CHANNELS else "Push"
    for canal in params_df.index
]

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
# OPTIMIZADOR PUSH PRIORIZADO
# =====================================================

st.markdown('<div class="section-title">Optimización Presupuesto Push (Prioridad Interna)</div>', unsafe_allow_html=True)

extra_total = st.slider(
    "Presupuesto total a distribuir en Push (€)",
    0, 100000, 20000, step=5000
)

push_channels = params_df[params_df["Tipo"] == "Push"].index

valid_channels = []

for canal in push_channels:

    a = params_df.loc[canal, "a"]
    b = params_df.loc[canal, "b"]
    cr = CR_VENTA.get(canal, 0.05)

    current_daily_spend = params_df.loc[canal, "Spend_Mensual_Promedio"] / DAYS_IN_MONTH

    mcpl = marginal_cpl(current_daily_spend, a, b)
    mcpv = mcpl / cr

    if mcpv <= TARGET_CPV:
        valid_channels.append((canal, mcpv))

if len(valid_channels) == 0:
    st.warning("Ningún canal Push cumple criterio de rentabilidad marginal.")
else:

    # Separar prioritarios
    priority = [c for c in valid_channels if c[0] in PRIORITY_CHANNELS]
    others = [c for c in valid_channels if c[0] not in PRIORITY_CHANNELS]

    allocation = {}

    def allocate(channel_list, budget):
        total_inverse = sum(1 / c[1] for c in channel_list)
        result = {}
        for canal, mcpv in channel_list:
            weight = (1 / mcpv) / total_inverse
            result[canal] = weight * budget
        return result

    if priority:
        allocation = allocate(priority, extra_total)
    else:
        allocation = allocate(valid_channels, extra_total)

    allocation_df = pd.DataFrame.from_dict(
        allocation,
        orient="index",
        columns=["Asignación Recomendada (€)"]
    )

    st.dataframe(allocation_df.style.format("{:,.0f}"))
