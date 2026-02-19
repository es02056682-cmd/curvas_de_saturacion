import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Performance Scaling Dashboard",
    layout="wide"
)

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
# FIT MODEL BY CHANNEL
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

st.markdown("""
<h1 style='text-align: center;'>📊 Performance Scaling Dashboard</h1>
<hr>
""", unsafe_allow_html=True)

# =====================================================
# USER INPUT
# =====================================================

colA, colB = st.columns([2,2])

with colA:
    canal = st.selectbox("Selecciona canal", params_df.index)

with colB:
    extra_budget = st.slider(
        "Presupuesto extra mensual (€)",
        min_value=0,
        max_value=50000,
        value=5000,
        step=1000
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

# =====================================================
# METRICS SECTION
# =====================================================

st.markdown("### 📈 Impacto en Leads")

col1, col2, col3 = st.columns(3)

col1.metric("Leads Incrementales", f"{incremental_leads:,.0f}")
col2.metric("Nuevo CPL (€)", f"{new_cpl:.2f}")
col3.metric("Elasticidad (b)", f"{b:.2f}")

st.markdown("### 💰 Impacto en Ventas")

col4, col5, col6 = st.columns(3)

col4.metric("Ventas Incrementales", f"{incremental_ventas:,.0f}")
col5.metric("Nuevo CPV (€)", f"{new_cpv:.2f}")
col6.metric("CR Venta", f"{cr*100:.2f}%")

# =====================================================
# STATUS INDICATORS
# =====================================================

if new_cpl <= TARGET_CPL:
    st.success("🟢 CPL dentro de objetivo")
elif new_cpl <= TARGET_CPL * 1.15:
    st.warning("🟡 CPL acercándose al límite")
else:
    st.error("🔴 Canal saturado en captación")

if new_cpv <= TARGET_CPV:
    st.success("🟢 CPV dentro de objetivo")
else:
    st.warning("⚠️ CPV por encima del objetivo")

# =====================================================
# VISUALIZATION
# =====================================================

spend_range = np.linspace(0, current_monthly_spend * 1.6, 200)

leads_range = monthly_leads(spend_range, a, b)
ventas_range = leads_range * cr

chart_df = pd.DataFrame({
    "Spend": spend_range,
    "Leads": leads_range,
    "Ventas": ventas_range
}).set_index("Spend")

st.markdown("### 📊 Curvas de Saturación")

st.line_chart(chart_df)
