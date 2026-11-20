"""
Investify — Pronóstico con Deep Learning y trading simulado
Proyecto final · Deep Learning para Finanzas

Ejecutar:  streamlit run app.py
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import nucleo as nc
import mexc_bot as mb

RUTA_LOGO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.png")
st.set_page_config(page_title="Investify · Pronóstico y trading con IA",
                   page_icon=RUTA_LOGO if os.path.exists(RUTA_LOGO) else "📈",
                   layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Identidad visual de Investify
# ---------------------------------------------------------------------------
VERDE, ROJO, AZUL, NARANJA, GRIS = "#10b981", "#f43f5e", "#6366f1", "#f59e0b", "#64748b"
INDIGO, TURQUESA = "#6366f1", "#14b8a6"

LOGO_SVG = """<svg width="{s}" height="{s}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6366f1"/>
<stop offset="1" stop-color="#14b8a6"/></linearGradient></defs>
<rect width="64" height="64" rx="16" fill="url(#g)"/>
<polyline points="14,44 25,33 34,39 50,21" fill="none" stroke="#fff" stroke-width="5.5"
 stroke-linecap="round" stroke-linejoin="round"/>
<polygon points="53,17 41,19 51,29" fill="#fff"/></svg>"""

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp, .stMarkdown, p, li, label, h1, h2, h3, h4, h5, input, textarea, button,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-baseweb="tab"] {
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}
[data-testid="stIconMaterial"] {font-family: 'Material Symbols Rounded' !important;}

/* Fondo y contenedor */
.stApp {background: radial-gradient(1200px 600px at 10% -10%, rgba(99,102,241,0.10), transparent 60%),
                     radial-gradient(900px 500px at 100% 0%, rgba(20,184,166,0.08), transparent 60%), #070b14;}
.block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1400px;}
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stAppDeployButton"] {visibility: hidden; display: none;}
header[data-testid="stHeader"] {background: transparent;}

/* Barra lateral */
[data-testid="stSidebar"] {background: linear-gradient(180deg, #0c1222 0%, #090e1a 100%);
    border-right: 1px solid rgba(148,163,184,0.10);}
[data-testid="stSidebar"] .marca {display:flex; align-items:center; gap:10px; margin: 4px 0 2px 0;}
[data-testid="stSidebar"] .marca span {font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em;}
.etiqueta-lateral {font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
    color: #64748b; margin: 18px 0 6px 0;}

/* Encabezado */
.encabezado {display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
    padding: 18px 22px; border-radius: 20px; margin-bottom: 14px;
    background: linear-gradient(120deg, rgba(99,102,241,0.16), rgba(20,184,166,0.10));
    border: 1px solid rgba(148,163,184,0.14);}
.encabezado .izq {display:flex; align-items:center; gap:14px;}
.encabezado h1 {font-size: 1.9rem; font-weight: 800; letter-spacing: -0.03em; margin: 0; padding: 0; line-height: 1.1;}
.encabezado h1 .acento {background: linear-gradient(90deg, #818cf8, #2dd4bf); -webkit-background-clip: text;
    background-clip: text; color: transparent;}
.encabezado p {margin: 2px 0 0 0; color: #94a3b8; font-size: 0.92rem;}
.estado {display:flex; gap:8px; flex-wrap:wrap;}
.pastilla {padding: 6px 12px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; color: #cbd5e1;
    background: rgba(15,23,42,0.7); border: 1px solid rgba(148,163,184,0.18);}
.punto {display:inline-block; width:8px; height:8px; border-radius:50%; background:#10b981; margin-right:6px;
    box-shadow: 0 0 0 0 rgba(16,185,129,0.6); animation: latido 2s infinite;}
@keyframes latido {0% {box-shadow: 0 0 0 0 rgba(16,185,129,0.55);} 70% {box-shadow: 0 0 0 8px rgba(16,185,129,0);}
    100% {box-shadow: 0 0 0 0 rgba(16,185,129,0);}}

/* Cinta de precios */
.cinta {overflow: hidden; white-space: nowrap; border-radius: 14px; margin-bottom: 18px; padding: 9px 0;
    background: rgba(15,23,42,0.65); border: 1px solid rgba(148,163,184,0.12);
    mask-image: linear-gradient(90deg, transparent, #000 4%, #000 96%, transparent);
    -webkit-mask-image: linear-gradient(90deg, transparent, #000 4%, #000 96%, transparent);}
.cinta-pista {display: inline-block; animation: desliza 90s linear infinite;}
.cinta:hover .cinta-pista {animation-play-state: paused;}
.cinta .item {display:inline-block; margin: 0 22px; font-size: 0.86rem; color: #cbd5e1;}
.cinta .item b {color: #f1f5f9; font-weight: 700; margin-right: 6px;}
.sube {color: #10b981; font-weight: 600;} .baja {color: #f43f5e; font-weight: 600;}
@keyframes desliza {from {transform: translateX(0);} to {transform: translateX(-50%);}}

/* Pestañas tipo "pill" */
.stTabs [data-baseweb="tab-list"] {gap: 6px; background: rgba(15,23,42,0.6); padding: 6px; border-radius: 14px;
    border: 1px solid rgba(148,163,184,0.12);}
.stTabs [data-baseweb="tab"] {height: 40px; padding: 0 18px; border-radius: 10px; color: #94a3b8; font-weight: 600;}
.stTabs [data-baseweb="tab"]:hover {color: #e2e8f0; background: rgba(148,163,184,0.08);}
.stTabs [aria-selected="true"] {color: #fff !important;
    background: linear-gradient(90deg, rgba(99,102,241,0.9), rgba(20,184,166,0.85)) !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}
.stTabs [data-baseweb="tab-panel"] {padding-top: 18px;}

/* Tarjetas de métricas */
[data-testid="stMetric"] {background: linear-gradient(180deg, rgba(30,41,59,0.55), rgba(15,23,42,0.55));
    border: 1px solid rgba(148,163,184,0.14); border-radius: 16px; padding: 16px 18px;
    transition: transform .15s ease, border-color .15s ease;}
[data-testid="stMetric"]:hover {transform: translateY(-2px); border-color: rgba(99,102,241,0.5);}
[data-testid="stMetricLabel"] p {font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #94a3b8;}
[data-testid="stMetricValue"] {font-weight: 700; letter-spacing: -0.02em;}

/* Títulos de sección */
.seccion {margin: 6px 0 14px 0;}
.seccion h2 {font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em; margin: 0; padding: 0;}
.seccion h2::before {content: ""; display: inline-block; width: 6px; height: 20px; border-radius: 3px;
    margin-right: 10px; vertical-align: -3px; background: linear-gradient(180deg, #818cf8, #2dd4bf);}
.seccion p {margin: 4px 0 0 16px; color: #94a3b8; font-size: 0.9rem;}
.subseccion {font-size: 0.95rem; font-weight: 700; margin: 6px 0 8px 0; color: #e2e8f0;}

/* Formularios, tablas y botones */
[data-testid="stForm"] {background: rgba(15,23,42,0.55); border: 1px solid rgba(148,163,184,0.14);
    border-radius: 16px; padding: 18px;}
[data-testid="stDataFrame"] {border: 1px solid rgba(148,163,184,0.14); border-radius: 14px; overflow: hidden;}
.stButton button, [data-testid="stFormSubmitButton"] button {border-radius: 10px; font-weight: 600;
    border: 1px solid rgba(148,163,184,0.25); transition: all .15s ease;}
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {
    background: linear-gradient(90deg, #6366f1, #14b8a6); border: none; color: white;}
.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(99,102,241,0.25);}
[data-testid="stExpander"] {border-radius: 14px; border: 1px solid rgba(148,163,184,0.14);}

/* Tarjeta de pronóstico */
.tarjeta {border-radius: 18px; padding: 20px 22px; border: 1px solid rgba(16,185,129,0.35);
    background: linear-gradient(145deg, rgba(16,185,129,0.14), rgba(99,102,241,0.08));}
.tarjeta.efectivo {border-color: rgba(148,163,184,0.25);
    background: linear-gradient(145deg, rgba(100,116,139,0.16), rgba(99,102,241,0.06));}
.tarjeta h3 {margin: 0 0 6px 0; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #94a3b8;}
.tarjeta .grande {font-size: 2.3rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.1;}
.chip {display: inline-block; padding: 5px 12px; border-radius: 999px; font-size: 0.72rem; font-weight: 800;
    letter-spacing: 0.06em;}
.chip.comprar {background: rgba(16,185,129,0.18); color: #34d399; border: 1px solid rgba(16,185,129,0.35);}
.chip.efectivo {background: rgba(100,116,139,0.2); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.3);}
.nota {font-size: 0.78rem; color: #64748b;}
.pie {text-align: center; color: #475569; font-size: 0.78rem; margin-top: 36px; padding-top: 16px;
    border-top: 1px solid rgba(148,163,184,0.1);}
</style>
""", unsafe_allow_html=True)


def layout_grafica(fig, alto=520):
    fig.update_layout(template="plotly_dark", height=alto, margin=dict(l=10, r=10, t=36, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                                  bgcolor="rgba(0,0,0,0)"),
                      hovermode="x unified", font=dict(family="Inter, sans-serif", size=12, color="#cbd5e1"),
                      hoverlabel=dict(bgcolor="#0f172a", bordercolor="#334155", font=dict(family="Inter")))
    fig.update_xaxes(showgrid=False, linecolor="rgba(148,163,184,0.2)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.08)", zeroline=False)
    return fig


def seccion(titulo, subtitulo=""):
    st.markdown(f'<div class="seccion"><h2>{titulo}</h2>{f"<p>{subtitulo}</p>" if subtitulo else ""}</div>',
                unsafe_allow_html=True)


def subseccion(titulo):
    st.markdown(f'<div class="subseccion">{titulo}</div>', unsafe_allow_html=True)


def pct(x, dec=2):
    return "—" if x is None or pd.isna(x) else f"{x:+.{dec}%}"


def dinero(x, moneda="USD"):
    if x is None or pd.isna(x):
        return "—"
    simbolo = {"USD": "$", "MXN": "MX$", "JPY": "¥", "CAD": "C$", "CHF": "CHF "}.get(moneda, "")
    return f"{simbolo}{x:,.4f}" if abs(x) < 10 else f"{simbolo}{x:,.2f}"


# ---------------------------------------------------------------------------
# Datos (con caché)
# ---------------------------------------------------------------------------
try:
    FICHA = nc.cargar_ficha()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

TICKERS = FICHA["ticker"].tolist()
INFO = FICHA.set_index("ticker")
MERCADOS = ["Acciones", "Criptomonedas", "Divisas"]


@st.cache_data(ttl=900, show_spinner="Descargando precios de Yahoo Finance…")
def datos_mercado(tickers, periodo):
    return nc.descargar_precios(list(tickers), periodo), datetime.now().strftime("%H:%M")


@st.cache_data(ttl=3600, show_spinner=False)
def pronostico_cacheado(ticker, df, mercado):
    return nc.pronosticar(ticker, df, mercado)


@st.cache_data(show_spinner=False)
def resultado(nombre):
    return nc.leer_resultado(nombre)


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f'<div class="marca">{LOGO_SVG.format(s=34)}<span>Investify</span></div>', unsafe_allow_html=True)
    st.caption("Pronóstico con redes neuronales · 40 activos")
    st.markdown('<div class="etiqueta-lateral">Mercado</div>', unsafe_allow_html=True)
    mercado_sel = st.segmented_control("Mercado", ["Todos"] + MERCADOS, default="Todos",
                                       label_visibility="collapsed") or "Todos"
    opciones = TICKERS if mercado_sel == "Todos" else FICHA.loc[FICHA.mercado == mercado_sel, "ticker"].tolist()
    st.markdown('<div class="etiqueta-lateral">Activo</div>', unsafe_allow_html=True)
    ticker = st.selectbox("Activo", opciones, format_func=lambda t: f"{t} · {INFO.loc[t, 'nombre']}",
                          label_visibility="collapsed")
    st.markdown('<div class="etiqueta-lateral">Gráfica</div>', unsafe_allow_html=True)
    periodo = st.select_slider("Historia en la gráfica", options=["6mo", "1y", "2y", "5y"], value="1y",
                               format_func=lambda p: {"6mo": "6 meses", "1y": "1 año", "2y": "2 años", "5y": "5 años"}[p])
    ver_sma = st.toggle("SMA 20", value=True)
    ver_bb = st.toggle("Bandas de Bollinger", value=True)
    ver_rsi = st.toggle("RSI 14", value=True)
    ver_macd = st.toggle("MACD", value=False)
    st.divider()
    if st.button("Actualizar datos"):
        st.cache_data.clear()
        st.rerun()
    st.markdown('<p class="nota">Proyecto académico de Deep Learning para Finanzas. Las señales no son '
                'recomendaciones de inversión y el portafolio es simulado.</p>', unsafe_allow_html=True)

# Se descargan 2 años para todos (suficiente para indicadores y gráficas) y 5 años solo si se pide
precios, hora_datos = datos_mercado(tuple(TICKERS), "5y" if periodo == "5y" else "2y")
if not precios:
    st.error("No se pudieron descargar datos de Yahoo Finance. Revisa tu conexión y presiona «Actualizar datos».")
    st.stop()
faltantes = [t for t in TICKERS if t not in precios]

# Pronósticos de los 40 activos (se calculan una vez por hora)
pronosticos = {}
with st.spinner("Calculando pronósticos con los modelos…"):
    for t in TICKERS:
        if t in precios:
            pronosticos[t] = pronostico_cacheado(t, precios[t], INFO.loc[t, "mercado"])
n_compra = sum(1 for p in pronosticos.values() if p and p.get("senal") == "comprar")

# Encabezado
st.markdown(f"""
<div class="encabezado">
  <div class="izq">{LOGO_SVG.format(s=52)}
    <div><h1>Invest<span class="acento">ify</span></h1>
    <p>Pronóstico del siguiente día con redes neuronales MLP y LSTM · trading simulado</p></div>
  </div>
  <div class="estado">
    <span class="pastilla"><span class="punto"></span>Datos de Yahoo Finance · {hora_datos}</span>
    <span class="pastilla">{len(precios)} activos</span>
    <span class="pastilla">{n_compra} señales de compra</span>
  </div>
</div>""", unsafe_allow_html=True)

# Cinta de precios
items = []
for t in TICKERS:
    if t in precios and len(precios[t]) > 1:
        c = precios[t]["cierre"]
        cambio = c.iloc[-1] / c.iloc[-2] - 1
        clase, flecha = ("sube", "▲") if cambio >= 0 else ("baja", "▼")
        items.append(f'<span class="item"><b>{t.replace("-USD", "").replace("=X", "")}</b>'
                     f'{c.iloc[-1]:,.2f} <span class="{clase}">{flecha} {abs(cambio):.2%}</span></span>')
st.markdown(f'<div class="cinta"><div class="cinta-pista">{"".join(items) * 2}</div></div>', unsafe_allow_html=True)

if faltantes:
    st.warning(f"Sin datos por ahora: {', '.join(faltantes)}. Intenta «Actualizar datos» en unos minutos.")
if not nc.tf_disponible():
    st.info("TensorFlow no está instalado: se muestran los pronósticos guardados del Notebook 5.")

tasas = nc.tasas_usd(precios)
precios_usd = {}
for t in TICKERS:
    if t in precios and INFO.loc[t, "moneda_cotizada"] in tasas:
        precios_usd[t] = nc.precio_usd(float(precios[t]["cierre"].iloc[-1]), INFO.loc[t, "moneda_cotizada"], tasas)

tab_mercado, tab_analisis, tab_trading, tab_bot, tab_te, tab_modelo = st.tabs(
    ["Mercado", "Análisis", "Portafolio", "Bot MEXC", "Resultados", "Modelo"])

# ===========================================================================
# 1. MERCADO
# ===========================================================================
with tab_mercado:
    filas = []
    for t in TICKERS:
        if t not in precios:
            continue
        df = precios[t]
        c = df["cierre"]
        p = pronosticos.get(t) or {}
        ind = nc.calcular_indicadores(df)
        filas.append({
            "Ticker": t, "Nombre": INFO.loc[t, "nombre"], "Mercado": INFO.loc[t, "mercado"],
            "Precio": float(c.iloc[-1]),
            "1 día": float(c.iloc[-1] / c.iloc[-2] - 1) * 100 if len(c) > 1 else np.nan,
            "1 mes": float(c.iloc[-1] / c.iloc[-22] - 1) * 100 if len(c) > 22 else np.nan,
            "Tendencia (3 meses)": c.iloc[-63:].round(6).tolist(),
            "RSI 14": float(ind["rsi14"].iloc[-1]),
            "Pronóstico mañana": p.get("estimado_dl", np.nan) * 100 if p else np.nan,
            "Señal": "Comprar" if p.get("senal") == "comprar" else "Efectivo",
        })
    tabla = pd.DataFrame(filas)
    if mercado_sel != "Todos":
        tabla = tabla[tabla.Mercado == mercado_sel]

    seccion("Panorama del mercado", "Precios, tendencia, RSI y el pronóstico de la red para el siguiente periodo")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Señales de compra", f"{(tabla['Señal'] == 'Comprar').sum()} de {len(tabla)}")
    mejor = tabla.loc[tabla["1 día"].idxmax()] if len(tabla) else None
    peor = tabla.loc[tabla["1 día"].idxmin()] if len(tabla) else None
    k2.metric("Mejor del día", mejor["Ticker"] if mejor is not None else "—",
              f"{mejor['1 día']:+.2f}%" if mejor is not None else None)
    k3.metric("Peor del día", peor["Ticker"] if peor is not None else "—",
              f"{peor['1 día']:+.2f}%" if peor is not None else None)
    k4.metric("Pronóstico promedio", f"{tabla['Pronóstico mañana'].mean():+.3f}%")

    def _color(v):
        if isinstance(v, str):
            return "color: #34d399; font-weight: 700" if v == "Comprar" else "color: #94a3b8"
        if pd.isna(v):
            return ""
        return "color: #34d399" if v > 0 else ("color: #fb7185" if v < 0 else "")

    estilo = tabla.style.map(_color, subset=["1 día", "1 mes", "Pronóstico mañana", "Señal"])
    st.dataframe(
        estilo, hide_index=True, width="stretch", height=min(42 + 35 * len(tabla), 760),
        column_config={
            "Precio": st.column_config.NumberColumn(format="%.4f"),
            "1 día": st.column_config.NumberColumn(format="%+.2f%%"),
            "1 mes": st.column_config.NumberColumn(format="%+.2f%%"),
            "Tendencia (3 meses)": st.column_config.LineChartColumn(width="medium"),
            "RSI 14": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
            "Pronóstico mañana": st.column_config.NumberColumn(format="%+.3f%%",
                                                               help="Rendimiento estimado por la red para el siguiente periodo"),
        })
    st.caption("Precios ajustados por dividendos y splits. El último dato puede ser del día en curso si el mercado está abierto.")

# ===========================================================================
# 2. ANÁLISIS
# ===========================================================================
with tab_analisis:
    if ticker not in precios:
        st.warning(f"No hay datos de {ticker} en este momento.")
    else:
        moneda = INFO.loc[ticker, "moneda_cotizada"]
        df = precios[ticker]
        ind = nc.calcular_indicadores(df)
        p = pronosticos.get(ticker)
        c = df["cierre"]

        seccion(f"{INFO.loc[ticker, 'nombre']} <span style='color:#64748b;font-weight:600'>· {ticker}</span>",
                f"{INFO.loc[ticker, 'mercado']} · Moneda: {moneda} · {INFO.loc[ticker, 'calendario']}")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Último cierre", dinero(c.iloc[-1], moneda), pct(c.iloc[-1] / c.iloc[-2] - 1))
        m2.metric("Distancia a SMA 20", pct(ind["dist_sma20"].iloc[-1]))
        m3.metric("Volatilidad 20 días", f"{ind['vol20'].iloc[-1]:.2%}")
        m4.metric("RSI 14", f"{ind['rsi14'].iloc[-1]:.0f}",
                  "sobrecompra" if ind["rsi14"].iloc[-1] > 70 else ("sobreventa" if ind["rsi14"].iloc[-1] < 30 else "neutral"),
                  delta_color="off")
        m5.metric("ATR 14", dinero(ind["atr14"].iloc[-1], moneda) if "atr14" in ind else "—")

        col_graf, col_pron = st.columns([3, 1.15])
        with col_pron:
            if p:
                clase = "comprar" if p["senal"] == "comprar" else "efectivo"
                color = VERDE if p["estimado_dl"] >= 0 else ROJO
                st.markdown(f"""
                <div class="tarjeta {clase}">
                  <h3>Pronóstico para {p['proximo_periodo']}</h3>
                  <div class="grande" style="color:{color}">{p['estimado_dl']:+.3%}</div>
                  <div style="margin:6px 0 10px 0;">Precio estimado: <b>{dinero(p['precio_estimado'], moneda)}</b></div>
                  <span class="chip {clase}">{'▲ COMPRAR' if clase == 'comprar' else 'PERMANECER EN EFECTIVO'}</span>
                </div>""", unsafe_allow_html=True)
                st.write("")
                if p.get("red"):
                    st.markdown(f"**Modelo:** {p['red']} ({p.get('neuronas', '')} neuronas) · versión "
                                f"{'B: rezagos + indicadores' if p['version'] == 'B' else 'A: solo rezagos'}")
                    regla_txt = ("pronóstico > δ **y** cierre > SMA 20" if str(p["regla"]).startswith("D")
                                 else "pronóstico > δ")
                    st.markdown(f"**Regla:** comprar si {regla_txt}, con δ = {p['umbral']:.4%}")
                    st.markdown(f"**Regresión lineal (referencia):** {p['estimado_lineal']:+.3%} → {p['senal_lineal']}")
                st.caption(f"Corte: {p['fecha_corte']} · Fuente: {p['fuente']}")
                if p.get("aviso"):
                    st.warning(p["aviso"])
            else:
                st.info("No hay pronóstico disponible para este activo.")

        with col_graf:
            desde = df.index.max() - pd.DateOffset(months={"6mo": 6, "1y": 12, "2y": 24, "5y": 60}[periodo])
            d, i2 = df[df.index >= desde], ind[ind.index >= desde]
            filas_g = 1 + int(ver_rsi) + int(ver_macd)
            alturas = [0.6] + [0.2] * (filas_g - 1)
            fig = make_subplots(rows=filas_g, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                                row_heights=[a / sum(alturas) for a in alturas])
            if {"apertura", "maximo", "minimo"}.issubset(d.columns) and d["apertura"].notna().all():
                fig.add_trace(go.Candlestick(x=d.index, open=d["apertura"], high=d["maximo"], low=d["minimo"],
                                             close=d["cierre"], name="Precio", increasing_line_color=VERDE,
                                             decreasing_line_color=ROJO), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=d.index, y=d["cierre"], name="Cierre", line=dict(color=AZUL)), row=1, col=1)
            if ver_bb:
                fig.add_trace(go.Scatter(x=i2.index, y=i2["bb_sup"], name="Bollinger", line=dict(color=GRIS, width=1),
                                         hoverinfo="skip"), row=1, col=1)
                fig.add_trace(go.Scatter(x=i2.index, y=i2["bb_inf"], name="Bollinger inf.", line=dict(color=GRIS, width=1),
                                         fill="tonexty", fillcolor="rgba(148,163,184,0.08)", showlegend=False,
                                         hoverinfo="skip"), row=1, col=1)
            if ver_sma:
                fig.add_trace(go.Scatter(x=i2.index, y=i2["sma20"], name="SMA 20", line=dict(color=NARANJA, width=1.6)),
                              row=1, col=1)
            if p:
                fig.add_trace(go.Scatter(x=[pd.Timestamp(p["proximo_periodo"])], y=[p["precio_estimado"]],
                                         mode="markers", name="Pronóstico",
                                         marker=dict(size=11, symbol="star", color=VERDE if p["estimado_dl"] >= 0 else ROJO)),
                              row=1, col=1)
            fila = 2
            if ver_rsi:
                fig.add_trace(go.Scatter(x=i2.index, y=i2["rsi14"], name="RSI 14", line=dict(color="#a78bfa")), row=fila, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color=ROJO, opacity=0.5, row=fila, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color=VERDE, opacity=0.5, row=fila, col=1)
                fig.update_yaxes(range=[0, 100], title_text="RSI", row=fila, col=1)
                fila += 1
            if ver_macd:
                colores = np.where(i2["macd_hist"] >= 0, VERDE, ROJO)
                fig.add_trace(go.Bar(x=i2.index, y=i2["macd_hist"], name="Histograma", marker_color=colores), row=fila, col=1)
                fig.add_trace(go.Scatter(x=i2.index, y=i2["macd"], name="MACD", line=dict(color=AZUL)), row=fila, col=1)
                fig.add_trace(go.Scatter(x=i2.index, y=i2["macd_senal"], name="Señal MACD", line=dict(color=NARANJA)),
                              row=fila, col=1)
                fig.update_yaxes(title_text="MACD", row=fila, col=1)
            fig.update_layout(xaxis_rangeslider_visible=False)
            st.plotly_chart(layout_grafica(fig, 430 + 140 * (filas_g - 1)), width="stretch")

        with st.expander("¿Cómo se calculan los indicadores?"):
            st.markdown("""
| Indicador | Cálculo |
|---|---|
| **SMA 20** | Promedio de los últimos 20 cierres. Distancia = cierre / SMA − 1 |
| **Volatilidad 20** | Desviación estándar muestral de los últimos 20 rendimientos diarios, sin anualizar |
| **RSI 14** | 100 − 100 / (1 + RS), con RS = promedio de ganancias / promedio de pérdidas (promedio simple de 14) |
| **MACD** | EMA 12 − EMA 26; señal = EMA 9 del MACD |
| **Bollinger** | SMA 20 ± 2 desviaciones estándar de los cierres |
| **ATR 14** | Promedio suavizado (Wilder) del rango verdadero |

Todos usan solo información disponible hasta el cierre de cada día. Un RSI alto **no garantiza** que el precio vaya a caer.
""")

# ===========================================================================
# 3. TRADING SIMULADO
# ===========================================================================
with tab_trading:
    if "portafolio" not in st.session_state:
        st.session_state.portafolio = nc.cargar_portafolio()
    port = st.session_state.portafolio
    tabla_pos, res = nc.valorar(port, precios_usd)

    seccion("Portafolio simulado", "Cuenta en dólares · sin apalancamiento ni ventas en corto · comisión de 0.10% "
                                   "por compra y por venta · se opera al último precio disponible")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Patrimonio", f"${res['patrimonio']:,.2f}", pct(res["rendimiento"]))
    r2.metric("Efectivo", f"${res['efectivo']:,.2f}")
    r3.metric("Invertido", f"${res['invertido']:,.2f}")
    r4.metric("Comisiones pagadas", f"${res['comisiones']:,.2f}")
    r5.metric("Win rate", "no definido" if pd.isna(res["win_rate"]) else f"{res['win_rate']:.0%}",
              f"{res['operaciones_cerradas']} operaciones cerradas", delta_color="off")

    operables = [t for t in TICKERS if t in precios_usd and t not in nc.SOLO_ANALISIS]
    c_compra, c_venta, c_auto = st.columns(3)
    with c_compra:
        with st.form("form_compra", clear_on_submit=False):
            subseccion("Comprar")
            t_c = st.selectbox("Activo", operables, index=operables.index(ticker) if ticker in operables else 0,
                               format_func=lambda t: f"{t} · ${precios_usd[t]:,.4f}")
            monto = st.number_input("Monto en USD (incluye comisión)", min_value=0.0,
                                    max_value=float(max(port["efectivo"], 0.0)),
                                    value=float(min(10_000.0, max(port["efectivo"], 0.0))), step=500.0)
            if st.form_submit_button("Comprar", type="primary"):
                try:
                    u = nc.comprar(port, t_c, monto, precios_usd[t_c])
                    nc.guardar_portafolio(port)
                    st.toast(f"Compraste {u:,.6f} de {t_c}")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
    with c_venta:
        with st.form("form_venta"):
            subseccion("Vender")
            con_pos = list(port["posiciones"].keys())
            t_v = st.selectbox("Posición", con_pos if con_pos else ["(sin posiciones)"])
            frac = st.slider("Porcentaje a vender", 1, 100, 100, format="%d%%")
            if st.form_submit_button("Vender"):
                try:
                    if not con_pos:
                        raise ValueError("No tienes posiciones abiertas.")
                    neto = nc.vender(port, t_v, frac / 100, precios_usd[t_v])
                    nc.guardar_portafolio(port)
                    st.toast(f"Venta de {t_v}: recibiste ${neto:,.2f}")
                    st.rerun()
                except (ValueError, KeyError) as e:
                    st.error(str(e))
    with c_auto:
        with st.form("form_auto"):
            subseccion("Seguir al modelo")
            st.caption("Vende las posiciones con señal «efectivo» y compra los activos con señal «comprar».")
            monto_auto = st.number_input("USD por activo", min_value=100.0, value=5_000.0, step=500.0)
            if st.form_submit_button("Ejecutar señales"):
                acciones = nc.ejecutar_senales(port, pronosticos, precios_usd, monto_auto)
                nc.guardar_portafolio(port)
                st.toast(f"{len(acciones)} operaciones ejecutadas" if acciones else "Nada que ejecutar")
                st.rerun()

    st.write("")
    c_tab, c_dona = st.columns([2, 1])
    with c_tab:
        subseccion("Posiciones abiertas")
        if len(tabla_pos):
            st.dataframe(tabla_pos, hide_index=True, width="stretch", column_config={
                "unidades": st.column_config.NumberColumn(format="%.6f"),
                "precio_usd": st.column_config.NumberColumn("precio (USD)", format="$%.4f"),
                "valor_usd": st.column_config.NumberColumn("valor (USD)", format="$%.2f"),
                "costo_usd": st.column_config.NumberColumn("costo (USD)", format="$%.2f"),
                "ganancia_usd": st.column_config.NumberColumn("ganancia (USD)", format="$%+.2f"),
                "ganancia_%": st.column_config.NumberColumn("ganancia", format="%+.2f%%")})
        else:
            st.info("Aún no tienes posiciones. Compra un activo o deja que el modelo opere por ti.")
    with c_dona:
        subseccion("Distribución")
        etiquetas = ["Efectivo"] + tabla_pos["ticker"].tolist()
        valores = [port["efectivo"]] + tabla_pos["valor_usd"].fillna(0).tolist()
        paleta = ["#334155", "#6366f1", "#14b8a6", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6", "#06b6d4", "#84cc16", "#f97316"]
        fig = go.Figure(go.Pie(labels=etiquetas, values=valores, hole=0.62, textinfo="percent", sort=False,
                               marker=dict(colors=(paleta * 5)[:len(etiquetas)], line=dict(color="#070b14", width=2))))
        st.plotly_chart(layout_grafica(fig, 300), width="stretch")

    if port["operaciones"]:
        subseccion("Historial de operaciones")
        hist = pd.DataFrame(port["operaciones"]).iloc[::-1]
        st.dataframe(hist, hide_index=True, width="stretch", column_config={
            "unidades": st.column_config.NumberColumn(format="%.6f"),
            "precio_usd": st.column_config.NumberColumn(format="$%.4f"),
            "monto_usd": st.column_config.NumberColumn(format="$%.2f"),
            "comision_usd": st.column_config.NumberColumn(format="$%.2f"),
            "resultado_usd": st.column_config.NumberColumn(format="$%+.2f")})
    if st.button("Reiniciar portafolio a $100,000"):
        st.session_state.portafolio = nc.portafolio_nuevo()
        nc.guardar_portafolio(st.session_state.portafolio)
        st.rerun()
    st.caption("USD/MXN, USD/JPY, USD/CAD y USD/CHF son solo de análisis: desde una cuenta en dólares su valor es constante. "
               "Los activos cotizados en otras monedas se convierten a USD con el tipo de cambio del último cierre.")

# ===========================================================================
# BOT MEXC (precios reales de MEXC, dinero de práctica)
# ===========================================================================
@st.cache_data(ttl=2, show_spinner=False)
def precios_bot(pares):
    return mb.obtener_precios(list(pares))


@st.cache_data(ttl=2, show_spinner=False)
def velas_bot(par, intervalo):
    return mb.obtener_velas(par, intervalo, 120)


def tabla_operaciones(ops, alto=None):
    df = pd.DataFrame(ops)
    columnas = ["fecha", "par", "tipo", "motivo", "precio", "unidades", "monto_usdt", "comision_usdt",
                "ganancia_usdt", "ganancia_%", "fuente", "bot_id"]
    df = df[[c for c in columnas if c in df.columns]]
    estilo = df.style.map(lambda v: "" if v is None or pd.isna(v) else ("color: #34d399" if v > 0 else "color: #fb7185"),
                          subset=[c for c in ["ganancia_usdt", "ganancia_%"] if c in df.columns])
    estilo = estilo.map(lambda v: "color: #34d399; font-weight: 700" if v == "compra" else "color: #fbbf24; font-weight: 700",
                        subset=["tipo"])
    kwargs = {"height": alto} if alto else {}
    st.dataframe(estilo, hide_index=True, width="stretch", column_config={
        "precio": st.column_config.NumberColumn(format="%.6g"),
        "unidades": st.column_config.NumberColumn(format="%.6f"),
        "monto_usdt": st.column_config.NumberColumn("monto (USDT)", format="%.2f"),
        "comision_usdt": st.column_config.NumberColumn("comisión (USDT)", format="%.4f"),
        "ganancia_usdt": st.column_config.NumberColumn("ganancia (USDT)", format="%+.2f"),
        "ganancia_%": st.column_config.NumberColumn("ganancia", format="%+.2f%%"),
        "bot_id": st.column_config.TextColumn("sesión del bot")}, **kwargs)


with tab_bot:
    seccion("Bot de trading · MEXC", "Precios reales de MEXC en tiempo real · capital de práctica de 10,000 USDT · "
                                     "no se envían órdenes reales")
    if "mexc" not in st.session_state:
        st.session_state.mexc = mb.cargar_estado()
        st.session_state.mexc_corriendo = st.session_state.mexc["bot"] is not None
    est = st.session_state.mexc

    with st.expander("Configurar bot", expanded=est["bot"] is None):
        with st.form("form_mexc"):
            pares_sel = st.multiselect("Pares de MEXC", list(mb.PARES), default=list(mb.PARES))
            c1, c2, c3 = st.columns(3)
            monto = c1.number_input("Monto por operación (USDT)", min_value=mb.MONTO_MINIMO,
                                    max_value=mb.CAPITAL_MEXC, value=1_000.0, step=100.0)
            tc = c2.slider("Tope de compra: comprar si baja desde su máximo", 0.2, 5.0, 0.5, 0.1, format="%.1f%%")
            tv = c3.slider("Tope de venta: vender al ganar", 0.3, 5.0, 0.8, 0.1, format="%.1f%%")
            d1, d2, d3 = st.columns(3)
            usar_sl = d1.toggle("Usar stop-loss", value=False,
                                help="Apagado: el bot solo vende con ganancia, pero una compra puede quedarse abierta en pérdida.")
            sl = d2.slider("Stop-loss: vender si pierde", 0.5, 20.0, 5.0, 0.5, format="%.1f%%")
            vel = d3.select_slider("Actualizar precios cada", options=[2, 5, 10, 30], value=5,
                                   format_func=lambda x: f"{x} segundos")
            arrancar = st.form_submit_button("Iniciar bot" if est["bot"] is None else "Reiniciar bot con esta configuración",
                                             type="primary")
        if arrancar:
            try:
                pr, fuente = precios_bot(tuple(pares_sel))
                est["bot"] = mb.bot_nuevo(pr, monto, tc / 100, tv / 100, sl / 100 if usar_sl else None)
                est["bot"].update({"fuente": fuente, "intervalo_s": vel})
                st.session_state.mexc_corriendo = True
                mb.guardar_estado(est)
                st.rerun()
            except (ValueError, mb.FuenteNoDisponible) as e:
                st.error(str(e))
        st.caption("El capital de práctica es de 10,000 USDT. El bot opera mientras la app está abierta en el navegador; "
                   "lo que hace se guarda y continúa cuando vuelves a abrirla.")

    if est["bot"] is not None:
        b1, b2, _ = st.columns([1, 1, 4])
        if b1.button("Pausar bot" if st.session_state.mexc_corriendo else "Reanudar bot"):
            st.session_state.mexc_corriendo = not st.session_state.mexc_corriendo
            st.rerun()
        if b2.button("Detener y cerrar posiciones"):
            pr, fuente = precios_bot(tuple(est["bot"]["activos"]))
            for par, a in est["bot"]["activos"].items():
                if a["unidades"] > 0 and par in pr:
                    bruto = a["unidades"] * pr[par]
                    comision = bruto * mb.COSTO
                    ganancia = bruto - comision - a["monto"]
                    est["bot"]["efectivo"] += bruto - comision
                    mb._registrar(est, {"bot_id": est["bot"]["id"], "fecha": mb.ahora().strftime("%Y-%m-%d %H:%M:%S"),
                                        "par": par, "tipo": "venta", "motivo": "cierre manual", "precio": pr[par],
                                        "unidades": a["unidades"], "monto_usdt": bruto - comision,
                                        "comision_usdt": comision, "ganancia_usdt": ganancia,
                                        "ganancia_%": ganancia / a["monto"] * 100, "fuente": fuente})
                    a.update({"unidades": 0.0, "precio_compra": None, "monto": 0.0, "maximo_reciente": pr[par]})
            st.session_state.mexc_corriendo = False
            mb.guardar_estado(est)
            st.rerun()

    intervalo_s = (est["bot"] or {}).get("intervalo_s", 5)

    @st.fragment(run_every=intervalo_s if (est["bot"] is not None and st.session_state.mexc_corriendo) else None)
    def panel_mexc():
        est = st.session_state.mexc
        bot = est["bot"]
        if bot is not None:
            try:
                pr, fuente = precios_bot(tuple(bot["activos"]))
                if st.session_state.mexc_corriendo:
                    for op in mb.bot_actualizar(est, pr, fuente):
                        st.toast(f"{'Compra' if op['tipo'] == 'compra' else 'Venta'} de {op['par']} a {op['precio']:,.6g}"
                                 + (f" · {op['ganancia_usdt']:+.2f} USDT" if op["ganancia_usdt"] is not None else ""))
                    mb.guardar_estado(est)
            except (mb.FuenteNoDisponible, KeyError, ValueError) as e:
                fuente = "sin conexión"
                st.error(f"No se pudieron obtener precios: {e}")

            estado_txt = "● En vivo" if st.session_state.mexc_corriendo else "En pausa"
            st.caption(f"{estado_txt} · Fuente: {fuente} · actualizado {mb.ahora():%H:%M:%S} · cada {intervalo_s} s · "
                       f"tope de compra −{bot['tope_compra']:.1%} · tope de venta +{bot['tope_venta']:.1%} · "
                       f"stop-loss {'−' + format(bot['stop_loss'], '.1%') if bot['stop_loss'] else 'apagado'}")
            if fuente.startswith("Yahoo"):
                st.warning("MEXC no está disponible desde este servidor (MEXC no opera en EE. UU., donde está Streamlit Cloud). "
                           "Se usan precios de Yahoo Finance con hasta 1 minuto de retraso. Para precios de MEXC en tiempo real, "
                           "corre la app en tu computadora.")

            # ---------------- Resumen de ganancias ----------------
            r = mb.resumen(bot)
            subseccion("Resumen de ganancias")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Capital inicial", f"{r['capital']:,.2f} USDT")
            k2.metric("Patrimonio actual", f"{r['patrimonio']:,.2f} USDT", pct(r["rendimiento"]))
            k3.metric("Ganancia realizada", f"{r['ganancia_realizada']:+,.2f} USDT",
                      f"{r['ganadoras']} de {r['ventas']} ventas con ganancia", delta_color="off")
            k4.metric("Ganancia no realizada", f"{r['ganancia_no_realizada']:+,.2f} USDT",
                      f"{len(r['abiertas'])} posiciones abiertas", delta_color="off")
            k5, k6, k7, k8 = st.columns(4)
            k5.metric("Ganancia total", f"{r['ganancia_total']:+,.2f} USDT", pct(r["rendimiento"]))
            k6.metric("Operaciones", f"{r['compras']} compras · {r['ventas']} ventas")
            k7.metric("Comisiones pagadas", f"{r['comisiones']:,.2f} USDT")
            horas, minutos = divmod(int(r["minutos"]), 60)
            k8.metric("Tiempo activo", f"{horas} h {minutos} min",
                      f"mejor venta {r['mejor']:+.2f} USDT" if r["mejor"] is not None else None, delta_color="off")

            g1, g2 = st.columns([1.6, 1])
            with g1:
                curva = pd.DataFrame(bot["curva"], columns=["tiempo", "patrimonio"])
                curva["tiempo"] = pd.to_datetime(curva["tiempo"])
                fig = go.Figure(go.Scatter(x=curva.tiempo, y=curva.patrimonio, name="Patrimonio", fill="tozeroy",
                                           line=dict(color=TURQUESA, width=2), fillcolor="rgba(20,184,166,0.10)"))
                fig.add_hline(y=r["capital"], line_dash="dot", line_color=GRIS, annotation_text="capital inicial")
                fig.update_yaxes(range=[curva.patrimonio.min() * 0.995, curva.patrimonio.max() * 1.005])
                fig.update_layout(title=dict(text="Patrimonio del bot (USDT)", font=dict(size=13)))
                st.plotly_chart(layout_grafica(fig, 280), width="stretch")
            with g2:
                pp = r["por_par"]
                fig = go.Figure(go.Bar(x=pp["par"] if len(pp) else [], y=pp["ganancia_usdt"] if len(pp) else [],
                                       marker_color=[VERDE if v >= 0 else ROJO for v in (pp["ganancia_usdt"] if len(pp) else [])]))
                fig.update_layout(title=dict(text="Ganancia realizada por par (USDT)", font=dict(size=13)))
                st.plotly_chart(layout_grafica(fig, 280), width="stretch")

            # ---------------- Gráfica en tiempo real ----------------
            subseccion("Gráfica en tiempo real")
            s1, s2 = st.columns([2, 1])
            par_ver = s1.selectbox("Par", list(bot["activos"]), key="mexc_par")
            int_lbl = s2.selectbox("Velas de", list(mb.INTERVALOS), key="mexc_int")
            try:
                velas, fuente_v = velas_bot(par_ver, mb.INTERVALOS[int_lbl])
                fig = go.Figure(go.Candlestick(x=velas.tiempo, open=velas.apertura, high=velas.maximo, low=velas.minimo,
                                               close=velas.cierre, name=par_ver, increasing_line_color=VERDE,
                                               decreasing_line_color=ROJO))
                desde = velas.tiempo.min()
                ops_par = [o for o in bot["operaciones"] if o["par"] == par_ver and pd.Timestamp(o["fecha"]) >= desde]
                for tipo, simbolo, color, nombre in (("compra", "triangle-up", VERDE, "Compra del bot"),
                                                     ("venta", "triangle-down", NARANJA, "Venta del bot")):
                    sub = [o for o in ops_par if o["tipo"] == tipo]
                    if sub:
                        fig.add_trace(go.Scatter(x=[pd.Timestamp(o["fecha"]) for o in sub], y=[o["precio"] for o in sub],
                                                 mode="markers", name=nombre,
                                                 marker=dict(symbol=simbolo, size=14, color=color, line=dict(width=1, color="#0b1220"))))
                nv = mb.niveles(bot, par_ver)
                if nv["compra"]:
                    fig.add_hline(y=nv["compra"], line_dash="dot", line_color=VERDE, annotation_text="tope de compra")
                if nv["venta"]:
                    fig.add_hline(y=nv["venta"], line_dash="dot", line_color=NARANJA, annotation_text="tope de venta")
                    fig.add_hline(y=nv["precio_compra"], line_dash="dash", line_color=GRIS, annotation_text="precio de compra")
                if nv["stop"]:
                    fig.add_hline(y=nv["stop"], line_dash="dot", line_color=ROJO, annotation_text="stop-loss")
                fig.update_layout(xaxis_rangeslider_visible=False, yaxis_title="USDT",
                                  title=dict(text=f"{par_ver} · último {bot['activos'][par_ver]['ultimo']:,.6g} USDT · {fuente_v}",
                                             font=dict(size=13)))
                st.plotly_chart(layout_grafica(fig, 440), width="stretch")
            except (mb.FuenteNoDisponible, KeyError, ValueError) as e:
                st.error(f"No se pudo cargar la gráfica: {e}")

            # ---------------- Posiciones y últimas operaciones ----------------
            subseccion("Posiciones abiertas")
            if len(r["abiertas"]):
                st.dataframe(r["abiertas"], hide_index=True, width="stretch", column_config={
                    "precio_compra": st.column_config.NumberColumn("compra", format="%.6g"),
                    "precio_actual": st.column_config.NumberColumn("actual", format="%.6g"),
                    "objetivo_venta": st.column_config.NumberColumn("tope de venta", format="%.6g"),
                    "invertido_usdt": st.column_config.NumberColumn("invertido (USDT)", format="%.2f"),
                    "valor_usdt": st.column_config.NumberColumn("valor (USDT)", format="%.2f"),
                    "resultado_%": st.column_config.NumberColumn("resultado", format="%+.2f%%")})
            else:
                st.caption("Sin posiciones abiertas: el bot espera que algún precio baje hasta su tope de compra.")

            subseccion("Últimas operaciones")
            if bot["operaciones"]:
                tabla_operaciones(bot["operaciones"][-15:][::-1])
            else:
                st.caption("Todavía no hay operaciones en esta sesión del bot.")
        else:
            st.info("Configura el bot y presiona **Iniciar bot**.")

        # ---------------- Movimientos históricos ----------------
        subseccion("Movimientos históricos")
        if est["historial"]:
            hist = est["historial"][::-1]
            st.caption(f"{len(hist)} movimientos de todas las sesiones del bot, del más reciente al más antiguo.")
            tabla_operaciones(hist, alto=420)
            st.download_button("Descargar historial (CSV)", pd.DataFrame(hist).to_csv(index=False).encode("utf-8"),
                               file_name="movimientos_bot_mexc.csv", mime="text/csv")
        else:
            st.caption("El historial aparecerá aquí cuando el bot haga su primera operación.")

    panel_mexc()

    with st.expander("Opciones avanzadas"):
        st.caption("Borra el historial guardado de todas las sesiones del bot. No se puede deshacer.")
        if st.button("Borrar historial de movimientos"):
            est["historial"] = []
            if est["bot"] is not None:
                est["bot"]["operaciones"] = []
            mb.guardar_estado(est)
            st.rerun()

# ===========================================================================
# 4. RESULTADOS EN TE (backtest del Notebook 5)
# ===========================================================================
with tab_te:
    res_te = resultado("resultados_trading_TE.csv")
    sim = resultado("simulacion.csv")
    met = resultado("metricas_TR_VA_TE.csv")
    if res_te is None:
        st.info("Copia los archivos de resultados del Notebook 5 a la carpeta `resultados/` para ver esta sección.")
    else:
        seccion("Evaluación final en el conjunto de prueba (TE)",
                "Periodo nunca usado para entrenar ni para elegir · capital de 100,000 · comisión de 0.10% · "
                "ejecución en la apertura de t+1")
        piv = res_te.pivot(index="ticker", columns="estrategia", values="rendimiento_neto")
        piv["mercado"] = INFO["mercado"]
        if mercado_sel != "Todos":
            piv = piv[piv.mercado == mercado_sel]
        n = len(piv)
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Rendimiento promedio · Sistema DL", pct(piv["Sistema DL final"].mean()))
        e2.metric("DL supera a la estrategia lineal", f"{(piv['Sistema DL final'] > piv['Estrategia lineal']).sum()} de {n}")
        e3.metric("DL supera a comprar y mantener", f"{(piv['Sistema DL final'] > piv['Comprar y mantener']).sum()} de {n}")
        e4.metric("Activos con ≥ +20%", f"{(piv['Sistema DL final'] >= 0.20).sum()} de {n}")

        orden = piv.sort_values("Sistema DL final", ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=orden.index, x=orden["Comprar y mantener"] * 100, name="Comprar y mantener",
                             orientation="h", marker_color=GRIS))
        fig.add_trace(go.Bar(y=orden.index, x=orden["Sistema DL final"] * 100, name="Sistema DL", orientation="h",
                             marker_color=[VERDE if v >= 0 else ROJO for v in orden["Sistema DL final"]]))
        fig.add_vline(x=20, line_dash="dot", line_color=NARANJA, annotation_text="meta +20%")
        fig.update_layout(barmode="group", xaxis_title="Rendimiento neto en TE (%)")
        st.plotly_chart(layout_grafica(fig, max(420, 22 * n)), width="stretch")

        if sim is not None and ticker in set(sim.ticker):
            s = sim[sim.ticker == ticker].copy()
            s["fecha"] = pd.to_datetime(s["fecha"])
            subseccion(f"Patrimonio en TE · {ticker}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s.fecha, y=s.patrimonio, name="Sistema DL", line=dict(color=VERDE, width=2)))
            if ticker in precios:
                d = precios[ticker].reindex(s.fecha)
                if d["cierre"].notna().all():
                    px0 = s["precio_ejecucion"].iloc[0]
                    unidades = nc.CAPITAL_INICIAL / (px0 * (1 + nc.COSTO))
                    fig.add_trace(go.Scatter(x=s.fecha, y=unidades * d["cierre"].to_numpy(), name="Comprar y mantener",
                                             line=dict(color=GRIS)))
            fig.add_hline(y=nc.CAPITAL_INICIAL, line_dash="dot", line_color=GRIS, annotation_text="Efectivo")
            st.plotly_chart(layout_grafica(fig, 360), width="stretch")
            fila = res_te[(res_te.ticker == ticker) & (res_te.estrategia == "Sistema DL final")].iloc[0]
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Rendimiento neto", pct(fila["rendimiento_neto"]))
            f2.metric("Máximo drawdown", pct(fila["max_drawdown"]))
            f3.metric("Operaciones", int(fila["operaciones"]))
            f4.metric("Win rate", "no definido" if pd.isna(fila["win_rate"]) else f"{fila['win_rate']:.0%}")

        with st.expander("Tabla completa de estrategias en TE"):
            st.dataframe(res_te, hide_index=True, width="stretch")

# ===========================================================================
# 5. MODELO
# ===========================================================================
with tab_modelo:
    seccion("Cómo funciona Investify", "Del dato al pronóstico y a la señal")
    a, b, c = st.columns(3)
    a.markdown("**1 · Datos**  \nPrecios diarios ajustados de Yahoo Finance. Entradas: los 20 rendimientos más recientes "
               "(y, en la versión B, distancia a la SMA 20, volatilidad 20 y RSI 14).")
    b.markdown("**2 · Modelos**  \nMLP y LSTM entrenados en una GPU T4 con separación temporal 70/15/15 (TR/VA/TE), "
               "parada temprana y 3 semillas. Se comparan con el pronóstico cero y la regresión lineal.")
    c.markdown("**3 · Señal**  \nComprar si el rendimiento estimado supera un umbral δ elegido en validación "
               "(con o sin filtro SMA 20); si no, permanecer en efectivo.")
    met = resultado("metricas_TR_VA_TE.csv")
    if met is not None and ticker in set(met.ticker):
        subseccion(f"Métricas de {ticker} (escala original del rendimiento)")
        m = met[met.ticker == ticker].pivot_table(index="modelo", columns="conjunto",
                                                  values=["MAE", "RMSE", "precision_direccional"])
        m = m.reindex(columns=["TR", "VA", "TE"], level=1)
        st.dataframe(m.style.format("{:.5f}"), width="stretch")
    p = pronosticos.get(ticker)
    if p and p.get("red"):
        st.markdown(f"**Configuración congelada de {ticker}:** red {p['red']} · {p.get('neuronas', '')} neuronas · "
                    f"versión {p['version']} · regla {p['regla']} · δ = {p['umbral']:.4%}")
    st.divider()
    st.markdown('<p class="nota">Proyecto académico de Deep Learning para Finanzas. Los rendimientos pasados no garantizan '
                'resultados futuros. Las señales no son asesoría financiera.</p>', unsafe_allow_html=True)

st.markdown('<div class="pie">Investify · Proyecto final de Deep Learning para Finanzas · '
            'Datos: Yahoo Finance · Las señales no son asesoría financiera</div>', unsafe_allow_html=True)
