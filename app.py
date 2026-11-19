"""
Mercado IA — Pronóstico con Deep Learning y trading simulado
Proyecto final · Deep Learning para Finanzas

Ejecutar:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import nucleo as nc

st.set_page_config(page_title="Mercado IA · Pronóstico y trading", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Estilo
# ---------------------------------------------------------------------------
VERDE, ROJO, AZUL, NARANJA, GRIS = "#22c55e", "#ef4444", "#3b82f6", "#f59e0b", "#94a3b8"
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(148,163,184,0.08); border: 1px solid rgba(148,163,184,0.18);
    border-radius: 14px; padding: 14px 16px;}
[data-testid="stMetricLabel"] p {font-size: 0.82rem; opacity: 0.8;}
.titulo {font-size: 2.0rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0;}
.subtitulo {opacity: 0.7; margin-top: 0.1rem; margin-bottom: 1.2rem;}
.tarjeta {border-radius: 16px; padding: 18px 20px; border: 1px solid rgba(148,163,184,0.25);
    background: linear-gradient(135deg, rgba(34,197,94,0.10), rgba(59,130,246,0.06));}
.tarjeta.efectivo {background: linear-gradient(135deg, rgba(148,163,184,0.12), rgba(59,130,246,0.05));}
.tarjeta h3 {margin: 0 0 4px 0; font-size: 0.9rem; opacity: 0.75; font-weight: 600;}
.tarjeta .grande {font-size: 2.1rem; font-weight: 800; line-height: 1.1;}
.chip {display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 700;}
.chip.comprar {background: rgba(34,197,94,0.18); color: #22c55e;}
.chip.efectivo {background: rgba(148,163,184,0.2); color: #cbd5e1;}
.nota {font-size: 0.8rem; opacity: 0.65;}
</style>
""", unsafe_allow_html=True)


def layout_grafica(fig, alto=520):
    fig.update_layout(template="plotly_dark", height=alto, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
                      hovermode="x unified", font=dict(size=12))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)")
    return fig


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
    return nc.descargar_precios(list(tickers), periodo)


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
    st.markdown("## 📈 Mercado IA")
    st.caption("Pronóstico con MLP / LSTM · 40 activos · datos de Yahoo Finance")
    mercado_sel = st.radio("Mercado", ["Todos"] + MERCADOS, horizontal=False)
    opciones = TICKERS if mercado_sel == "Todos" else FICHA.loc[FICHA.mercado == mercado_sel, "ticker"].tolist()
    ticker = st.selectbox("Activo", opciones, format_func=lambda t: f"{t} · {INFO.loc[t, 'nombre']}")
    periodo = st.select_slider("Historia en la gráfica", options=["6mo", "1y", "2y", "5y"], value="1y")
    st.markdown("**Indicadores**")
    ver_sma = st.toggle("SMA 20", value=True)
    ver_bb = st.toggle("Bandas de Bollinger", value=True)
    ver_rsi = st.toggle("RSI 14", value=True)
    ver_macd = st.toggle("MACD", value=False)
    st.divider()
    if st.button("🔄 Actualizar datos"):
        st.cache_data.clear()
        st.rerun()
    st.markdown('<p class="nota">Proyecto académico. Las señales no son recomendaciones de inversión. '
                'El portafolio es simulado: no se usa dinero real.</p>', unsafe_allow_html=True)

# Se descargan 2 años para todos (suficiente para indicadores y gráficas) y 5 años solo si se pide
precios = datos_mercado(tuple(TICKERS), "5y" if periodo == "5y" else "2y")
if not precios:
    st.error("No se pudieron descargar datos de Yahoo Finance. Revisa tu conexión y presiona «Actualizar datos».")
    st.stop()
faltantes = [t for t in TICKERS if t not in precios]

st.markdown('<p class="titulo">Mercado IA</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitulo">Pronóstico del rendimiento del siguiente día con redes neuronales '
            'y portafolio de trading simulado</p>', unsafe_allow_html=True)
if faltantes:
    st.warning(f"Sin datos por ahora: {', '.join(faltantes)}. Intenta «Actualizar datos» en unos minutos.")
if not nc.tf_disponible():
    st.info("TensorFlow no está instalado: se muestran los pronósticos guardados del Notebook 5.")

# Pronósticos de los 40 activos (se calculan una vez por hora)
pronosticos = {}
with st.spinner("Calculando pronósticos con los modelos…"):
    for t in TICKERS:
        if t in precios:
            pronosticos[t] = pronostico_cacheado(t, precios[t], INFO.loc[t, "mercado"])

tasas = nc.tasas_usd(precios)
precios_usd = {}
for t in TICKERS:
    if t in precios and INFO.loc[t, "moneda_cotizada"] in tasas:
        precios_usd[t] = nc.precio_usd(float(precios[t]["cierre"].iloc[-1]), INFO.loc[t, "moneda_cotizada"], tasas)

tab_mercado, tab_analisis, tab_trading, tab_te, tab_modelo = st.tabs(
    ["🌎 Mercado", "📊 Análisis", "💼 Trading simulado", "🧪 Resultados en TE", "🧠 Modelo"])

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
            "Señal": "🟢 Comprar" if p.get("senal") == "comprar" else "⚪ Efectivo",
        })
    tabla = pd.DataFrame(filas)
    if mercado_sel != "Todos":
        tabla = tabla[tabla.Mercado == mercado_sel]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Activos con señal de compra", f"{(tabla['Señal'] == '🟢 Comprar').sum()} de {len(tabla)}")
    mejor = tabla.loc[tabla["1 día"].idxmax()] if len(tabla) else None
    peor = tabla.loc[tabla["1 día"].idxmin()] if len(tabla) else None
    k2.metric("Mejor del día", mejor["Ticker"] if mejor is not None else "—",
              f"{mejor['1 día']:+.2f}%" if mejor is not None else None)
    k3.metric("Peor del día", peor["Ticker"] if peor is not None else "—",
              f"{peor['1 día']:+.2f}%" if peor is not None else None)
    k4.metric("Pronóstico promedio", f"{tabla['Pronóstico mañana'].mean():+.3f}%")

    st.dataframe(
        tabla, hide_index=True, width="stretch", height=min(42 + 35 * len(tabla), 760),
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

        st.markdown(f"### {INFO.loc[ticker, 'nombre']}  ·  `{ticker}`")
        st.caption(f"{INFO.loc[ticker, 'mercado']} · Moneda: {moneda} · Calendario: {INFO.loc[ticker, 'calendario']}")

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
                  <span class="chip {clase}">{'🟢 COMPRAR' if clase == 'comprar' else '⚪ PERMANECER EN EFECTIVO'}</span>
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

    st.markdown("### 💼 Portafolio simulado")
    st.caption("Cuenta en dólares · sin apalancamiento ni ventas en corto · comisión de 0.10% por compra y por venta · "
               "se opera al último precio disponible.")
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
            st.markdown("#### 🟢 Comprar")
            t_c = st.selectbox("Activo", operables, index=operables.index(ticker) if ticker in operables else 0,
                               format_func=lambda t: f"{t} · ${precios_usd[t]:,.4f}")
            monto = st.number_input("Monto en USD (incluye comisión)", min_value=0.0,
                                    max_value=float(max(port["efectivo"], 0.0)),
                                    value=float(min(10_000.0, max(port["efectivo"], 0.0))), step=500.0)
            if st.form_submit_button("Comprar", type="primary"):
                try:
                    u = nc.comprar(port, t_c, monto, precios_usd[t_c])
                    nc.guardar_portafolio(port)
                    st.toast(f"Compraste {u:,.6f} de {t_c}", icon="✅")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
    with c_venta:
        with st.form("form_venta"):
            st.markdown("#### 🔴 Vender")
            con_pos = list(port["posiciones"].keys())
            t_v = st.selectbox("Posición", con_pos if con_pos else ["(sin posiciones)"])
            frac = st.slider("Porcentaje a vender", 1, 100, 100, format="%d%%")
            if st.form_submit_button("Vender"):
                try:
                    if not con_pos:
                        raise ValueError("No tienes posiciones abiertas.")
                    neto = nc.vender(port, t_v, frac / 100, precios_usd[t_v])
                    nc.guardar_portafolio(port)
                    st.toast(f"Venta de {t_v}: recibiste ${neto:,.2f}", icon="💵")
                    st.rerun()
                except (ValueError, KeyError) as e:
                    st.error(str(e))
    with c_auto:
        with st.form("form_auto"):
            st.markdown("#### 🤖 Seguir al modelo")
            st.caption("Vende las posiciones con señal «efectivo» y compra los activos con señal «comprar».")
            monto_auto = st.number_input("USD por activo", min_value=100.0, value=5_000.0, step=500.0)
            if st.form_submit_button("Ejecutar señales"):
                acciones = nc.ejecutar_senales(port, pronosticos, precios_usd, monto_auto)
                nc.guardar_portafolio(port)
                st.toast(f"{len(acciones)} operaciones ejecutadas" if acciones else "Nada que ejecutar", icon="🤖")
                st.rerun()

    st.write("")
    c_tab, c_dona = st.columns([2, 1])
    with c_tab:
        st.markdown("#### Posiciones abiertas")
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
        st.markdown("#### Distribución")
        etiquetas = ["Efectivo"] + tabla_pos["ticker"].tolist()
        valores = [port["efectivo"]] + tabla_pos["valor_usd"].fillna(0).tolist()
        fig = go.Figure(go.Pie(labels=etiquetas, values=valores, hole=0.6, textinfo="percent", sort=False))
        st.plotly_chart(layout_grafica(fig, 300), width="stretch")

    if port["operaciones"]:
        st.markdown("#### Historial de operaciones")
        hist = pd.DataFrame(port["operaciones"]).iloc[::-1]
        st.dataframe(hist, hide_index=True, width="stretch", column_config={
            "unidades": st.column_config.NumberColumn(format="%.6f"),
            "precio_usd": st.column_config.NumberColumn(format="$%.4f"),
            "monto_usd": st.column_config.NumberColumn(format="$%.2f"),
            "comision_usd": st.column_config.NumberColumn(format="$%.2f"),
            "resultado_usd": st.column_config.NumberColumn(format="$%+.2f")})
    if st.button("♻️ Reiniciar portafolio a $100,000"):
        st.session_state.portafolio = nc.portafolio_nuevo()
        nc.guardar_portafolio(st.session_state.portafolio)
        st.rerun()
    st.caption("USD/MXN, USD/JPY, USD/CAD y USD/CHF son solo de análisis: desde una cuenta en dólares su valor es constante. "
               "Los activos cotizados en otras monedas se convierten a USD con el tipo de cambio del último cierre.")

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
        st.markdown("### 🧪 Evaluación final en el conjunto de prueba (TE)")
        st.caption("Periodo nunca usado para entrenar ni para elegir. Capital de 100,000, comisión de 0.10% y ejecución en la apertura de t+1.")
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
            st.markdown(f"#### Patrimonio en TE · {ticker}")
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
    st.markdown("### 🧠 Cómo funciona")
    a, b, c = st.columns(3)
    a.markdown("**1 · Datos**  \nPrecios diarios ajustados de Yahoo Finance. Entradas: los 20 rendimientos más recientes "
               "(y, en la versión B, distancia a la SMA 20, volatilidad 20 y RSI 14).")
    b.markdown("**2 · Modelos**  \nMLP y LSTM entrenados en una GPU T4 con separación temporal 70/15/15 (TR/VA/TE), "
               "parada temprana y 3 semillas. Se comparan con el pronóstico cero y la regresión lineal.")
    c.markdown("**3 · Señal**  \nComprar si el rendimiento estimado supera un umbral δ elegido en validación "
               "(con o sin filtro SMA 20); si no, permanecer en efectivo.")
    met = resultado("metricas_TR_VA_TE.csv")
    if met is not None and ticker in set(met.ticker):
        st.markdown(f"#### Métricas de {ticker} (escala original del rendimiento)")
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
