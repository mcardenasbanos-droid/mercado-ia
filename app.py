"""
Investify — Plataforma de pronóstico con Deep Learning y trading con IA
Proyecto final · Deep Learning para Finanzas

Ejecutar:  streamlit run app.py
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import nucleo as nc
import mexc_bot as mb
import mexc_cuenta as mxc
import polymarket_bot as pb
import polymarket_real as pr
import notificaciones as wa
import ui
from ui import VERDE, ROJO, AMARILLO, TENUE, TEXTO, BORDE, PANEL, INDIGO, TURQUESA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_LOGO = os.path.join(BASE_DIR, "assets", "logo.png")
RUTA_SECRETOS = os.path.join(BASE_DIR, ".streamlit", "secrets.toml")

st.set_page_config(page_title="Investify · Trading con IA", page_icon=RUTA_LOGO if os.path.exists(RUTA_LOGO) else "📈",
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown(f"<style>{ui.CSS}</style>", unsafe_allow_html=True)


def html(contenido):
    st.markdown(contenido, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Formatos y gráficas con estilo de exchange
# ---------------------------------------------------------------------------
def fp(x):
    """Formato de precio según su magnitud."""
    if x is None or pd.isna(x):
        return "—"
    x = float(x)
    if abs(x) >= 1000:
        return f"{x:,.2f}"
    if abs(x) >= 1:
        return f"{x:,.4f}"
    return f"{x:,.6f}"


def fpct(x, dec=2):
    return "—" if x is None or pd.isna(x) else f"{x:+.{dec}%}"


def fvol(x):
    if x is None or pd.isna(x):
        return "—"
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"{x / div:,.2f}{suf}"
    return f"{x:,.0f}"


def estilo_exchange(fig, alto, filas=1):
    fig.update_layout(template="plotly_dark", height=alto, margin=dict(l=4, r=78, t=8, b=22),
                      paper_bgcolor=PANEL, plot_bgcolor=PANEL, hovermode="x", dragmode="pan",
                      font=dict(family="Inter, Segoe UI, sans-serif", size=11, color=TENUE),
                      legend=dict(orientation="h", x=0, y=1.0, yanchor="top", bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                      hoverlabel=dict(bgcolor="#1e2329", bordercolor="#2b3139", font=dict(color=TEXTO)),
                      xaxis_rangeslider_visible=False, bargap=0.15)
    # Se aplica a todos los ejes: funciona igual en gráficas simples (go.Figure) y con subgráficas (make_subplots)
    fig.update_xaxes(showgrid=True, gridcolor="#161a1e", linecolor=BORDE, showspikes=True, spikemode="across",
                     spikesnap="cursor", spikecolor="#5e6673", spikethickness=1, spikedash="dot")
    fig.update_yaxes(side="right", showgrid=True, gridcolor="#161a1e", zeroline=False, linecolor=BORDE,
                     showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="#5e6673",
                     spikethickness=1, spikedash="dot")
    return fig


def etiqueta_eje(fig, y, texto, fondo, color="#fff", fila_y="y"):
    """Etiqueta tipo exchange sobre el eje derecho (precio actual, entrada, TP...)."""
    fig.add_annotation(xref="paper", yref=fila_y, x=1, y=y, text=f"<b>{texto}</b>", showarrow=False, xanchor="left",
                       xshift=4, bgcolor=fondo, bordercolor=fondo, borderpad=3, font=dict(size=10, color=color))


def figura_velas(v, alto=460, marcadores=None, niveles=None, indicadores=None, subgraficas=()):
    """v: DataFrame con tiempo, apertura, maximo, minimo, cierre y (opcional) volumen."""
    con_vol = "volumen" in v and v["volumen"].fillna(0).sum() > 0
    filas = 1 + int(con_vol) + len(subgraficas)
    alturas = [0.64] + ([0.14] if con_vol else []) + [0.22] * len(subgraficas)
    fig = make_subplots(rows=filas, cols=1, shared_xaxes=True, vertical_spacing=0.012,
                        row_heights=[a / sum(alturas) for a in alturas])
    fig.add_trace(go.Candlestick(x=v["tiempo"], open=v["apertura"], high=v["maximo"], low=v["minimo"], close=v["cierre"],
                                 name="Precio", increasing=dict(line=dict(color=VERDE, width=1), fillcolor=VERDE),
                                 decreasing=dict(line=dict(color=ROJO, width=1), fillcolor=ROJO), showlegend=False),
                  row=1, col=1)
    for nombre, (x, y, color, ancho, dash) in (indicadores or {}).items():
        fig.add_trace(go.Scatter(x=x, y=y, name=nombre, line=dict(color=color, width=ancho, dash=dash), hoverinfo="skip"),
                      row=1, col=1)
    fila = 2
    if con_vol:
        colores = np.where(v["cierre"] >= v["apertura"], "rgba(14,203,129,0.45)", "rgba(246,70,93,0.45)")
        fig.add_trace(go.Bar(x=v["tiempo"], y=v["volumen"], marker_color=colores, name="Volumen", showlegend=False),
                      row=2, col=1)
        fila = 3
    for sub in subgraficas:
        for tr in sub["trazas"]:
            fig.add_trace(tr, row=fila, col=1)
        for nivel, color in sub.get("lineas", []):
            fig.add_hline(y=nivel, line=dict(color=color, width=1, dash="dot"), row=fila, col=1)
        if sub.get("rango"):
            fig.update_yaxes(range=sub["rango"], row=fila, col=1)
        fila += 1
    for m in marcadores or []:
        fig.add_trace(go.Scatter(x=m["x"], y=m["y"], mode="markers+text", name=m["nombre"], text=m["texto"],
                                 textposition=m["pos"], textfont=dict(color=m["color"], size=10),
                                 marker=dict(symbol=m["simbolo"], size=13, color=m["color"],
                                             line=dict(width=1, color="#0b0e11"))), row=1, col=1)
    lo, hi = float(v["minimo"].min()), float(v["maximo"].max())
    pad = (hi - lo) * 0.12 or hi * 0.002
    y_min, y_max = lo - pad, hi + pad
    for y, texto, color in niveles or []:
        txt_color = "#0b0e11" if color == AMARILLO else "#fff"
        if y_min <= y <= y_max:
            fig.add_hline(y=y, line=dict(color=color, width=1, dash="dash"), row=1, col=1)
            etiqueta_eje(fig, y, texto, color, txt_color)
        else:                                   # fuera de la vista: etiqueta en el borde para no aplastar las velas
            borde = y_max if y > y_max else y_min
            etiqueta_eje(fig, borde, f"{texto} {fp(y)} {'↑' if y > y_max else '↓'}", color, txt_color)
    fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
    ultimo = float(v["cierre"].iloc[-1])
    color_u = VERDE if v["cierre"].iloc[-1] >= v["apertura"].iloc[-1] else ROJO
    fig.add_hline(y=ultimo, line=dict(color=color_u, width=1, dash="dot"), row=1, col=1)
    etiqueta_eje(fig, ultimo, fp(ultimo), color_u)
    return estilo_exchange(fig, alto, filas)


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


@st.cache_data(ttl=900, show_spinner="Cargando mercados…")
def datos_mercado(tickers):
    return nc.descargar_precios(list(tickers), "2y")


@st.cache_data(ttl=3600, show_spinner=False)
def pronostico_cacheado(ticker, df, mercado):
    return nc.pronosticar(ticker, df, mercado)


@st.cache_data(show_spinner=False)
def resultado(nombre):
    return nc.leer_resultado(nombre)


@st.cache_data(ttl=2, show_spinner=False)
def precios_bot(pares):
    return mb.obtener_precios(list(pares))


@st.cache_data(ttl=2, show_spinner=False)
def velas_bot(par, intervalo):
    return mb.obtener_velas(par, intervalo, 120)


@st.cache_data(ttl=10, show_spinner=False)
def stats_bot(pares):
    return mb.estadisticas_24h(list(pares))


def llaves_mexc():
    api, secreto, problema = mxc.leer_llaves(RUTA_SECRETOS)
    if api and secreto:
        return api, secreto, None
    try:                                       # respaldo: secretos de Streamlit o variables de entorno
        sec = st.secrets["mexc"]
        return sec.get("api_key", ""), sec.get("secret_key", ""), None
    except Exception:
        pass
    api, secreto = os.environ.get("MEXC_API_KEY", ""), os.environ.get("MEXC_SECRET_KEY", "")
    return api, secreto, (None if api and secreto else problema)


@st.cache_resource(show_spinner=False)
def cliente_mexc(api_key, secret_key):
    return mxc.ClienteMEXC(api_key, secret_key)


@st.cache_data(ttl=20, show_spinner=False)
def saldos_mexc(api_key, secret_key):
    return cliente_mexc(api_key, secret_key).cuenta()


@st.cache_data(ttl=30, show_spinner=False)
def operaciones_mexc(api_key, secret_key, par):
    return cliente_mexc(api_key, secret_key).mis_operaciones(par, 50)


precios = datos_mercado(tuple(TICKERS))
if not precios:
    st.error("No se pudieron descargar datos de Yahoo Finance. Revisa tu conexión y recarga la página.")
    st.stop()
faltantes = [t for t in TICKERS if t not in precios]

pronosticos = {}
with st.spinner("Calculando pronósticos con los modelos…"):
    for t in TICKERS:
        if t in precios:
            pronosticos[t] = pronostico_cacheado(t, precios[t], INFO.loc[t, "mercado"])

tasas = nc.tasas_usd(precios)
precios_usd = {t: nc.precio_usd(float(precios[t]["cierre"].iloc[-1]), INFO.loc[t, "moneda_cotizada"], tasas)
               for t in TICKERS if t in precios and INFO.loc[t, "moneda_cotizada"] in tasas}

CLAVE, SECRETO, PROBLEMA_LLAVES = llaves_mexc()
WA_CFG = wa.leer_config(RUTA_SECRETOS)
CLIENTE = cliente_mexc(CLAVE, SECRETO) if CLAVE and SECRETO else None

# Aviso al celular de los activos que la IA marca como recomendables (una sola vez por activo y periodo)
RECOMENDACIONES = wa.recomendaciones(pronosticos, INFO["nombre"].to_dict())
if WA_CFG["listo"]:
    _ruta_env = os.path.join(BASE_DIR, "avisos_enviados.json")
    _en_envio = st.session_state.setdefault("recs_en_envio", set())   # evita mandarlas doble
    _recs_nuevas = [r for r in wa.recomendaciones_nuevas(RECOMENDACIONES, _ruta_env) if wa._clave_rec(r) not in _en_envio]
    if _recs_nuevas:
        _en_envio.update(wa._clave_rec(r) for r in _recs_nuevas)
        wa.enviar_en_segundo_plano(WA_CFG, [wa.mensaje_recomendacion(r) for r in _recs_nuevas],
                                   al_enviar=lambda i, r=_recs_nuevas: wa.marcar_enviadas([r[i]], _ruta_env))

if "mexc" not in st.session_state:
    st.session_state.mexc = mb.cargar_estado()
    st.session_state.mexc_corriendo = st.session_state.mexc["bot"] is not None
if "portafolio" not in st.session_state:
    st.session_state.portafolio = nc.cargar_portafolio()
EST = st.session_state.mexc
PORT = st.session_state.portafolio


def cambio_diario(t):
    c = precios[t]["cierre"]
    return float(c.iloc[-1] / c.iloc[-2] - 1) if len(c) > 1 else np.nan


# ---------------------------------------------------------------------------
# Encabezado y cinta
# ---------------------------------------------------------------------------
bot_g = EST["bot"]
pastillas = []
if bot_g is not None:
    pastillas.append(("Bot operando" if st.session_state.mexc_corriendo else "Bot en pausa",
                      "live" if st.session_state.mexc_corriendo else "warn"))
    pastillas.append(("MODO REAL", "real") if bot_g.get("modo") == "real" else ("MODO PAPER", "paper"))
    eq_etq, eq_val = "Equity del bot", f"{mb.patrimonio(bot_g):,.2f} USDT"
else:
    pastillas.append(("Mercados en vivo", "live"))
    eq_etq, eq_val = "Portafolio simulado", f"${nc.valorar(PORT, precios_usd)[1]['patrimonio']:,.2f}"
html(ui.encabezado(pastillas, eq_etq, eq_val))
html(ui.cinta([(t.replace("-USD", "").replace("=X", ""), fp(precios[t]["cierre"].iloc[-1]), cambio_diario(t))
               for t in TICKERS if t in precios and len(precios[t]) > 1]))
if faltantes:
    html(ui.aviso(f"Sin datos por ahora: {', '.join(faltantes)}. Recarga la página en unos minutos.", "warn"))

tab_mercados, tab_analisis, tab_portafolio, tab_bot, tab_poly, tab_resultados, tab_modelo = st.tabs(
    ["Mercados", "Análisis", "Portafolio", "Bot MEXC", "Bot Polymarket", "Resultados", "Modelo"])

# ===========================================================================
# MERCADOS
# ===========================================================================
with tab_mercados:
    mk = st.segmented_control("Mercado", ["Todos"] + MERCADOS, default="Todos", key="mk_mercado",
                              label_visibility="collapsed") or "Todos"
    filas = []
    for t in TICKERS:
        if t not in precios:
            continue
        c = precios[t]["cierre"]
        p = pronosticos.get(t) or {}
        ind = nc.calcular_indicadores(precios[t])
        filas.append({
            "Activo": t, "Nombre": INFO.loc[t, "nombre"], "Mercado": INFO.loc[t, "mercado"], "Precio": float(c.iloc[-1]),
            "24h %": cambio_diario(t) * 100,
            "1 mes %": float(c.iloc[-1] / c.iloc[-22] - 1) * 100 if len(c) > 22 else np.nan,
            "Tendencia 3M": c.iloc[-63:].round(6).tolist(), "RSI 14": float(ind["rsi14"].iloc[-1]),
            "Pronóstico IA": p.get("estimado_dl", np.nan) * 100 if p else np.nan,
            "Señal": "Comprar" if p.get("senal") == "comprar" else "Efectivo",
        })
    tabla_m = pd.DataFrame(filas)
    if mk != "Todos":
        tabla_m = tabla_m[tabla_m.Mercado == mk]
    hay = len(tabla_m) and tabla_m["24h %"].notna().any()
    mejor = tabla_m.loc[tabla_m["24h %"].idxmax()] if hay else None
    peor = tabla_m.loc[tabla_m["24h %"].idxmin()] if hay else None
    prom = tabla_m["Pronóstico IA"].mean() / 100 if len(tabla_m) else np.nan
    html(ui.franja([
        ("Activos", f"{len(tabla_m)}", "neu", mk if mk != "Todos" else "acciones, criptos y divisas"),
        ("Señales de compra (IA)", f"{(tabla_m['Señal'] == 'Comprar').sum()}", "up", "modelo MLP / LSTM"),
        ("Mayor alza", f"{mejor['Activo']} {mejor['24h %']:+.2f}%" if mejor is not None else "—", "up",
         mejor["Nombre"] if mejor is not None else ""),
        ("Mayor baja", f"{peor['Activo']} {peor['24h %']:+.2f}%" if peor is not None else "—", "dn",
         peor["Nombre"] if peor is not None else ""),
        ("Pronóstico promedio", fpct(prom, 3), ui.tono(prom), "rendimiento estimado para mañana"),
    ]))

    def _color(v):
        if isinstance(v, str):
            return f"color: {VERDE}; font-weight: 700" if v == "Comprar" else f"color: {TENUE}"
        if pd.isna(v):
            return ""
        return f"color: {VERDE}" if v > 0 else (f"color: {ROJO}" if v < 0 else "")

    st.dataframe(
        tabla_m.style.map(_color, subset=["24h %", "1 mes %", "Pronóstico IA", "Señal"]), hide_index=True,
        width="stretch", height=min(40 + 35 * len(tabla_m), 780),
        column_config={
            "Precio": st.column_config.NumberColumn(format="%.4f"),
            "24h %": st.column_config.NumberColumn(format="%+.2f%%"),
            "1 mes %": st.column_config.NumberColumn(format="%+.2f%%"),
            "Tendencia 3M": st.column_config.LineChartColumn(width="medium"),
            "RSI 14": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
            "Pronóstico IA": st.column_config.NumberColumn(format="%+.3f%%", help="Rendimiento estimado por la red para el siguiente periodo"),
        })
    st.caption("Datos diarios de Yahoo Finance, ajustados por dividendos y splits. El pronóstico es del modelo final de cada activo.")

# ===========================================================================
# ANÁLISIS
# ===========================================================================
with tab_analisis:
    a1, a2, a3, a4 = st.columns([1.5, 2.2, 1.6, 0.9], vertical_alignment="bottom")
    an_mk = a1.segmented_control("Mercado", ["Todos"] + MERCADOS, default="Todos", key="an_mercado") or "Todos"
    opciones = [t for t in TICKERS if t in precios and (an_mk == "Todos" or INFO.loc[t, "mercado"] == an_mk)]
    ticker = a2.selectbox("Activo", opciones, format_func=lambda t: f"{t} · {INFO.loc[t, 'nombre']}", key="an_activo")
    periodo = a3.segmented_control("Periodo", ["3M", "6M", "1A", "2A"], default="6M", key="an_periodo") or "6M"
    with a4.popover("Indicadores"):
        ver_sma = st.toggle("SMA 20", value=True)
        ver_bb = st.toggle("Bandas de Bollinger", value=True)
        ver_rsi = st.toggle("RSI 14", value=True)
        ver_macd = st.toggle("MACD", value=False)

    if ticker:
        moneda = INFO.loc[ticker, "moneda_cotizada"]
        df = precios[ticker]
        ind = nc.calcular_indicadores(df)
        p = pronosticos.get(ticker)
        c = df["cierre"]
        ch = cambio_diario(ticker)
        anio = c.iloc[-252:]
        html(ui.barra_par(ticker, f"{INFO.loc[ticker, 'nombre']} · {INFO.loc[ticker, 'mercado']} · {moneda}",
                          fp(c.iloc[-1]), ui.tono(ch), [
                              ("Cambio diario", fpct(ch), ui.tono(ch)),
                              ("Máx. 52 semanas", fp(anio.max()), "neu"),
                              ("Mín. 52 semanas", fp(anio.min()), "neu"),
                              ("Distancia SMA 20", fpct(ind["dist_sma20"].iloc[-1]), ui.tono(ind["dist_sma20"].iloc[-1])),
                              ("Volatilidad 20d", f"{ind['vol20'].iloc[-1]:.2%}", "neu"),
                              ("RSI 14", f"{ind['rsi14'].iloc[-1]:.0f}", "neu"),
                          ]))
        g, pr_col = st.columns([3.3, 1.15], gap="small")
        with g:
            meses = {"3M": 3, "6M": 6, "1A": 12, "2A": 24}[periodo]
            desde = df.index.max() - pd.DateOffset(months=meses)
            d, i2 = df[df.index >= desde], ind[ind.index >= desde]
            v = pd.DataFrame({"tiempo": d.index, "apertura": d.get("apertura", d["cierre"]), "maximo": d.get("maximo", d["cierre"]),
                              "minimo": d.get("minimo", d["cierre"]), "cierre": d["cierre"],
                              "volumen": d["volumen"] if "volumen" in d else 0.0}).fillna({"apertura": d["cierre"]})
            indicadores = {}
            if ver_sma:
                indicadores["SMA 20"] = (i2.index, i2["sma20"], AMARILLO, 1.4, "solid")
            if ver_bb:
                indicadores["Bollinger sup."] = (i2.index, i2["bb_sup"], "#5e6673", 1, "dot")
                indicadores["Bollinger inf."] = (i2.index, i2["bb_inf"], "#5e6673", 1, "dot")
            subs = []
            if ver_rsi:
                subs.append({"trazas": [go.Scatter(x=i2.index, y=i2["rsi14"], name="RSI 14", line=dict(color="#a78bfa", width=1.3))],
                             "lineas": [(70, ROJO), (30, VERDE)], "rango": [0, 100]})
            if ver_macd:
                subs.append({"trazas": [
                    go.Bar(x=i2.index, y=i2["macd_hist"], name="Histograma MACD",
                           marker_color=np.where(i2["macd_hist"] >= 0, VERDE, ROJO)),
                    go.Scatter(x=i2.index, y=i2["macd"], name="MACD", line=dict(color=INDIGO, width=1.2)),
                    go.Scatter(x=i2.index, y=i2["macd_senal"], name="Señal", line=dict(color=AMARILLO, width=1.2))]})
            marcas = []
            if p:
                marcas.append({"x": [pd.Timestamp(p["proximo_periodo"])], "y": [p["precio_estimado"]], "nombre": "Pronóstico IA",
                               "texto": ["IA"], "pos": "top center", "simbolo": "star",
                               "color": VERDE if p["estimado_dl"] >= 0 else ROJO})
            fig = figura_velas(v, 470 + 120 * len(subs), marcadores=marcas, indicadores=indicadores, subgraficas=subs)
            st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": True})
        with pr_col:
            with st.container(border=True):
                html(ui.titulo_panel("Pronóstico IA", p["proximo_periodo"] if p else ""))
                if p:
                    compra = p["senal"] == "comprar"
                    html(ui.valor_grande("Rendimiento estimado", f'<span class="{ui.tono(p["estimado_dl"])}">{p["estimado_dl"]:+.3%}</span>'))
                    html(ui.etiqueta("▲ SEÑAL: COMPRAR" if compra else "SEÑAL: PERMANECER EN EFECTIVO", "buy" if compra else "neu"))
                    filas_kv = [("Precio estimado", fp(p["precio_estimado"]), "neu"), ("Último cierre", fp(p["cierre"]), "neu")]
                    if p.get("red"):
                        filas_kv += [("Modelo", f"{p['red']} · {p.get('neuronas', '')} neuronas", "neu"),
                                     ("Versión", "B · rezagos + indicadores" if p["version"] == "B" else "A · solo rezagos", "neu"),
                                     ("Regla", "pronóstico > δ y cierre > SMA20" if str(p["regla"]).startswith("D") else "pronóstico > δ", "neu"),
                                     ("Umbral δ", f"{p['umbral']:.3%}", "neu"),
                                     ("Regresión lineal", f"{p['estimado_lineal']:+.3%}", ui.tono(p["estimado_lineal"]))]
                    filas_kv.append(("Fecha de corte", p["fecha_corte"], "neu"))
                    html(ui.lista_kv(filas_kv))
                    if p.get("aviso"):
                        html(ui.aviso(p["aviso"], "warn"))
                else:
                    html(ui.aviso("No hay pronóstico disponible para este activo."))
        with st.expander("¿Cómo se calculan los indicadores?"):
            st.markdown("""
| Indicador | Cálculo |
|---|---|
| **SMA 20** | Promedio de los últimos 20 cierres. Distancia = cierre / SMA − 1 |
| **Volatilidad 20** | Desviación estándar muestral de los últimos 20 rendimientos diarios, sin anualizar |
| **RSI 14** | 100 − 100 / (1 + RS), con RS = promedio de ganancias / promedio de pérdidas (promedio simple de 14) |
| **MACD** | EMA 12 − EMA 26; señal = EMA 9 del MACD |
| **Bollinger** | SMA 20 ± 2 desviaciones estándar de los cierres |

Todos usan solo información disponible hasta el cierre de cada día. Un RSI alto **no garantiza** que el precio vaya a caer.
""")

# ===========================================================================
# PORTAFOLIO
# ===========================================================================
with tab_portafolio:
    tabla_pos, res = nc.valorar(PORT, precios_usd)
    html(ui.franja([
        ("Patrimonio", f"${res['patrimonio']:,.2f}", "neu", "cuenta simulada en USD"),
        ("Rendimiento", fpct(res["rendimiento"]), ui.tono(res["rendimiento"]), f"capital inicial ${PORT['capital_inicial']:,.0f}"),
        ("Efectivo", f"${res['efectivo']:,.2f}", "neu", "disponible para comprar"),
        ("Invertido", f"${res['invertido']:,.2f}", "neu", f"{len(tabla_pos)} posiciones"),
        ("Comisiones", f"${res['comisiones']:,.2f}", "neu", "0.10% por operación"),
        ("Win rate", "—" if pd.isna(res["win_rate"]) else f"{res['win_rate']:.0%}", "neu",
         f"{res['operaciones_cerradas']} operaciones cerradas"),
    ]))
    operables = [t for t in TICKERS if t in precios_usd and t not in nc.SOLO_ANALISIS]
    izq, der = st.columns([1.1, 2.4], gap="medium")
    with izq:
        with st.container(border=True):
            html(ui.titulo_panel("Operar", "precio de mercado"))
            t_comprar, t_vender, t_auto = st.tabs(["Comprar", "Vender", "Seguir a la IA"])
            with t_comprar:
                with st.form("form_compra", border=False):
                    t_c = st.selectbox("Activo", operables, format_func=lambda t: f"{t} · ${precios_usd[t]:,.4f}")
                    monto = st.number_input("Monto en USD (incluye comisión)", min_value=0.0,
                                            max_value=float(max(PORT["efectivo"], 0.0)),
                                            value=float(min(10_000.0, max(PORT["efectivo"], 0.0))), step=500.0)
                    if st.form_submit_button("Comprar", type="primary"):
                        try:
                            u = nc.comprar(PORT, t_c, monto, precios_usd[t_c])
                            nc.guardar_portafolio(PORT)
                            st.toast(f"Compraste {u:,.6f} de {t_c}")
                            st.rerun()
                        except ValueError as e:
                            st.error(str(e))
            with t_vender:
                with st.form("form_venta", border=False):
                    con_pos = list(PORT["posiciones"].keys())
                    t_v = st.selectbox("Posición", con_pos if con_pos else ["(sin posiciones)"])
                    frac = st.slider("Porcentaje a vender", 1, 100, 100, format="%d%%")
                    if st.form_submit_button("Vender"):
                        try:
                            if not con_pos:
                                raise ValueError("No tienes posiciones abiertas.")
                            neto = nc.vender(PORT, t_v, frac / 100, precios_usd[t_v])
                            nc.guardar_portafolio(PORT)
                            st.toast(f"Venta de {t_v}: recibiste ${neto:,.2f}")
                            st.rerun()
                        except (ValueError, KeyError) as e:
                            st.error(str(e))
            with t_auto:
                with st.form("form_auto", border=False):
                    st.caption("Vende las posiciones con señal «efectivo» y compra los activos con señal «comprar».")
                    monto_auto = st.number_input("USD por activo", min_value=100.0, value=5_000.0, step=500.0)
                    if st.form_submit_button("Ejecutar señales de la IA", type="primary"):
                        acciones = nc.ejecutar_senales(PORT, pronosticos, precios_usd, monto_auto)
                        nc.guardar_portafolio(PORT)
                        st.toast(f"{len(acciones)} operaciones ejecutadas" if acciones else "Nada que ejecutar")
                        st.rerun()
        if len(tabla_pos):
            etq = ["Efectivo"] + tabla_pos["ticker"].tolist()
            val = [PORT["efectivo"]] + tabla_pos["valor_usd"].fillna(0).tolist()
            paleta = ["#2b3139", INDIGO, TURQUESA, VERDE, AMARILLO, "#ec4899", "#8b5cf6", "#06b6d4", "#84cc16", "#f97316"]
            fig = go.Figure(go.Pie(labels=etq, values=val, hole=0.68, textinfo="none", sort=False,
                                   marker=dict(colors=(paleta * 5)[:len(etq)], line=dict(color=PANEL, width=2))))
            fig.add_annotation(text=f"<b>${res['patrimonio']:,.0f}</b><br><span style='font-size:10px'>patrimonio</span>",
                               showarrow=False, font=dict(color=TEXTO, size=14))
            fig.update_layout(height=260, margin=dict(l=0, r=0, t=6, b=0), paper_bgcolor="rgba(0,0,0,0)",
                              showlegend=True, legend=dict(font=dict(color=TENUE, size=10)), template="plotly_dark")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with der:
        html(ui.titulo_panel("Posiciones abiertas", f"{len(tabla_pos)}"))
        pos = tabla_pos.rename(columns={"ticker": "Activo", "unidades": "Cantidad", "precio_usd": "Precio (USD)",
                                        "valor_usd": "Valor (USD)", "costo_usd": "Costo (USD)",
                                        "ganancia_usd": "PNL (USD)", "ganancia_%": "PNL %"})
        html(ui.tabla(pos, formatos={"Cantidad": "{:,.6f}", "Precio (USD)": fp, "Valor (USD)": "{:,.2f}",
                                     "Costo (USD)": "{:,.2f}", "PNL (USD)": "{:+,.2f}", "PNL %": "{:+.2f}%"},
                      colores=["PNL (USD)", "PNL %"], negritas=["Activo"]))
        html(ui.titulo_panel("Historial de operaciones", f"{len(PORT['operaciones'])}"))
        hist = pd.DataFrame(PORT["operaciones"], columns=["fecha", "ticker", "tipo", "unidades", "precio_usd", "monto_usd",
                                                          "comision_usd", "resultado_usd"]).iloc[::-1]
        hist = hist.rename(columns={"fecha": "Fecha", "ticker": "Activo", "tipo": "Lado", "unidades": "Cantidad",
                                    "precio_usd": "Precio (USD)", "monto_usd": "Monto (USD)", "comision_usd": "Comisión",
                                    "resultado_usd": "PNL (USD)"})
        hist["Lado"] = hist["Lado"].map({"compra": "Comprar", "venta": "Vender"})
        html(ui.tabla(hist, formatos={"Cantidad": "{:,.6f}", "Precio (USD)": fp, "Monto (USD)": "{:,.2f}",
                                      "Comisión": "{:,.2f}", "PNL (USD)": "{:+,.2f}"},
                      insignias={"Lado": {"Comprar": "buy", "Vender": "sell"}}, colores=["PNL (USD)"], alto=360))
        if st.button("Reiniciar portafolio a $100,000"):
            st.session_state.portafolio = nc.portafolio_nuevo()
            nc.guardar_portafolio(st.session_state.portafolio)
            st.rerun()
        st.caption("USD/MXN, USD/JPY, USD/CAD y USD/CHF son solo de análisis: desde una cuenta en dólares su valor es constante.")

# ===========================================================================
# BOT DE TRADING (MEXC)
# ===========================================================================
def tabla_ops(ops, alto=None):
    df = pd.DataFrame(ops, columns=["fecha", "par", "tipo", "precio", "unidades", "monto_usdt", "ganancia_usdt",
                                    "ganancia_%", "comision_usdt", "estado", "modo", "motivo"])
    t = pd.DataFrame({
        "Fecha": df["fecha"], "Par": df["par"], "Lado": df["tipo"].map({"compra": "Comprar", "venta": "Vender"}),
        "Precio": df["precio"], "Cantidad": df["unidades"], "Monto (USDT)": df["monto_usdt"],
        "PNL (USDT)": df["ganancia_usdt"], "PNL %": df["ganancia_%"], "Comisión": df["comision_usdt"],
        "Estado": df["estado"].fillna("Ejecutada"), "Modo": df["modo"].fillna("paper").str.capitalize(), "Motivo": df["motivo"],
    })
    html(ui.tabla(t, formatos={"Precio": fp, "Cantidad": "{:,.6g}", "Monto (USDT)": "{:,.2f}", "PNL (USDT)": "{:+,.2f}",
                               "PNL %": "{:+.2f}%", "Comisión": "{:,.4f}"},
                  insignias={"Lado": {"Comprar": "buy", "Vender": "sell"}, "Estado": {"Ejecutada": "ok", "Rechazada": "bad"},
                             "Modo": {"Real": "real", "Paper": "neu"}},
                  colores=["PNL (USDT)", "PNL %"], negritas=["Par"], alto=alto))


def formulario_bot(est):
    """Configuración del bot (modo paper o real)."""
    with st.form("form_mexc", border=False):
        pares_sel = st.multiselect("Pares de MEXC", list(mb.PARES), default=list(mb.PARES))
        c1, c2 = st.columns(2)
        monto = c1.number_input("Monto por orden (USDT)", min_value=mb.MONTO_MINIMO, max_value=mb.CAPITAL_MEXC,
                                value=5_000.0, step=500.0)
        vel = c2.select_slider("Actualizar cada", options=[2, 5, 10, 30], value=5, format_func=lambda x: f"{x} s")
        c3, c4 = st.columns(2)
        tc = c3.slider("Tope de compra (baja desde su máximo)", 0.2, 5.0, 0.5, 0.1, format="%.1f%%")
        tv = c4.slider("Tope de venta (ganancia)", 0.3, 5.0, 0.8, 0.1, format="%.1f%%")
        c5, c6 = st.columns(2)
        usar_sl = c5.toggle("Usar stop-loss", value=False, help="Apagado: el bot solo vende con ganancia.")
        sl = c6.slider("Stop-loss", 0.5, 20.0, 5.0, 0.5, format="%.1f%%")
        st.markdown("**Modo real** · dinero real en tu cuenta Spot de MEXC")
        e1, e2, e3 = st.columns(3)
        pedir_real = e1.toggle("Activar modo real", value=False, disabled=CLIENTE is None,
                               help="Requiere tus llaves en .streamlit/secrets.toml. Solo funciona en tu computadora.")
        monto_real = e2.number_input("USDT por orden (real)", min_value=5.0, max_value=1_000.0, value=5.0, step=1.0)
        tope_real = e3.number_input("Tope total (USDT)", min_value=5.0, max_value=10_000.0, value=50.0, step=5.0)
        confirmacion = st.text_input("Para el modo real escribe: OPERAR CON DINERO REAL", disabled=CLIENTE is None)
        enviar = st.form_submit_button("Iniciar bot" if est["bot"] is None else "Reiniciar bot con esta configuración",
                                       type="primary")
    if enviar:
        try:
            pr, fuente = precios_bot(tuple(pares_sel))
            if pedir_real:
                if confirmacion.strip().upper() != "OPERAR CON DINERO REAL":
                    raise ValueError("Para activar el modo real escribe exactamente: OPERAR CON DINERO REAL")
                if fuente != "MEXC":
                    raise ValueError("El modo real necesita conexión directa con MEXC (corre la app en tu computadora).")
                usdt = CLIENTE.saldo("USDT")
                if usdt < monto_real:
                    raise ValueError(f"Tu cuenta Spot tiene {usdt:,.2f} USDT; necesitas al menos {monto_real:,.2f} USDT.")
                est["bot"] = mb.bot_nuevo(pr, monto_real, tc / 100, tv / 100, sl / 100 if usar_sl else None,
                                          capital=min(tope_real, usdt), modo="real")
            else:
                est["bot"] = mb.bot_nuevo(pr, monto, tc / 100, tv / 100, sl / 100 if usar_sl else None)
            est["bot"].update({"fuente": fuente, "intervalo_s": vel})
            st.session_state.mexc_corriendo = True
            mb.guardar_estado(est)
            st.rerun()
        except (ValueError, mb.FuenteNoDisponible, mxc.ErrorMEXC) as e:
            st.error(str(e))


def cerrar_posiciones(est):
    bot = est["bot"]
    pr, fuente = precios_bot(tuple(bot["activos"]))
    real = bot.get("modo") == "real"
    for par, a in bot["activos"].items():
        if a["unidades"] <= 0 or par not in pr:
            continue
        if real:
            try:
                r_ = CLIENTE.vender_mercado(par, a["unidades"])
            except mxc.ErrorMEXC as e:
                st.error(f"{par}: {e}")
                continue
            neto, px, vendidas, comision, oid = r_["total_usdt"], r_["precio"], r_["cantidad"], None, r_["order_id"]
            costo = a["monto"] * min(vendidas / a["unidades"], 1.0)
        else:
            bruto = a["unidades"] * pr[par]
            comision = bruto * mb.COSTO
            neto, px, vendidas, oid, costo = bruto - comision, pr[par], a["unidades"], None, a["monto"]
        bot["efectivo"] += neto
        mb._registrar(est, {"bot_id": bot["id"], "fecha": mb.ahora().strftime("%Y-%m-%d %H:%M:%S"), "par": par,
                            "tipo": "venta", "motivo": "cierre manual", "precio": px, "unidades": vendidas,
                            "monto_usdt": neto, "comision_usdt": comision, "ganancia_usdt": neto - costo,
                            "ganancia_%": (neto / costo - 1) * 100, "fuente": fuente, "modo": "real" if real else "paper",
                            "estado": "Ejecutada", "order_id": oid})
        a.update({"unidades": 0.0, "precio_compra": None, "monto": 0.0, "maximo_reciente": pr[par]})
    st.session_state.mexc_corriendo = False
    mb.guardar_estado(est)


def panel_cuenta_mexc():
    if CLIENTE is None:
        html(ui.aviso("Para ver tu saldo real y activar el modo real, crea el archivo <code>.streamlit/secrets.toml</code> en tu "
                      "computadora con tus llaves (instrucciones en el README). Nunca lo subas a GitHub."))
        if PROBLEMA_LLAVES:
            html(ui.aviso(f"Diagnóstico: {ui.esc(PROBLEMA_LLAVES)}", "warn"))
        return
    try:
        saldos = saldos_mexc(CLAVE, SECRETO)
        try:
            pr_all, _ = precios_bot(tuple(f"{a}USDT" for a in saldos if a != "USDT"))
        except mb.FuenteNoDisponible:
            pr_all = {}
        filas_s = [{"Activo": a, "Disponible": s_["libre"], "En órdenes": s_["bloqueado"],
                    "Valor (USDT)": (s_["libre"] + s_["bloqueado"]) * (1.0 if a == "USDT" else pr_all.get(f"{a}USDT", np.nan))}
                   for a, s_ in saldos.items()]
        tabla_s = pd.DataFrame(filas_s, columns=["Activo", "Disponible", "En órdenes", "Valor (USDT)"])
        total = float(np.nansum(tabla_s["Valor (USDT)"])) if len(tabla_s) else 0.0
        usdt = saldos.get("USDT", {"libre": 0.0})["libre"]
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        c1.markdown(ui.franja([("Valor de tu cuenta Spot", f"{total:,.2f} USDT", "neu", "saldo real en MEXC"),
                               ("USDT disponible", f"{usdt:,.2f} USDT", "neu", "para el modo real")]), unsafe_allow_html=True)
        if c2.button("Probar conexión", help="Orden de prueba que MEXC valida pero NO ejecuta."):
            try:
                CLIENTE.probar_orden("BTCUSDT", 5)
                st.success("Conexión correcta: tus llaves y permisos funcionan. No se ejecutó ninguna operación.")
            except mxc.ErrorMEXC as e:
                st.error(str(e))
        html(ui.tabla(tabla_s, formatos={"Disponible": "{:,.8g}", "En órdenes": "{:,.8g}", "Valor (USDT)": "{:,.2f}"},
                      negritas=["Activo"]))
        par_h = st.selectbox("Mis operaciones reales de", list(mb.PARES), key="mexc_hist_real")
        ops_real = operaciones_mexc(CLAVE, SECRETO, par_h)
        if ops_real:
            dfr = pd.DataFrame(ops_real)
            if "time" in dfr:
                dfr["time"] = pd.to_datetime(dfr["time"], unit="ms", utc=True).dt.tz_convert(mb.ZONA).dt.tz_localize(None)
            st.dataframe(dfr, hide_index=True, width="stretch")
        else:
            st.caption(f"No tienes operaciones reales en {par_h}.")
    except mxc.ErrorMEXC as e:
        st.error(str(e))


with tab_bot:
    intervalo_s = (EST["bot"] or {}).get("intervalo_s", 5)

    @st.fragment(run_every=intervalo_s if (EST["bot"] is not None and st.session_state.mexc_corriendo) else None)
    def terminal_bot():
        est = st.session_state.mexc
        bot = est["bot"]
        pares = list(bot["activos"]) if bot else list(mb.PARES)

        # ---------- precios en vivo y bot ----------
        try:
            pr, fuente = precios_bot(tuple(pares))
        except (mb.FuenteNoDisponible, KeyError, ValueError) as e:
            pr, fuente = {}, "sin conexión"
            html(ui.aviso(f"No se pudieron obtener precios: {ui.esc(e)}", "warn"))
        if bot is not None and st.session_state.mexc_corriendo and pr:
            if bot.get("modo") == "real" and (CLIENTE is None or fuente != "MEXC"):
                st.session_state.mexc_corriendo = False
                html(ui.aviso("Modo real en pausa: no hay conexión directa con MEXC o faltan las llaves.", "real"))
            else:
                ops_nuevas = mb.bot_actualizar(est, pr, fuente, ejecutor=CLIENTE if bot.get("modo") == "real" else None)
                avisos = [wa.mensaje_mexc(o, bot) for o in ops_nuevas if o.get("estado", "Ejecutada") == "Ejecutada"]
                if avisos and WA_CFG["listo"]:
                    wa.enviar_en_segundo_plano(WA_CFG, avisos)
                for op in ops_nuevas:
                    if op.get("estado") == "Rechazada":
                        st.toast(f"Orden rechazada en {op['par']}: {op['motivo']}")
                    else:
                        st.toast(f"{'Compra' if op['tipo'] == 'compra' else 'Venta'} {op['par']} a {fp(op['precio'])}"
                                 + (f" · {op['ganancia_usdt']:+.2f} USDT" if op["ganancia_usdt"] is not None else ""))
                mb.guardar_estado(est)
        try:
            stats = stats_bot(tuple(pares))
        except mb.FuenteNoDisponible:
            stats = {}
            for par in pares:                       # respaldo: datos diarios de Yahoo
                t = mb.PARES.get(par)
                if t in precios and len(precios[t]) > 1:
                    d = precios[t]
                    stats[par] = {"ultimo": float(d["cierre"].iloc[-1]), "cambio": cambio_diario(t),
                                  "maximo": float(d["maximo"].iloc[-1]) if "maximo" in d else np.nan,
                                  "minimo": float(d["minimo"].iloc[-1]) if "minimo" in d else np.nan, "volumen": np.nan}
        if fuente.startswith("Yahoo"):
            html(ui.aviso("MEXC no está disponible desde este servidor (MEXC no opera en EE. UU., donde está Streamlit Cloud). "
                          "Se usan precios de Yahoo Finance con hasta 1 minuto de retraso. Para MEXC en tiempo real, "
                          "corre la app en tu computadora.", "warn"))

        # ---------- franja superior ----------
        if bot is not None:
            r = mb.resumen(bot)
            html(ui.franja([
                ("Capital inicial", f"{r['capital']:,.2f} USDT", "neu", "modo real" if bot.get("modo") == "real" else "modo paper"),
                ("Equity total", f"{r['patrimonio']:,.2f} USDT", "neu", f"disponible {r['efectivo']:,.2f}"),
                ("PNL realizado", f"{r['ganancia_realizada']:+,.2f} USDT", ui.tono(r["ganancia_realizada"]),
                 f"{r['ganadoras']} de {r['ventas']} ventas con ganancia"),
                ("PNL no realizado", f"{r['ganancia_no_realizada']:+,.2f} USDT", ui.tono(r["ganancia_no_realizada"]),
                 f"{len(r['abiertas'])} posiciones abiertas"),
                ("Rendimiento", fpct(r["rendimiento"]), ui.tono(r["rendimiento"]), f"{r['ganancia_total']:+,.2f} USDT"),
            ]))

        # ---------- par seleccionado ----------
        if st.session_state.get("par_sel") not in pares:
            st.session_state.par_sel = pares[0]
        par = st.session_state.par_sel
        s_ = stats.get(par, {})
        ultimo = pr.get(par, s_.get("ultimo"))
        html(ui.barra_par(par.replace("USDT", "/USDT"), "Spot · MEXC", fp(ultimo), ui.tono(s_.get("cambio")), [
            ("Cambio 24h", fpct(s_.get("cambio")), ui.tono(s_.get("cambio"))),
            ("Máx. 24h", fp(s_.get("maximo")), "neu"), ("Mín. 24h", fp(s_.get("minimo")), "neu"),
            ("Volumen 24h (USDT)", fvol(s_.get("volumen")), "neu"), ("Fuente", fuente, "neu"),
            ("Actualizado", f"{mb.ahora():%H:%M:%S}", "neu"),
        ]))

        # ---------- terminal de 3 columnas ----------
        c_lista, c_graf, c_bot = st.columns([1.3, 3.1, 1.3], gap="small")
        with c_lista:
            with st.container(border=True):
                html(ui.titulo_panel("Pares", "Precio · 24h"))
                wl = pd.DataFrame([{"Par": p_.replace("USDT", ""), "Precio": pr.get(p_, stats.get(p_, {}).get("ultimo", np.nan)),
                                    "24h %": stats.get(p_, {}).get("cambio", np.nan) * 100} for p_ in pares])
                ev = st.dataframe(
                    wl.style.map(lambda v_: "" if pd.isna(v_) else (f"color:{VERDE}" if v_ > 0 else f"color:{ROJO}"),
                                 subset=["24h %"]),
                    hide_index=True, width="stretch", height=38 + 35 * len(wl), key="wl_bot",
                    on_select="rerun", selection_mode="single-row",
                    column_config={"Par": st.column_config.TextColumn(width="small"),
                                   "Precio": st.column_config.NumberColumn(format="%.6g"),
                                   "24h %": st.column_config.NumberColumn(format="%+.2f%%", width="small")})
                filas_sel = getattr(getattr(ev, "selection", None), "rows", []) if ev is not None else []
                if filas_sel:
                    nuevo = wl.iloc[filas_sel[0]]["Par"] + "USDT"
                    if nuevo != st.session_state.par_sel:
                        st.session_state.par_sel = nuevo
                        st.rerun(scope="fragment")
        with c_graf:
            with st.container(border=True):
                intervalo = st.segmented_control("Intervalo", list(mb.INTERVALOS), default="1 minuto", key="bot_int",
                                                 label_visibility="collapsed") or "1 minuto"
                try:
                    v, fuente_v = velas_bot(par, mb.INTERVALOS[intervalo])
                    marcas, niveles = [], []
                    if bot is not None:
                        desde = v["tiempo"].min()
                        ops_par = [o for o in bot["operaciones"] if o["par"] == par and o.get("estado", "Ejecutada") == "Ejecutada"
                                   and pd.Timestamp(o["fecha"]) >= desde]
                        for tipo, sim, color, txt, pos in (("compra", "triangle-up", VERDE, "B", "bottom center"),
                                                           ("venta", "triangle-down", ROJO, "S", "top center")):
                            sub = [o for o in ops_par if o["tipo"] == tipo]
                            if sub:
                                marcas.append({"x": [pd.Timestamp(o["fecha"]) for o in sub], "y": [o["precio"] for o in sub],
                                               "nombre": "Compra del bot" if tipo == "compra" else "Venta del bot",
                                               "texto": [txt] * len(sub), "pos": pos, "simbolo": sim, "color": color})
                        nv = mb.niveles(bot, par)
                        if nv["compra"]:
                            niveles.append((nv["compra"], "Compra", VERDE))
                        if nv["venta"]:
                            niveles += [(nv["venta"], "TP", AMARILLO), (nv["precio_compra"], "Entrada", "#5e6673")]
                        if nv["stop"]:
                            niveles.append((nv["stop"], "SL", ROJO))
                    fig = figura_velas(v, 470, marcadores=marcas, niveles=niveles)
                    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": True},
                                    key=f"graf_bot_{par}_{intervalo}")
                except (mb.FuenteNoDisponible, KeyError, ValueError) as e:
                    html(ui.aviso(f"No se pudo cargar la gráfica: {ui.esc(e)}", "warn"))
        with c_bot:
            with st.container(border=True):
                if bot is None:
                    html(ui.titulo_panel("Bot automático", "Detenido"))
                    html(ui.valor_grande("Capital de práctica", f"{mb.CAPITAL_MEXC:,.2f}", "USDT"))
                    html(ui.aviso("Configura los topes de compra y de venta y presiona <b>Iniciar bot</b>. "
                                  "Opera solo, con precios reales de MEXC."))
                    with st.popover("Configurar e iniciar"):
                        formulario_bot(est)
                else:
                    corriendo = st.session_state.mexc_corriendo
                    html(ui.titulo_panel("Bot automático", "● Operando" if corriendo else "En pausa", "up" if corriendo else ""))
                    html(ui.valor_grande("Equity del bot", f"{r['patrimonio']:,.2f}", "USDT",
                                         f"{r['ganancia_total']:+,.2f} ({r['rendimiento']:+.2%})", ui.tono(r["ganancia_total"])))
                    horas, minutos = divmod(int(r["minutos"]), 60)
                    html(ui.lista_kv([
                        ("Modo", "REAL" if bot.get("modo") == "real" else "Paper", "dn" if bot.get("modo") == "real" else "neu"),
                        ("Tope de compra", f"−{bot['tope_compra']:.1%}", "neu"),
                        ("Tope de venta", f"+{bot['tope_venta']:.1%}", "neu"),
                        ("Monto por orden", f"{bot['monto_operacion']:,.0f} USDT", "neu"),
                        ("Stop-loss", f"−{bot['stop_loss']:.1%}" if bot["stop_loss"] else "Apagado", "neu"),
                        ("Operaciones", f"{r['compras']} compras · {r['ventas']} ventas", "neu"),
                        ("Ventas con ganancia", f"{r['ganadoras'] / r['ventas']:.0%}" if r["ventas"] else "—", "up"),
                        ("Tiempo activo", f"{horas} h {minutos} min", "neu"),
                        ("Avisos WhatsApp", "Activos" if WA_CFG["listo"] else "Apagados", "up" if WA_CFG["listo"] else "neu"),
                    ]))
                    b1, b2 = st.columns(2)
                    if b1.button("Pausar" if corriendo else "Reanudar", key="btn_pausa"):
                        st.session_state.mexc_corriendo = not corriendo
                        st.rerun()
                    if b2.button("Detener", key="btn_detener",
                                 help="Vende todas las posiciones abiertas y pausa el bot."):
                        cerrar_posiciones(est)
                        st.rerun()
                    with st.popover("Configurar bot"):
                        formulario_bot(est)

        # ---------- pestañas inferiores ----------
        n_ab = len(mb.resumen(bot)["abiertas"]) if bot else 0
        t1, t2, t3, t4, t5 = st.tabs(["Últimas operaciones", f"Posiciones abiertas ({n_ab})", "Historial completo",
                                      "Rendimiento", "Mi cuenta MEXC"])
        with t1:
            if bot and bot["operaciones"]:
                tabla_ops(bot["operaciones"][-15:][::-1])
            else:
                html(ui.aviso("Todavía no hay operaciones en esta sesión del bot."))
        with t2:
            if bot and n_ab:
                ab = mb.resumen(bot)["abiertas"].rename(columns={
                    "par": "Par", "precio_compra": "Entrada", "precio_actual": "Precio actual", "objetivo_venta": "TP",
                    "invertido_usdt": "Invertido (USDT)", "valor_usdt": "Valor (USDT)", "resultado_%": "PNL %"})
                html(ui.tabla(ab, formatos={"Entrada": fp, "Precio actual": fp, "TP": fp, "Invertido (USDT)": "{:,.2f}",
                                            "Valor (USDT)": "{:,.2f}", "PNL %": "{:+.2f}%"}, colores=["PNL %"], negritas=["Par"]))
            else:
                html(ui.aviso("Sin posiciones abiertas: el bot espera que algún precio baje hasta su tope de compra."))
        with t3:
            if est["historial"]:
                hist = est["historial"][::-1]
                tabla_ops(hist, alto=440)
                d1, d2 = st.columns([1, 4])
                d1.download_button("Descargar CSV", pd.DataFrame(hist).to_csv(index=False).encode("utf-8"),
                                   file_name="movimientos_bot_mexc.csv", mime="text/csv")
                d2.caption(f"{len(hist)} movimientos de todas las sesiones del bot.")
            else:
                html(ui.aviso("El historial aparecerá aquí cuando el bot haga su primera operación."))
        with t4:
            if bot is not None:
                g1, g2 = st.columns([1.7, 1])
                curva = pd.DataFrame(bot["curva"], columns=["tiempo", "patrimonio"])
                curva["tiempo"] = pd.to_datetime(curva["tiempo"])
                fig = go.Figure(go.Scatter(x=curva.tiempo, y=curva.patrimonio, fill="tozeroy", name="Equity",
                                           line=dict(color=VERDE if r["rendimiento"] >= 0 else ROJO, width=2),
                                           fillcolor="rgba(14,203,129,0.08)" if r["rendimiento"] >= 0 else "rgba(246,70,93,0.08)"))
                fig.add_hline(y=r["capital"], line=dict(color="#5e6673", dash="dot"))
                fig.update_yaxes(range=[curva.patrimonio.min() * 0.998, curva.patrimonio.max() * 1.002])
                g1.plotly_chart(estilo_exchange(fig, 300), width="stretch", config={"displayModeBar": False})
                pp = r["por_par"]
                if len(pp):
                    fig = go.Figure(go.Bar(x=pp["par"], y=pp["ganancia_usdt"],
                                           marker_color=[VERDE if x >= 0 else ROJO for x in pp["ganancia_usdt"]]))
                    g2.plotly_chart(estilo_exchange(fig, 300), width="stretch", config={"displayModeBar": False})
                else:
                    with g2:
                        html(ui.aviso("El PNL por par aparece cuando el bot cierre su primera venta "
                                      f"(sube {bot['tope_venta']:.1%} desde la compra)."))
                st.caption("Izquierda: equity del bot en el tiempo. Derecha: PNL realizado por par (USDT).")
            else:
                html(ui.aviso("Inicia el bot para ver su rendimiento."))
        with t5:
            panel_cuenta_mexc()

    terminal_bot()
    with st.expander("Avisos por WhatsApp"):
        if WA_CFG["listo"]:
            html(ui.aviso(f"Avisos activos para el número terminado en <b>…{WA_CFG['telefono'][-4:]}</b>. Recibirás un WhatsApp "
                          "cada vez que un bot compre o venda (MEXC: Long / cerrar Long · Polymarket: Up ≈ Long, Down ≈ Short)."))
        else:
            motivo_wa = WA_CFG["problema"] or 'Pon activo = "1" en la sección [whatsapp].'
            html(ui.aviso(f"Avisos apagados. {ui.esc(motivo_wa)} Instrucciones en el README (CallMeBot, gratis).", "warn"))
        if st.button("Enviar WhatsApp de prueba", disabled=not WA_CFG["listo"]):
            ok, detalle = wa.enviar(WA_CFG, "✅ *Investify*: prueba de avisos por WhatsApp. ¡Funciona!")
            (st.success if ok else st.error)(detalle)
        if wa.ULTIMOS:
            ult = wa.ULTIMOS[-1]
            st.caption(f"Último envío: {'✅' if ult['ok'] else '❌'} {ult['detalle']}")
    with st.expander("Opciones avanzadas del bot"):
        st.caption("Borra el historial guardado de todas las sesiones del bot. No se puede deshacer.")
        if st.button("Borrar historial de movimientos"):
            EST["historial"] = []
            if EST["bot"] is not None:
                EST["bot"]["operaciones"] = []
            mb.guardar_estado(EST)
            st.rerun()

# ===========================================================================
# BOT POLYMARKET (mercados "Bitcoin / Ethereum Up or Down" por hora)
# ===========================================================================
POLY_CFG = pr.leer_config(RUTA_SECRETOS)
if "poly" not in st.session_state:
    st.session_state.poly = pb.cargar_estado()
    st.session_state.poly_corriendo = st.session_state.poly["bot"] is not None


@st.cache_data(ttl=30, show_spinner=False)
def mercado_poly(activo, inicio_iso):
    return pb.mercado_hora(activo, pd.Timestamp(inicio_iso))


@st.cache_data(ttl=4, show_spinner=False)
def libro_poly(token):
    return pb.libro(token)


@st.cache_data(ttl=15, show_spinner=False)
def apertura_hora(par):
    v, fuente = mb.obtener_velas(par, "60m", 3)
    return float(v["apertura"].iloc[-1]), fuente


@st.cache_resource(show_spinner=False)
def ejecutor_poly(private_key):
    return pr.EjecutorPolymarket(private_key)


def vela_respaldo(apuesta):
    """Apertura y cierre de la vela de 1 hora de MEXC para la hora de la apuesta (respaldo aproximado)."""
    par = pb.ACTIVOS[apuesta["activo"]]["par"]
    v, _ = mb.obtener_velas(par, "60m", 12)
    inicio = pd.Timestamp(apuesta["inicio_et"]).tz_convert(mb.ZONA).tz_localize(None)
    fila = v[v["tiempo"] == inicio]
    if fila.empty:
        raise KeyError("vela no encontrada")
    return float(fila["apertura"].iloc[0]), float(fila["cierre"].iloc[0])


def tabla_apuestas(apuestas, alto=None):
    df = pd.DataFrame(apuestas, columns=["fecha", "activo", "lado", "precio", "acciones", "monto", "prob_modelo",
                                         "ventaja", "estado", "pago", "ganancia", "modo", "resolucion"])
    t = pd.DataFrame({
        "Fecha": df["fecha"], "Mercado": df["activo"] + " · 1h", "Lado": df["lado"], "Precio (¢)": df["precio"] * 100,
        "Acciones": df["acciones"], "Monto (USD)": df["monto"], "Prob. IA": df["prob_modelo"] * 100,
        "Ventaja": df["ventaja"] * 100, "Estado": df["estado"].str.capitalize(), "Pago (USD)": df["pago"],
        "PNL (USD)": df["ganancia"], "Modo": df["modo"].str.capitalize(), "Resolución": df["resolucion"]})
    html(ui.tabla(t, formatos={"Precio (¢)": "{:.1f}", "Acciones": "{:,.2f}", "Monto (USD)": "{:,.2f}",
                               "Prob. IA": "{:.1f}%", "Ventaja": "{:+.1f} pts", "Pago (USD)": "{:,.2f}",
                               "PNL (USD)": "{:+,.2f}"},
                  insignias={"Lado": {"Up": "buy", "Down": "sell"},
                             "Estado": {"Ganada": "buy", "Perdida": "sell", "Abierta": "ok", "Rechazada": "bad",
                                        "Sombra": "neu", "Sombra resuelta": "neu"},
                             "Modo": {"Real": "real", "Paper": "neu", "Sombra": "neu"}},
                  colores=["PNL (USD)"], negritas=["Mercado"], alto=alto))


def formulario_poly(est):
    horas_sombra = pb.horas_de_sombra(est)
    with st.form("form_poly", border=False):
        activos_sel = st.multiselect("Mercados", list(pb.ACTIVOS), default=list(pb.ACTIVOS),
                                     format_func=lambda a: f"{pb.ACTIVOS[a]['nombre']} Up or Down (1 h)")
        modo = st.radio("Modo", ["paper", "sombra", "real"], horizontal=True,
                        format_func=lambda m: {"paper": "Paper", "sombra": "Sombra (calcula, no envía)", "real": "Real"}[m])
        c1, c2 = st.columns(2)
        capital = c1.number_input("Capital paper (USD)", min_value=100.0, value=pb.CAPITAL_PAPER, step=500.0)
        monto = c2.number_input("Monto por apuesta (USD)", min_value=1.0, value=100.0, step=10.0,
                                help="En modo sombra y real se usa «USD por apuesta (real)».")
        c3, c4 = st.columns(2)
        margen = c3.slider("Ventaja mínima para apostar", 0.5, 15.0, 3.0, 0.5, format="%.1f pts",
                           help="Diferencia mínima entre la probabilidad de la IA y el precio del mercado.")
        vel = c4.select_slider("Actualizar cada", options=[5, 10, 30, 60], value=10, format_func=lambda x: f"{x} s")
        st.markdown("**Sombra y real** · USDC de tu billetera de Polymarket")
        e1, e2 = st.columns(2)
        monto_real = e1.number_input("USD por apuesta (real)", min_value=1.0, max_value=100.0, value=5.0, step=1.0)
        tope_real = e2.number_input("Tope total (USD)", min_value=5.0, max_value=1_000.0, value=50.0, step=5.0)
        confirmacion = st.text_input(f"Para el modo real escribe: {pr.FRASE_CONFIRMACION}")
        enviar = st.form_submit_button("Iniciar bot" if est["bot"] is None else "Reiniciar bot con esta configuración",
                                       type="primary")
    if enviar:
        try:
            if modo in ("sombra", "real") and not POLY_CFG["private_key"]:
                raise ValueError("Para sombra o real necesitas tu llave privada en .streamlit/secrets.toml ([polymarket]).")
            if modo == "real":
                ok, faltan = pr.llaves_modo_real(POLY_CFG, confirmacion, horas_sombra)
                if not ok:
                    raise ValueError("Modo real bloqueado. Falta: " + "; ".join(faltan) + ".")
            est["bot"] = pb.bot_nuevo(activos_sel, monto_real if modo != "paper" else monto, margen / 100,
                                      capital=capital, modo=modo, tope_total=tope_real)
            est["bot"]["intervalo_s"] = vel
            st.session_state.poly_corriendo = True
            pb.guardar_estado(est)
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    falta_real = pr.llaves_modo_real(POLY_CFG, pr.FRASE_CONFIRMACION, horas_sombra)[1]
    st.caption("Llaves del modo real: "
               + ("✅ llave privada" if POLY_CFG["private_key"] else "❌ llave privada") + " · "
               + ("✅ modo_real=1" if POLY_CFG["modo_real"] else "❌ modo_real=1") + " · "
               + (f"✅ sombra {horas_sombra:.1f} h" if horas_sombra >= pr.HORAS_SOMBRA_MIN
                  else f"❌ sombra {horas_sombra:.1f} de {pr.HORAS_SOMBRA_MIN} h") + " · frase de confirmación al iniciar")


with tab_poly:
    intervalo_p = (st.session_state.poly["bot"] or {}).get("intervalo_s", 10)

    @st.fragment(run_every=intervalo_p if (st.session_state.poly["bot"] is not None and st.session_state.poly_corriendo) else None)
    def terminal_poly():
        est = st.session_state.poly
        bot = est["bot"]
        activos = bot["activos"] if bot else list(pb.ACTIVOS)
        inicio = pb.inicio_hora_et()

        # ---------- datos en vivo ----------
        entradas, errores = {}, []
        for a in activos:
            info = pb.ACTIVOS[a]
            try:
                m = mercado_poly(a, inicio.isoformat())
            except pb.PolymarketNoDisponible as e:
                m = None
                errores.append(f"{a}: {e}")
            try:
                precio_act = precios_bot((info["par"],))[0][info["par"]]
                apertura, _ = apertura_hora(info["par"])
            except (mb.FuenteNoDisponible, KeyError, ValueError) as e:
                precio_act, apertura = np.nan, np.nan
                errores.append(f"{a}: sin precio ({e})")
            p_ia = pronosticos.get(info["yahoo"]) or {}
            vol = (nc.calcular_indicadores(precios[info["yahoo"]])["vol20"].iloc[-1]
                   if info["yahoo"] in precios else 0.03)
            entradas[a] = {"mercado": m, "precio_actual": precio_act, "apertura": apertura,
                           "rend_ia": p_ia.get("estimado_dl", 0.0), "vol_diaria": float(vol)}

        if bot is not None and st.session_state.poly_corriendo:
            ejecutor = ejecutor_poly(POLY_CFG["private_key"]) if bot["modo"] in ("sombra", "real") else None
            if bot["modo"] == "real" and not POLY_CFG["modo_real"]:
                st.session_state.poly_corriendo = False
                html(ui.aviso("Modo real en pausa: falta <code>modo_real = \"1\"</code> en secrets.toml.", "real"))
            else:
                validas = {k: v for k, v in entradas.items() if not np.isnan(v["precio_actual"]) and not np.isnan(v["apertura"])}
                nuevas_ap = pb.ciclo(est, validas, ejecutor=ejecutor, obtener_libro=libro_poly)
                resueltas = pb.resolver(est, vela_respaldo=vela_respaldo)
                avisos = ([wa.mensaje_polymarket(x) for x in nuevas_ap if x["estado"] != "rechazada"]
                          + [wa.mensaje_resolucion(x) for x in resueltas])
                if avisos and WA_CFG["listo"]:
                    wa.enviar_en_segundo_plano(WA_CFG, avisos)
                for ap in nuevas_ap:
                    st.toast(f"{'Apuesta rechazada' if ap['estado'] == 'rechazada' else 'Apuesta ' + ap['estado']}: "
                             f"{ap['activo']} {ap['lado']} · {ap['monto']:.2f} USD a {ap['precio'] * 100:.1f}¢")
                for ap in resueltas:
                    st.toast(f"{ap['activo']} {ap['lado']}: {ap['estado']} ({ap['ganancia']:+.2f} USD)")
                pb.guardar_estado(est)
        for e in errores:
            html(ui.aviso(ui.esc(e), "warn"))

        # ---------- franja ----------
        if bot is not None:
            r = pb.resumen(bot)
            etq_modo = {"paper": "modo paper", "sombra": "modo sombra (no envía órdenes)", "real": "MODO REAL"}[bot["modo"]]
            html(ui.franja([
                ("Capital", f"{r['capital']:,.2f} USD", "neu", etq_modo),
                ("Equity", f"{r['patrimonio']:,.2f} USD", "neu", f"{r['en_juego']:,.2f} USD en juego"),
                ("PNL realizado", f"{r['pnl']:+,.2f} USD", ui.tono(r["pnl"]), "apuestas resueltas"),
                ("Ganadas · perdidas", f"{r['ganadas']} · {r['perdidas']}", "neu", f"{r['abiertas']} abiertas"),
                ("Acierto", "—" if np.isnan(r["win_rate"]) else f"{r['win_rate']:.0%}", "neu", "de las apuestas resueltas"),
                ("Rendimiento", fpct(r["rendimiento"]), ui.tono(r["rendimiento"]),
                 f"sombra: {r['sombra']} calculadas" if bot["modo"] == "sombra" else ""),
            ]))

        # ---------- tarjetas de mercado + panel del bot ----------
        cols = st.columns([1.5] * len(activos) + [1.1], gap="small")
        for col, a in zip(cols, activos):
            info, e = pb.ACTIVOS[a], entradas[a]
            m = e["mercado"]
            with col.container(border=True):
                if m is None:
                    html(ui.titulo_panel(f"{info['nombre']} · Up or Down", "1 h"))
                    html(ui.aviso("No encontré el mercado de esta hora en Polymarket. Se vuelve a buscar en unos segundos."))
                    continue
                dec = (bot or {}).get("ultima_decision", {}).get(a, {})
                minutos = (pd.Timestamp(m["fin"]) - pb.ahora_et()).total_seconds() / 60
                html(ui.titulo_panel(f"{info['nombre']} · Up or Down", f"cierra en {max(minutos, 0):.0f} min"))
                st.markdown(f'<div style="color:#b7bdc6;font-size:.82rem;margin-bottom:6px">{ui.esc(m["pregunta"])}</div>',
                            unsafe_allow_html=True)
                cambio_h = e["precio_actual"] / e["apertura"] - 1 if e["apertura"] and not np.isnan(e["apertura"]) else np.nan
                html(ui.lista_kv([("Precio a superar (apertura)", fp(e["apertura"]), "neu"),
                                  ("Precio actual", fp(e["precio_actual"]), ui.tono(cambio_h)),
                                  ("Cambio en la hora", fpct(cambio_h, 3), ui.tono(cambio_h)),
                                  ("Pronóstico diario IA", fpct(e["rend_ia"], 3), ui.tono(e["rend_ia"]))]))
                if dec.get("p_up") is not None and not pd.isna(dec.get("p_up")):
                    p_up, ask_up, ask_dn = dec["p_up"], dec["ask_up"], dec["ask_down"]
                else:
                    fr = min(max(minutos / 60, 0), 1)
                    p_up = pb.probabilidad_up(e["precio_actual"], e["apertura"], fr, e["rend_ia"], e["vol_diaria"]) \
                        if not np.isnan(e["apertura"]) else np.nan
                    ask_up, ask_dn = m["precio_up"], m["precio_down"]
                html(ui.barra_prob("IA · probabilidad de Up", p_up, TURQUESA))
                html(ui.barra_prob("Mercado · precio de Up", ask_up, VERDE, f"Up {ask_up * 100:.1f}¢" if not pd.isna(ask_up) else ""))
                html(ui.barra_prob("Mercado · precio de Down", ask_dn, ROJO, f"Down {ask_dn * 100:.1f}¢" if not pd.isna(ask_dn) else ""))
                lado = dec.get("lado")
                html(ui.etiqueta(f"▲ APOSTAR UP · ventaja {dec['ventaja'] * 100:+.1f} pts" if lado == "Up" else
                                 f"▼ APOSTAR DOWN · ventaja {dec['ventaja'] * 100:+.1f} pts" if lado == "Down" else
                                 "ESPERAR · sin ventaja suficiente", "buy" if lado == "Up" else "sell" if lado == "Down" else "neu"))
                if bot is not None:
                    ap = next((x for x in bot["apuestas"] if x["slug"] == m["slug"] and x["estado"] != "rechazada"), None)
                    if ap:
                        html(ui.aviso(f"Apuesta {ui.esc(ap['estado'])}: <b>{ap['lado']}</b> · {ap['acciones']:,.2f} acciones a "
                                      f"{ap['precio'] * 100:.1f}¢ · paga <b>{ap['acciones']:,.2f} USD</b> si gana"))
        with cols[-1].container(border=True):
            if bot is None:
                html(ui.titulo_panel("Bot Polymarket", "Detenido"))
                html(ui.valor_grande("Capital paper", f"{pb.CAPITAL_PAPER:,.2f}", "USD"))
                html(ui.aviso("Apuesta en los mercados «Up or Down» de cada hora cuando la probabilidad de tu IA supera el "
                              "precio del mercado por la ventaja mínima. Opera solo."))
                with st.popover("Configurar e iniciar"):
                    formulario_poly(est)
            else:
                corriendo = st.session_state.poly_corriendo
                html(ui.titulo_panel("Bot Polymarket", "● Operando" if corriendo else "En pausa", "up" if corriendo else ""))
                html(ui.valor_grande("Equity", f"{r['patrimonio']:,.2f}", "USD", f"{r['pnl']:+,.2f} USD realizado", ui.tono(r["pnl"])))
                html(ui.lista_kv([("Modo", bot["modo"].upper() if bot["modo"] == "real" else bot["modo"].capitalize(),
                                   "dn" if bot["modo"] == "real" else "neu"),
                                  ("Monto por apuesta", f"{bot['monto_apuesta']:,.2f} USD", "neu"),
                                  ("Ventaja mínima", f"{bot['margen'] * 100:.1f} pts", "neu"),
                                  ("Mercados", ", ".join(bot["activos"]), "neu"),
                                  ("Hora del mercado (ET)", f"{inicio:%H:%M} – {(inicio + pd.Timedelta(hours=1)):%H:%M}", "neu")]))
                b1, b2 = st.columns(2)
                if b1.button("Pausar" if corriendo else "Reanudar", key="btn_pausa_poly"):
                    st.session_state.poly_corriendo = not corriendo
                    st.rerun()
                if b2.button("Apagar", key="btn_apagar_poly", help="Detiene el bot al instante. Las apuestas abiertas "
                                                                     "se resuelven solas al terminar su hora."):
                    st.session_state.poly_corriendo = False
                    st.rerun()
                with st.popover("Configurar bot"):
                    formulario_poly(est)

        # ---------- pestañas inferiores ----------
        n_ab = sum(1 for x in (bot or {}).get("apuestas", []) if x["estado"] == "abierta")
        p1, p2, p3, p4 = st.tabs([f"Apuestas abiertas ({n_ab})", "Historial completo", "Rendimiento", "Cómo decide"])
        with p1:
            ab = [x for x in (bot or {}).get("apuestas", []) if x["estado"] in ("abierta", "sombra")]
            tabla_apuestas(ab[::-1]) if ab else html(ui.aviso("Sin apuestas abiertas."))
        with p2:
            if est["historial"]:
                tabla_apuestas(est["historial"][::-1], alto=440)
                st.download_button("Descargar CSV", pd.DataFrame(est["historial"]).to_csv(index=False).encode("utf-8"),
                                   file_name="apuestas_polymarket.csv", mime="text/csv")
            else:
                html(ui.aviso("El historial aparecerá cuando el bot haga su primera apuesta."))
        with p3:
            if bot is not None:
                curva = pd.DataFrame(bot["curva"], columns=["tiempo", "equity"])
                curva["tiempo"] = pd.to_datetime(curva["tiempo"])
                fig = go.Figure(go.Scatter(x=curva.tiempo, y=curva.equity, fill="tozeroy", line=dict(color=VERDE, width=2),
                                           fillcolor="rgba(14,203,129,0.08)", name="Equity"))
                fig.add_hline(y=bot["capital"], line=dict(color="#5e6673", dash="dot"))
                fig.update_yaxes(range=[curva.equity.min() * 0.995, curva.equity.max() * 1.005])
                st.plotly_chart(estilo_exchange(fig, 300), width="stretch", config={"displayModeBar": False})
            else:
                html(ui.aviso("Inicia el bot para ver su rendimiento."))
        with p4:
            st.markdown("""
**1 · Mercado.** Cada hora Polymarket abre «Bitcoin Up or Down» y «Ethereum Up or Down»: gana **Up** si la vela de 1 hora
(BTC/USDT o ETH/USDT de Binance) cierra igual o arriba de su apertura. Cada acción ganadora paga **1 USD**.

**2 · Probabilidad de la IA.** El pronóstico **diario** de tu MLP/LSTM se convierte en deriva por hora (÷ 24) y la volatilidad
diaria de 20 días en volatilidad por hora (÷ √24). Con el precio actual y el tiempo que falta se calcula
P(cierre ≥ apertura) con una distribución normal.

**3 · Decisión.** Ventaja = probabilidad de la IA − precio del lado en el libro. Si la mayor ventaja supera el mínimo, compra ese
lado con una orden de mercado; si no, espera. Una apuesta por mercado y nunca en los últimos 5 minutos.

**4 · Resolución.** Se usa el resultado oficial de Polymarket; si tarda más de 20 minutos, la vela de 1 hora de MEXC como
respaldo aproximado (Polymarket resuelve con Binance).

**Modos:** *Paper* (simulado con el libro real) · *Sombra* (calcula y valida cada orden con el SDK oficial sin enviarla) ·
*Real* (requiere llave privada + `modo_real = "1"` + 24 h de sombra + frase de confirmación). Las cifras de cada modo
no se mezclan.
""")

    terminal_poly()

# ===========================================================================
# RESULTADOS (evaluación final en TE)
# ===========================================================================
with tab_resultados:
    res_te, sim = resultado("resultados_trading_TE.csv"), resultado("simulacion.csv")
    if res_te is None:
        html(ui.aviso("Copia los archivos de resultados del Notebook 5 a la carpeta <code>resultados/</code> para ver esta sección."))
    else:
        html(ui.seccion("Evaluación final en prueba (TE)", "Periodo nunca usado para entrenar ni para elegir · capital de "
                                                            "100,000 · comisión de 0.10% · ejecución en la apertura de t+1"))
        rk = st.segmented_control("Mercado", ["Todos"] + MERCADOS, default="Todos", key="res_mercado",
                                  label_visibility="collapsed") or "Todos"
        piv = res_te.pivot(index="ticker", columns="estrategia", values="rendimiento_neto")
        piv["mercado"] = INFO["mercado"]
        if rk != "Todos":
            piv = piv[piv.mercado == rk]
        n = len(piv)
        html(ui.franja([
            ("Rendimiento promedio · IA", fpct(piv["Sistema DL final"].mean()), ui.tono(piv["Sistema DL final"].mean()), "sistema DL en TE"),
            ("IA vs. estrategia lineal", f"{(piv['Sistema DL final'] > piv['Estrategia lineal']).sum()} de {n}", "neu", "activos donde gana la IA"),
            ("IA vs. comprar y mantener", f"{(piv['Sistema DL final'] > piv['Comprar y mantener']).sum()} de {n}", "neu", "activos donde gana la IA"),
            ("Meta +20%", f"{(piv['Sistema DL final'] >= 0.20).sum()} de {n}", "neu", "activos que la alcanzan"),
        ]))
        orden = piv.sort_values("Sistema DL final", ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=orden.index, x=orden["Comprar y mantener"] * 100, name="Comprar y mantener", orientation="h",
                             marker_color="#2b3139"))
        fig.add_trace(go.Bar(y=orden.index, x=orden["Sistema DL final"] * 100, name="Sistema IA", orientation="h",
                             marker_color=VERDE))
        fig.add_vline(x=20, line=dict(color=AMARILLO, dash="dot"), annotation_text="meta +20%")
        fig.update_layout(barmode="group")
        fig = estilo_exchange(fig, max(420, 22 * n))
        fig.update_yaxes(side="left")
        with st.container(border=True):
            html(ui.titulo_panel("Rendimiento neto en TE por activo (%)", ""))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        if sim is not None:
            t_res = st.selectbox("Ver detalle de", sorted(set(sim.ticker)), key="res_activo")
            s = sim[sim.ticker == t_res].copy()
            s["fecha"] = pd.to_datetime(s["fecha"])
            fila = res_te[(res_te.ticker == t_res) & (res_te.estrategia == "Sistema DL final")].iloc[0]
            html(ui.franja([
                ("Rendimiento neto", fpct(fila["rendimiento_neto"]), ui.tono(fila["rendimiento_neto"]), "después de comisiones"),
                ("Máximo drawdown", fpct(fila["max_drawdown"]), "dn", "peor caída desde un máximo"),
                ("Operaciones", f"{int(fila['operaciones'])}", "neu", "cerradas"),
                ("Win rate", "—" if pd.isna(fila["win_rate"]) else f"{fila['win_rate']:.0%}", "neu", "operaciones con ganancia"),
            ]))
            fig = go.Figure(go.Scatter(x=s.fecha, y=s.patrimonio, name="Sistema IA", line=dict(color=VERDE, width=2)))
            if t_res in precios:
                d = precios[t_res].reindex(s.fecha)
                if d["cierre"].notna().all():
                    u_bh = nc.CAPITAL_INICIAL / (s["precio_ejecucion"].iloc[0] * (1 + nc.COSTO))
                    fig.add_trace(go.Scatter(x=s.fecha, y=u_bh * d["cierre"].to_numpy(), name="Comprar y mantener",
                                             line=dict(color="#5e6673", width=1.5)))
            fig.add_hline(y=nc.CAPITAL_INICIAL, line=dict(color="#5e6673", dash="dot"))
            with st.container(border=True):
                html(ui.titulo_panel(f"Patrimonio en TE · {t_res}", ""))
                st.plotly_chart(estilo_exchange(fig, 340), width="stretch", config={"displayModeBar": False})
        with st.expander("Tabla completa de estrategias en TE"):
            st.dataframe(res_te, hide_index=True, width="stretch")

# ===========================================================================
# MODELO
# ===========================================================================
with tab_modelo:
    html(ui.seccion("Cómo funciona Investify", "Del dato al pronóstico y a la señal"))
    pasos = [("1 · Datos", "Precios diarios ajustados de Yahoo Finance para 40 activos. Entradas: los 20 rendimientos más "
                           "recientes y, en la versión B, distancia a la SMA 20, volatilidad 20 y RSI 14."),
             ("2 · Modelos", "MLP y LSTM entrenados en GPU T4 con separación temporal 70/15/15 (TR/VA/TE), parada temprana "
                             "y 3 semillas. Se comparan con el pronóstico cero y la regresión lineal."),
             ("3 · Señal", "Comprar si el rendimiento estimado supera un umbral δ elegido en validación (con o sin filtro "
                           "SMA 20); si no, permanecer en efectivo.")]
    for col, (tit, txt) in zip(st.columns(3), pasos):
        with col.container(border=True):
            html(ui.titulo_panel(tit, ""))
            st.markdown(f'<p style="color:#b7bdc6;font-size:.86rem;line-height:1.5;margin:0">{txt}</p>', unsafe_allow_html=True)
    met = resultado("metricas_TR_VA_TE.csv")
    if met is not None:
        t_mod = st.selectbox("Activo", sorted(set(met.ticker)), key="mod_activo")
        m = met[met.ticker == t_mod].pivot_table(index="modelo", columns="conjunto", values=["MAE", "RMSE", "precision_direccional"])
        m.columns = [f"{a} {b}".replace("precision_direccional", "Prec. dir.") for a, b in m.columns]
        m = m.reset_index().rename(columns={"modelo": "Modelo"})
        html(ui.titulo_panel(f"Métricas de {t_mod} (escala original del rendimiento)", ""))
        html(ui.tabla(m, formatos={c: "{:.5f}" for c in m.columns if c != "Modelo"}, negritas=["Modelo"]))
        p = pronosticos.get(t_mod)
        if p and p.get("red"):
            html(ui.lista_kv([("Red", p["red"], "neu"), ("Neuronas", f"{p.get('neuronas', '')}", "neu"),
                              ("Versión", p["version"], "neu"), ("Regla", p["regla"], "neu"), ("Umbral δ", f"{p['umbral']:.4%}", "neu")]))

html('<div class="x-foot">Investify · Proyecto final de Deep Learning para Finanzas · Datos: Yahoo Finance y MEXC · '
     'Las señales no son asesoría financiera</div>')
