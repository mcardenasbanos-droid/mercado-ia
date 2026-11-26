"""
ui.py — Sistema visual de Investify con estilo de exchange profesional (tipo MEXC / Binance).
Todos los componentes devuelven HTML para st.markdown(..., unsafe_allow_html=True).
"""
import html
import math

import pandas as pd

# Paleta de exchange
FONDO, PANEL, BORDE = "#0b0e11", "#0f1318", "#1e2329"
TEXTO, TENUE, TENUE2 = "#eaecef", "#848e9c", "#5e6673"
VERDE, ROJO, AMARILLO = "#0ecb81", "#f6465d", "#f0b90b"
INDIGO, TURQUESA = "#6366f1", "#14b8a6"

LOGO_SVG = """<svg width="{s}" height="{s}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="lg{s}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6366f1"/>
<stop offset="1" stop-color="#14b8a6"/></linearGradient></defs><rect width="64" height="64" rx="16" fill="url(#lg{s})"/>
<polyline points="14,44 25,33 34,39 50,21" fill="none" stroke="#fff" stroke-width="5.5" stroke-linecap="round"
stroke-linejoin="round"/><polygon points="53,17 41,19 51,29" fill="#fff"/></svg>"""


def _vacio(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def tono(v):
    if _vacio(v):
        return "neu"
    return "up" if v > 0 else ("dn" if v < 0 else "neu")


def esc(x):
    return html.escape(str(x))


# ---------------------------------------------------------------------------
# Encabezado, cinta y franja de indicadores
# ---------------------------------------------------------------------------
def encabezado(pastillas, etiqueta_equity, valor_equity):
    """pastillas: lista de (texto, tipo) con tipo en {'live','paper','real','warn','info'}."""
    p = "".join(f'<span class="x-pill {t}">{esc(txt)}</span>' for txt, t in pastillas)
    return (f'<div class="x-top"><div class="x-brand">{LOGO_SVG.format(s=30)}<span>Invest<i>ify</i></span>'
            f'<em>Trading con IA</em></div><div class="x-right">{p}<div class="x-eq"><small>{esc(etiqueta_equity)}</small>'
            f'<b>{valor_equity}</b></div></div></div>')


def cinta(items):
    """items: lista de (símbolo, precio_formateado, cambio_fracción)."""
    partes = "".join(f'<span><b>{esc(s)}</b>{p} <i class="{tono(c)}">{"" if _vacio(c) else f"{c:+.2%}"}</i></span>'
                     for s, p, c in items)
    return f'<div class="x-tape"><div class="x-tape-in">{partes}{partes}</div></div>'


def franja(items):
    """items: lista de (etiqueta, valor_html, tono 'up'/'dn'/'neu', subtítulo opcional)."""
    celdas = []
    for it in items:
        etq, val, t = it[0], it[1], it[2]
        sub = f'<em>{esc(it[3])}</em>' if len(it) > 3 and it[3] else ""
        celdas.append(f'<div class="x-k"><small>{esc(etq)}</small><b class="{t}">{val}</b>{sub}</div>')
    return f'<div class="x-strip" style="grid-template-columns:repeat({len(items)},minmax(0,1fr))">{"".join(celdas)}</div>'


def barra_par(simbolo, sufijo, precio, tono_precio, stats):
    """Barra del par como en los exchanges. stats: lista de (etiqueta, valor_html, tono)."""
    s = "".join(f'<div class="x-st"><small>{esc(e)}</small><b class="{t}">{v}</b></div>' for e, v, t in stats)
    return (f'<div class="x-pair"><div class="x-sym">{esc(simbolo)}<span>{esc(sufijo)}</span></div>'
            f'<div class="x-last {tono_precio}">{precio}</div>{s}</div>')


def titulo_panel(izq, der="", tono_der=""):
    return f'<div class="x-ph"><span>{esc(izq)}</span><span class="{tono_der}">{der}</span></div>'


def seccion(titulo, subtitulo=""):
    sub = f"<p>{esc(subtitulo)}</p>" if subtitulo else ""
    return f'<div class="x-sec"><h2>{titulo}</h2>{sub}</div>'


def lista_kv(pares):
    """pares: lista de (etiqueta, valor_html, tono)."""
    return '<div class="x-kv">' + "".join(
        f'<div><span>{esc(k)}</span><b class="{t}">{v}</b></div>' for k, v, t in pares) + "</div>"


def valor_grande(etiqueta, valor, unidad="", cambio_txt="", tono_cambio="neu"):
    u = f' <span class="x-unit">{esc(unidad)}</span>' if unidad else ""
    c = f'<div class="x-delta {tono_cambio}">{cambio_txt}</div>' if cambio_txt else ""
    return f'<div class="x-bigwrap"><small>{esc(etiqueta)}</small><div class="x-big">{valor}{u}</div>{c}</div>'


def barra_prob(etiqueta_txt, valor, color, detalle=""):
    """Barra horizontal de probabilidad (0 a 1)."""
    pct = 0 if _vacio(valor) else max(0.0, min(1.0, float(valor))) * 100
    txt = "—" if _vacio(valor) else f"{pct:.1f}%"
    return (f'<div class="x-prob"><div class="x-prob-top"><span>{esc(etiqueta_txt)}</span><b>{txt}</b></div>'
            f'<div class="x-prob-bar"><div style="width:{pct:.1f}%;background:{color}"></div></div>'
            f'{f"<em>{esc(detalle)}</em>" if detalle else ""}</div>')


def etiqueta(texto, tipo):
    return f'<span class="x-tag {tipo}">{esc(texto)}</span>'


def aviso(texto, tipo="info"):
    return f'<div class="x-note {tipo}">{texto}</div>'


# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------
def _fmt(v, fmt):
    if _vacio(v) or (isinstance(v, str) and v == ""):
        return '<span class="x-nil">—</span>'
    if callable(fmt):
        return fmt(v)
    try:
        return fmt.format(v)
    except (ValueError, TypeError):
        return esc(v)


def tabla(df, formatos=None, insignias=None, colores=None, alto=None, negritas=None):
    """Tabla HTML estilo exchange.

    formatos: {col: "{:,.2f}" | función}; insignias: {col: {valor: clase css}} (buy, sell, ok, bad, real, neu);
    colores: columnas numéricas verde/rojo; negritas: columnas en negritas."""
    formatos, insignias = formatos or {}, insignias or {}
    colores, negritas = set(colores or []), set(negritas or [])
    derecha = {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    cab = "".join(f'<th class="{"r" if c in derecha else ""}">{esc(c)}</th>' for c in df.columns)
    filas = []
    for _, fila in df.iterrows():
        celdas = []
        for c in df.columns:
            v = fila[c]
            if c in insignias and isinstance(v, str):
                contenido = f'<span class="x-b {insignias[c].get(v, "neu")}">{esc(v)}</span>'
            else:
                contenido = _fmt(v, formatos.get(c, "{}"))
            clases = ["r"] if c in derecha else []
            if c in colores and isinstance(v, (int, float)) and not _vacio(v):
                clases.append(tono(v))
            if c in negritas:
                clases.append("bold")
            celdas.append(f'<td class="{" ".join(clases)}">{contenido}</td>')
        filas.append(f'<tr>{"".join(celdas)}</tr>')
    if not filas:
        filas = [f'<tr><td colspan="{len(df.columns)}" class="x-empty">Sin datos todavía</td></tr>']
    estilo = f' style="max-height:{alto}px"' if alto else ""
    return (f'<div class="x-table"{estilo}><table><thead><tr>{cab}</tr></thead>'
            f'<tbody>{"".join(filas)}</tbody></table></div>')


# ---------------------------------------------------------------------------
# Tema completo
# ---------------------------------------------------------------------------
CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stMarkdown, p, li, label, h1, h2, h3, h4, input, textarea, button, [data-baseweb="tab"] {{
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;}}
[data-testid="stIconMaterial"] {{font-family: 'Material Symbols Rounded' !important;}}
.stApp {{background: {FONDO};}}
.block-container {{padding: 2.9rem 1.4rem 2.5rem 1.4rem; max-width: 1680px;}}
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stAppDeployButton"] {{display: none;}}
header[data-testid="stHeader"] {{background: transparent; height: 0;}}
[data-testid="stSidebar"] {{display: none;}}
[data-testid="stVerticalBlock"] {{gap: 0.65rem;}}
.up {{color: {VERDE} !important;}} .dn {{color: {ROJO} !important;}} .neu {{color: {TEXTO};}}

/* Encabezado */
.x-top {{display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 10px 2px 12px 2px;
    border-bottom: 1px solid {BORDE}; flex-wrap: wrap;}}
.x-brand {{display: flex; align-items: center; gap: 10px;}}
.x-brand span {{font-size: 1.3rem; font-weight: 800; letter-spacing: -0.03em; color: {TEXTO};}}
.x-brand i {{font-style: normal; background: linear-gradient(90deg, #818cf8, #2dd4bf); -webkit-background-clip: text;
    background-clip: text; color: transparent;}}
.x-brand em {{font-style: normal; font-size: 0.72rem; color: {TENUE}; border-left: 1px solid {BORDE}; padding-left: 10px;
    letter-spacing: .04em; text-transform: uppercase; font-weight: 600;}}
.x-right {{display: flex; align-items: center; gap: 10px; flex-wrap: wrap;}}
.x-pill {{padding: 6px 11px; border-radius: 6px; font-size: 0.74rem; font-weight: 700; background: {BORDE}; color: #b7bdc6;}}
.x-pill.live {{color: {VERDE};}}
.x-pill.live::before {{content: ""; display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: {VERDE};
    margin-right: 7px; box-shadow: 0 0 8px {VERDE}; animation: x-pulse 1.6s infinite;}}
@keyframes x-pulse {{0%, 100% {{opacity: 1;}} 50% {{opacity: .35;}}}}
.x-pill.paper {{color: {AMARILLO}; background: transparent; border: 1px solid rgba(240,185,11,.4);}}
.x-pill.real {{color: #fff; background: {ROJO};}}
.x-pill.warn {{color: #fcd34d; background: rgba(245,158,11,.12);}}
.x-eq {{text-align: right; line-height: 1.15; margin-left: 6px;}}
.x-eq small {{display: block; color: {TENUE}; font-size: .7rem;}} .x-eq b {{font-size: 1.02rem; color: {TEXTO};}}

/* Cinta */
.x-tape {{overflow: hidden; white-space: nowrap; border-bottom: 1px solid {BORDE}; padding: 7px 0; font-size: .78rem;
    color: #b7bdc6; mask-image: linear-gradient(90deg, transparent, #000 3%, #000 97%, transparent);}}
.x-tape-in {{display: inline-block; animation: x-slide 80s linear infinite;}}
.x-tape:hover .x-tape-in {{animation-play-state: paused;}}
.x-tape span {{margin-right: 28px;}} .x-tape b {{color: {TEXTO}; margin-right: 6px; font-weight: 700;}}
.x-tape i {{font-style: normal; font-weight: 600;}}
@keyframes x-slide {{from {{transform: translateX(0);}} to {{transform: translateX(-50%);}}}}

/* Navegación principal (pestañas) */
.stTabs [data-baseweb="tab-list"] {{gap: 26px; background: transparent; border-bottom: 1px solid {BORDE}; padding: 0 2px;}}
.stTabs [data-baseweb="tab"] {{height: 44px; padding: 0; background: transparent !important; color: {TENUE};
    font-weight: 600; font-size: .92rem;}}
.stTabs [data-baseweb="tab"]:hover {{color: {TEXTO};}}
.stTabs [aria-selected="true"] {{color: {TEXTO} !important;}}
.stTabs [data-baseweb="tab-highlight"] {{background: {AMARILLO} !important; height: 2px;}}
.stTabs [data-baseweb="tab-border"] {{display: none;}}
.stTabs [data-baseweb="tab-panel"] {{padding-top: 14px;}}

/* Franja de indicadores */
.x-strip {{display: grid; gap: 1px; background: {BORDE}; border: 1px solid {BORDE}; border-radius: 10px; overflow: hidden;}}
.x-k {{background: {PANEL}; padding: 16px 22px;}}
.x-k small {{display: block; color: {TENUE}; font-size: .85rem; margin-bottom: 6px; font-weight: 500;}}
.x-k b {{font-size: 1.6rem; font-weight: 700; color: {TEXTO}; font-variant-numeric: tabular-nums; letter-spacing: -.01em;}}
.x-k em {{display: block; font-style: normal; color: {TENUE2}; font-size: .82rem; margin-top: 4px;}}

/* Barra del par */
.x-pair {{display: flex; align-items: center; gap: 34px; padding: 12px 18px; background: {PANEL}; border: 1px solid {BORDE};
    border-radius: 10px; flex-wrap: wrap;}}
.x-sym {{font-size: 1.3rem; font-weight: 800; color: {TEXTO}; letter-spacing: -.02em;}}
.x-sym span {{font-size: .8rem; color: {TENUE}; font-weight: 600; margin-left: 7px;}}
.x-last {{font-size: 1.45rem; font-weight: 700; font-variant-numeric: tabular-nums;}}
.x-st small {{display: block; color: {TENUE}; font-size: .7rem; margin-bottom: 2px;}}
.x-st b {{font-weight: 600; font-size: .86rem; color: {TEXTO}; font-variant-numeric: tabular-nums;}}

/* Paneles */
.x-ph {{display: flex; justify-content: space-between; align-items: center; color: {TENUE}; font-size: .7rem;
    font-weight: 700; letter-spacing: .08em; text-transform: uppercase; margin: 2px 0 8px 0;}}
.x-sec {{margin: 4px 0 6px 0;}}
.x-sec h2 {{font-size: 1.2rem; font-weight: 800; letter-spacing: -.02em; color: {TEXTO}; margin: 0; padding: 0;}}
.x-sec p {{color: {TENUE}; font-size: .84rem; margin: 3px 0 0 0;}}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{border-color: {BORDE} !important;
    background: {PANEL}; border-radius: 10px;}}
.x-kv > div {{display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px dashed {BORDE};
    color: #b7bdc6; font-size: .84rem;}}
.x-kv b {{color: {TEXTO}; font-weight: 600; font-variant-numeric: tabular-nums;}}
.x-bigwrap small {{color: {TENUE}; font-size: .74rem;}}
.x-big {{font-size: 1.85rem; font-weight: 800; letter-spacing: -.03em; color: {TEXTO}; font-variant-numeric: tabular-nums;
    line-height: 1.15; margin: 2px 0;}}
.x-unit {{font-size: .8rem; color: {TENUE}; font-weight: 600;}}
.x-delta {{font-weight: 700; font-size: .86rem;}}
.x-tag {{display: inline-block; padding: 5px 10px; border-radius: 6px; font-size: .74rem; font-weight: 800; letter-spacing: .04em;}}
.x-tag.buy {{color: {VERDE}; background: rgba(14,203,129,.12);}} .x-tag.sell {{color: {ROJO}; background: rgba(246,70,93,.12);}}
.x-tag.neu {{color: #b7bdc6; background: {BORDE};}}
.x-note {{border-radius: 8px; padding: 10px 14px; font-size: .82rem; line-height: 1.45; border: 1px solid;}}
.x-note.info {{color: #b7bdc6; background: rgba(132,142,156,.07); border-color: {BORDE};}}
.x-note.warn {{color: #fcd34d; background: rgba(240,185,11,.07); border-color: rgba(240,185,11,.3);}}
.x-note.real {{color: #fecdd3; background: rgba(246,70,93,.1); border-color: rgba(246,70,93,.45);}}

/* Tablas */
.x-table {{overflow: auto; border: 1px solid {BORDE}; border-radius: 10px; background: {PANEL};}}
.x-table table {{width: 100%; border-collapse: collapse; font-size: .82rem; font-variant-numeric: tabular-nums;}}
.x-table th {{position: sticky; top: 0; z-index: 1; background: {PANEL}; color: {TENUE}; font-weight: 500; text-align: left;
    padding: 10px 14px; font-size: .74rem; border-bottom: 1px solid {BORDE}; white-space: nowrap;}}
.x-table td {{padding: 9px 14px; border-bottom: 1px solid #161a1e; color: {TEXTO}; white-space: nowrap;}}
.x-table tbody tr:hover td {{background: #151a20;}}
.x-table tbody tr:last-child td {{border-bottom: none;}}
.x-table .r {{text-align: right;}} .x-table td.bold {{font-weight: 700;}}
.x-table .x-nil {{color: {TENUE2};}} .x-table .x-empty {{text-align: center; color: {TENUE}; padding: 26px;}}
.x-b {{font-weight: 700;}} .x-b.buy {{color: {VERDE};}} .x-b.sell {{color: {ROJO};}}
.x-b.ok {{color: #b7bdc6; font-weight: 500;}} .x-b.bad {{color: {ROJO};}}
.x-b.real {{color: #fff; background: {ROJO}; padding: 2px 7px; border-radius: 4px; font-size: .72rem;}}
.x-b.neu {{color: {TENUE}; font-weight: 500;}}

/* Widgets de Streamlit */
.stButton button, [data-testid="stFormSubmitButton"] button, [data-testid="stDownloadButton"] button,
[data-testid="stPopover"] button {{border-radius: 6px; font-weight: 700; border: 1px solid {BORDE}; background: {BORDE};
    color: {TEXTO}; transition: filter .12s ease;}}
.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover, [data-testid="stPopover"] button:hover {{
    filter: brightness(1.18); border-color: #2b3139; color: #fff;}}
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {{
    background: {VERDE}; border-color: {VERDE}; color: #0b0e11;}}
.st-key-btn_detener button, .st-key-form_venta [data-testid="stFormSubmitButton"] button {{background: {ROJO} !important;
    border-color: {ROJO} !important; color: #fff !important;}}
[data-testid="stFormSubmitButton"] button, .st-key-btn_pausa button, .st-key-btn_detener button,
[data-testid="stPopover"] > div > button, [data-testid="stPopover"] button {{width: 100%;}}
.st-key-btn_pausa button {{background: {VERDE}; border-color: {VERDE}; color: #0b0e11;}}
[data-testid="stForm"] {{background: {PANEL}; border: 1px solid {BORDE}; border-radius: 10px; padding: 16px;}}
[data-testid="stExpander"] details {{border: 1px solid {BORDE}; border-radius: 10px; background: {PANEL};}}
[data-baseweb="select"] > div, [data-baseweb="input"] > div, .stNumberInput input, .stTextInput input {{
    background: #161a1e !important; border-color: {BORDE} !important; border-radius: 6px !important;}}
[data-testid="stDataFrame"] {{border: 1px solid {BORDE}; border-radius: 10px; overflow: hidden;}}
[data-testid="stWidgetLabel"] p {{color: {TENUE}; font-size: .78rem; font-weight: 600;}}
.x-prob {{margin: 8px 0;}}
.x-prob-top {{display: flex; justify-content: space-between; font-size: .8rem; color: #b7bdc6; margin-bottom: 4px;}}
.x-prob-top b {{color: {TEXTO}; font-variant-numeric: tabular-nums;}}
.x-prob-bar {{height: 8px; border-radius: 4px; background: #1e2329; overflow: hidden;}}
.x-prob-bar div {{height: 100%; border-radius: 4px;}}
.x-prob em {{font-style: normal; font-size: .7rem; color: {TENUE2};}}
.x-foot {{text-align: center; color: {TENUE2}; font-size: .74rem; margin-top: 28px; padding-top: 14px;
    border-top: 1px solid {BORDE};}}
"""
