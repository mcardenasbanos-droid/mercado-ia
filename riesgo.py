"""
riesgo.py — Matemática financiera del bot: indicadores de las velas, métricas de desempeño,
costos, SL/TP con ATR, tamaño de posición (riesgo fijo y Kelly fraccional), límites de riesgo
deterministas y bitácora inmutable con hash encadenado.

Todo son funciones puras (sin red ni Streamlit) para poder probarlas con números conocidos.
"""
import hashlib
import json
import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Costos (MEXC Spot; supuestos conservadores documentados en el README)
# ---------------------------------------------------------------------------
COMISION = 0.001          # 0.10% por lado (taker conservador; MEXC cobra menos en Spot)
DESLIZAMIENTO = 0.0005    # 0.05% por lado: la orden a mercado se llena peor que el último precio
COSTO_IDA_VUELTA = 2 * (COMISION + DESLIZAMIENTO)   # 0.30% por operación completa


# ---------------------------------------------------------------------------
# Indicadores sobre velas CERRADAS
# ---------------------------------------------------------------------------
def velas_cerradas(v):
    """Quita la última vela (todavía en formación): el bot decide solo con velas cerradas."""
    return v.iloc[:-1] if len(v) > 1 else v


def atr_wilder(v, n=14):
    """ATR de Wilder: media suavizada (alfa = 1/n) del rango verdadero."""
    alto, bajo, cierre = v["maximo"].astype(float), v["minimo"].astype(float), v["cierre"].astype(float)
    previo = cierre.shift(1)
    tr = pd.concat([alto - bajo, (alto - previo).abs(), (bajo - previo).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi_wilder(cierre, n=14):
    """RSI de Wilder (mismo método que TradingView)."""
    d = cierre.astype(float).diff()
    sube = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    baja = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = sube / baja.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).where(baja != 0, 100.0)


def macd_hist(cierre, rapida=12, lenta=26, senal=9):
    c = cierre.astype(float)
    macd = c.ewm(span=rapida, adjust=False).mean() - c.ewm(span=lenta, adjust=False).mean()
    return macd - macd.ewm(span=senal, adjust=False).mean()


def contexto_velas(v):
    """Resumen de las velas cerradas que usan los agentes y el riesgo. None si no hay datos suficientes."""
    if v is None or len(v) < 30:
        return None
    c = velas_cerradas(v)
    cierre = c["cierre"].astype(float)
    atr = atr_wilder(c).iloc[-1]
    rsi = rsi_wilder(cierre).iloc[-1]
    if not (np.isfinite(atr) and np.isfinite(rsi)):
        return None
    return {"atr": float(atr), "rsi": float(rsi), "macd_hist": float(macd_hist(cierre).iloc[-1]),
            "sma20": float(cierre.tail(20).mean()), "cierre": float(cierre.iloc[-1]),
            "vela": str(c.iloc[-1].get("tiempo", ""))}


# ---------------------------------------------------------------------------
# SL / TP con ATR y RR neto
# ---------------------------------------------------------------------------
def niveles_atr(precio, atr, k_sl=1.5, k_tp=3.0, costo_ida_vuelta=COSTO_IDA_VUELTA):
    """SL bajo el precio a k_sl·ATR (el «invalidador»: un movimiento normal no lo toca) y TP a k_tp·ATR.
    Devuelve también el RR neto: (ganancia en TP − costos) / (pérdida en SL + costos)."""
    sl = precio - k_sl * atr
    tp = precio + k_tp * atr
    costos = precio * costo_ida_vuelta
    ganancia, perdida = (tp - precio) - costos, (precio - sl) + costos
    rr = ganancia / perdida if perdida > 0 else 0.0
    return {"sl": sl, "tp": tp, "rr_neto": rr, "acierto_requerido": 1 / (1 + rr) if rr > 0 else 1.0}


# ---------------------------------------------------------------------------
# Tamaño de posición
# ---------------------------------------------------------------------------
def wilson(aciertos, n, z=1.96):
    """Intervalo de confianza de Wilson (95%) para la tasa de acierto."""
    if n <= 0:
        return 0.0, 1.0
    p = aciertos / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    margen = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centro - margen), min(1.0, centro + margen)


def kelly_fraccional(p, rr, fraccion=0.5):
    """Kelly f* = p − (1 − p)/RR, multiplicado por la fracción (≤ ½). Nunca negativo."""
    if rr <= 0:
        return 0.0
    return max(0.0, (p - (1 - p) / rr) * fraccion)


MIN_OPS_KELLY = 30         # antes de 30 ventas no hay estadística: se usa riesgo fijo


def tamano_posicion(equity, efectivo, precio, sl, riesgo_pct, monto_max, ventas=None, rr=None,
                    costo_ida_vuelta=COSTO_IDA_VUELTA):
    """USDT a invertir para que, si toca el SL, se pierda como máximo riesgo_pct del equity (con costos).
    Con ≥ 30 ventas usa Kelly fraccional (½) sobre la COTA INFERIOR de Wilson del acierto, con el
    mismo tope de riesgo. Topes finales: monto_max por orden y efectivo disponible."""
    riesgo = riesgo_pct
    regla = f"riesgo fijo {riesgo_pct:.2%}"
    if ventas is not None and len(ventas) >= MIN_OPS_KELLY and rr:
        gan = sum(1 for g in ventas if g > 0)
        p_bajo, _ = wilson(gan, len(ventas))
        k = kelly_fraccional(p_bajo, rr)
        riesgo = min(riesgo_pct, k)
        regla = f"Kelly ½ (p≥{p_bajo:.0%}) → {riesgo:.2%}"
    perdida_por_usdt = max((precio - sl) / precio, 0) + costo_ida_vuelta
    if perdida_por_usdt <= 0 or riesgo <= 0:
        return 0.0, regla + " · sin ventaja"
    monto = equity * riesgo / perdida_por_usdt
    monto = min(monto, monto_max, efectivo)
    return float(max(monto, 0.0)), regla


# ---------------------------------------------------------------------------
# Límites de riesgo deterministas (tienen veto sobre la IA)
# ---------------------------------------------------------------------------
LIMITES_DEFECTO = {"perdida_diaria_max": 0.02, "drawdown_max": 0.05, "max_posiciones": 4, "racha_max": 4}


def veto_riesgo(bot, equity, hoy):
    """Devuelve el motivo del veto (texto) o None. Se revisa ANTES de cada entrada."""
    lim = {**LIMITES_DEFECTO, **bot.get("limites", {})}
    if bot.get("kill"):
        return f"kill switch activo ({bot.get('kill_motivo', 'manual')}): rearma manualmente"
    abiertas = sum(1 for a in bot["activos"].values() if a["unidades"] > 0)
    if abiertas >= lim["max_posiciones"]:
        return f"máximo de {lim['max_posiciones']} posiciones abiertas"
    ventas = [o for o in bot["operaciones"] if o["tipo"] == "venta" and o.get("estado", "Ejecutada") == "Ejecutada"]
    del_dia = sum(o["ganancia_usdt"] for o in ventas if str(o["fecha"]).startswith(hoy))
    if del_dia <= -lim["perdida_diaria_max"] * bot["capital"]:
        return f"pérdida del día {del_dia:,.2f} USDT ≥ {lim['perdida_diaria_max']:.0%} del capital"
    pico = max([bot["capital"]] + [p for _, p in bot.get("curva", [])])
    if pico > 0 and equity / pico - 1 <= -lim["drawdown_max"]:
        return f"drawdown {equity / pico - 1:.2%} supera el máximo de {lim['drawdown_max']:.0%}"
    racha = 0
    for o in reversed(ventas):
        if o["ganancia_usdt"] < 0:
            racha += 1
        else:
            break
    if racha >= lim["racha_max"]:
        return f"racha de {racha} pérdidas seguidas"
    return None


def debe_activar_kill(motivo):
    """Los límites de pérdida activan el kill switch (se queda apagado hasta que la persona lo rearme)."""
    return motivo is not None and (motivo.startswith("pérdida del día") or motivo.startswith("drawdown")
                                   or motivo.startswith("racha"))


# ---------------------------------------------------------------------------
# Métricas de desempeño
# ---------------------------------------------------------------------------
def max_drawdown(curva):
    s = pd.Series([float(x) for x in curva], dtype=float)
    if len(s) < 2:
        return 0.0
    return float((s / s.cummax() - 1).min())


def metricas(ventas_pnl, ventas_pct, curva_equity, capital):
    """ventas_pnl: ganancias netas (USDT) de cada venta, con costos. ventas_pct: lo mismo en fracción."""
    n = len(ventas_pnl)
    g = [x for x in ventas_pnl if x > 0]
    p = [x for x in ventas_pnl if x <= 0]
    gan_bajo, gan_alto = wilson(len(g), n)
    prom_g = float(np.mean(g)) if g else 0.0
    prom_p = float(-np.mean(p)) if p else 0.0
    rr = prom_g / prom_p if prom_p > 0 else (float("inf") if prom_g > 0 else 0.0)
    r = np.array(ventas_pct, dtype=float)
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(n)) if n >= 3 and r.std(ddof=1) > 0 else float("nan")
    return {
        "operaciones": n,
        "win_rate": len(g) / n if n else float("nan"),
        "wilson_bajo": gan_bajo, "wilson_alto": gan_alto,
        "expectativa": float(np.mean(ventas_pnl)) if n else float("nan"),
        "profit_factor": (sum(g) / -sum(p)) if p and sum(p) < 0 else (float("inf") if g else float("nan")),
        "rr_neto": rr,
        "acierto_equilibrio": (1 / (1 + rr)) if rr not in (0.0, float("inf")) and rr > 0 else float("nan"),
        "max_drawdown": max_drawdown(curva_equity),
        "sharpe_operaciones": sharpe,
        "pnl_total": float(sum(ventas_pnl)),
        "rendimiento": float(curva_equity[-1] / capital - 1) if curva_equity else 0.0,
    }


# ---------------------------------------------------------------------------
# Bitácora inmutable con hash encadenado
# ---------------------------------------------------------------------------
def _hash(previo, evento):
    cuerpo = json.dumps(evento, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256((previo + cuerpo).encode("utf-8")).hexdigest()


def bitacora_agregar(bitacora, evento):
    """Agrega un evento; su hash depende del anterior, así que alterar uno rompe toda la cadena."""
    previo = bitacora[-1]["hash"] if bitacora else "0" * 64
    ev = {k: v for k, v in evento.items() if k not in ("hash", "hash_previo")}
    ev["n"] = len(bitacora) + 1
    ev["hash_previo"] = previo
    ev["hash"] = _hash(previo, {k: v for k, v in ev.items() if k != "hash"})
    bitacora.append(ev)
    return ev


def bitacora_verificar(bitacora):
    """(True, None) si la cadena está intacta; si no, (False, número del primer evento alterado)."""
    previo = "0" * 64
    for ev in bitacora:
        if ev.get("hash_previo") != previo or ev.get("hash") != _hash(previo, {k: v for k, v in ev.items() if k != "hash"}):
            return False, ev.get("n")
        previo = ev["hash"]
    return True, None
