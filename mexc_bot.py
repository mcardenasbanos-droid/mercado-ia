"""
mexc_bot.py — Bot de trading con PRECIOS REALES de MEXC y dinero de PRÁCTICA.

- Precios y velas en tiempo real desde la API pública de MEXC (no requiere llaves).
- Si MEXC no responde (por ejemplo, desde servidores en EE. UU., donde MEXC no opera),
  se usa Yahoo Finance como respaldo y se indica en la app.
- Capital de práctica: 10,000 USDT. No se envían órdenes reales.
- Tope de compra: compra cuando el precio cae X% desde su máximo reciente.
- Tope de venta: vende cuando gana Y% sobre su precio de compra (toma de ganancia).
- Stop-loss opcional (apagado por defecto).
"""
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf

URL_MEXC = "https://api.mexc.com"
CAPITAL_MEXC = 10_000.0
COSTO = 0.001                     # 0.10% por compra y por venta (supuesto del proyecto)
MONTO_MINIMO = 5.0                # USDT mínimos por orden
TOPE_VENTA_MINIMO = 0.003         # 0.3%: cubre las dos comisiones y deja ganancia
ZONA = "America/Mexico_City"

PARES = {  # par de MEXC: ticker equivalente de Yahoo (respaldo)
    "BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "SOLUSDT": "SOL-USD", "BNBUSDT": "BNB-USD",
    "XRPUSDT": "XRP-USD", "ADAUSDT": "ADA-USD", "DOGEUSDT": "DOGE-USD", "LTCUSDT": "LTC-USD",
    "LINKUSDT": "LINK-USD", "AVAXUSDT": "AVAX-USD",
}
INTERVALOS = {"1 minuto": "1m", "5 minutos": "5m", "15 minutos": "15m", "1 hora": "60m"}
_YAHOO_INTERVALO = {"1m": ("1m", "1d"), "5m": ("5m", "5d"), "15m": ("15m", "5d"), "60m": ("60m", "1mo")}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_ESTADO = os.path.join(BASE_DIR, "bot_mexc.json")

_sesion = requests.Session()
_sesion.headers.update({"User-Agent": "Investify/1.0 (proyecto academico)"})


class FuenteNoDisponible(Exception):
    pass


def ahora():
    return pd.Timestamp.now(tz=ZONA).tz_localize(None)


# ---------------------------------------------------------------------------
# Datos de mercado
# ---------------------------------------------------------------------------
def precios_mexc(pares):
    try:
        r = _sesion.get(f"{URL_MEXC}/api/v3/ticker/price", timeout=5)
        r.raise_for_status()
        datos = r.json()
    except (requests.RequestException, ValueError) as e:
        raise FuenteNoDisponible(f"MEXC no respondió: {e}") from e
    if not isinstance(datos, list):
        raise FuenteNoDisponible(f"Respuesta inesperada de MEXC: {str(datos)[:120]}")
    todos = {d.get("symbol"): d.get("price") for d in datos if isinstance(d, dict)}
    precios = {p: float(todos[p]) for p in pares if todos.get(p) not in (None, "")}
    if not precios:
        raise FuenteNoDisponible("MEXC no devolvió precios para los pares elegidos.")
    return precios


def velas_mexc(par, intervalo="1m", limite=120):
    try:
        r = _sesion.get(f"{URL_MEXC}/api/v3/klines",
                        params={"symbol": par, "interval": intervalo, "limit": limite}, timeout=5)
        r.raise_for_status()
        filas = r.json()
    except (requests.RequestException, ValueError) as e:
        raise FuenteNoDisponible(f"MEXC no respondió: {e}") from e
    if not isinstance(filas, list) or not filas:
        raise FuenteNoDisponible(f"MEXC no devolvió velas para {par}.")
    df = pd.DataFrame([f[:6] for f in filas], columns=["tiempo", "apertura", "maximo", "minimo", "cierre", "volumen"])
    df["tiempo"] = pd.to_datetime(df["tiempo"], unit="ms", utc=True).dt.tz_convert(ZONA).dt.tz_localize(None)
    for c in ["apertura", "maximo", "minimo", "cierre", "volumen"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()


def _yahoo_minuto(tickers, intervalo="1m"):
    yi, periodo = _YAHOO_INTERVALO.get(intervalo, ("1m", "1d"))
    crudo = yf.download(list(tickers), period=periodo, interval=yi, auto_adjust=True,
                        group_by="ticker", progress=False, threads=True)
    salida = {}
    if crudo is None or crudo.empty:
        return salida
    for t in tickers:
        if isinstance(crudo.columns, pd.MultiIndex):
            if t not in crudo.columns.get_level_values(0):
                continue
            df = crudo[t]
        else:
            df = crudo
        df = df.dropna(subset=["Close"])
        if len(df):
            salida[t] = df
    return salida


def precios_yahoo(pares):
    datos = _yahoo_minuto([PARES[p] for p in pares])
    inv = {v: k for k, v in PARES.items()}
    return {inv[t]: float(df["Close"].iloc[-1]) for t, df in datos.items()}


def velas_yahoo(par, intervalo="1m", limite=120):
    datos = _yahoo_minuto([PARES[par]], intervalo)
    if PARES[par] not in datos:
        raise FuenteNoDisponible(f"Sin velas para {par} en Yahoo.")
    df = datos[PARES[par]].tail(limite).reset_index()
    col_t = df.columns[0]
    t = pd.to_datetime(df[col_t])
    t = t.dt.tz_convert(ZONA).dt.tz_localize(None) if t.dt.tz is not None else t
    return pd.DataFrame({"tiempo": t, "apertura": df["Open"], "maximo": df["High"], "minimo": df["Low"],
                         "cierre": df["Close"], "volumen": df.get("Volume", 0)})


def obtener_precios(pares):
    """Devuelve (precios, fuente). Usa MEXC y, si falla, Yahoo Finance."""
    try:
        return precios_mexc(pares), "MEXC"
    except FuenteNoDisponible:
        return precios_yahoo(pares), "Yahoo Finance (respaldo)"


def obtener_velas(par, intervalo="1m", limite=120):
    try:
        return velas_mexc(par, intervalo, limite), "MEXC"
    except FuenteNoDisponible:
        return velas_yahoo(par, intervalo, limite), "Yahoo Finance (respaldo)"


# ---------------------------------------------------------------------------
# Estado del bot (se guarda en bot_mexc.json para no perder nada al cerrar la app)
# ---------------------------------------------------------------------------
def estado_vacio():
    return {"bot": None, "historial": []}


def cargar_estado():
    if os.path.exists(RUTA_ESTADO):
        try:
            with open(RUTA_ESTADO) as f:
                est = json.load(f)
            if isinstance(est, dict) and "historial" in est:
                return est
        except (json.JSONDecodeError, OSError):
            pass
    return estado_vacio()


def guardar_estado(est):
    try:
        with open(RUTA_ESTADO, "w") as f:
            json.dump(est, f)
    except OSError:
        pass   # en Streamlit Cloud el disco puede no persistir; el estado sigue en la sesión


def bot_nuevo(precios, monto_operacion, tope_compra, tope_venta, stop_loss=None, capital=CAPITAL_MEXC):
    if not precios:
        raise ValueError("Elige al menos un par para el bot.")
    if tope_venta < TOPE_VENTA_MINIMO:
        raise ValueError(f"El tope de venta debe ser al menos {TOPE_VENTA_MINIMO:.1%} para cubrir las comisiones.")
    if not 0 < tope_compra <= 0.10:
        raise ValueError("El tope de compra debe estar entre 0% y 10%.")
    if monto_operacion < MONTO_MINIMO:
        raise ValueError(f"El monto por operación debe ser al menos {MONTO_MINIMO:.0f} USDT.")
    if stop_loss is not None and not 0 < stop_loss <= 0.5:
        raise ValueError("El stop-loss debe estar entre 0% y 50%.")
    t = ahora()
    return {
        "id": t.strftime("%Y-%m-%d %H:%M:%S"), "inicio": t.isoformat(), "capital": float(capital),
        "efectivo": float(capital), "monto_operacion": float(monto_operacion),
        "tope_compra": float(tope_compra), "tope_venta": float(tope_venta),
        "stop_loss": None if stop_loss is None else float(stop_loss),
        "activos": {p: {"ultimo": float(px), "maximo_reciente": float(px), "unidades": 0.0,
                        "precio_compra": None, "monto": 0.0} for p, px in precios.items()},
        "operaciones": [], "curva": [[t.isoformat(), float(capital)]], "fuente": "",
    }


def _registrar(est, op):
    est["bot"]["operaciones"].append(op)
    est["historial"].append(op)


def patrimonio(bot):
    return bot["efectivo"] + sum(a["unidades"] * a["ultimo"] for a in bot["activos"].values())


def bot_actualizar(est, precios, fuente="MEXC", costo=COSTO):
    """Aplica los topes con los precios reales recibidos. Devuelve las operaciones nuevas."""
    bot = est["bot"]
    if bot is None:
        return []
    bot["fuente"] = fuente
    t = ahora()
    nuevas = []
    for par, a in bot["activos"].items():
        if par not in precios:
            continue
        px = float(precios[par])
        a["ultimo"] = px
        if a["unidades"] == 0:
            a["maximo_reciente"] = max(a["maximo_reciente"], px)
            nivel_compra = a["maximo_reciente"] * (1 - bot["tope_compra"])
            if px <= nivel_compra:
                monto = min(bot["monto_operacion"], bot["efectivo"])
                if monto >= MONTO_MINIMO:
                    unidades = monto / (px * (1 + costo))
                    comision = unidades * px * costo
                    a.update({"unidades": unidades, "precio_compra": px, "monto": monto})
                    bot["efectivo"] -= monto
                    op = {"bot_id": bot["id"], "fecha": t.strftime("%Y-%m-%d %H:%M:%S"), "par": par, "tipo": "compra",
                          "motivo": f"bajó {bot['tope_compra']:.2%} desde su máximo", "precio": px,
                          "unidades": unidades, "monto_usdt": monto, "comision_usdt": comision,
                          "ganancia_usdt": None, "ganancia_%": None, "fuente": fuente}
                    _registrar(est, op)
                    nuevas.append(op)
        else:
            objetivo = a["precio_compra"] * (1 + bot["tope_venta"])
            limite_perdida = a["precio_compra"] * (1 - bot["stop_loss"]) if bot["stop_loss"] else None
            motivo = None
            if px >= objetivo:
                motivo = f"tope de venta +{bot['tope_venta']:.2%}"
            elif limite_perdida is not None and px <= limite_perdida:
                motivo = f"stop-loss −{bot['stop_loss']:.2%}"
            if motivo:
                bruto = a["unidades"] * px
                comision = bruto * costo
                neto = bruto - comision
                ganancia = neto - a["monto"]
                bot["efectivo"] += neto
                op = {"bot_id": bot["id"], "fecha": t.strftime("%Y-%m-%d %H:%M:%S"), "par": par, "tipo": "venta",
                      "motivo": motivo, "precio": px, "unidades": a["unidades"], "monto_usdt": neto,
                      "comision_usdt": comision, "ganancia_usdt": ganancia, "ganancia_%": ganancia / a["monto"] * 100,
                      "fuente": fuente}
                _registrar(est, op)
                nuevas.append(op)
                a.update({"unidades": 0.0, "precio_compra": None, "monto": 0.0, "maximo_reciente": px})
    # Curva de patrimonio: un punto cada 15 segundos como máximo
    ultimo_t = pd.Timestamp(bot["curva"][-1][0]) if bot["curva"] else None
    if ultimo_t is None or (t - ultimo_t).total_seconds() >= 15 or nuevas:
        bot["curva"].append([t.isoformat(), patrimonio(bot)])
        if len(bot["curva"]) > 6000:
            bot["curva"] = bot["curva"][-6000:]
    return nuevas


def niveles(bot, par):
    a = bot["activos"][par]
    if a["unidades"] > 0:
        return {"compra": None, "venta": a["precio_compra"] * (1 + bot["tope_venta"]),
                "stop": a["precio_compra"] * (1 - bot["stop_loss"]) if bot["stop_loss"] else None,
                "precio_compra": a["precio_compra"]}
    return {"compra": a["maximo_reciente"] * (1 - bot["tope_compra"]), "venta": None, "stop": None, "precio_compra": None}


def resumen(bot):
    ventas = [o for o in bot["operaciones"] if o["tipo"] == "venta"]
    compras = [o for o in bot["operaciones"] if o["tipo"] == "compra"]
    realizada = float(sum(o["ganancia_usdt"] for o in ventas))
    no_realizada = float(sum(a["unidades"] * a["ultimo"] - a["monto"] for a in bot["activos"].values() if a["unidades"] > 0))
    pat = patrimonio(bot)
    inicio = pd.Timestamp(bot["inicio"])
    minutos = max((ahora() - inicio).total_seconds() / 60, 0)
    por_par = (pd.DataFrame(ventas).groupby("par")["ganancia_usdt"].agg(["sum", "count"])
               .rename(columns={"sum": "ganancia_usdt", "count": "ventas"}).reset_index()
               if ventas else pd.DataFrame(columns=["par", "ganancia_usdt", "ventas"]))
    abiertas = pd.DataFrame([
        {"par": p, "precio_compra": a["precio_compra"], "precio_actual": a["ultimo"],
         "objetivo_venta": a["precio_compra"] * (1 + bot["tope_venta"]), "invertido_usdt": a["monto"],
         "valor_usdt": a["unidades"] * a["ultimo"], "resultado_%": (a["ultimo"] / a["precio_compra"] - 1) * 100}
        for p, a in bot["activos"].items() if a["unidades"] > 0],
        columns=["par", "precio_compra", "precio_actual", "objetivo_venta", "invertido_usdt", "valor_usdt", "resultado_%"])
    return {
        "capital": bot["capital"], "patrimonio": pat, "efectivo": bot["efectivo"],
        "invertido": pat - bot["efectivo"], "ganancia_realizada": realizada, "ganancia_no_realizada": no_realizada,
        "ganancia_total": pat - bot["capital"], "rendimiento": pat / bot["capital"] - 1,
        "comisiones": float(sum(o["comision_usdt"] for o in bot["operaciones"])),
        "compras": len(compras), "ventas": len(ventas),
        "ganadoras": sum(o["ganancia_usdt"] > 0 for o in ventas),
        "mejor": max((o["ganancia_usdt"] for o in ventas), default=None),
        "minutos": minutos, "por_par": por_par, "abiertas": abiertas,
    }
