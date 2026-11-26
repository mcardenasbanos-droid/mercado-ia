"""
mexc_bot.py — Bot de trading con PRECIOS REALES de MEXC y dinero de PRÁCTICA.

- Precios y velas en tiempo real desde la API pública de MEXC (no requiere llaves).
- Si MEXC no responde (por ejemplo, desde servidores en EE. UU., donde MEXC no opera),
  se usa Yahoo Finance como respaldo y se indica en la app.
- Capital de práctica (modo paper): 50,000 USDT. No se envían órdenes reales.
- Tope de compra: compra cuando el precio cae X% desde su máximo reciente.
- Tope de venta: vende cuando gana Y% sobre su precio de compra (toma de ganancia).
- Stop-loss opcional (apagado por defecto).
- MODO REAL (opcional, solo en tu computadora): las órdenes se envían a tu cuenta Spot de MEXC
  a través de mexc_cuenta.ClienteMEXC, con un monto fijo por orden y un tope total.
"""
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from mexc_cuenta import ErrorMEXC

URL_MEXC = "https://api.mexc.com"
CAPITAL_MEXC = 50_000.0
COSTO = 0.001                     # 0.10% por compra y por venta (supuesto del proyecto)
MONTO_MINIMO = 5.0                # USDT mínimos por orden
TOPE_VENTA_MINIMO = 0.003         # 0.3%: cubre las dos comisiones y deja ganancia
ZONA = "America/Mexico_City"
ESPERA_TRAS_ERROR_MIN = 5          # minutos sin reintentar un par después de una orden rechazada

PARES = {  # par de MEXC: ticker equivalente de Yahoo (respaldo)
    "BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "SOLUSDT": "SOL-USD", "BNBUSDT": "BNB-USD",
    "XRPUSDT": "XRP-USD", "ADAUSDT": "ADA-USD", "DOGEUSDT": "DOGE-USD", "LTCUSDT": "LTC-USD",
    "LINKUSDT": "LINK-USD", "AVAXUSDT": "AVAX-USD",
}
# Pares disponibles en la Demo Trading de MEXC (futuros USDT-M). El bot opera todos sus pares en modo paper,
# pero solo estos se pueden copiar a la demo.
PARES_DEMO_MEXC = ["BTCUSDT", "ETHUSDT"]

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


def estadisticas_24h(pares):
    """Cambio, máximo, mínimo y volumen de las últimas 24 h desde MEXC: {par: {...}}."""
    try:
        r = _sesion.get(f"{URL_MEXC}/api/v3/ticker/24hr", timeout=6)
        r.raise_for_status()
        datos = r.json()
    except (requests.RequestException, ValueError) as e:
        raise FuenteNoDisponible(f"MEXC no respondió: {e}") from e
    if not isinstance(datos, list):
        raise FuenteNoDisponible("Respuesta inesperada de MEXC (24 h).")
    salida = {}
    for d in datos:
        par = d.get("symbol")
        if par not in pares:
            continue
        try:
            ultimo, apertura = float(d["lastPrice"]), float(d["openPrice"])
            salida[par] = {"ultimo": ultimo, "cambio": ultimo / apertura - 1 if apertura else float("nan"),
                           "maximo": float(d.get("highPrice", "nan")), "minimo": float(d.get("lowPrice", "nan")),
                           "volumen": float(d.get("quoteVolume", "nan"))}
        except (KeyError, TypeError, ValueError):
            continue
    if not salida:
        raise FuenteNoDisponible("MEXC no devolvió estadísticas de 24 h.")
    return salida


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
    pares = [p for p in pares if p in PARES]
    if not pares:
        return {}
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
    if not pares:
        return {}, "MEXC"
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


def bot_nuevo(precios, monto_operacion, tope_compra, tope_venta, stop_loss=None, capital=CAPITAL_MEXC,
              modo="paper"):
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
                        "precio_compra": None, "monto": 0.0, "espera_hasta": None} for p, px in precios.items()},
        "operaciones": [], "curva": [[t.isoformat(), float(capital)]], "fuente": "", "modo": modo,
    }


def _registrar(est, op):
    est["bot"]["operaciones"].append(op)
    est["historial"].append(op)


def patrimonio(bot):
    return bot["efectivo"] + sum(a["unidades"] * a["ultimo"] for a in bot["activos"].values())


def _en_espera(a, t):
    return a.get("espera_hasta") is not None and t < pd.Timestamp(a["espera_hasta"])


def _rechazo(est, bot, a, par, tipo, px, error, fuente, t):
    a["espera_hasta"] = (t + pd.Timedelta(minutes=ESPERA_TRAS_ERROR_MIN)).isoformat()
    op = {"bot_id": bot["id"], "fecha": t.strftime("%Y-%m-%d %H:%M:%S"), "par": par, "tipo": tipo,
          "motivo": str(error), "precio": px, "unidades": 0.0, "monto_usdt": 0.0, "comision_usdt": None,
          "ganancia_usdt": None, "ganancia_%": None, "fuente": fuente, "modo": bot.get("modo", "paper"),
          "estado": "Rechazada", "order_id": None}
    _registrar(est, op)
    return op


def bot_actualizar(est, precios, fuente="MEXC", costo=COSTO, ejecutor=None):
    """Aplica los topes con los precios reales recibidos. Devuelve las operaciones nuevas.

    ejecutor=None → modo paper (simulado). ejecutor=ClienteMEXC → modo real (órdenes en tu cuenta)."""
    bot = est["bot"]
    if bot is None:
        return []
    real = bot.get("modo") == "real" and ejecutor is not None
    bot["fuente"] = fuente
    t = ahora()
    nuevas = []
    for par, a in bot["activos"].items():
        if par not in precios:
            continue
        px = float(precios[par])
        a["ultimo"] = px
        if _en_espera(a, t):
            continue
        if a["unidades"] == 0:
            a["maximo_reciente"] = max(a["maximo_reciente"], px)
            nivel_compra = a["maximo_reciente"] * (1 - bot["tope_compra"])
            if px <= nivel_compra:
                monto = min(bot["monto_operacion"], bot["efectivo"])
                if monto >= MONTO_MINIMO:
                    order_id = None
                    if real:
                        try:
                            res = ejecutor.comprar_mercado(par, monto)
                        except ErrorMEXC as e:
                            nuevas.append(_rechazo(est, bot, a, par, "compra", px, e, fuente, t))
                            continue
                        unidades, px_ejec, monto, comision = res["cantidad"], res["precio"], res["total_usdt"], None
                        order_id = res["order_id"]
                    else:
                        unidades = monto / (px * (1 + costo))
                        comision = unidades * px * costo
                        px_ejec = px
                    a.update({"unidades": unidades, "precio_compra": px_ejec, "monto": monto, "espera_hasta": None})
                    bot["efectivo"] -= monto
                    op = {"bot_id": bot["id"], "fecha": t.strftime("%Y-%m-%d %H:%M:%S"), "par": par, "tipo": "compra",
                          "motivo": f"bajó {bot['tope_compra']:.2%} desde su máximo", "precio": px_ejec,
                          "unidades": unidades, "monto_usdt": monto, "comision_usdt": comision,
                          "ganancia_usdt": None, "ganancia_%": None, "fuente": fuente,
                          "modo": "real" if real else "paper", "estado": "Ejecutada", "order_id": order_id}
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
                order_id = None
                if real:
                    try:
                        res = ejecutor.vender_mercado(par, a["unidades"])
                    except ErrorMEXC as e:
                        nuevas.append(_rechazo(est, bot, a, par, "venta", px, e, fuente, t))
                        continue
                    neto, px_ejec, comision, order_id = res["total_usdt"], res["precio"], None, res["order_id"]
                    vendidas = res["cantidad"]
                    # MEXC redondea la cantidad al paso mínimo del par: el residuo pequeño queda en tu cuenta
                    costo_vendido = a["monto"] * min(vendidas / a["unidades"], 1.0)
                else:
                    bruto = a["unidades"] * px
                    comision = bruto * costo
                    neto, px_ejec, vendidas, costo_vendido = bruto - comision, px, a["unidades"], a["monto"]
                ganancia = neto - costo_vendido
                bot["efectivo"] += neto
                op = {"bot_id": bot["id"], "fecha": t.strftime("%Y-%m-%d %H:%M:%S"), "par": par, "tipo": "venta",
                      "motivo": motivo, "precio": px_ejec, "unidades": vendidas, "monto_usdt": neto,
                      "comision_usdt": comision, "ganancia_usdt": ganancia, "ganancia_%": ganancia / costo_vendido * 100,
                      "fuente": fuente, "modo": "real" if real else "paper", "estado": "Ejecutada", "order_id": order_id}
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
    ejecutadas = [o for o in bot["operaciones"] if o.get("estado", "Ejecutada") == "Ejecutada"]
    ventas = [o for o in ejecutadas if o["tipo"] == "venta"]
    compras = [o for o in ejecutadas if o["tipo"] == "compra"]
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
        "comisiones": float(sum(o["comision_usdt"] or 0 for o in ejecutadas)),
        "rechazadas": len(bot["operaciones"]) - len(ejecutadas),
        "compras": len(compras), "ventas": len(ventas),
        "ganadoras": sum(o["ganancia_usdt"] > 0 for o in ventas),
        "mejor": max((o["ganancia_usdt"] for o in ventas), default=None),
        "minutos": minutos, "por_par": por_par, "abiertas": abiertas,
    }


def clave_op(op):
    """Identificador estable de una operación (para marcarla como copiada a MEXC Demo)."""
    return f"{op.get('bot_id', '')}|{op['fecha']}|{op['par']}|{op['tipo']}"


def pendientes_demo(est):
    """Operaciones ejecutadas en modo paper de los pares de la demo que aún no se copian a MEXC Demo."""
    copiadas = set(est.get("copiadas", [])) | set(est.get("omitidas", []))
    bot = est.get("bot") or {}
    return [o for o in bot.get("operaciones", [])
            if o["par"] in PARES_DEMO_MEXC and o.get("estado", "Ejecutada") == "Ejecutada"
            and o.get("modo", "paper") == "paper" and clave_op(o) not in copiadas]
