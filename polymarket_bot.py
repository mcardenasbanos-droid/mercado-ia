"""
polymarket_bot.py — Bot de apuestas en los mercados "Bitcoin / Ethereum Up or Down" (por hora) de Polymarket.

- Datos públicos: Gamma API (mercado de cada hora) y CLOB (libro de órdenes). No necesita llaves.
- Señal: el pronóstico diario de tu IA (MLP/LSTM) se convierte en una deriva por hora y, con la
  volatilidad reciente, en la probabilidad de que la vela de 1 hora cierre arriba de su apertura.
  El bot apuesta solo si esa probabilidad supera el precio del mercado por un margen.
- Cartera PAPER: simulador local (sin dinero real), con precios reales del libro de Polymarket.
- Este módulo NO envía órdenes reales: eso solo lo hace polymarket_real.py.
"""
import json
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests

URL_GAMMA = "https://gamma-api.polymarket.com"
URL_CLOB = "https://clob.polymarket.com"
ZONA_ET = "America/New_York"          # los mercados se nombran con la hora de Nueva York (ET)
ZONA_LOCAL = "America/Mexico_City"

ACTIVOS = {
    "BTC": {"slug": "bitcoin-up-or-down", "par": "BTCUSDT", "yahoo": "BTC-USD", "nombre": "Bitcoin"},
    "ETH": {"slug": "ethereum-up-or-down", "par": "ETHUSDT", "yahoo": "ETH-USD", "nombre": "Ethereum"},
}
MESES = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
         "november", "december"]

CAPITAL_PAPER = 10_000.0
MINUTOS_SIN_APOSTAR_AL_FINAL = 5      # no se abren apuestas en los últimos minutos de la hora
MINUTOS_PARA_RESPALDO = 20            # si Polymarket no ha resuelto, se usa la vela de MEXC (aprox.)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_ESTADO = os.path.join(BASE_DIR, "polymarket_estado.json")

_sesion = requests.Session()
_sesion.headers.update({"User-Agent": "Investify/1.0 (proyecto academico)"})


class PolymarketNoDisponible(Exception):
    pass


# ---------------------------------------------------------------------------
# Tiempo y nombres de mercado
# ---------------------------------------------------------------------------
def ahora_et():
    return pd.Timestamp.now(tz=ZONA_ET)


def inicio_hora_et(t=None):
    t = t or ahora_et()
    return t.floor("h")


def slugs_candidatos(base, inicio):
    """Polymarket nombra cada mercado así: bitcoin-up-or-down-november-26-2026-9am-et (con o sin año)."""
    h = inicio.hour
    etiqueta = f"{h % 12 or 12}{'am' if h < 12 else 'pm'}"
    mes = MESES[inicio.month - 1]
    return [f"{base}-{mes}-{inicio.day}-{inicio.year}-{etiqueta}-et", f"{base}-{mes}-{inicio.day}-{etiqueta}-et"]


def _lista(valor):
    if valor is None:
        return []
    if isinstance(valor, list):
        return valor
    try:
        return json.loads(valor)
    except (TypeError, ValueError):
        return []


# ---------------------------------------------------------------------------
# Datos públicos de Polymarket
# ---------------------------------------------------------------------------
def _get(url, params=None):
    try:
        r = _sesion.get(url, params=params, timeout=6)
    except requests.RequestException as e:
        raise PolymarketNoDisponible(f"Polymarket no respondió: {e}") from e
    if r.status_code == 404:
        return None
    if r.status_code >= 400:
        raise PolymarketNoDisponible(f"Polymarket respondió HTTP {r.status_code}.")
    try:
        return r.json()
    except ValueError as e:
        raise PolymarketNoDisponible("Respuesta inesperada de Polymarket.") from e


def mercado_hora(activo, inicio=None):
    """Mercado 'Up or Down' de la hora que empieza en `inicio` (ET). Devuelve un dict o None si no existe."""
    inicio = inicio or inicio_hora_et()
    for slug in slugs_candidatos(ACTIVOS[activo]["slug"], inicio):
        m = _get(f"{URL_GAMMA}/markets/slug/{slug}")
        if isinstance(m, list):
            m = m[0] if m else None
        if not isinstance(m, dict) or not m.get("clobTokenIds"):
            continue
        etiquetas = [str(x).lower() for x in _lista(m.get("outcomes"))]
        precios = [float(x) for x in _lista(m.get("outcomePrices"))] or [np.nan, np.nan]
        tokens = _lista(m.get("clobTokenIds"))
        if len(tokens) < 2:
            continue
        i_up = etiquetas.index("up") if "up" in etiquetas else 0
        i_dn = 1 - i_up
        return {"activo": activo, "slug": slug, "pregunta": m.get("question", slug),
                "inicio": inicio.isoformat(), "fin": (inicio + pd.Timedelta(hours=1)).isoformat(),
                "token_up": str(tokens[i_up]), "token_down": str(tokens[i_dn]),
                "precio_up": precios[i_up] if len(precios) > i_up else np.nan,
                "precio_down": precios[i_dn] if len(precios) > i_dn else np.nan,
                "cerrado": bool(m.get("closed")), "activo_m": bool(m.get("active", True)),
                "aceptando": bool(m.get("acceptingOrders", True))}
    return None


def libro(token_id):
    """Libro de órdenes del token: mejores precios, profundidad y tamaño mínimo de orden."""
    b = _get(f"{URL_CLOB}/book", {"token_id": token_id})
    if not isinstance(b, dict):
        raise PolymarketNoDisponible("No hay libro de órdenes para este mercado.")
    asks = sorted(((float(a["price"]), float(a["size"])) for a in b.get("asks", [])), key=lambda x: x[0])
    bids = sorted(((float(a["price"]), float(a["size"])) for a in b.get("bids", [])), key=lambda x: -x[0])
    return {"asks": asks, "bids": bids, "mejor_compra": asks[0][0] if asks else np.nan,
            "mejor_venta": bids[0][0] if bids else np.nan,
            "min_acciones": float(b.get("min_order_size") or 5)}


def costo_compra(asks, monto):
    """Recorre el libro: cuántas acciones se obtienen con `monto` USD y a qué precio promedio."""
    restante, acciones = monto, 0.0
    for precio, tam in asks:
        if precio <= 0:
            continue
        costo_nivel = precio * tam
        if costo_nivel >= restante:
            acciones += restante / precio
            restante = 0.0
            break
        acciones += tam
        restante -= costo_nivel
    if restante > 1e-9 or acciones <= 0:
        return None                                   # no hay liquidez suficiente
    return {"acciones": acciones, "precio_promedio": monto / acciones}


# ---------------------------------------------------------------------------
# Señal: del pronóstico diario de la IA a la probabilidad de "Sube" en la hora
# ---------------------------------------------------------------------------
def _phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def probabilidad_up(precio_actual, precio_apertura, fraccion_restante, rend_diario_ia, vol_diaria):
    """P(cierre ≥ apertura) suponiendo un paseo aleatorio con deriva.

    deriva por hora = pronóstico diario de la IA / 24 ; volatilidad por hora = volatilidad diaria / √24.
    Usa solo información disponible ahora (precio actual y apertura de la vela en curso)."""
    if precio_actual <= 0 or precio_apertura <= 0:
        return np.nan
    if fraccion_restante <= 0:
        return 1.0 if precio_actual >= precio_apertura else 0.0
    mu_h = (rend_diario_ia or 0.0) / 24
    s_h = max(vol_diaria or 0.0, 1e-6) / math.sqrt(24)
    z = (math.log(precio_actual / precio_apertura) + mu_h * fraccion_restante) / (s_h * math.sqrt(fraccion_restante))
    return min(max(_phi(z), 0.001), 0.999)


def decidir(p_up, precio_up, precio_down, margen):
    """Compara la probabilidad del modelo con el precio de cada lado. Devuelve ('Up'|'Down'|None, ventaja)."""
    ventaja_up = p_up - precio_up if not np.isnan(precio_up) else -1
    ventaja_dn = (1 - p_up) - precio_down if not np.isnan(precio_down) else -1
    if max(ventaja_up, ventaja_dn) < margen:
        return None, max(ventaja_up, ventaja_dn)
    return ("Up", ventaja_up) if ventaja_up >= ventaja_dn else ("Down", ventaja_dn)


# ---------------------------------------------------------------------------
# Estado de la cartera (paper / sombra / real)
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
        pass


def bot_nuevo(activos, monto_apuesta, margen, capital=CAPITAL_PAPER, modo="paper", tope_total=None):
    if not activos:
        raise ValueError("Elige al menos un mercado (BTC o ETH).")
    if not 0.005 <= margen <= 0.30:
        raise ValueError("El margen mínimo debe estar entre 0.5% y 30%.")
    if monto_apuesta < 1:
        raise ValueError("El monto por apuesta debe ser de al menos 1 USD.")
    if modo not in ("paper", "sombra", "real"):
        raise ValueError("Modo desconocido.")
    t = pd.Timestamp.now(tz=ZONA_LOCAL).tz_localize(None)
    cap = float(tope_total if modo in ("sombra", "real") and tope_total else capital)
    return {"id": t.strftime("%Y-%m-%d %H:%M:%S"), "inicio": t.isoformat(), "modo": modo, "capital": cap,
            "efectivo": cap, "monto_apuesta": float(monto_apuesta), "margen": float(margen), "activos": list(activos),
            "apuestas": [], "curva": [[t.isoformat(), cap]], "ultima_decision": {}}


def _registrar(est, apuesta):
    est["bot"]["apuestas"].append(apuesta)
    est["historial"].append(apuesta)


def _sincronizar_historial(est, apuesta):
    for h in est["historial"]:
        if h["id"] == apuesta["id"]:
            h.update(apuesta)


def ya_aposto(bot, slug):
    return any(a["slug"] == slug and a["estado"] != "rechazada" for a in bot["apuestas"])


def apostar(est, mercado, lado, p_up, ventaja, libro_lado, ejecutor=None):
    """Abre una apuesta. ejecutor=None → paper; ejecutor con modo sombra/real → polymarket_real."""
    bot = est["bot"]
    monto = min(bot["monto_apuesta"], bot["efectivo"])
    if monto < 1:
        return None
    sim = costo_compra(libro_lado["asks"], monto)
    if sim is None or sim["acciones"] < libro_lado["min_acciones"]:
        return None                                   # sin liquidez o por debajo del mínimo de Polymarket
    t = pd.Timestamp.now(tz=ZONA_LOCAL).tz_localize(None)
    token = mercado["token_up"] if lado == "Up" else mercado["token_down"]
    apuesta = {"id": f"{bot['id']}|{mercado['slug']}", "bot_id": bot["id"], "fecha": t.strftime("%Y-%m-%d %H:%M:%S"),
               "activo": mercado["activo"], "slug": mercado["slug"], "pregunta": mercado["pregunta"], "lado": lado,
               "token": token, "precio": sim["precio_promedio"], "acciones": sim["acciones"], "monto": monto,
               "prob_modelo": p_up if lado == "Up" else 1 - p_up, "ventaja": ventaja, "inicio_et": mercado["inicio"],
               "fin_et": mercado["fin"], "estado": "abierta", "pago": None, "ganancia": None,
               "modo": bot["modo"], "order_id": None, "resolucion": None}
    if ejecutor is not None and bot["modo"] in ("sombra", "real"):
        try:
            r = ejecutor.comprar(token, monto, enviar=bot["modo"] == "real")
        except Exception as e:                        # cualquier error del ejecutor: se registra y no se opera
            apuesta.update({"estado": "rechazada", "resolucion": f"{type(e).__name__}: {e}"})
            _registrar(est, apuesta)
            return apuesta
        apuesta.update({"order_id": r.get("order_id"), "precio": r.get("precio", apuesta["precio"]),
                        "acciones": r.get("acciones", apuesta["acciones"])})
        if bot["modo"] == "sombra":
            apuesta["estado"] = "sombra"              # calculada y validada, pero NO enviada
            _registrar(est, apuesta)
            return apuesta
    bot["efectivo"] -= monto
    _registrar(est, apuesta)
    return apuesta


def resolver(est, consultar_mercado=None, vela_respaldo=None):
    """Cierra las apuestas cuya hora terminó. Usa la resolución oficial de Polymarket y, si tarda,
    la vela de 1 hora de MEXC como respaldo aproximado (Polymarket resuelve con Binance)."""
    bot = est["bot"]
    if bot is None:
        return []
    consultar_mercado = consultar_mercado or (lambda a: _get(f"{URL_GAMMA}/markets/slug/{a['slug']}"))
    ahora = ahora_et()
    cerradas = []
    for a in bot["apuestas"]:
        if a["estado"] not in ("abierta", "sombra") or ahora < pd.Timestamp(a["fin_et"]):
            continue
        ganador, fuente = None, None
        try:
            m = consultar_mercado(a)
            if isinstance(m, list):
                m = m[0] if m else None
            if isinstance(m, dict) and m.get("closed"):
                etiquetas = [str(x).lower() for x in _lista(m.get("outcomes"))]
                precios = [float(x) for x in _lista(m.get("outcomePrices"))]
                if precios and max(precios) >= 0.99:
                    ganador = etiquetas[int(np.argmax(precios))].capitalize()
                    fuente = "Polymarket (oficial)"
        except PolymarketNoDisponible:
            pass
        espera = (ahora - pd.Timestamp(a["fin_et"])).total_seconds() / 60
        if ganador is None and vela_respaldo is not None and espera >= MINUTOS_PARA_RESPALDO:
            try:
                apertura, cierre = vela_respaldo(a)
                ganador, fuente = ("Up" if cierre >= apertura else "Down"), "vela de MEXC (aprox.)"
            except Exception:                         # sin respaldo disponible: se intenta en el siguiente ciclo
                ganador = None
        if ganador is None:
            continue
        gano = a["lado"] == ganador
        pago = a["acciones"] * 1.0 if gano else 0.0
        if a["estado"] == "sombra":
            a.update({"estado": "sombra resuelta", "pago": pago, "ganancia": pago - a["monto"],
                      "resolucion": f"{ganador} · {fuente}"})
        else:
            a.update({"estado": "ganada" if gano else "perdida", "pago": pago, "ganancia": pago - a["monto"],
                      "resolucion": f"{ganador} · {fuente}"})
            if bot["modo"] == "paper":
                bot["efectivo"] += pago
            elif gano:
                a["resolucion"] += " · cobra tus acciones ganadoras en Polymarket"
        _sincronizar_historial(est, a)
        cerradas.append(a)
    t = pd.Timestamp.now(tz=ZONA_LOCAL).tz_localize(None)
    if cerradas or not bot["curva"] or (t - pd.Timestamp(bot["curva"][-1][0])).total_seconds() >= 60:
        bot["curva"].append([t.isoformat(), patrimonio(bot)])
        bot["curva"] = bot["curva"][-5000:]
    return cerradas


def patrimonio(bot):
    """Efectivo + costo de las apuestas abiertas (se valoran a su costo hasta resolverse)."""
    abiertas = sum(a["monto"] for a in bot["apuestas"] if a["estado"] == "abierta")
    return bot["efectivo"] + abiertas


def resumen(bot):
    cerradas = [a for a in bot["apuestas"] if a["estado"] in ("ganada", "perdida")]
    ganadas = [a for a in cerradas if a["estado"] == "ganada"]
    abiertas = [a for a in bot["apuestas"] if a["estado"] == "abierta"]
    sombra = [a for a in bot["apuestas"] if a["estado"].startswith("sombra")]
    pnl = float(sum(a["ganancia"] for a in cerradas))
    pat = patrimonio(bot)
    return {"capital": bot["capital"], "patrimonio": pat, "efectivo": bot["efectivo"], "pnl": pnl,
            "rendimiento": pat / bot["capital"] - 1 if bot["capital"] else 0.0,
            "apuestas": len(cerradas) + len(abiertas), "cerradas": len(cerradas), "ganadas": len(ganadas),
            "perdidas": len(cerradas) - len(ganadas), "abiertas": len(abiertas),
            "win_rate": len(ganadas) / len(cerradas) if cerradas else np.nan,
            "en_juego": float(sum(a["monto"] for a in abiertas)),
            "sombra": len(sombra), "sombra_pnl": float(sum(a["ganancia"] or 0 for a in sombra)),
            "rechazadas": sum(1 for a in bot["apuestas"] if a["estado"] == "rechazada")}


def horas_de_sombra(est):
    """Horas acumuladas en modo sombra (requisito para encender el modo real)."""
    sombras = [a for a in est["historial"] if a.get("modo") == "sombra"]
    if not sombras:
        return 0.0
    inicio = min(pd.Timestamp(a["fecha"]) for a in sombras)
    fin = max(pd.Timestamp(a["fecha"]) for a in sombras)
    return (fin - inicio).total_seconds() / 3600


# ---------------------------------------------------------------------------
# Ciclo del bot
# ---------------------------------------------------------------------------
def evaluar(mercado, precio_actual, apertura, rend_ia, vol_diaria, margen, obtener_libro=None):
    """Probabilidad del modelo vs. precios del libro para el mercado de la hora."""
    obtener_libro = obtener_libro or libro
    restante_h = (pd.Timestamp(mercado["fin"]) - ahora_et()).total_seconds() / 3600
    fraccion = min(max(restante_h, 0.0), 1.0)
    p_up = probabilidad_up(precio_actual, apertura, fraccion, rend_ia, vol_diaria)
    lib_up, lib_dn = obtener_libro(mercado["token_up"]), obtener_libro(mercado["token_down"])
    lado, ventaja = decidir(p_up, lib_up["mejor_compra"], lib_dn["mejor_compra"], margen)
    return {"p_up": p_up, "ask_up": lib_up["mejor_compra"], "ask_down": lib_dn["mejor_compra"], "lado": lado,
            "ventaja": ventaja, "minutos_restantes": restante_h * 60, "libro_up": lib_up, "libro_down": lib_dn,
            "precio_actual": precio_actual, "apertura": apertura}


def ciclo(est, entradas, ejecutor=None, obtener_libro=None):
    """entradas: {activo: {'mercado', 'precio_actual', 'apertura', 'rend_ia', 'vol_diaria'}}.
    Evalúa cada mercado y abre como máximo una apuesta por mercado (por hora)."""
    bot = est["bot"]
    nuevas = []
    if bot is None:
        return nuevas
    for activo in bot["activos"]:
        e = entradas.get(activo)
        if not e or not e.get("mercado"):
            bot["ultima_decision"][activo] = {"texto": "Sin mercado abierto para esta hora"}
            continue
        m = e["mercado"]
        try:
            ev = evaluar(m, e["precio_actual"], e["apertura"], e["rend_ia"], e["vol_diaria"], bot["margen"], obtener_libro)
        except PolymarketNoDisponible as err:
            bot["ultima_decision"][activo] = {"texto": f"Sin libro de órdenes: {err}"}
            continue
        texto = (f"Apostar {ev['lado']}" if ev["lado"] else "Esperar: sin ventaja suficiente")
        bot["ultima_decision"][activo] = {k: ev[k] for k in ("p_up", "ask_up", "ask_down", "lado", "ventaja",
                                                             "minutos_restantes", "precio_actual", "apertura")}
        bot["ultima_decision"][activo].update({"texto": texto, "slug": m["slug"], "pregunta": m["pregunta"]})
        if (ev["lado"] and not ya_aposto(bot, m["slug"]) and ev["minutos_restantes"] > MINUTOS_SIN_APOSTAR_AL_FINAL
                and not m.get("cerrado") and m.get("aceptando", True)):
            lib = ev["libro_up"] if ev["lado"] == "Up" else ev["libro_down"]
            a = apostar(est, m, ev["lado"], ev["p_up"], ev["ventaja"], lib, ejecutor)
            if a:
                nuevas.append(a)
    return nuevas
