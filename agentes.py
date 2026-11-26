"""
agentes.py — La IA como cerebro de las operaciones del bot MEXC.

Cuatro agentes con UN solo rol cada uno opinan sobre cada entrada; un coordinador combina sus votos con
pesos y produce una salida estructurada (JSON validado con probabilidad 0–1 y razón). Encima de todo,
el módulo de riesgo (riesgo.py) es determinista y tiene VETO: la IA nunca pone órdenes ni cambia tamaños
o límites; solo propone.

Agentes (deterministas, temperatura 0 por construcción: mismas entradas → misma salida):
- modelo_dl  : pronóstico de tu MLP/LSTM (Notebook 5) para el siguiente día.
- tecnico    : RSI 14 (Wilder) y MACD en velas cerradas de 1 minuto → reversión a la media.
- tendencia  : precio contra su SMA 20 → no comprar caídas dentro de una tendencia bajista fuerte.
- volatilidad: ATR relativo → evita mercados demasiado quietos (no cubren costos) o demasiado violentos.

Aprendizaje de pesos: cuando se cierra una operación, cada agente que votó suma un acierto si su voto
tuvo el mismo signo que el PnL. Peso = piso + (1 − piso) · (aciertos + 1) / (votos + 2)  (Laplace),
con piso 0.10 para que ningún agente desaparezca. Solo se aplica con ≥ 10 operaciones auditadas.
"""
import math

AGENTES = ("modelo_dl", "tecnico", "tendencia", "volatilidad")
PISO_PESO = 0.10
MIN_OPS_PESOS = 10
UMBRAL_ENTRADA = 0.55          # probabilidad mínima del coordinador para proponer una compra
VERSION = "agentes-v1.0"


def _voto(agente, voto, confianza, razon):
    return {"agente": agente, "voto": max(-1.0, min(1.0, float(voto))),
            "confianza": max(0.0, min(1.0, float(confianza))), "razon": razon}


def agente_modelo_dl(ia):
    """ia = {'senal': 'comprar'|'efectivo', 'estimado': rendimiento del día, 'umbral': δ} o None."""
    if not ia or ia.get("estimado") is None or ia["estimado"] != ia["estimado"]:
        return _voto("modelo_dl", 0, 0, "sin pronóstico disponible: se abstiene")
    est = float(ia["estimado"])
    umbral = abs(float(ia.get("umbral") or 0.002)) or 0.002
    conf = min(1.0, abs(est) / (3 * umbral))
    if ia.get("senal") == "comprar":
        return _voto("modelo_dl", 1, max(conf, 0.4), f"la red estima {est:+.2%} (> δ {umbral:.2%}): LONG")
    if est < 0:
        return _voto("modelo_dl", -1, conf, f"la red estima {est:+.2%}: sesgo bajista")
    return _voto("modelo_dl", 0, 0.2, f"la red estima {est:+.2%} (< δ): sin ventaja")


def agente_tecnico(ctx):
    if not ctx:
        return _voto("tecnico", 0, 0, "sin velas suficientes: se abstiene")
    rsi, h = ctx["rsi"], ctx["macd_hist"]
    if rsi <= 35:
        return _voto("tecnico", 1, min(1.0, (40 - rsi) / 20), f"RSI {rsi:.0f}: sobreventa, rebote probable")
    if rsi >= 65:
        return _voto("tecnico", -1, min(1.0, (rsi - 60) / 20), f"RSI {rsi:.0f}: sobrecompra")
    return _voto("tecnico", 1 if h > 0 else -1, 0.3, f"RSI {rsi:.0f} neutral; MACD {'al alza' if h > 0 else 'a la baja'}")


def agente_tendencia(ctx, precio):
    if not ctx or not ctx.get("sma20"):
        return _voto("tendencia", 0, 0, "sin SMA 20: se abstiene")
    dist = precio / ctx["sma20"] - 1
    atr_rel = ctx["atr"] / precio if precio else 0
    if dist < -3 * atr_rel:
        return _voto("tendencia", -1, 0.8, f"precio {dist:+.2%} bajo la SMA 20: caída fuerte, no atrapar el cuchillo")
    if dist >= 0:
        return _voto("tendencia", 1, 0.5, f"precio {dist:+.2%} sobre la SMA 20: tendencia a favor")
    return _voto("tendencia", 1, 0.3, f"precio {dist:+.2%}: retroceso moderado dentro del rango")


def agente_volatilidad(ctx, precio, costo_ida_vuelta):
    if not ctx or not precio:
        return _voto("volatilidad", 0, 0, "sin ATR: se abstiene")
    atr_rel = ctx["atr"] / precio
    if atr_rel * 3.0 < 2 * costo_ida_vuelta:
        return _voto("volatilidad", -1, 0.7, f"ATR {atr_rel:.3%}: el TP no cubriría bien los costos")
    if atr_rel > 0.02:
        return _voto("volatilidad", -1, 0.6, f"ATR {atr_rel:.2%}: volatilidad extrema")
    return _voto("volatilidad", 1, 0.4, f"ATR {atr_rel:.3%}: volatilidad operable")


def pesos_desde_auditoria(auditoria):
    """auditoria = {agente: {'votos': n, 'aciertos': k}}. Regla documentada arriba."""
    total = min((auditoria.get(a, {}).get("votos", 0) for a in AGENTES), default=0)
    if total < MIN_OPS_PESOS:
        return {a: 1.0 for a in AGENTES}
    return {a: PISO_PESO + (1 - PISO_PESO) * (auditoria[a]["aciertos"] + 1) / (auditoria[a]["votos"] + 2)
            for a in AGENTES}


def coordinar(votos, pesos):
    """Combina votos con pesos → probabilidad 0–1 de que la entrada termine en ganancia."""
    num = sum(pesos.get(v["agente"], 1.0) * v["voto"] * v["confianza"] for v in votos)
    den = sum(pesos.get(v["agente"], 1.0) * v["confianza"] for v in votos)
    if den == 0:
        return {"decision": "abstenerse", "probabilidad": 0.5, "confianza": 0.0,
                "razon": "todos los agentes se abstuvieron: decisión determinista de respaldo = no operar"}
    puntaje = num / den
    prob = 0.5 + 0.5 * puntaje
    confianza = den / sum(pesos.get(v["agente"], 1.0) for v in votos)
    a_favor = [v["agente"] for v in votos if v["voto"] > 0 and v["confianza"] > 0]
    en_contra = [v["agente"] for v in votos if v["voto"] < 0 and v["confianza"] > 0]
    decision = "comprar" if prob >= UMBRAL_ENTRADA else "esperar"
    razon = (f"a favor: {', '.join(a_favor) or 'ninguno'} · en contra: {', '.join(en_contra) or 'ninguno'} · "
             f"P = {prob:.0%} vs mínimo {UMBRAL_ENTRADA:.0%}")
    return {"decision": decision, "probabilidad": round(prob, 4), "confianza": round(confianza, 4), "razon": razon}


ESQUEMA = {"decision": str, "probabilidad": float, "confianza": float, "razon": str, "votos": list, "version": str}


def validar_salida(salida):
    """Valida el JSON del coordinador. Si algo no cumple, la salida se reemplaza por «abstenerse»."""
    try:
        for k, tipo in ESQUEMA.items():
            if not isinstance(salida[k], tipo):
                raise ValueError(k)
        if salida["decision"] not in ("comprar", "esperar", "abstenerse"):
            raise ValueError("decision")
        if not (0 <= salida["probabilidad"] <= 1 and 0 <= salida["confianza"] <= 1):
            raise ValueError("rango")
        for v in salida["votos"]:
            if v["agente"] not in AGENTES or not -1 <= v["voto"] <= 1 or not 0 <= v["confianza"] <= 1:
                raise ValueError("voto")
        return salida
    except (KeyError, TypeError, ValueError) as e:
        return {"decision": "abstenerse", "probabilidad": 0.5, "confianza": 0.0, "votos": [],
                "razon": f"salida inválida ({e}): se abstiene", "version": VERSION}


def decidir_entrada(ia, ctx, precio, pesos, costo_ida_vuelta):
    votos = [agente_modelo_dl(ia), agente_tecnico(ctx), agente_tendencia(ctx, precio),
             agente_volatilidad(ctx, precio, costo_ida_vuelta)]
    salida = coordinar(votos, pesos)
    salida.update({"votos": votos, "version": VERSION})
    return validar_salida(salida)


def auditar_cierre(auditoria, votos, pnl):
    """Suma votos y aciertos por agente al cerrar una operación."""
    for v in votos or []:
        if v.get("confianza", 0) <= 0 or v.get("voto", 0) == 0:
            continue
        reg = auditoria.setdefault(v["agente"], {"votos": 0, "aciertos": 0})
        reg["votos"] += 1
        reg["aciertos"] += int(math.copysign(1, v["voto"]) == math.copysign(1, pnl) and pnl != 0)
    return auditoria
