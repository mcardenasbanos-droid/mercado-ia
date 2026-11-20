"""
nucleo.py — Lógica de la app (sin interfaz):
datos de Yahoo Finance, indicadores técnicos, pronósticos con los modelos del Notebook 5
y el portafolio de trading simulado.
"""
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import yfinance as yf

VENTANA = 20
IND = ["dist_sma20", "vol20", "rsi14"]
COSTO = 0.001            # 0.10% por compra y por venta (supuesto académico del PDF)
CAPITAL_INICIAL = 100_000.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_MODELOS = os.path.join(BASE_DIR, "modelos")
DIR_RESULTADOS = os.path.join(BASE_DIR, "resultados")
RUTA_PORTAFOLIO = os.path.join(BASE_DIR, "portafolio.json")

# Pares con base USD: desde una cuenta en dólares su valor es constante (solo análisis)
SOLO_ANALISIS = {"MXN=X", "JPY=X", "CAD=X", "CHF=X"}
# Ticker de Yahoo que da cuántas unidades de cada moneda vale 1 USD
FX_A_USD = {"MXN": "MXN=X", "JPY": "JPY=X", "CAD": "CAD=X", "CHF": "CHF=X"}


def nombre_archivo(ticker):
    return ticker.replace("=", "_").replace("^", "")


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def cargar_ficha():
    ruta = os.path.join(DIR_RESULTADOS, "ficha_activos.csv")
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            "No se encontró resultados/ficha_activos.csv. Descomprime paquete_app.zip dentro de la carpeta de la app.")
    return pd.read_csv(ruta)


def _limpiar(df):
    df = df.rename(columns={"Open": "apertura", "High": "maximo", "Low": "minimo",
                            "Close": "cierre", "Volume": "volumen"})
    df = df[[c for c in ["apertura", "maximo", "minimo", "cierre", "volumen"] if c in df.columns]].copy()
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "fecha"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df["cierre"].notna() & (df["cierre"] > 0)]
    if "volumen" in df.columns and (df["volumen"].fillna(0) == 0).mean() > 0.9:
        df = df.drop(columns="volumen")             # divisas: Yahoo no da volumen real
    return df


def descargar_precios(tickers, periodo="2y"):
    """Descarga velas diarias ajustadas (como en el Notebook 1). Devuelve {ticker: DataFrame}."""
    crudo = yf.download(list(tickers), period=periodo, interval="1d", auto_adjust=True,
                        group_by="ticker", progress=False, threads=True)
    datos = {}
    if crudo is None or crudo.empty:
        return datos
    for t in tickers:
        try:
            if isinstance(crudo.columns, pd.MultiIndex):
                if t not in crudo.columns.get_level_values(0):
                    continue
                df = crudo[t].copy()
            else:
                df = crudo.copy()
        except KeyError:
            continue
        df = _limpiar(df)
        if len(df) > 0:
            datos[t] = df
    return datos


# ---------------------------------------------------------------------------
# Indicadores técnicos (catálogo de la sección 7 del PDF), solo con historia hasta t
# ---------------------------------------------------------------------------
def calcular_indicadores(df):
    c = df["cierre"]
    ind = pd.DataFrame(index=df.index)
    ind["rend"] = c.pct_change()
    ind["sma20"] = c.rolling(20).mean()
    ind["dist_sma20"] = c / ind["sma20"] - 1
    ind["vol20"] = ind["rend"].rolling(20).std(ddof=1)
    cambio = c.diff()
    gan = cambio.clip(lower=0).rolling(14).mean()
    per = (-cambio).clip(lower=0).rolling(14).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100 - 100 / (1 + gan / per)
    rsi = rsi.where(per != 0, np.where(gan > 0, 100.0, 50.0))
    ind["rsi14"] = rsi.where(gan.notna() & per.notna())
    desv20 = c.rolling(20).std(ddof=1)
    ind["bb_sup"] = ind["sma20"] + 2 * desv20
    ind["bb_inf"] = ind["sma20"] - 2 * desv20
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    ind["macd"] = ema12 - ema26
    ind["macd_senal"] = ind["macd"].ewm(span=9, adjust=False).mean()
    ind["macd_hist"] = ind["macd"] - ind["macd_senal"]
    if {"maximo", "minimo"}.issubset(df.columns):
        previo = c.shift(1)
        rango = pd.concat([df["maximo"] - df["minimo"], (df["maximo"] - previo).abs(),
                           (df["minimo"] - previo).abs()], axis=1).max(axis=1)
        ind["atr14"] = rango.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()   # suavizado de Wilder
    return ind


# ---------------------------------------------------------------------------
# Modelos y pronóstico del siguiente día
# ---------------------------------------------------------------------------
_cache_modelos = {}


def tf_disponible():
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


def cargar_modelo(ticker):
    if ticker in _cache_modelos:
        return _cache_modelos[ticker]
    import tensorflow as tf
    ruta_cfg = os.path.join(DIR_MODELOS, f"{nombre_archivo(ticker)}_config.joblib")
    ruta_mod = os.path.join(DIR_MODELOS, f"{nombre_archivo(ticker)}.keras")
    cfg = joblib.load(ruta_cfg)
    modelo = tf.keras.models.load_model(ruta_mod, compile=False)
    _cache_modelos[ticker] = (cfg, modelo)
    return cfg, modelo


def _variables_ultima_fila(df, ind):
    """Rezagos de rendimiento e indicadores en t (última fecha) y en t−k."""
    fila = {f"r_lag{k}": ind["rend"].iloc[-1 - k] for k in range(VENTANA)}
    for i in IND:
        for k in range(VENTANA):
            fila[f"{i}_lag{k}"] = ind[i].iloc[-1 - k]
    fila["cierre"] = df["cierre"].iloc[-1]
    fila["sma20"] = ind["sma20"].iloc[-1]
    return pd.DataFrame([fila])


def _senal(estimado, cierre, sma20, umbral, regla):
    compra = estimado > umbral
    if str(regla).startswith("D"):
        compra = compra and (cierre > sma20)
    return "comprar" if compra else "efectivo"


def pronostico_archivo(ticker):
    ruta = os.path.join(DIR_RESULTADOS, "prediccion_futura.csv")
    if not os.path.exists(ruta):
        return None
    pf = pd.read_csv(ruta)
    fila = pf[pf.ticker == ticker]
    if fila.empty:
        return None
    f = fila.iloc[0]
    return {"ticker": ticker, "fuente": "Notebook 5 (prediccion_futura.csv)", "fecha_corte": str(f["fecha_corte"]),
            "proximo_periodo": str(f["proximo_periodo"]), "cierre": float(f["cierre_corte"]),
            "estimado_dl": float(f["rendimiento_estimado"]), "estimado_lineal": np.nan,
            "senal": f["senal"], "precio_estimado": float(f["precio_estimado"]),
            "red": "", "version": "", "regla": "", "umbral": np.nan}


def pronosticar(ticker, df, mercado):
    """Pronóstico del rendimiento del siguiente periodo con el modelo final del Notebook 5."""
    ind = calcular_indicadores(df)
    try:
        cfg, modelo = cargar_modelo(ticker)
    except (ImportError, FileNotFoundError, OSError, ValueError) as e:
        respaldo = pronostico_archivo(ticker)
        if respaldo:
            respaldo["aviso"] = f"No se pudo cargar el modelo ({type(e).__name__}); se muestra el pronóstico guardado."
        return respaldo
    X_df = _variables_ultima_fila(df, ind)
    if not np.isfinite(X_df[cfg["variables"]].to_numpy()).all():
        return None
    X = cfg["esc_x"].transform(X_df[cfg["variables"]].to_numpy())
    if cfg["red"] == "LSTM":
        X = X[:, ::-1].reshape(1, VENTANA, 1).copy() if cfg["version"] == "A" else X.reshape(1, VENTANA, -1)
    estimado = float(cfg["esc_y"].inverse_transform(modelo.predict(X, verbose=0).reshape(-1, 1))[0, 0])
    lin = float(cfg["lineal_modelo"].predict(cfg["lineal_esc_x"].transform(X_df[[f"r_lag{k}" for k in range(VENTANA)]].to_numpy()))[0])
    fecha_corte = df.index[-1]
    proximo = fecha_corte + (pd.Timedelta(days=1) if mercado == "Criptomonedas" else pd.offsets.BDay(1))
    cierre, sma20 = float(X_df["cierre"].iloc[0]), float(X_df["sma20"].iloc[0])
    return {"ticker": ticker, "fuente": "modelo en vivo", "fecha_corte": str(fecha_corte.date()),
            "proximo_periodo": str(proximo.date()), "cierre": cierre, "sma20": sma20,
            "estimado_dl": estimado, "estimado_lineal": lin,
            "senal": _senal(estimado, cierre, sma20, cfg["umbral"], cfg["regla"]),
            "senal_lineal": _senal(lin, cierre, sma20, cfg["lineal_umbral"], cfg["lineal_regla"]),
            "precio_estimado": cierre * (1 + estimado), "red": cfg["red"], "version": cfg["version"],
            "neuronas": cfg.get("neuronas"), "regla": cfg["regla"], "umbral": cfg["umbral"]}


# ---------------------------------------------------------------------------
# Resultados del proyecto (Notebook 5)
# ---------------------------------------------------------------------------
def leer_resultado(nombre):
    ruta = os.path.join(DIR_RESULTADOS, nombre)
    return pd.read_csv(ruta) if os.path.exists(ruta) else None


# ---------------------------------------------------------------------------
# Portafolio simulado (cuenta en USD, sin cortos ni apalancamiento, 0.10% por operación)
# ---------------------------------------------------------------------------
def portafolio_nuevo(capital=CAPITAL_INICIAL):
    return {"capital_inicial": capital, "efectivo": capital, "posiciones": {}, "operaciones": [],
            "creado": datetime.now().strftime("%Y-%m-%d %H:%M")}


def cargar_portafolio():
    if os.path.exists(RUTA_PORTAFOLIO):
        try:
            with open(RUTA_PORTAFOLIO) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return portafolio_nuevo()


def guardar_portafolio(port):
    try:
        with open(RUTA_PORTAFOLIO, "w") as f:
            json.dump(port, f, indent=2)
    except OSError:
        pass   # en Streamlit Cloud el disco puede ser de solo lectura; queda en la sesión


def tasas_usd(precios):
    """Unidades de cada moneda por 1 USD, con el último cierre disponible."""
    tasas = {"USD": 1.0}
    for moneda, t in FX_A_USD.items():
        if t in precios and len(precios[t]):
            tasas[moneda] = float(precios[t]["cierre"].iloc[-1])
    return tasas


def precio_usd(precio, moneda, tasas):
    if moneda not in tasas:
        raise ValueError(f"No hay tipo de cambio para {moneda}.")
    return precio / tasas[moneda]


def comprar(port, ticker, monto_usd, px_usd, costo=COSTO):
    if ticker in SOLO_ANALISIS:
        raise ValueError(f"{ticker} es un par con base USD: desde una cuenta en dólares no se puede operar.")
    if monto_usd <= 0:
        raise ValueError("El monto debe ser mayor que cero.")
    if monto_usd > port["efectivo"] + 1e-9:
        raise ValueError(f"Efectivo insuficiente: tienes ${port['efectivo']:,.2f} (sin apalancamiento).")
    unidades = monto_usd / (px_usd * (1 + costo))
    comision = unidades * px_usd * costo
    pos = port["posiciones"].get(ticker, {"unidades": 0.0, "costo_total": 0.0})
    pos["unidades"] += unidades
    pos["costo_total"] += monto_usd                                   # incluye la comisión
    port["posiciones"][ticker] = pos
    port["efectivo"] -= monto_usd
    port["operaciones"].append({"fecha": datetime.now().strftime("%Y-%m-%d %H:%M"), "ticker": ticker,
                                "tipo": "compra", "unidades": unidades, "precio_usd": px_usd,
                                "monto_usd": monto_usd, "comision_usd": comision, "resultado_usd": None})
    return unidades


def vender(port, ticker, fraccion, px_usd, costo=COSTO):
    pos = port["posiciones"].get(ticker)
    if not pos or pos["unidades"] <= 0:
        raise ValueError(f"No tienes posición en {ticker} (no se permiten ventas en corto).")
    if not 0 < fraccion <= 1:
        raise ValueError("La fracción a vender debe estar entre 0 y 100%.")
    unidades = pos["unidades"] * fraccion
    bruto = unidades * px_usd
    comision = bruto * costo
    neto = bruto - comision
    costo_vendido = pos["costo_total"] * fraccion
    resultado = neto - costo_vendido
    pos["unidades"] -= unidades
    pos["costo_total"] -= costo_vendido
    if pos["unidades"] <= 1e-12:
        del port["posiciones"][ticker]
    port["efectivo"] += neto
    port["operaciones"].append({"fecha": datetime.now().strftime("%Y-%m-%d %H:%M"), "ticker": ticker,
                                "tipo": "venta", "unidades": unidades, "precio_usd": px_usd,
                                "monto_usd": neto, "comision_usd": comision, "resultado_usd": resultado})
    return neto


def valorar(port, precios_usd_actuales):
    filas = []
    for t, pos in port["posiciones"].items():
        px = precios_usd_actuales.get(t, np.nan)
        valor = pos["unidades"] * px
        filas.append({"ticker": t, "unidades": pos["unidades"], "precio_usd": px, "valor_usd": valor,
                      "costo_usd": pos["costo_total"], "ganancia_usd": valor - pos["costo_total"],
                      "ganancia_%": (valor / pos["costo_total"] - 1) * 100 if pos["costo_total"] else np.nan})
    tabla = pd.DataFrame(filas, columns=["ticker", "unidades", "precio_usd", "valor_usd", "costo_usd",
                                         "ganancia_usd", "ganancia_%"])
    invertido = float(np.nansum(tabla["valor_usd"])) if len(tabla) else 0.0
    patrimonio = port["efectivo"] + invertido
    ops = [o for o in port["operaciones"] if o["tipo"] == "venta"]
    resumen = {"patrimonio": patrimonio, "efectivo": port["efectivo"], "invertido": invertido,
               "rendimiento": patrimonio / port["capital_inicial"] - 1,
               "comisiones": sum(o["comision_usd"] for o in port["operaciones"]),
               "operaciones_cerradas": len(ops),
               "win_rate": (np.mean([o["resultado_usd"] > 0 for o in ops]) if ops else np.nan)}
    return tabla, resumen


def ejecutar_senales(port, pronosticos, precios_usd_actuales, monto_por_activo):
    """Sigue las señales del modelo: vende posiciones con señal 'efectivo' y compra las de 'comprar'."""
    acciones = []
    for t, p in pronosticos.items():
        if p is None or t in SOLO_ANALISIS or t not in precios_usd_actuales:
            continue
        px = precios_usd_actuales[t]
        tiene = t in port["posiciones"]
        if p["senal"] == "efectivo" and tiene:
            vender(port, t, 1.0, px)
            acciones.append(f"Venta de {t}")
    for t, p in pronosticos.items():
        if p is None or t in SOLO_ANALISIS or t not in precios_usd_actuales:
            continue
        if p["senal"] == "comprar" and t not in port["posiciones"]:
            monto = min(monto_por_activo, port["efectivo"])
            if monto < 1:
                acciones.append(f"Sin efectivo para comprar {t}")
                continue
            comprar(port, t, monto, precios_usd_actuales[t])
            acciones.append(f"Compra de {t} por ${monto:,.2f}")
    return acciones
