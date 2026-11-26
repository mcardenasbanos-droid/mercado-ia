import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import pytest
import riesgo as rg
import agentes as ag
import mexc_bot as mb


def test_wilson_y_kelly():
    lo, hi = rg.wilson(8, 10)
    assert lo == pytest.approx(0.490, abs=0.002) and hi == pytest.approx(0.943, abs=0.002)
    assert rg.kelly_fraccional(0.6, 1.0) == pytest.approx(0.1)          # (0.6-0.4)/1 · ½
    assert rg.kelly_fraccional(0.3, 1.0) == 0.0


def test_indicadores_contra_referencia():
    v = pd.DataFrame({"maximo": [11.0] * 40, "minimo": [9.0] * 40, "cierre": [10.0] * 40})
    assert rg.atr_wilder(v).iloc[-1] == pytest.approx(2.0)                # rango constante = ATR
    assert rg.rsi_wilder(pd.Series(np.arange(1, 41, dtype=float))).iloc[-1] == pytest.approx(100.0)
    # RSI de Wilder con ganancias y pérdidas iguales alternadas → 50
    alt = pd.Series(np.cumsum([1, -1] * 40, dtype=float) + 100)
    assert rg.rsi_wilder(alt).iloc[-1] == pytest.approx(50, abs=3)


def test_niveles_y_tamano():
    n = rg.niveles_atr(100, 1, k_sl=1.5, k_tp=2.5)
    assert n["sl"] == 98.5 and n["tp"] == 102.5
    assert n["rr_neto"] == pytest.approx((2.5 - 0.3) / (1.5 + 0.3))
    monto, regla = rg.tamano_posicion(50_000, 50_000, 100, 99, 0.005, 2_500)
    assert monto == 2_500 and "riesgo fijo" in regla                     # 250 / 1.3% = 19,230 → tope 2,500
    monto, _ = rg.tamano_posicion(50_000, 50_000, 100, 99, 0.005, 100_000)
    assert monto == pytest.approx(250 / 0.013)
    ventas = [10] * 20 + [-8] * 15
    _, regla = rg.tamano_posicion(50_000, 50_000, 100, 99, 0.005, 2_500, ventas, 1.2)
    assert regla.startswith("Kelly")


def test_metricas():
    m = rg.metricas([10, -5, 20, -5], [0.01, -0.005, 0.02, -0.005], [100, 110, 105, 125, 120], 100)
    assert m["win_rate"] == 0.5 and m["profit_factor"] == pytest.approx(3.0)
    assert m["expectativa"] == pytest.approx(5.0) and m["max_drawdown"] == pytest.approx(105 / 110 - 1)
    assert m["acierto_equilibrio"] == pytest.approx(1 / (1 + 3.0))


def test_bitacora_detecta_alteraciones():
    b = []
    for i in range(5):
        rg.bitacora_agregar(b, {"tipo": "compra", "i": i})
    assert rg.bitacora_verificar(b) == (True, None)
    b[2]["i"] = 99
    assert rg.bitacora_verificar(b) == (False, 3)


def test_agentes_salida_valida_y_abstencion():
    pesos = ag.pesos_desde_auditoria({})
    d = ag.decidir_entrada(None, None, 100, pesos, rg.COSTO_IDA_VUELTA)
    assert d["decision"] == "abstenerse"
    ctx = {"rsi": 28, "macd_hist": 0.1, "sma20": 100.5, "atr": 0.5, "cierre": 100}
    d = ag.decidir_entrada({"senal": "comprar", "estimado": 0.006, "umbral": 0.002}, ctx, 100, pesos, rg.COSTO_IDA_VUELTA)
    assert d["decision"] == "comprar" and 0.55 <= d["probabilidad"] <= 1 and len(d["votos"]) == 4
    assert ag.validar_salida({"decision": "vender"})["decision"] == "abstenerse"


def test_pesos_aprenden_con_piso():
    aud = {a: {"votos": 20, "aciertos": 2} for a in ag.AGENTES}
    aud["tecnico"] = {"votos": 20, "aciertos": 18}
    w = ag.pesos_desde_auditoria(aud)
    assert w["tecnico"] > w["modelo_dl"] >= ag.PISO_PESO


def test_veto_y_kill_switch():
    bot = mb.bot_nuevo({"BTCUSDT": 100.0}, 2500, 0.002, 0.005)
    bot["operaciones"] = [{"tipo": "venta", "estado": "Ejecutada", "fecha": mb.ahora().strftime("%Y-%m-%d %H:%M:%S"),
                           "ganancia_usdt": -1200.0}]
    v = rg.veto_riesgo(bot, 48_800, mb.ahora().strftime("%Y-%m-%d"))
    assert v.startswith("pérdida del día") and rg.debe_activar_kill(v)
    bot["kill"] = True
    assert rg.veto_riesgo(bot, 50_000, "x").startswith("kill switch")


def test_bot_pro_ciclo_completo(monkeypatch):
    est = mb.estado_vacio()
    est["bot"] = mb.bot_nuevo({"SOLUSDT": 100.0}, 2_500, 0.002, 0.005)
    ctx = {"SOLUSDT": {"rsi": 28, "macd_hist": 0.1, "sma20": 100.2, "atr": 0.8, "cierre": 99.7}}
    ia = {"SOLUSDT": {"senal": "comprar", "estimado": 0.006, "umbral": 0.002}}
    assert mb.bot_actualizar(est, {"SOLUSDT": 100.0}, contexto=ctx, ia=ia) == []
    compra = mb.bot_actualizar(est, {"SOLUSDT": 99.7}, contexto=ctx, ia=ia)
    assert len(compra) == 1 and compra[0]["tipo"] == "compra" and compra[0]["rr_neto"] >= 1
    a = est["bot"]["activos"]["SOLUSDT"]
    assert a["sl"] < 99.7 < a["tp"] and compra[0]["monto_usdt"] <= 2_500
    venta = mb.bot_actualizar(est, {"SOLUSDT": a["tp"] + 0.01}, contexto=ctx, ia=ia)
    assert venta[0]["tipo"] == "venta" and venta[0]["ganancia_usdt"] > 0 and "take profit" in venta[0]["motivo"]
    assert est["auditoria"]["modelo_dl"]["aciertos"] == 1
    assert rg.bitacora_verificar(est["bitacora"])[0] and [e["tipo"] for e in est["bitacora"]] == ["compra", "venta"]
    r = mb.resumen(est["bot"])
    assert r["metricas"]["operaciones"] == 1 and r["metricas"]["win_rate"] == 1.0


def test_ia_en_contra_no_compra_y_queda_en_bitacora():
    est = mb.estado_vacio()
    est["bot"] = mb.bot_nuevo({"SOLUSDT": 100.0}, 2_500, 0.002, 0.005)
    ctx = {"SOLUSDT": {"rsi": 72, "macd_hist": -0.1, "sma20": 103, "atr": 0.8, "cierre": 99.7}}
    ia = {"SOLUSDT": {"senal": "efectivo", "estimado": -0.01, "umbral": 0.002}}
    assert mb.bot_actualizar(est, {"SOLUSDT": 99.0}, contexto=ctx, ia=ia) == []
    assert est["bitacora"][-1]["tipo"] == "ia_descarta"
    mb.bot_actualizar(est, {"SOLUSDT": 98.9}, contexto=ctx, ia=ia)
    assert len(est["bitacora"]) == 1                                    # no repite el mismo descarte


def test_salida_por_tiempo_maximo():
    est = mb.estado_vacio()
    est["bot"] = mb.bot_nuevo({"SOLUSDT": 100.0}, 2_500, 0.002, 0.005, tiempo_max_min=0)
    ctx = {"SOLUSDT": {"rsi": 28, "macd_hist": 0.1, "sma20": 100.2, "atr": 0.8, "cierre": 99.7}}
    ia = {"SOLUSDT": {"senal": "comprar", "estimado": 0.006, "umbral": 0.002}}
    mb.bot_actualizar(est, {"SOLUSDT": 99.7}, contexto=ctx, ia=ia)
    venta = mb.bot_actualizar(est, {"SOLUSDT": 99.75}, contexto=ctx, ia=ia)
    assert venta and "tiempo máximo" in venta[0]["motivo"]


def test_calibracion_polymarket():
    import polymarket_bot as pb
    ap = [{"prob_modelo": 0.7, "estado": "ganada"}, {"prob_modelo": 0.7, "estado": "perdida"},
          {"prob_modelo": 0.55, "estado": "ganada"}]
    c = pb.calibracion(ap)
    assert c["n"] == 3 and c["brier"] == pytest.approx(((0.3) ** 2 + 0.7 ** 2 + 0.45 ** 2) / 3)
    assert {f["rango"] for f in c["tabla"]} == {"50%–60%", "70%–80%"}


def test_excel_historial():
    import io
    from openpyxl import load_workbook
    import exportar
    est = mb.estado_vacio()
    est["bot"] = mb.bot_nuevo({"SOLUSDT": 100.0}, 2_500, 0.002, 0.005)
    ctx = {"SOLUSDT": {"rsi": 28, "macd_hist": 0.1, "sma20": 100.2, "atr": 0.8, "cierre": 99.7}}
    ia = {"SOLUSDT": {"senal": "comprar", "estimado": 0.006, "umbral": 0.002}}
    mb.bot_actualizar(est, {"SOLUSDT": 99.7}, contexto=ctx, ia=ia)
    tp = est["bot"]["activos"]["SOLUSDT"]["tp"]
    mb.bot_actualizar(est, {"SOLUSDT": tp + 0.01}, contexto=ctx, ia=ia)
    wb = load_workbook(io.BytesIO(exportar.excel_historial(est["historial"][::-1], mb.resumen(est["bot"]), est["bitacora"])))
    assert wb.sheetnames == ["Historial", "Resumen", "Bitácora"]
    h = wb["Historial"]
    assert h["A1"].value == "Fecha" and h.max_row == 3
    titulos = [c.value for c in h[1]]
    pnl = h.cell(row=2, column=titulos.index("PnL (USDT)") + 1).value
    assert pnl > 0 and wb["Bitácora"].max_row == 3
