"""
exportar.py — Excel (.xlsx) del bot MEXC: historial de compras y ventas, resumen con métricas y bitácora.
Necesita openpyxl (está en requirements.txt).
"""
import io
import math

import pandas as pd

COLUMNAS = [  # (campo, título en Excel, formato de número)
    ("fecha", "Fecha", None), ("par", "Par", None), ("tipo", "Lado", None), ("precio", "Precio", "#,##0.000000"),
    ("unidades", "Cantidad", "#,##0.000000"), ("monto_usdt", "Monto (USDT)", "#,##0.00"),
    ("comision_usdt", "Comisión (USDT)", "#,##0.0000"), ("ganancia_usdt", "PnL (USDT)", "+#,##0.00;-#,##0.00"),
    ("ganancia_%", "PnL %", "+0.00;-0.00"), ("sl", "SL", "#,##0.000000"), ("tp", "TP", "#,##0.000000"),
    ("rr_neto", "RR neto", "0.00"), ("prob_ia", "Prob. IA", "0%"), ("tamano", "Regla de tamaño", None),
    ("duracion_min", "Duración (min)", "0.0"), ("motivo", "Motivo", None), ("estado", "Estado", None),
    ("modo", "Modo", None), ("fuente", "Fuente", None), ("order_id", "Orden MEXC", None), ("bot_id", "Sesión del bot", None),
]


def _limpio(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def excel_historial(historial, resumen=None, bitacora=None):
    """Devuelve los bytes de un .xlsx con 3 hojas: Historial, Resumen y Bitácora."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    encabezado = PatternFill("solid", fgColor="0B0E11")
    blanco = Font(bold=True, color="EAECEF")
    verde, rojo = Font(color="0B8A5A"), Font(color="C8283D")

    ws = wb.active
    ws.title = "Historial"
    df = pd.DataFrame(historial)
    cols = [c for c in COLUMNAS if c[0] in df.columns]
    ws.append([t for _, t, _ in cols])
    for reg in df.to_dict("records"):
        ws.append([_limpio(reg.get(c)) for c, _, _ in cols])
    for j, (c, t, fmt) in enumerate(cols, start=1):
        celda = ws.cell(row=1, column=j)
        celda.fill, celda.font = encabezado, blanco
        celda.alignment = Alignment(horizontal="center")
        ancho = max([len(t)] + [len(str(ws.cell(row=i, column=j).value or "")) for i in range(2, min(ws.max_row, 200) + 1)])
        ws.column_dimensions[get_column_letter(j)].width = min(max(10, ancho + 2), 60)
        for i in range(2, ws.max_row + 1):
            cel = ws.cell(row=i, column=j)
            if fmt:
                cel.number_format = fmt
            if c in ("ganancia_usdt", "ganancia_%") and isinstance(cel.value, (int, float)):
                cel.font = verde if cel.value >= 0 else rojo
            if c == "tipo":
                cel.font = verde if cel.value == "compra" else rojo
    ws.freeze_panes = "A2"
    if ws.max_row > 1:
        ws.auto_filter.ref = ws.dimensions

    hr = wb.create_sheet("Resumen")
    hr.append(["Concepto", "Valor"])
    if resumen:
        m = resumen.get("metricas", {})
        filas = [("Capital inicial (USDT)", resumen.get("capital")), ("Equity (USDT)", resumen.get("patrimonio")),
                 ("PnL realizado (USDT)", resumen.get("ganancia_realizada")),
                 ("PnL no realizado (USDT)", resumen.get("ganancia_no_realizada")),
                 ("Rendimiento", resumen.get("rendimiento")), ("Comisiones (USDT)", resumen.get("comisiones")),
                 ("Compras", resumen.get("compras")), ("Ventas", resumen.get("ventas")),
                 ("Win rate", m.get("win_rate")), ("Win rate · IC 95% bajo (Wilson)", m.get("wilson_bajo")),
                 ("Win rate · IC 95% alto (Wilson)", m.get("wilson_alto")),
                 ("Expectativa por operación (USDT)", m.get("expectativa")), ("Profit factor", m.get("profit_factor")),
                 ("RR neto", m.get("rr_neto")), ("Acierto de equilibrio", m.get("acierto_equilibrio")),
                 ("Drawdown máximo", m.get("max_drawdown")), ("Sharpe por operación", m.get("sharpe_operaciones"))]
        for k, v in filas:
            hr.append([k, _limpio(v)])
    else:
        hr.append(["Sin bot activo", None])
    for i in range(2, hr.max_row + 1):
        k = hr.cell(row=i, column=1).value or ""
        hr.cell(row=i, column=2).number_format = ("0.00%" if any(x in k for x in ("Win", "Rendimiento", "equilibrio", "Drawdown"))
                                                   else "#,##0.00")
    for c in hr[1]:
        c.fill, c.font = encabezado, blanco
    hr.column_dimensions["A"].width, hr.column_dimensions["B"].width = 36, 18

    hb = wb.create_sheet("Bitácora")
    hb.append(["#", "Fecha", "Evento", "Par", "Detalle", "Hash previo", "Hash"])
    for e in bitacora or []:
        detalle = e.get("motivo") or (e.get("decision") or {}).get("razon") or ""
        hb.append([e.get("n"), e.get("fecha"), e.get("tipo"), e.get("par"), detalle, e.get("hash_previo"), e.get("hash")])
    for c in hb[1]:
        c.fill, c.font = encabezado, blanco
    for col, w in zip("ABCDEFG", (6, 20, 16, 12, 70, 20, 20)):
        hb.column_dimensions[col].width = w
    hb.freeze_panes = "A2"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
