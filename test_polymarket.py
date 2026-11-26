"""Pruebas del bot de Polymarket. Usan clientes FALSOS: ninguna prueba toca Polymarket ni dinero real.

Correr:  python -m pytest tests -q
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import polymarket_bot as pb       # noqa: E402
import polymarket_real as pr      # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------- nombres de mercado
def test_slugs_con_y_sin_anio():
    inicio = pd.Timestamp("2026-11-26 09:00", tz=pb.ZONA_ET)
    assert pb.slugs_candidatos("bitcoin-up-or-down", inicio) == [
        "bitcoin-up-or-down-november-26-2026-9am-et", "bitcoin-up-or-down-november-26-9am-et"]


@pytest.mark.parametrize("hora,etiqueta", [(0, "12am"), (11, "11am"), (12, "12pm"), (23, "11pm")])
def test_etiqueta_de_hora(hora, etiqueta):
    inicio = pd.Timestamp(f"2026-03-01 {hora:02d}:00", tz=pb.ZONA_ET)
    assert pb.slugs_candidatos("x", inicio)[1].endswith(f"-{etiqueta}-et")


# ----------------------------------------------------------------- probabilidad y decisión
def test_probabilidad_simetrica_sin_deriva():
    assert pb.probabilidad_up(100, 100, 1.0, 0.0, 0.03) == pytest.approx(0.5)


def test_probabilidad_sube_con_precio_arriba_y_deriva_positiva():
    base = pb.probabilidad_up(100, 100, 0.5, 0.0, 0.03)
    assert pb.probabilidad_up(100.5, 100, 0.5, 0.0, 0.03) > base
    assert pb.probabilidad_up(100, 100, 0.5, 0.02, 0.03) > base


def test_probabilidad_al_final_de_la_hora_es_cierta():
    assert pb.probabilidad_up(101, 100, 0.0, 0.0, 0.03) == 1.0
    assert pb.probabilidad_up(99, 100, 0.0, 0.0, 0.03) == 0.0


def test_decidir_respeta_el_margen():
    assert pb.decidir(0.55, 0.53, 0.48, 0.03)[0] is None          # ventaja de 2% < 3%
    assert pb.decidir(0.60, 0.53, 0.48, 0.03)[0] == "Up"
    assert pb.decidir(0.40, 0.55, 0.50, 0.03)[0] == "Down"


def test_costo_compra_recorre_el_libro():
    r = pb.costo_compra([(0.5, 10), (0.6, 100)], 11)             # 10 acc a 0.5 = 5 USD, luego 6 USD a 0.6
    assert r["acciones"] == pytest.approx(20)
    assert pb.costo_compra([(0.5, 2)], 100) is None               # sin liquidez suficiente


# ----------------------------------------------------------------- cartera paper
def _mercado(slug="bitcoin-up-or-down-november-26-2026-9am-et", fin_horas=1):
    inicio = pb.inicio_hora_et() - pd.Timedelta(hours=2)
    return {"activo": "BTC", "slug": slug, "pregunta": "BTC Up or Down", "inicio": inicio.isoformat(),
            "fin": (inicio + pd.Timedelta(hours=fin_horas)).isoformat(), "token_up": "T_UP", "token_down": "T_DN",
            "precio_up": 0.5, "precio_down": 0.5}


LIBRO = {"asks": [(0.5, 1000)], "bids": [(0.49, 1000)], "min_acciones": 5}


def test_apuesta_paper_ganada_y_perdida():
    est = pb.estado_vacio()
    est["bot"] = pb.bot_nuevo(["BTC"], 100, 0.03)
    a = pb.apostar(est, _mercado(), "Up", 0.6, 0.1, LIBRO)
    assert a["acciones"] == pytest.approx(200) and est["bot"]["efectivo"] == pytest.approx(9_900)
    assert pb.ya_aposto(est["bot"], a["slug"])                    # no se duplica la apuesta del mismo mercado
    oficial = {"closed": True, "outcomes": '["Up", "Down"]', "outcomePrices": '["1", "0"]'}
    pb.resolver(est, consultar_mercado=lambda _: oficial)
    assert a["estado"] == "ganada" and est["bot"]["efectivo"] == pytest.approx(10_100)

    b = pb.apostar(est, _mercado(slug="otro"), "Down", 0.4, 0.1, LIBRO)
    pb.resolver(est, consultar_mercado=lambda _: oficial)
    assert b["estado"] == "perdida" and b["ganancia"] == pytest.approx(-100)
    r = pb.resumen(est["bot"])
    assert r["ganadas"] == 1 and r["perdidas"] == 1 and r["pnl"] == pytest.approx(0)


def test_no_apuesta_debajo_del_minimo_de_acciones():
    est = pb.estado_vacio()
    est["bot"] = pb.bot_nuevo(["BTC"], 1, 0.03)
    assert pb.apostar(est, _mercado(), "Up", 0.6, 0.1, {"asks": [(0.5, 100)], "min_acciones": 5}) is None


def test_resolucion_de_respaldo_con_vela():
    est = pb.estado_vacio()
    est["bot"] = pb.bot_nuevo(["BTC"], 100, 0.03)
    a = pb.apostar(est, _mercado(), "Up", 0.6, 0.1, LIBRO)
    pb.resolver(est, consultar_mercado=lambda _: {"closed": False}, vela_respaldo=lambda _: (100, 101))
    assert a["estado"] == "ganada" and "MEXC" in a["resolucion"]


def test_no_se_resuelve_antes_de_tiempo():
    est = pb.estado_vacio()
    est["bot"] = pb.bot_nuevo(["BTC"], 100, 0.03)
    m = _mercado()
    m["fin"] = (pb.ahora_et() + pd.Timedelta(minutes=30)).isoformat()
    a = pb.apostar(est, m, "Up", 0.6, 0.1, LIBRO)
    pb.resolver(est, consultar_mercado=lambda _: {"closed": True, "outcomes": '["Up","Down"]', "outcomePrices": '["1","0"]'})
    assert a["estado"] == "abierta"


# ----------------------------------------------------------------- modo sombra / real (cliente falso)
class ClienteFalso:
    def __init__(self, ok=True):
        self.ok, self.ordenes = ok, []

    def estimate_market_price(self, **k):
        return 0.5

    def place_market_order(self, **k):
        self.ordenes.append(k)

    def __getattr__(self, n):
        raise AttributeError(n)


class RespuestaFalsa:
    def __init__(self, ok):
        self.ok, self.order_id, self.status, self.code, self.message = ok, "0xabc" if ok else None, "matched", 400, "no"


class ClienteFalsoOrdenes(ClienteFalso):
    def place_market_order(self, **k):
        self.ordenes.append(k)
        return RespuestaFalsa(self.ok)


class GeoFalso:
    def __init__(self, bloqueado=False):
        self.bloqueado = bloqueado

    def get(self, *a, **k):
        b = self.bloqueado

        class R:
            def json(self_):
                return {"blocked": b, "country": "MX"}
        return R()


def test_modo_sombra_no_envia_ordenes():
    cli = ClienteFalsoOrdenes()
    est = pb.estado_vacio()
    est["bot"] = pb.bot_nuevo(["BTC"], 5, 0.03, modo="sombra", tope_total=50)
    a = pb.apostar(est, _mercado(), "Up", 0.6, 0.1, LIBRO, ejecutor=pr.EjecutorPolymarket("pk", cliente=cli))
    assert a["estado"] == "sombra" and cli.ordenes == [] and est["bot"]["efectivo"] == 50


def test_modo_real_envia_una_orden_y_respeta_el_tope():
    cli = ClienteFalsoOrdenes()
    est = pb.estado_vacio()
    est["bot"] = pb.bot_nuevo(["BTC"], 5, 0.03, modo="real", tope_total=50)
    ej = pr.EjecutorPolymarket("pk", cliente=cli, sesion_geo=GeoFalso())
    a = pb.apostar(est, _mercado(), "Up", 0.6, 0.1, LIBRO, ejecutor=ej)
    assert a["estado"] == "abierta" and a["order_id"] == "0xabc" and len(cli.ordenes) == 1
    assert cli.ordenes[0]["amount"] == "5.00" and est["bot"]["efectivo"] == pytest.approx(45)


def test_orden_rechazada_o_geobloqueo_no_gasta():
    for ej in (pr.EjecutorPolymarket("pk", cliente=ClienteFalsoOrdenes(ok=False), sesion_geo=GeoFalso()),
               pr.EjecutorPolymarket("pk", cliente=ClienteFalsoOrdenes(), sesion_geo=GeoFalso(bloqueado=True))):
        est = pb.estado_vacio()
        est["bot"] = pb.bot_nuevo(["BTC"], 5, 0.03, modo="real", tope_total=50)
        a = pb.apostar(est, _mercado(), "Up", 0.6, 0.1, LIBRO, ejecutor=ej)
        assert a["estado"] == "rechazada" and est["bot"]["efectivo"] == 50


def test_tres_llaves():
    cfg = {"private_key": "pk", "modo_real": True}
    assert pr.llaves_modo_real(cfg, pr.FRASE_CONFIRMACION, 30)[0]
    assert not pr.llaves_modo_real(cfg, "otra cosa", 30)[0]
    assert not pr.llaves_modo_real(cfg, pr.FRASE_CONFIRMACION, 2)[0]
    assert not pr.llaves_modo_real({"private_key": "pk", "modo_real": False}, pr.FRASE_CONFIRMACION, 30)[0]


def test_leer_config_no_expone_la_llave(tmp_path):
    ruta = tmp_path / "secrets.toml"
    ruta.write_text('[polymarket]\nprivate_key = "0xSECRETA"\nmodo_real = "1"\n', encoding="utf-8")
    cfg = pr.leer_config(str(ruta))
    assert cfg["modo_real"] and cfg["private_key"] == "0xSECRETA" and cfg["problema"] is None
    vacio = pr.leer_config(str(tmp_path / "no_existe.toml"))
    assert "0xSECRETA" not in (vacio["problema"] or "")


def test_solo_un_modulo_envia_ordenes_reales():
    """Ningún archivo, salvo polymarket_real.py, puede llamar a place_market_order / place_limit_order."""
    for nombre in os.listdir(RAIZ):
        if nombre.endswith(".py") and nombre != "polymarket_real.py":
            texto = open(os.path.join(RAIZ, nombre), encoding="utf-8").read()
            assert not re.search(r"place_(market|limit)_order", texto), f"{nombre} no debe enviar órdenes"


def test_ciclo_apuesta_una_vez_por_mercado():
    est = pb.estado_vacio()
    est["bot"] = pb.bot_nuevo(["BTC"], 100, 0.03)
    m = _mercado()
    m["fin"] = (pb.ahora_et() + pd.Timedelta(minutes=30)).isoformat()
    libros = {"T_UP": {"asks": [(0.40, 1000)], "bids": [], "mejor_compra": 0.40, "mejor_venta": np.nan, "min_acciones": 5},
              "T_DN": {"asks": [(0.62, 1000)], "bids": [], "mejor_compra": 0.62, "mejor_venta": np.nan, "min_acciones": 5}}
    entradas = {"BTC": {"mercado": m, "precio_actual": 100.2, "apertura": 100, "rend_ia": 0.01, "vol_diaria": 0.03}}
    nuevas = pb.ciclo(est, entradas, obtener_libro=libros.get)
    assert len(nuevas) == 1 and nuevas[0]["lado"] == "Up"
    assert pb.ciclo(est, entradas, obtener_libro=libros.get) == []        # misma hora: no repite
    assert est["bot"]["ultima_decision"]["BTC"]["texto"].startswith("Apostar")


def test_ciclo_no_apuesta_en_los_ultimos_minutos():
    est = pb.estado_vacio()
    est["bot"] = pb.bot_nuevo(["BTC"], 100, 0.03)
    m = _mercado()
    m["fin"] = (pb.ahora_et() + pd.Timedelta(minutes=2)).isoformat()
    lib = {"asks": [(0.10, 1000)], "bids": [], "mejor_compra": 0.10, "mejor_venta": np.nan, "min_acciones": 5}
    entradas = {"BTC": {"mercado": m, "precio_actual": 101, "apertura": 100, "rend_ia": 0.0, "vol_diaria": 0.03}}
    assert pb.ciclo(est, entradas, obtener_libro=lambda _: lib) == []


# ----------------------------------------------------------------- avisos de WhatsApp
import notificaciones as wa  # noqa: E402


class SesionWA:
    def __init__(self, codigo=200, texto="Message queued. You will receive it in a few seconds."):
        self.codigo, self.texto, self.llamadas = codigo, texto, []

    def get(self, url, params=None, timeout=None):
        self.llamadas.append(params)
        c, t = self.codigo, self.texto

        class R:
            status_code, text = c, t
        return R()


def test_mensajes_long_short():
    bot = {"tope_venta": 0.005, "stop_loss": None}
    compra = {"tipo": "compra", "par": "BTCUSDT", "precio": 60000.0, "monto_usdt": 2500.0, "modo": "paper"}
    venta = {"tipo": "venta", "par": "BTCUSDT", "precio": 60300.0, "monto_usdt": 2510.0, "ganancia_usdt": 10.0,
             "ganancia_%": 0.4, "motivo": "tope de venta +0.50%", "modo": "paper"}
    assert "LONG · COMPRAR BTCUSDT" in wa.mensaje_mexc(compra, bot) and "60,300.00" in wa.mensaje_mexc(compra, bot)
    assert "CERRAR LONG" in wa.mensaje_mexc(venta, bot) and "+10.00" in wa.mensaje_mexc(venta, bot)
    ap = {"activo": "ETH", "lado": "Down", "precio": 0.45, "monto": 100.0, "acciones": 222.2, "prob_modelo": 0.6,
          "ventaja": 0.15, "modo": "paper"}
    assert "DOWN (≈ SHORT)" in wa.mensaje_polymarket(ap)


def test_envio_whatsapp_y_errores(tmp_path):
    ruta = tmp_path / "secrets.toml"
    ruta.write_text('[whatsapp]\ntelefono = "+52 55 1234 5678"\napikey = "1234567"\nactivo = "1"\n', encoding="utf-8")
    cfg = wa.leer_config(str(ruta))
    assert cfg["listo"] and cfg["telefono"] == "+525512345678"
    s = SesionWA()
    assert wa.enviar(cfg, "hola", sesion=s)[0] and s.llamadas[0]["phone"] == "+525512345678"
    assert not wa.enviar(cfg, "hola", sesion=SesionWA(203, "APIKey is invalid"))[0]
    apagado = wa.leer_config(str(tmp_path / "no.toml"))
    assert not apagado["listo"] and not wa.enviar(apagado, "x", sesion=s)[0]
    ruta.write_text('[whatsapp]\ntelefono = "+5215512345678"\napikey = "TU_APIKEY"\nactivo = "1"\n', encoding="utf-8")
    ejemplo = wa.leer_config(str(ruta))
    assert not ejemplo["listo"] and "TU_APIKEY" in ejemplo["problema"]


class _RespTw:
    def __init__(self, code, js): self.status_code, self._js, self.text = code, js, str(js)
    def json(self): return self._js


class SesionTw:
    def __init__(self, code=201, js=None): self.code, self.js, self.llamadas = code, js or {"sid": "SM1"}, []
    def post(self, url, data=None, auth=None, timeout=None):
        self.llamadas.append((url, data, auth)); return _RespTw(self.code, self.js)


def test_whatsapp_twilio(tmp_path):
    ruta = tmp_path / "secrets.toml"
    ruta.write_text('[whatsapp]\nproveedor = "twilio"\ntelefono = "+52 1 55 1234 5678"\n'
                    'twilio_sid = "AC' + "a" * 32 + '"\ntwilio_token = "' + "b" * 32 + '"\nactivo = "1"\n', encoding="utf-8")
    cfg = wa.leer_config(str(ruta))
    assert cfg["listo"], cfg["problema"]
    s = SesionTw()
    ok, _ = wa.enviar(cfg, "hola", sesion=s)
    url, datos, _ = s.llamadas[0]
    assert ok and datos["To"] == "whatsapp:+5215512345678" and datos["From"] == "whatsapp:+14155238886"
    ok, det = wa.enviar(cfg, "hola", sesion=SesionTw(400, {"code": 63015, "message": "not in sandbox"}))
    assert not ok and "join" in det and "b" * 32 not in det
    ruta.write_text('[whatsapp]\nproveedor = "twilio"\ntelefono = "+52155"\ntwilio_sid = "mx0abc"\n', encoding="utf-8")
    assert not wa.leer_config(str(ruta))["listo"]


class SesionNtfy:
    def __init__(self): self.llamadas = []
    def post(self, url, data=None, headers=None, timeout=None, **k):
        self.llamadas.append((url, data, headers)); return _RespTw(200, {})


def test_avisos_ntfy(tmp_path):
    ruta = tmp_path / "secrets.toml"
    ruta.write_text('[whatsapp]\nproveedor = "ntfy"\ntema = "investify-mariana-4821"\nactivo = "1"\n', encoding="utf-8")
    cfg = wa.leer_config(str(ruta))
    assert cfg["listo"], cfg["problema"]
    s = SesionNtfy()
    assert wa.enviar(cfg, "🟢 *LONG · COMPRAR SOLUSDT*\nPrecio 1", sesion=s)[0]
    url, data, headers = s.llamadas[0]
    assert url.endswith("/investify-mariana-4821") and b"*" not in data and "LONG" in data.decode()
    for v in headers.values():
        v.encode("latin-1")
    ruta.write_text('[whatsapp]\nproveedor = "ntfy"\ntema = "abc"\n', encoding="utf-8")
    assert not wa.leer_config(str(ruta))["listo"]


def test_recomendaciones_ia(tmp_path):
    pron = {"AAPL": {"senal": "comprar", "estimado_dl": 0.012, "umbral": 0.002, "cierre": 200, "sma20": 190,
                     "precio_estimado": 202.4, "proximo_periodo": "2026-11-27"},
            "TSLA": {"senal": "efectivo", "estimado_dl": -0.015, "umbral": 0.002, "cierre": 300, "sma20": 320,
                     "precio_estimado": 295.5, "proximo_periodo": "2026-11-27"},
            "MSFT": {"senal": "efectivo", "estimado_dl": 0.0005, "umbral": 0.002, "cierre": 400, "sma20": 390,
                     "precio_estimado": 400.2, "proximo_periodo": "2026-11-27"},
            "X": None}
    recs = wa.recomendaciones(pron, {"AAPL": "Apple"})
    assert [(r["ticker"], r["lado"]) for r in recs] == [("TSLA", "SHORT"), ("AAPL", "LONG")]
    msg = wa.mensaje_recomendaciones(recs)
    assert "LONG" in msg and "SHORT" in msg and "Apple" in msg
    ruta = str(tmp_path / "env.json")
    assert len(wa.recomendaciones_nuevas(recs, ruta)) == 2
    assert len(wa.recomendaciones_nuevas(recs, ruta)) == 2      # si no llegó, se vuelve a intentar
    wa.marcar_enviadas(recs, ruta)
    assert wa.recomendaciones_nuevas(recs, ruta) == []


class SesionTg:
    def __init__(self, updates): self.updates, self.envios = updates, []
    def get(self, url, timeout=None, **k):
        if "/getMe" in url: return _RespTw(200, {"ok": True, "result": {"username": "investify_prueba_bot"}})
        assert "/getUpdates" in url; return _RespTw(200, {"ok": True, "result": self.updates})
    def post(self, url, data=None, timeout=None, **k):
        self.envios.append((url, data)); return _RespTw(200, {"ok": True})


def test_avisos_telegram(tmp_path):
    ruta = tmp_path / "secrets.toml"
    tok = "123456789:" + "A" * 35
    ruta.write_text(f'[whatsapp]\nproveedor = "telegram"\ntoken = "{tok}"\nactivo = "1"\n', encoding="utf-8")
    cfg = wa.leer_config(str(ruta))
    assert cfg["listo"], cfg["problema"]
    ok, det = wa.enviar(cfg, "hola", sesion=SesionTg([]))
    assert not ok and "@investify_prueba_bot" in det and tok not in det
    s = SesionTg([{"message": {"chat": {"id": 555}}}])
    assert wa.enviar(cfg, "*LONG*", sesion=s)[0]
    assert s.envios[0][1] == {"chat_id": "555", "text": "LONG"} and cfg["chat_id"] == "555"
    cfg["chat_id"] = ""; import os; os.path.exists("telegram_chat.txt") and os.remove("telegram_chat.txt")
    s2 = SesionTg([{"my_chat_member": {"chat": {"id": 777}}}])
    assert wa.enviar(cfg, "x", sesion=s2)[0] and cfg["chat_id"] == "777"
    os.path.exists("telegram_chat.txt") and os.remove("telegram_chat.txt")
    ruta.write_text('[whatsapp]\nproveedor = "telegram"\ntoken = "mx0vgl123"\n', encoding="utf-8")
    assert not wa.leer_config(str(ruta))["listo"]


def test_avisos_separados(monkeypatch, tmp_path):
    import time
    enviados, marcados = [], []
    monkeypatch.setattr(wa, "enviar", lambda cfg, t, sesion=None: (enviados.append(t) or True, "Enviado"))
    monkeypatch.setattr(wa, "PAUSA_ENTRE_MENSAJES", 0)
    recs = wa.recomendaciones({"AAPL": {"senal": "comprar", "estimado_dl": 0.01, "cierre": 1, "sma20": 1,
                                        "precio_estimado": 1.01, "proximo_periodo": "d"},
                               "MSFT": {"senal": "comprar", "estimado_dl": 0.02, "cierre": 1, "sma20": 1,
                                        "precio_estimado": 1.02, "proximo_periodo": "d"}})
    wa.enviar_en_segundo_plano({}, [wa.mensaje_recomendacion(r) for r in recs], al_enviar=marcados.append)
    for _ in range(50):
        if len(marcados) == 2: break
        time.sleep(0.05)
    assert marcados == [0, 1] and len(enviados) == 2
    assert enviados[0].startswith("*🟢 LONG · COMPRAR MSFT*") and "AAPL" in enviados[1]


def test_config_desde_secrets_de_la_nube(tmp_path):
    tok = "123456789:" + "B" * 35
    cfg = wa.leer_config(str(tmp_path / "no_existe.toml"), {"proveedor": "telegram", "token": tok, "chat_id": "555"})
    assert cfg["listo"] and cfg["chat_id"] == "555"
    assert not wa.leer_config(str(tmp_path / "no_existe.toml"))["listo"]
