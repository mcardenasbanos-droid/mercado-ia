"""
polymarket_real.py — ÚNICO módulo que puede enviar órdenes REALES a Polymarket.

Reglas:
- La llave privada vive en .streamlit/secrets.toml ([polymarket] private_key = "...") o en la variable de
  entorno POLYMARKET_PRIVATE_KEY. Nunca se imprime, nunca se guarda en registros.
- El modo real necesita TRES llaves: modo_real = "1" en secrets.toml, la frase de confirmación escrita por ti
  y haber pasado la fase de sombra (HORAS_SOMBRA_MIN horas calculando apuestas sin enviarlas).
- Antes de cada orden real se revisa el geobloqueo de Polymarket.
- Usa el SDK oficial `polymarket-client` (pip install polymarket-client).
"""
import os

import requests

FRASE_CONFIRMACION = "APOSTAR CON DINERO REAL"
HORAS_SOMBRA_MIN = 24
URL_GEOBLOQUEO = "https://polymarket.com/api/geoblock"


class ErrorPolymarketReal(Exception):
    pass


def leer_config(ruta_secrets):
    """Devuelve {'private_key', 'modo_real', 'problema'} sin exponer la llave en mensajes."""
    import tomllib
    pk, modo, problema = os.environ.get("POLYMARKET_PRIVATE_KEY", ""), os.environ.get("POLYMARKET_MODO_REAL", ""), None
    if os.path.exists(ruta_secrets):
        try:
            with open(ruta_secrets, encoding="utf-8-sig") as f:
                sec = tomllib.loads(f.read()).get("polymarket", {})
            pk = str(sec.get("private_key", pk) or "").strip()
            modo = str(sec.get("modo_real", modo) or "").strip()
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
            problema = f"secrets.toml tiene un error de formato ({e})."
    if not pk and problema is None:
        problema = "No hay llave privada de Polymarket en .streamlit/secrets.toml (sección [polymarket])."
    return {"private_key": pk, "modo_real": modo == "1", "problema": problema}


def llaves_modo_real(config, confirmacion, horas_sombra):
    """Revisa las tres llaves. Devuelve (todas_listas, lista de lo que falta)."""
    faltan = []
    if not config.get("private_key"):
        faltan.append("llave privada en secrets.toml")
    if not config.get("modo_real"):
        faltan.append('modo_real = "1" en secrets.toml')
    if (confirmacion or "").strip().upper() != FRASE_CONFIRMACION:
        faltan.append(f"escribir la frase «{FRASE_CONFIRMACION}»")
    if horas_sombra < HORAS_SOMBRA_MIN:
        faltan.append(f"fase de sombra de {HORAS_SOMBRA_MIN} h (llevas {horas_sombra:.1f} h)")
    return not faltan, faltan


def revisar_geobloqueo(sesion=None):
    s = sesion or requests
    try:
        r = s.get(URL_GEOBLOQUEO, timeout=6)
        datos = r.json()
    except (requests.RequestException, ValueError) as e:
        raise ErrorPolymarketReal(f"No se pudo verificar el geobloqueo de Polymarket: {e}") from e
    if datos.get("blocked"):
        raise ErrorPolymarketReal(f"Polymarket no permite operar desde tu ubicación ({datos.get('country', '?')}).")
    return datos


class EjecutorPolymarket:
    """Compra acciones 'Up' o 'Down' con orden de mercado FAK.

    enviar=False → modo sombra: estima el precio con el SDK oficial pero NO envía la orden."""

    def __init__(self, private_key, cliente=None, sesion_geo=None):
        if not private_key and cliente is None:
            raise ErrorPolymarketReal("Falta la llave privada de Polymarket.")
        self._pk = private_key
        self._cliente = cliente
        self._ctx = None
        self._sesion_geo = sesion_geo

    def _conectar(self):
        if self._cliente is None:
            try:
                from polymarket import SecureClient
            except ImportError as e:
                raise ErrorPolymarketReal("Falta el SDK oficial: en la ventana negra escribe  pip install polymarket-client") from e
            self._ctx = SecureClient.create(private_key=self._pk)
            self._cliente = self._ctx.__enter__()
        return self._cliente

    def cerrar(self):
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx, self._cliente = None, None

    def comprar(self, token, monto, enviar):
        c = self._conectar()
        monto_txt = f"{monto:.2f}"
        estimado = c.estimate_market_price(asset_id=token, side="BUY", amount=monto_txt, order_type="FAK")
        precio = float(estimado)
        if precio <= 0 or precio >= 1:
            raise ErrorPolymarketReal(f"Precio estimado inválido ({precio}).")
        if not enviar:
            return {"order_id": None, "precio": precio, "acciones": monto / precio, "estado": "sombra"}
        revisar_geobloqueo(self._sesion_geo)
        r = c.place_market_order(asset_id=token, side="BUY", amount=monto_txt, max_price=estimado, order_type="FAK")
        if not getattr(r, "ok", False):
            raise ErrorPolymarketReal(f"Polymarket rechazó la orden ({getattr(r, 'code', '?')}): {getattr(r, 'message', '')}")
        return {"order_id": getattr(r, "order_id", None), "precio": precio, "acciones": monto / precio,
                "estado": getattr(r, "status", "")}
