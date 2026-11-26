"""
mexc_cuenta.py — Conexión PRIVADA a tu cuenta real de MEXC (Spot) con la API oficial v3.

Las llaves NUNCA se escriben en el código: se leen de .streamlit/secrets.toml (solo en tu computadora)
o de las variables de entorno MEXC_API_KEY y MEXC_SECRET_KEY.

Firma: HMAC-SHA256 con tu Secret Key sobre todos los parámetros (query string + cuerpo),
en minúsculas, y la Access Key en el encabezado X-MEXC-APIKEY.
"""
import hmac
import hashlib
import math
import time
from decimal import Decimal, ROUND_DOWN
from urllib.parse import urlencode

import requests

URL = "https://api.mexc.com"


class ErrorMEXC(Exception):
    def __init__(self, mensaje, codigo=None):
        super().__init__(mensaje)
        self.codigo = codigo


def _explicar(codigo, mensaje):
    pistas = {
        700001: "La llave API no es válida o se escribió mal.",
        700002: "La firma no coincide: revisa que la Secret Key esté completa y sin espacios.",
        700003: "La hora de tu computadora está desfasada. Sincroniza el reloj de Windows.",
        700006: "Tu IP no está autorizada para esta llave (revisa «Link IP Address» en MEXC).",
        700007: "La llave no tiene el permiso necesario (lectura u «Order Placing» de Spot).",
        30004: "Saldo insuficiente en tu cuenta Spot.",
        30005: "Saldo insuficiente en tu cuenta Spot.",
        30002: "El monto es menor al mínimo permitido por MEXC para ese par.",
        10007: "Ese par no está disponible para operar por API.",
    }
    return pistas.get(codigo, mensaje)


class ClienteMEXC:
    def __init__(self, api_key, secret_key, timeout=8):
        if not api_key or not secret_key:
            raise ErrorMEXC("Faltan las llaves de MEXC.")
        self.api_key = api_key.strip()
        self.secret = secret_key.strip()
        self.timeout = timeout
        self.sesion = requests.Session()
        self.sesion.headers.update({"X-MEXC-APIKEY": self.api_key, "Content-Type": "application/json"})
        self._desfase = 0
        self._reglas = {}

    # ------------------------------------------------------------------ base
    def _sincronizar(self):
        try:
            r = self.sesion.get(f"{URL}/api/v3/time", timeout=self.timeout)
            self._desfase = int(r.json()["serverTime"]) - int(time.time() * 1000)
        except (requests.RequestException, ValueError, KeyError):
            self._desfase = 0

    def _firmar(self, params):
        params = {k: v for k, v in params.items() if v is not None}
        params["recvWindow"] = 10000
        params["timestamp"] = int(time.time() * 1000) + self._desfase
        consulta = urlencode(params)
        firma = hmac.new(self.secret.encode(), consulta.encode(), hashlib.sha256).hexdigest()
        return f"{consulta}&signature={firma}"

    def _privado(self, metodo, ruta, params=None, reintento=True):
        consulta = self._firmar(params or {})
        try:
            r = self.sesion.request(metodo, f"{URL}{ruta}?{consulta}", timeout=self.timeout)
            datos = r.json()
        except requests.RequestException as e:
            raise ErrorMEXC(f"No se pudo conectar con MEXC: {e}") from e
        except ValueError as e:
            raise ErrorMEXC(f"MEXC respondió algo inesperado (HTTP {r.status_code}).") from e
        if isinstance(datos, dict) and datos.get("code") not in (None, 0, 200) and "msg" in datos:
            codigo = datos.get("code")
            if codigo == 700003 and reintento:            # reloj desfasado: sincronizar y reintentar una vez
                self._sincronizar()
                return self._privado(metodo, ruta, params, reintento=False)
            raise ErrorMEXC(f"MEXC ({codigo}): {_explicar(codigo, datos.get('msg'))}", codigo)
        if r.status_code >= 400:
            raise ErrorMEXC(f"MEXC respondió HTTP {r.status_code}: {str(datos)[:200]}")
        return datos

    # --------------------------------------------------------------- lectura
    def cuenta(self):
        """Saldos de la cuenta Spot: {activo: {'libre': x, 'bloqueado': y}} solo con saldo > 0."""
        datos = self._privado("GET", "/api/v3/account")
        saldos = {}
        for b in datos.get("balances", []):
            libre, bloqueado = float(b.get("free", 0)), float(b.get("locked", 0))
            if libre + bloqueado > 0:
                saldos[b["asset"]] = {"libre": libre, "bloqueado": bloqueado}
        return saldos

    def saldo(self, activo):
        return self.cuenta().get(activo, {"libre": 0.0})["libre"]

    def mis_operaciones(self, par, limite=50):
        return self._privado("GET", "/api/v3/myTrades", {"symbol": par, "limit": limite})

    def orden(self, par, order_id):
        return self._privado("GET", "/api/v3/order", {"symbol": par, "orderId": order_id})

    def reglas(self, par):
        if par not in self._reglas:
            try:
                r = self.sesion.get(f"{URL}/api/v3/exchangeInfo", params={"symbol": par}, timeout=self.timeout)
                info = r.json()["symbols"][0]
            except (requests.RequestException, ValueError, KeyError, IndexError) as e:
                raise ErrorMEXC(f"No se pudieron leer las reglas de {par}.") from e
            self._reglas[par] = info
        return self._reglas[par]

    def _ajustar_cantidad(self, par, cantidad):
        info = self.reglas(par)
        paso = info.get("baseSizePrecision")
        if paso not in (None, "", "0"):
            paso = Decimal(str(paso))
            return float((Decimal(str(cantidad)) / paso).to_integral_value(rounding=ROUND_DOWN) * paso)
        decimales = int(info.get("baseAssetPrecision", 8))
        factor = 10 ** decimales
        return math.floor(cantidad * factor) / factor

    # ---------------------------------------------------------------- órdenes
    def probar_orden(self, par="BTCUSDT", monto_usdt=5):
        """Valida permisos y firma con /api/v3/order/test. NO ejecuta ninguna operación."""
        return self._privado("POST", "/api/v3/order/test",
                             {"symbol": par, "side": "BUY", "type": "MARKET", "quoteOrderQty": f"{monto_usdt:.2f}"})

    def _resultado(self, par, respuesta):
        order_id = respuesta.get("orderId")
        detalle = respuesta
        for _ in range(5):                              # la orden de mercado se llena casi al instante
            try:
                detalle = self.orden(par, order_id)
                if float(detalle.get("executedQty", 0)) > 0:
                    break
            except ErrorMEXC:
                pass
            time.sleep(0.4)
        cantidad = float(detalle.get("executedQty", 0))
        total = float(detalle.get("cummulativeQuoteQty", 0))
        if cantidad <= 0:
            raise ErrorMEXC(f"La orden {order_id} se envió pero aún no aparece como ejecutada en MEXC.")
        return {"order_id": order_id, "cantidad": cantidad, "total_usdt": total, "precio": total / cantidad,
                "estado": detalle.get("status", "")}

    def comprar_mercado(self, par, monto_usdt):
        r = self._privado("POST", "/api/v3/order",
                          {"symbol": par, "side": "BUY", "type": "MARKET", "quoteOrderQty": f"{monto_usdt:.2f}"})
        return self._resultado(par, r)

    def vender_mercado(self, par, cantidad):
        activo = par.replace("USDT", "")
        disponible = self.saldo(activo)                 # la comisión pudo descontarse en la cripto
        cantidad = self._ajustar_cantidad(par, min(cantidad, disponible))
        if cantidad <= 0:
            raise ErrorMEXC(f"No hay {activo} disponible para vender.")
        r = self._privado("POST", "/api/v3/order",
                          {"symbol": par, "side": "SELL", "type": "MARKET", "quantity": f"{cantidad:f}"})
        return self._resultado(par, r)


def leer_llaves(ruta):
    """Lee las llaves de un archivo secrets.toml. Devuelve (api_key, secret_key, diagnóstico).

    Acepta [mexc] con api_key/secret_key (o access_key/secret) o las llaves sueltas.
    El diagnóstico nunca muestra el valor de las llaves."""
    import os
    import tomllib
    if not os.path.exists(ruta):
        carpeta = os.path.dirname(ruta)
        parecidos = [f for f in (os.listdir(carpeta) if os.path.isdir(carpeta) else []) if f.startswith("secrets")]
        return "", "", (f"No encontré el archivo en {ruta}."
                         + (f" Encontré: {', '.join(parecidos)}. Renómbralo exactamente a secrets.toml." if parecidos else ""))
    try:
        with open(ruta, encoding="utf-8-sig") as f:          # utf-8-sig tolera el BOM de Windows
            datos = tomllib.loads(f.read())
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        return "", "", (f"El archivo secrets.toml tiene un error de formato ({e}). Revisa que cada llave esté "
                         f'entre comillas rectas "así" y que la primera línea sea [mexc].')
    sec = datos.get("mexc", datos.get("MEXC", datos))
    api = sec.get("api_key") or sec.get("access_key") or sec.get("MEXC_API_KEY") or ""
    secreto = sec.get("secret_key") or sec.get("secret") or sec.get("MEXC_SECRET_KEY") or ""
    if api and secreto:
        return str(api).strip(), str(secreto).strip(), None
    return "", "", (f"El archivo existe pero no tiene api_key y secret_key dentro de [mexc]. "
                    f"Nombres que encontré: {', '.join(map(str, sec.keys())) or 'ninguno'}.")
