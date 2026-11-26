"""
notificaciones.py — Avisos por WhatsApp (CallMeBot, gratis para uso personal) cuando un bot compra o vende.

Configuración en .streamlit/secrets.toml (nunca en el código ni en GitHub):
    [whatsapp]
    telefono = "+5215512345678"
    apikey = "123456"
    activo = "1"
"""
import os
import threading
from collections import deque

import requests

URL_CALLMEBOT = "https://api.callmebot.com/whatsapp.php"
ULTIMOS = deque(maxlen=20)          # resultado de los últimos envíos (para mostrar en la app)


CLAVES_AVISO = {"token", "tema", "apikey", "twilio_sid", "proveedor"}


def buscar_seccion(todo):
    """Encuentra la configuración de avisos aunque la sección se llame [WhatsApp], [telegram], [avisos]…
    o aunque las llaves estén sueltas sin sección. Nunca toma [mexc] ni [polymarket]."""
    if not isinstance(todo, dict):
        return {}
    for nombre in ("whatsapp", "telegram", "avisos", "notificaciones"):
        for k, v in todo.items():
            if isinstance(v, dict) and k.strip().lower() == nombre:
                return dict(v)
    for k, v in todo.items():
        if isinstance(v, dict) and k.lower() not in ("mexc", "polymarket") and CLAVES_AVISO & {x.lower() for x in v}:
            return dict(v)
    sueltas = {k: v for k, v in todo.items() if not isinstance(v, dict)}
    return sueltas if CLAVES_AVISO & {x.lower() for x in sueltas} else {}


def _limpiar_tel(t):
    t = "".join(ch for ch in str(t or "") if ch.isdigit() or ch == "+")
    return ("+" + t) if t and not t.startswith("+") else t


def leer_config(ruta_secrets, seccion_nube=None):
    """Lee [whatsapp] del archivo secrets.toml o, si no existe, de los Secrets de Streamlit Cloud
    (seccion_nube). proveedor = "telegram", "ntfy", "callmebot" (por defecto) o "twilio". Nunca expone llaves."""
    import tomllib
    sec = {}
    cfg = {"problema": None}
    if os.path.exists(ruta_secrets):
        try:
            with open(ruta_secrets, encoding="utf-8-sig") as f:
                todo = tomllib.loads(f.read())
            sec = buscar_seccion(todo)
            cfg["secciones"] = sorted(k for k, v in todo.items() if isinstance(v, dict))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
            cfg["problema"] = f"secrets.toml tiene un error de formato ({e})."
    if not sec and seccion_nube:
        sec = buscar_seccion(dict(seccion_nube)) or {}
    if not sec and not os.path.exists(ruta_secrets) and not os.environ.get("WHATSAPP_PROVEEDOR"):
        cfg["problema"] = ("No hay archivo .streamlit/secrets.toml. Es normal en la página pública de Streamlit Cloud: "
                           "los avisos funcionan en tu computadora, o agrega la sección [whatsapp] en los Secrets de la app (ver README).")
    g = lambda k, env="": str(sec.get(k, os.environ.get(env, "")) or "").strip()
    proveedor = g("proveedor", "WHATSAPP_PROVEEDOR").lower().strip()
    if not proveedor:                          # si no lo escribiste, se deduce de las llaves
        proveedor = ("telegram" if g("token") else "ntfy" if g("tema") else
                     "twilio" if g("twilio_sid") else "callmebot")
    cfg["proveedor"] = proveedor
    cfg["telefono"] = _limpiar_tel(g("telefono", "WHATSAPP_TELEFONO"))
    cfg["activo"] = (g("activo", "WHATSAPP_ACTIVO") or "1") == "1"

    if cfg["proveedor"] == "telegram":
        import re
        cfg["token"] = g("token", "TELEGRAM_TOKEN").replace(" ", "")
        cfg["chat_id"] = g("chat_id", "TELEGRAM_CHAT_ID")
        cfg["telefono"] = "Telegram"      # solo para mostrar en la app
        if cfg["problema"] is None and not re.fullmatch(r"\d{5,}:[A-Za-z0-9_-]{25,}", cfg["token"]):
            cfg["problema"] = ('Falta el token de tu bot de Telegram en [whatsapp] (token = "123456789:AA..."). '
                               'Te lo da @BotFather cuando creas el bot.')
        llaves_ok = bool(cfg["token"])
    elif cfg["proveedor"] == "ntfy":
        tema = "".join(ch for ch in g("tema", "NTFY_TEMA") if ch.isalnum() or ch in "-_")
        cfg["tema"] = tema
        cfg["telefono"] = tema            # solo para mostrar "…" en la app
        if cfg["problema"] is None and len(tema) < 8:
            cfg["problema"] = ('Pon en [whatsapp] un tema largo y difícil de adivinar, por ejemplo '
                               'tema = "investify-tunombre-4821", y suscríbete a ese mismo tema en la app ntfy.')
        llaves_ok = bool(tema)
    elif cfg["proveedor"] == "twilio":
        cfg["twilio_sid"] = g("twilio_sid", "TWILIO_SID")
        cfg["twilio_token"] = g("twilio_token", "TWILIO_TOKEN")
        cfg["twilio_numero"] = _limpiar_tel(g("twilio_numero", "TWILIO_NUMERO") or "+14155238886")
        if cfg["problema"] is None:
            if not cfg["twilio_sid"].startswith("AC"):
                cfg["problema"] = ("twilio_sid debe ser el Account SID de Twilio (empieza con AC). "
                                   "Lo ves en console.twilio.com, en la página principal.")
            elif len(cfg["twilio_token"]) < 20:
                cfg["problema"] = "Falta twilio_token (el Auth Token de Twilio, en console.twilio.com)."
            elif not cfg["telefono"]:
                cfg["problema"] = "Falta telefono (tu celular con +52) en la sección [whatsapp]."
        llaves_ok = bool(cfg["twilio_sid"] and cfg["twilio_token"])
    else:
        crudo = g("apikey", "WHATSAPP_APIKEY")
        cfg["apikey"] = "".join(ch for ch in crudo if ch.isdigit())
        if cfg["problema"] is None:
            if crudo and not cfg["apikey"]:
                cfg["problema"] = ('La apikey de [whatsapp] no tiene números: parece que sigue el texto de ejemplo '
                                   '(TU_APIKEY). Pon la apikey numérica que te mandó CallMeBot por WhatsApp.')
            elif cfg["apikey"] and not (5 <= len(cfg["apikey"]) <= 10):
                cfg["problema"] = (f'La apikey tiene {len(cfg["apikey"])} dígitos; las de CallMeBot suelen tener 6 o 7. '
                                   'Ojo: NO es la apikey de MEXC; es la que CallMeBot te manda por WhatsApp.')
            elif not (cfg["telefono"] and cfg["apikey"]):
                if not sec:
                    vistas = ", ".join(f"[{x}]" for x in cfg.get("secciones", [])) or "ninguna"
                    cfg["problema"] = (f'No encontré la sección [whatsapp] (secciones encontradas: {vistas}). Agrega: [whatsapp] / '
                                       'proveedor = "telegram" / token = "..." / chat_id = "..." / activo = "1" (cada uno en su renglón). '
                                       'En Streamlit Cloud: Settings → Secrets → Save y luego Reboot app.')
                else:
                    cfg["problema"] = (f'La sección [whatsapp] solo tiene: {", ".join(sorted(sec)) or "nada"}. '
                                       'Para Telegram necesita token = "..." (el que te dio @BotFather).')
        llaves_ok = bool(cfg["apikey"])
    cfg["listo"] = bool(cfg["telefono"] and llaves_ok and cfg["activo"]) and cfg["problema"] is None
    return cfg


def _enviar_telegram(cfg, texto, s):
    """Bot propio de Telegram. Si no hay chat_id, lo toma del último mensaje que le mandaste al bot (/start)."""
    base = f"https://api.telegram.org/bot{cfg['token']}"
    ruta_chat = "telegram_chat.txt"           # guarda tu chat para no depender de /start cada día
    if not cfg.get("chat_id") and os.path.exists(ruta_chat):
        with open(ruta_chat, encoding="utf-8") as f:
            cfg["chat_id"] = f.read().strip()
    if not cfg.get("chat_id"):
        r = s.get(f"{base}/getUpdates", timeout=15)
        if r.status_code == 401:
            return False, "Telegram dice que el token no es válido. Cópialo otra vez de @BotFather."
        try:
            res = r.json().get("result", [])
        except Exception:
            res = []
        chats = []
        for u in res:
            for k in ("message", "edited_message", "my_chat_member", "callback_query", "channel_post"):
                obj = u.get(k) or {}
                obj = obj.get("message", obj) if k == "callback_query" else obj
                if obj.get("chat", {}).get("id"):
                    chats.append(obj["chat"]["id"])
        chats = [c for c in chats if c]
        if not chats:
            try:
                nombre_bot = s.get(f"{base}/getMe", timeout=15).json().get("result", {}).get("username", "")
            except Exception:
                nombre_bot = ""
            cual = f"@{nombre_bot}" if nombre_bot else "tu bot"
            return False, (f"Tu token es del bot {cual}, pero ese bot no ha recibido mensajes tuyos. "
                           f"Busca {cual} en Telegram, mándale «hola» y vuelve a probar.")
        cfg["chat_id"] = str(chats[-1])
        try:
            with open(ruta_chat, "w", encoding="utf-8") as f:
                f.write(cfg["chat_id"])
        except OSError:
            pass
    r = s.post(f"{base}/sendMessage", data={"chat_id": cfg["chat_id"], "text": texto.replace("*", "")}, timeout=15)
    if r.status_code == 200:
        return True, "Enviado"
    if r.status_code == 401:
        return False, "Telegram dice que el token no es válido. Cópialo otra vez de @BotFather."
    return False, f"Telegram respondió HTTP {r.status_code}. Abre tu bot y presiona INICIAR."


def _enviar_ntfy(cfg, texto, s):
    """Notificación push gratis con la app ntfy (sin cuenta ni llaves)."""
    limpio = texto.replace("*", "")
    primera = limpio.strip().splitlines()[0] if limpio.strip() else "Investify"
    sube = any(k in primera for k in ("LONG · COMPRAR", "UP", "GANADA", "prueba"))
    r = s.post(f"https://ntfy.sh/{cfg['tema']}", data=limpio.encode("utf-8"), timeout=15,
               headers={"Title": "Investify bot", "Priority": "high",
                        "Tags": "chart_with_upwards_trend" if sube else "chart_with_downwards_trend"})
    if r.status_code == 200:
        return True, "Enviado"
    return False, f"ntfy respondió HTTP {r.status_code}: {(getattr(r, 'text', '') or '')[:120]}"


def _enviar_twilio(cfg, texto, s):
    url = f"https://api.twilio.com/2010-04-01/Accounts/{cfg['twilio_sid']}/Messages.json"
    datos = {"From": "whatsapp:" + cfg["twilio_numero"], "To": "whatsapp:" + cfg["telefono"], "Body": texto}
    r = s.post(url, data=datos, auth=(cfg["twilio_sid"], cfg["twilio_token"]), timeout=15)
    if r.status_code in (200, 201):
        return True, "Enviado"
    try:
        info = r.json()
        msg, codigo = str(info.get("message", ""))[:160], info.get("code")
    except Exception:
        msg, codigo = (getattr(r, "text", "") or "")[:160], None
    pista = {20003: " → Revisa twilio_sid y twilio_token.",
             63015: " → Tu celular no está unido al sandbox: manda el mensaje 'join ...' al número de Twilio.",
             63016: " → Pasaron más de 24 h: vuelve a mandar 'join ...' al número de Twilio.",
             21211: " → El teléfono está mal escrito; prueba +521 en lugar de +52 (o al revés)."}.get(codigo, "")
    return False, f"Twilio respondió HTTP {r.status_code} (código {codigo}): {msg}{pista}"


def enviar(cfg, texto, sesion=None):
    """Envía el mensaje. Devuelve (ok, detalle). Nunca muestra llaves."""
    if not cfg.get("listo"):
        return False, cfg.get("problema") or "Avisos de WhatsApp apagados."
    s = sesion or requests
    try:
        if cfg.get("proveedor") == "telegram":
            return _enviar_telegram(cfg, texto, s)
        if cfg.get("proveedor") == "ntfy":
            return _enviar_ntfy(cfg, texto, s)
        if cfg.get("proveedor") == "twilio":
            return _enviar_twilio(cfg, texto, s)
        r = s.get(URL_CALLMEBOT, params={"phone": cfg["telefono"], "text": texto, "apikey": cfg["apikey"]}, timeout=15)
    except requests.RequestException as e:
        return False, f"No se pudo conectar con el servicio de WhatsApp: {type(e).__name__}"
    cuerpo = (getattr(r, "text", "") or "").lower()
    ok = r.status_code == 200 and "apikey is invalid" not in cuerpo and "error" not in cuerpo[:200]
    return ok, "Enviado" if ok else f"CallMeBot respondió HTTP {r.status_code}: {cuerpo[:120]}"


_CANDADO = threading.Lock()          # los mensajes salen en orden, uno por uno
PAUSA_ENTRE_MENSAJES = 1.2           # Telegram permite ~1 mensaje por segundo a un mismo chat


def enviar_en_segundo_plano(cfg, mensajes, al_enviar=None):
    """Manda cada aviso como mensaje SEPARADO (para que cada uno sea su propia notificación).
    mensajes: un texto o una lista de textos. al_enviar(i) se llama por cada mensaje que sí llegó."""
    import time
    lista = [mensajes] if isinstance(mensajes, str) else [m for m in mensajes if m]

    def _tarea():
        with _CANDADO:
            for i, texto in enumerate(lista):
                if i:
                    time.sleep(PAUSA_ENTRE_MENSAJES)
                ok, detalle = enviar(cfg, texto)
                ULTIMOS.append({"ok": ok, "detalle": detalle, "texto": texto[:80]})
                if ok and al_enviar:
                    al_enviar(i)
    if lista:
        threading.Thread(target=_tarea, daemon=True).start()


def _fp(x):
    x = float(x)
    return f"{x:,.2f}" if abs(x) >= 100 else (f"{x:,.4f}" if abs(x) >= 1 else f"{x:,.6f}")


def mensaje_mexc(op, bot):
    """MEXC Spot solo permite Long (comprar y luego vender); no hay Short."""
    modo = "REAL" if op.get("modo") == "real" else "Paper"
    if op["tipo"] == "compra":
        tp = op["precio"] * (1 + bot["tope_venta"])
        sl = f"\nStop-loss: {_fp(op['precio'] * (1 - bot['stop_loss']))}" if bot.get("stop_loss") else ""
        return (f"🟢 *LONG · COMPRAR {op['par']}*\nPrecio: {_fp(op['precio'])} USDT\nMonto: {op['monto_usdt']:,.2f} USDT\n"
                f"Tope de venta (TP): {_fp(tp)}{sl}\nModo: {modo} · Bot MEXC")
    g = op.get("ganancia_usdt") or 0.0
    return (f"🔴 *CERRAR LONG · VENDER {op['par']}*\nPrecio: {_fp(op['precio'])} USDT\nRecibido: {op['monto_usdt']:,.2f} USDT\n"
            f"PNL: {g:+,.2f} USDT ({(op.get('ganancia_%') or 0):+.2f}%)\nMotivo: {op['motivo']}\nModo: {modo} · Bot MEXC")


def mensaje_polymarket(ap):
    """En Polymarket: Up ≈ Long (apuestas a que sube) y Down ≈ Short (apuestas a que baja)."""
    nombre = {"BTC": "Bitcoin", "ETH": "Ethereum"}.get(ap["activo"], ap["activo"])
    lado = "📈 *UP (≈ LONG)*" if ap["lado"] == "Up" else "📉 *DOWN (≈ SHORT)*"
    modo = {"real": "REAL", "sombra": "Sombra (no enviada)"}.get(ap.get("modo"), "Paper")
    return (f"{lado} · {nombre} 1 h\nPrecio: {ap['precio'] * 100:.1f}¢ por acción\nMonto: {ap['monto']:,.2f} USD "
            f"({ap['acciones']:,.2f} acciones)\nProbabilidad IA: {ap['prob_modelo'] * 100:.1f}% · ventaja "
            f"{ap['ventaja'] * 100:+.1f} pts\nPaga {ap['acciones']:,.2f} USD si gana\nModo: {modo} · Bot Polymarket")


def mensaje_resolucion(ap):
    nombre = {"BTC": "Bitcoin", "ETH": "Ethereum"}.get(ap["activo"], ap["activo"])
    icono = "✅" if ap["estado"] == "ganada" else ("❌" if ap["estado"] == "perdida" else "🔎")
    return f"{icono} *{nombre} {ap['lado']}: {ap['estado'].upper()}*\nPNL: {(ap.get('ganancia') or 0):+,.2f} USD · Bot Polymarket"


def juntar(mensajes):
    """Junta varios avisos de un mismo ciclo en un solo WhatsApp (CallMeBot limita los envíos)."""
    return "\n\n".join(mensajes)


# ---------------------------------------------------------------------------
# Avisos de activos recomendables por la IA (una vez por activo y periodo)
# ---------------------------------------------------------------------------
RUTA_ENVIADOS = "avisos_enviados.json"
UMBRAL_SHORT_MIN = 0.002            # caída estimada mínima (0.2 %) para sugerir SHORT


def recomendaciones(pronosticos, nombres=None, maximo=8):
    """LONG: la IA da señal «comprar». SHORT: estima una caída clara y el precio está bajo su SMA20."""
    nombres = nombres or {}
    recs = []
    for t, p in (pronosticos or {}).items():
        if not p:
            continue
        est = p.get("estimado_dl")
        if est is None or est != est:          # NaN
            continue
        umbral = p.get("umbral")
        umbral = abs(umbral) if umbral is not None and umbral == umbral else 0.0
        cierre, sma = p.get("cierre"), p.get("sma20")
        if p.get("senal") == "comprar":
            lado = "LONG"
        elif est < -max(umbral, UMBRAL_SHORT_MIN) and (sma is None or sma != sma or cierre < sma):
            lado = "SHORT"
        else:
            continue
        recs.append({"ticker": t, "nombre": nombres.get(t, t), "lado": lado, "estimado": float(est),
                     "cierre": float(cierre or 0), "objetivo": float(p.get("precio_estimado") or 0),
                     "periodo": str(p.get("proximo_periodo", ""))})
    recs.sort(key=lambda r: abs(r["estimado"]), reverse=True)
    return recs[:maximo]


def mensaje_recomendacion(r):
    """Un aviso por activo; la primera línea es lo que se ve en la notificación del celular."""
    nombre = "" if r["nombre"] == r["ticker"] else f" ({r['nombre']})"
    if r["lado"] == "LONG":
        titulo = f"🟢 LONG · COMPRAR {r['ticker']}"
    else:
        titulo = f"🔴 SHORT · VENDER {r['ticker']}"
    return (f"*{titulo}*  {r['estimado']:+.2%}\n"
            f"Recomendación de la IA{nombre}\n"
            f"Precio: {r['cierre']:,.4g} → objetivo: {r['objetivo']:,.4g}\n"
            f"Para: {r['periodo']}\n"
            "Pronóstico del modelo, no es asesoría financiera.")


def mensaje_recomendaciones(recs):
    """Todas juntas (se usa solo para mostrar o probar)."""
    return "\n\n".join(mensaje_recomendacion(r) for r in recs)


def _clave_rec(r):
    return f"{r['ticker']}|{r['lado']}|{r['periodo']}"


def _leer_enviados(ruta):
    import json
    try:
        with open(ruta, encoding="utf-8") as f:
            return set(json.load(f))
    except (OSError, ValueError):
        return set()


def recomendaciones_nuevas(recs, ruta=RUTA_ENVIADOS):
    """Las que todavía no se han avisado (mismo activo, lado y periodo)."""
    enviados = _leer_enviados(ruta)
    return [r for r in recs if _clave_rec(r) not in enviados]


def marcar_enviadas(recs, ruta=RUTA_ENVIADOS):
    """Se llama solo cuando el aviso sí llegó, para que no se pierda si falló."""
    import json
    enviados = _leer_enviados(ruta) | {_clave_rec(r) for r in recs}
    try:
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(sorted(enviados)[-500:], f)
    except OSError:
        pass
