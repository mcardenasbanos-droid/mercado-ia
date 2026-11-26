# Investify · Pronóstico con Deep Learning y bots de trading

**Investify** es una app web hecha en Streamlit para el proyecto final de *Deep Learning para Finanzas con Agente de IA*.
Pronostica el rendimiento del siguiente periodo de **40 activos** con redes **MLP y LSTM** y muestra sus indicadores técnicos.
También opera con **bots conectados a datos reales de MEXC y de Polymarket**, y avisa al celular (Telegram) de cada
recomendación y de cada operación.

> Proyecto académico. Las señales y recomendaciones **no son asesoría financiera**.

**Repositorio:** https://github.com/mcardenasbanos-droid/mercado-ia
**Versión actual:** **v2.5** (26 de noviembre de 2026). El detalle está en [Historial de versiones](#historial-de-versiones).

---

## Contenido
1. [Qué hace la app](#qué-hace-la-app)
2. [Activos](#activos)
3. [Cómo funciona (arquitectura)](#cómo-funciona-arquitectura)
4. [Modelos de Deep Learning](#modelos-de-deep-learning)
5. [Bot MEXC](#bot-mexc)
6. [Bot Polymarket](#bot-polymarket)
7. [Avisos al celular](#avisos-al-celular)
8. [Historial de versiones](#historial-de-versiones)
9. [Tecnologías y versiones](#tecnologías-y-versiones)
10. [Instalación y uso](#instalación-y-uso)
11. [Configuración de llaves (secrets.toml)](#configuración-de-llaves-secretstoml)
12. [Pruebas](#pruebas)
13. [Seguridad](#seguridad)
14. [Limitaciones conocidas](#limitaciones-conocidas)
15. [Problemas comunes](#problemas-comunes)
16. [Autoevaluación con la rúbrica](#autoevaluación-con-la-rúbrica)
17. [Rutas de API usadas](#rutas-de-api-usadas)

---

## Qué hace la app
La interfaz tiene estilo de exchange profesional: tema oscuro, cinta de precios, velas japonesas y paneles. Se organiza en 7 pestañas:

| Pestaña | Qué muestra |
|---|---|
| **Mercados** | Tabla de los 40 activos con precio, cambio diario, señal de la IA y filtros por mercado. |
| **Análisis** | Velas de cada activo con indicadores (SMA 20, Bandas de Bollinger, RSI 14, MACD, ATR 14) y el pronóstico del siguiente periodo. |
| **Portafolio** | Portafolio simulado de 100,000 USD: compra y venta manual, botón «Ejecutar señales de la IA», posiciones e historial. |
| **Bot MEXC** | Terminal de trading con precios en tiempo real de MEXC, lista de pares, velas (1m/5m/15m/1h), panel del bot, operaciones, posiciones, historial completo (CSV), rendimiento y cuenta MEXC. |
| **Bot Polymarket** | Bot que apuesta en los mercados horarios «Bitcoin/Ethereum Up or Down» con la probabilidad que calcula la IA. |
| **Resultados** | Evaluación final en el conjunto de prueba (TE): comparación de la IA contra las referencias (comprar y mantener, efectivo). |
| **Modelo** | Cómo funciona Investify: datos → modelo → señal; configuración final de cada activo. |

---

## Activos
**40 activos** descargados de Yahoo Finance desde 2015:

| Mercado | Activos |
|---|---|
| **20 acciones de EE. UU.** | AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AMD, INTC, NFLX, JPM, BAC, V, MA, JNJ, PFE, WMT, KO, DIS, XOM |
| **10 criptomonedas** | BTC, ETH, SOL, BNB, XRP, ADA, DOGE, LTC, LINK, AVAX (contra USD) |
| **10 divisas** | USD/MXN, EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF, NZD/USD, EUR/MXN, EUR/JPY |

USD/MXN, USD/JPY, USD/CAD y USD/CHF son **solo de análisis**: desde una cuenta en dólares su valor no cambia.

---

## Cómo funciona (arquitectura)
```
 Google Colab (GPU T4)                                App local / Streamlit Cloud
 ┌──────────────────────────────┐   paquete_app.zip   ┌─────────────────────────────────────────────┐
 │ NB1 datos y referencias      │  ─────────────────▶ │ modelos/  (.keras + _config.joblib)         │
 │ NB2 MLP y LSTM (3 configs)   │                     │ resultados/ (CSV del Notebook 5)            │
 │ NB3 umbrales y reglas (VA)   │                     │                                             │
 │ NB4 diseño del agente        │                     │ nucleo.py   datos Yahoo, indicadores,       │
 │ NB5 indicadores, TE, futuro  │                     │             pronóstico, portafolio          │
 │ NB6 fallos y entrega         │                     │ app.py + ui.py   interfaz estilo exchange   │
 └──────────────────────────────┘                     │ mexc_bot.py / mexc_cuenta.py  ──▶ API MEXC   │
                                                      │ polymarket_bot.py / _real.py ──▶ Polymarket │
                                                      │ notificaciones.py  ──▶ Telegram (celular)   │
                                                      └─────────────────────────────────────────────┘
```

### Estructura de la carpeta
```
app_bolsa/
├── app.py                 ← interfaz (7 pestañas)
├── ui.py                  ← diseño estilo exchange (colores, tablas, paneles)
├── nucleo.py              ← datos, indicadores, pronóstico y portafolio
├── mexc_bot.py            ← precios públicos de MEXC y lógica del bot (estrategias simple y pro)
├── agentes.py             ← IA como cerebro: 4 agentes + coordinador con pesos aprendidos
├── riesgo.py              ← ATR/RSI/MACD, métricas, costos, SL/TP, tamaño (Kelly), límites, bitácora con hash
├── mexc_cuenta.py         ← conexión privada a tu cuenta MEXC (firma HMAC-SHA256)
├── polymarket_bot.py      ← bot de Polymarket (datos públicos, señal, cartera paper)
├── polymarket_real.py     ← ÚNICO módulo que puede enviar apuestas reales
├── notificaciones.py      ← avisos al celular (Telegram, ntfy, CallMeBot o Twilio)
├── tests/                 ← 44 pruebas automáticas (pytest)
├── .github/workflows/ci.yml ← CI: compila y corre las pruebas en cada push
├── iniciar.bat            ← arranque en 1 comando (Windows)
├── secrets_ejemplo.toml   ← plantilla de llaves (sin llaves reales)
├── requirements.txt
├── assets/logo.png
├── .streamlit/config.toml ← tema oscuro
├── modelos/               ← modelos finales (de paquete_app.zip)
└── resultados/            ← CSV de resultados (de paquete_app.zip)
```

---

## Modelos de Deep Learning
Los modelos siguen el protocolo del PDF y se entrenan en Google Colab con GPU T4:

- **Objetivo:** rendimiento del siguiente periodo, *y = P(t+1)/P(t) − 1*.
- **Entradas:**
  - **Versión A:** los 20 rendimientos más recientes.
  - **Versión B:** además, distancia a la SMA 20, volatilidad de 20 días y RSI 14. La LSTM B recibe (n, 20, 4).
- **División temporal** sin mezclar fechas: **70% entrenamiento (TR) · 15% validación (VA) · 15% prueba (TE)**.
- **Redes:**
  - **MLP** y **LSTM**, con 3 configuraciones cada una (16, 32 y 64 neuronas).
  - Early stopping (paciencia 15, máximo 200 épocas) y 3 semillas (42, 7, 123).
  - La mejor configuración se elige por RMSE en VA.
- **Referencias:** modelo lineal, comprar y mantener, y efectivo.
- **Señal:** se compra si el rendimiento estimado supera el umbral δ.
  - La regla **C** usa solo el umbral.
  - La regla **D** exige además que el precio esté arriba de la SMA 20.
  - δ y la regla se eligen **solo en VA**, con 3 ventanas de validación expandibles.
- **Simulador:**
  - 100,000 USD de capital y 0.10% de comisión por operación.
  - Ejecución a la apertura de t+1.
- **Evaluación final:** una sola vez en TE con la configuración congelada. La app la muestra en la pestaña **Resultados**; los CSV están en `resultados/`.

---

## Bot MEXC
- **Datos reales:**
  - Precios, velas de 1m/5m/15m/1h y estadísticas de 24 h vienen de la **API pública Spot v3 de MEXC**: `/ticker/price`, `/klines`, `/ticker/24hr`.
  - Se actualiza automáticamente cada pocos segundos.
- **10 pares:** BTC, ETH, SOL, BNB, XRP, ADA, DOGE, LTC, LINK y AVAX contra USDT.
- **Estrategia Pro (por defecto): IA + riesgo.**
  - **Tesis:** reversión a la media intradía. Tras una caída rápida desde su máximo reciente, una cripto líquida tiende a rebotar.
    Funciona en mercados laterales o alcistas; falla en tendencias bajistas fuertes.
  - **1 · Disparador:** el precio baja el *tope de compra* (0.2%) desde su máximo reciente.
  - **2 · Agentes de IA** (`agentes.py`), cada uno con un solo rol, votan con confianza 0–1 y razón:
    `modelo_dl` (tu MLP/LSTM), `tecnico` (RSI 14 y MACD), `tendencia` (SMA 20) y `volatilidad` (ATR).
    El coordinador los combina con pesos y entrega un JSON validado: *decisión, probabilidad, confianza, razón, votos*.
    Compra solo si P ≥ 55%. Si todos se abstienen, la decisión de respaldo es **no operar**.
  - **3 · Riesgo con veto** (`riesgo.py`, determinista). Bloquea la entrada si se cumple cualquiera de estas condiciones:
    kill switch activo, pérdida del día ≥ 2%, drawdown ≥ 5%, 4 posiciones abiertas, 4 pérdidas seguidas o RR neto < 1.
    Los tres límites de pérdida activan el **kill switch**, que solo una persona rearma.
  - **4 · Ejecución:**
    - **SL = entrada − 1.5·ATR** y **TP = entrada + 3·ATR**, con el ATR de Wilder en velas **cerradas** de 15 min.
    - **Tamaño:** se arriesga 0.5% del equity hasta el SL, contando costos.
    - Con ≥ 30 operaciones se usa **Kelly ½** sobre la cota inferior de Wilson del acierto, con tope por orden.
  - **Salidas, en este orden:** SL > TP > señal contraria de la IA (si cambia de opinión) > tiempo máximo (240 min).
  - **Costos:** 0.10% de comisión + 0.05% de deslizamiento por lado (0.30% ida y vuelta) en cada cálculo.
  - La IA **propone**; nunca pone órdenes ni cambia tamaños o límites.
  - Los pesos de los agentes se **auditan contra el PnL real**: peso = 0.10 + 0.90·(aciertos+1)/(votos+2), desde 10 operaciones.
- **Estrategia Simple:** compra con el tope de compra y vende con un tope de venta fijo (la versión anterior).
- **Modos:**
  1. **Paper** (por defecto): 50,000 USDT de práctica, con precios reales y 0.10% de comisión.
  2. **Real** (opcional):
     - Órdenes Spot reales a mercado.
     - Mínimo 5 USDT por orden y un tope total configurable.
     - Requiere llaves y escribir `OPERAR CON DINERO REAL`.
- **Resultados:**
  - Últimas operaciones, posiciones con SL y TP, historial completo en CSV y curva de equity.
  - **Métricas netas de costos:** win rate con intervalo de Wilson al 95%, expectativa por operación, profit factor, RR neto,
    acierto de equilibrio, drawdown máximo y Sharpe por operación.
- **IA y bitácora:** pesos y aciertos de cada agente, y una **bitácora inmutable**. Cada propuesta, veto, kill switch y orden
  lleva un SHA-256 encadenado al evento anterior; la app verifica la cadena y se puede descargar en JSON.
- **Mi cuenta MEXC:** muestra el saldo real si configuras tus llaves (solo lectura). También hay una prueba de conexión que no ejecuta órdenes.

> La API de MEXC **no tiene cuenta demo ni sandbox** (lo dice MEXC oficialmente), por eso las operaciones del modo paper no
> aparecen en «Demo Trading» de la página de MEXC. Además, MEXC no opera desde EE. UU.: en Streamlit Cloud el bot usa Yahoo
> Finance como respaldo, y **en una computadora en México usa MEXC en tiempo real**.

---

## Bot Polymarket
Apuesta en los mercados **«Bitcoin Up or Down»** y **«Ethereum Up or Down»** de cada hora.

- **Datos reales:**
  - El mercado de la hora viene de la API Gamma de Polymarket.
  - Los precios del libro de órdenes vienen de la API CLOB.
- **Probabilidad de la IA:**
  - Se calcula *P(Up) = Φ((ln(P/apertura) + μₕ·τ) / (σₕ·√τ))*.
  - μₕ es el pronóstico diario de la IA ÷ 24 y σₕ es la volatilidad de 20 días ÷ √24.
- **Decisión:**
  - Apuesta al lado (Up o Down) cuya probabilidad supera su precio por la *ventaja mínima*.
  - Hace una apuesta por mercado y nunca en los últimos 5 minutos.
- **Resolución:** resultado oficial de Polymarket. Si no hay resultado después de 20 minutos, usa la vela de 1 hora de MEXC.
- **Modos:**
  1. **Paper**: 10,000 USD simulados con precios reales del libro.
  2. **Sombra**: valida cada apuesta con el SDK oficial **sin enviarla**.
  3. **Real**: requiere tres llaves: `modo_real = "1"`, 24 h de sombra y escribir `APOSTAR CON DINERO REAL`.
- Polymarket tampoco tiene cuenta demo.

---

## Avisos al celular
`notificaciones.py` manda un **mensaje separado** por cada aviso, para que cada uno sea su propia notificación en el celular:

| Aviso | Ejemplo de la primera línea |
|---|---|
| Recomendación de la IA (al abrir la app; una vez por activo y periodo; las 8 más fuertes) | 🟢 **LONG · COMPRAR AMD** +0.55% / 🔴 **SHORT · VENDER NVDA** −0.30% |
| Compra del bot MEXC | 🟢 **LONG · COMPRAR SOLUSDT** (precio, monto y TP) |
| Venta del bot MEXC | 🔴 **CERRAR LONG · VENDER SOLUSDT** (PnL y motivo) |
| Apuesta de Polymarket | 📈 **UP (≈ LONG)** / 📉 **DOWN (≈ SHORT)** |
| Resultado de Polymarket | ✅ **GANADA** / ❌ **PERDIDA** |

- **LONG** sale cuando la IA da señal de compra.
- **SHORT** sale cuando estima una caída mayor al umbral (mínimo 0.2%) con el precio debajo de su SMA 20.
- MEXC Spot solo permite Long.

### Configurar Telegram (recomendado, gratis)
1. En Telegram abre **@BotFather** (con palomita azul), envía `/newbot` y elige nombre y usuario (que termine en `bot`).
2. Copia el **token** que te da.
3. Abre tu bot y presiona **INICIAR**.
4. En `.streamlit/secrets.toml` agrega:
   ```toml
   [whatsapp]
   proveedor = "telegram"
   token = "TU_TOKEN"
   activo = "1"
   ```
5. Reinicia la app y abre **Bot MEXC → «Avisos por WhatsApp»** y presiona **«Enviar WhatsApp de prueba»** (los avisos llegan al servicio que elegiste). La app detecta tu chat sola y lo guarda en `telegram_chat.txt`.

**En la página pública (Streamlit Cloud):** en share.streamlit.io → tu app → ⋮ → **Settings → Secrets**, pega **solo** la
sección `[whatsapp]` con `proveedor`, `token`, `chat_id` (el número que quedó en tu `telegram_chat.txt`) y `activo`.
**Nunca pongas ahí las llaves de MEXC ni de Polymarket.** Los avisos llegan mientras alguien tenga abierta la página.

Otros servicios que también funcionan son `proveedor = "ntfy"` (con `tema`), `"callmebot"` (con `telefono` y `apikey`) y `"twilio"` (con `telefono`, `twilio_sid` y `twilio_token`). Los avisos se envían **mientras la app está abierta** (en tu computadora o en la página pública).

---

## Historial de versiones
| Versión | Fecha | Cambios |
|---|---|---|
| **v1.0** | 18/11/2026 | 6 notebooks del protocolo (40 activos, MLP/LSTM en T4, umbrales en VA, agente, evaluación en TE, pronóstico futuro, bitácora de fallos). Primera app en Streamlit: pronósticos, indicadores y portafolio simulado. |
| **v1.1** | 19/11/2026 | App publicada en Streamlit Community Cloud desde GitHub. Primer **bot de trading simulado** con topes de compra y venta. |
| **v1.2** | 20/11/2026 | Rediseño visual y nombre **Investify** (logo, tema oscuro, tarjetas y gráficas). |
| **v2.0** | nov 2026 | Bot conectado a **MEXC**: precios y velas en tiempo real, capital de 50,000 USDT, resumen de ganancias, tabla de últimas operaciones e historial completo. Lectura de la cuenta MEXC con llaves API y **modo real** opcional (5 USDT por orden). |
| **v2.1** | 26/11/2026 | Rediseño completo **estilo exchange**: cinta de precios, terminal de 3 columnas, velas con niveles TP y SL, pestañas inferiores y gráfica de rendimiento. |
| **v2.2** | 26/11/2026 | **Bot Polymarket** (BTC y ETH por hora) con la señal del pronóstico de la IA. Modos paper, sombra y real. |
| **v2.3** | 26/11/2026 | **Avisos al celular** de cada compra o venta (LONG/SHORT): CallMeBot, Twilio, ntfy y **Telegram**. |
| **v2.4** | 26/11/2026 | Avisos de **recomendaciones de la IA** (LONG/SHORT), un mensaje por aviso, envío confiable (solo se marca como enviado si llegó) y README completo. |
| **v2.5** | 26/11/2026 | Mejoras guiadas por la rúbrica: **estrategia Pro** (4 agentes de IA + coordinador con pesos aprendidos, riesgo determinista con veto y kill switch, SL/TP con ATR, tamaño por riesgo y Kelly ½), **métricas profesionales**, deslizamiento en los costos, **bitácora con hash encadenado**, calibración (Brier) del bot Polymarket, CI y arranque en 1 comando. |

### Errores corregidos durante el desarrollo
| Problema | Corrección |
|---|---|
| Google Drive no montaba en Colab | Permisos completos y `force_remount=True`. |
| Error de Plotly «reference traces by row and column» | Estilo de ejes aplicado a toda la figura sin `row` ni `col`. |
| La línea del TP aplastaba las velas | Rango del eje y ajustado a las velas; niveles fuera de vista como etiqueta. |
| El bot no compraba (umbral 0.5%) y nunca vendía (TP 5%) | Recomendación: compra al 0.2% y TP al 0.5%. |
| `ModuleNotFoundError: polymarket_bot` | Faltaba copiar los módulos nuevos a la carpeta. |
| f-string incompatible con Python 3.11 | Texto movido a una variable. |
| PnL del modo real con cantidad redondeada por MEXC | La ganancia se calcula con la cantidad realmente vendida. |
| CallMeBot lleno / apikey equivocada (se usó la de MEXC) | Validación de la apikey y soporte para Telegram, ntfy y Twilio. |

---

## Tecnologías y versiones
| Herramienta | Versión | Uso |
|---|---|---|
| Python | 3.11 o 3.12 | Lenguaje |
| Streamlit | ≥ 1.50 | Interfaz web (`st.fragment`, tablas seleccionables, pestañas) |
| TensorFlow / Keras | ≥ 2.16 (la misma que en Colab) | Modelos MLP y LSTM |
| scikit-learn | ≥ 1.4 (la misma que en Colab) | Escaladores y modelo lineal |
| yfinance | ≥ 0.2.54 | Datos de Yahoo Finance |
| pandas / numpy | ≥ 2.2 / ≥ 1.26 | Datos |
| Plotly | ≥ 5.22 | Velas e indicadores |
| requests | ≥ 2.31 | APIs de MEXC, Polymarket y Telegram |
| joblib, tzdata | — | Configuración de modelos y zonas horarias |
| polymarket-client | opcional (beta) | Modos sombra y real de Polymarket |
| Google Colab (GPU T4) | — | Entrenamiento |

Para ver tus versiones exactas: `pip freeze > versiones.txt`.

---

## Instalación y uso
1. **Modelos y resultados:** descomprime `paquete_app.zip` (lo genera el Notebook 6 en Drive) dentro de `app_bolsa/`, para que queden `modelos/` y `resultados/`.
2. **Versiones de Colab:** en Colab corre `import tensorflow as tf, sklearn; print(tf.__version__, sklearn.__version__)` y pon esas versiones exactas en `requirements.txt`.
3. **Instalar** (Windows):
   ```bash
   cd "ruta\a\app_bolsa"
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   En Mac o Linux, el paso de activar es `source .venv/bin/activate`.
**Arranque en 1 comando (Windows):** doble clic en `iniciar.bat`. Crea el entorno si no existe, instala y abre la app.
Sin llaves arranca en **modo paper**.

4. **Correr:**
   ```bash
   .venv\Scripts\activate
   streamlit run app.py
   ```
   Se abre en `http://localhost:8501`. Para detener la app usa `Ctrl + C`. **Cada terminal nueva necesita activar `.venv` primero.**
5. **Publicar (opcional):** en share.streamlit.io → Create app → este repositorio → `app.py` → Python 3.11/3.12 → Deploy.

---

## Configuración de llaves (secrets.toml)
Copia `secrets_ejemplo.toml` como **`.streamlit/secrets.toml`** y llena solo lo que uses:
```toml
[mexc]                 # opcional: saldo real y modo real
api_key = "..."
secret_key = "..."

[polymarket]           # opcional: modos sombra y real
private_key = "..."
modo_real = "0"

[whatsapp]             # avisos al celular
proveedor = "telegram"
token = "..."
activo = "1"
```
- **Permisos de la llave MEXC:** solo *View Account Details*, *View Order Details* y *Trade* (Spot). **Nunca** retiros ni futuros.
- En MEXC → **Mi cuenta MEXC → Probar conexión** puedes validar las llaves sin operar.

---

## Pruebas
```bash
pip install pytest
python -m pytest tests -q
```
Las pruebas (44) usan clientes falsos y nunca se conectan a nada real. Cubren:
- mercados y probabilidad de Polymarket;
- libro de órdenes, apuestas ganadas y perdidas, y resolución;
- que el modo sombra **no envía** nada y que solo `polymarket_real.py` puede enviar órdenes;
- geobloqueo y las tres llaves;
- mensajes y envíos por Telegram, ntfy, CallMeBot y Twilio;
- recomendaciones LONG/SHORT y avisos separados;
- ATR y RSI de Wilder contra valores de referencia, Wilson, Kelly, SL/TP con RR neto, tamaño de posición y métricas;
- veto del riesgo y kill switch, bitácora que detecta alteraciones, salida estructurada y pesos de los agentes;
- ciclo completo del bot Pro (entrada → TP con PnL y auditoría), descarte de la IA y salida por tiempo máximo;
- calibración (Brier) de Polymarket.

**CI:** `.github/workflows/ci.yml` compila todos los módulos y corre las pruebas en GitHub en cada push.

---

## Seguridad
- **Nunca subas `.streamlit/secrets.toml` a GitHub** (el repositorio es público), ni pongas llaves en los «Secrets» de Streamlit Cloud.
- `.gitignore` excluye `secrets.toml`, `bot_mexc.json`, `polymarket_estado.json`, `portafolio.json`, `avisos_enviados.json` y `telegram_chat.txt`.
- Los modos reales piden frase de confirmación y tienen montos máximos. La app nunca muestra llaves completas.

---

## Limitaciones conocidas
- Las operaciones del modo paper **no aparecen** en la cuenta Demo de MEXC: su API no tiene sandbox.
- En Streamlit Cloud (EE. UU.) MEXC está bloqueado y el bot usa Yahoo. **Para la demostración, correr la app en la computadora.**
- Los bots y los avisos funcionan **mientras la página está abierta**.
- La estrategia Pro opera poco a propósito: exige caída, P ≥ 55% de los agentes y RR neto ≥ 1 con costos. Para una
  demostración rápida existe la estrategia Simple.
- Sin conexión real no se verificaron el modo real de Polymarket ni la cantidad exacta ejecutada de cada orden real.
- Los pronósticos son de un periodo. El modelo no garantiza ganancias.

---

## Problemas comunes
| Problema | Solución |
|---|---|
| `ModuleNotFoundError: No module named 'yfinance'` | Activa el entorno: `.venv\Scripts\activate`. Si sigue, `pip install -r requirements.txt`. |
| «No se encontró resultados/ficha_activos.csv» | Descomprime `paquete_app.zip` dentro de `app_bolsa/`. |
| «No se pudo cargar el modelo…» | Usa las mismas versiones que Colab. Mientras tanto, la app usa los pronósticos guardados. |
| Faltan algunos activos | Yahoo limita las descargas: espera un minuto y presiona «Actualizar datos». |
| El bot MEXC no compra o no vende | Usa compra al 0.2% y TP al 0.5%. |
| Telegram: «presiona INICIAR» | Busca tu bot por su @usuario, mándale «hola» y vuelve a probar. |
| No suena la notificación | Revisa los ajustes de notificaciones del celular para Telegram y cierra Telegram en la computadora. |


---

## Autoevaluación con la rúbrica
Esta es la evaluación honesta de Investify contra la rúbrica *App de Trading con IA*.
Niveles: 4 Excelente · 3 Competente · 2 En desarrollo · 1 Insuficiente.
Donde algo queda fuera, se explica por qué.

### 1 · App: FrontEnd y BackEnd (20%)
| Indicador | Nivel | Evidencia / por qué |
|---|---|---|
| Arquitectura | 3 | Capas separadas: interfaz (`app.py`, `ui.py`), lógica (`mexc_bot`, `agentes`, `riesgo`, `polymarket_bot`) y gateway del broker (`mexc_cuenta`, `polymarket_real`). El navegador **nunca** llama al broker: todo pasa por el servidor de Streamlit. Hay diagrama, pero sin REST ni WebSocket (ver «Tiempo real»). |
| FrontEnd | 3–4 | Datos en vivo, velas con SMA, Bollinger, RSI, MACD y ATR, mensajes de carga y error, y respaldo con Yahoo. Streamlit se adapta a pantallas chicas, pero la terminal está pensada para escritorio. |
| BackEnd / API | 3 | Las rutas externas están documentadas (tabla abajo). Las órdenes se validan: mínimo 5 USDT, topes, redondeo al paso del par y frase para el modo real. Ante un rechazo, el par espera 5 minutos antes de reintentar. No hay API propia porque Streamlit no la necesita. |
| Tiempo real | 2 | Sondeo cada 2–30 s con `st.fragment`. **Sin WebSocket**: Streamlit vuelve a dibujar la página en cada ciclo y para un bot de minutos el sondeo basta. Siguiente paso: el WebSocket de MEXC en un proceso aparte. |
| Seguridad | 3–4 | Llaves en `.streamlit/secrets.toml`, fuera de Git y del cliente. Modo paper por defecto. El real exige frase y topes, y las pruebas verifican que el modo sombra no envía nada y que solo un módulo puede enviar órdenes. |
| Persistencia y trazabilidad | 4 | Bitácora inmutable con hash encadenado SHA-256, verificación en la app y exportación en JSON; historial en CSV. |
| Calidad y entrega | 4 | README completo, `iniciar.bat` (1 comando), modo paper sin llaves, 44 pruebas y CI en GitHub Actions. No hay revisión de tipos (mypy). |

### 2 · Consideraciones financieras (25%)
| Indicador | Nivel | Evidencia / por qué |
|---|---|---|
| Velas y datos | 3–4 | OHLCV de MEXC por temporalidad (1m/5m/15m/1h) en hora de Ciudad de México. ATR, RSI y SMA usan **solo velas cerradas**. El disparador usa el último precio, a propósito, para reaccionar a la caída. |
| Análisis técnico | 4 | RSI y ATR de Wilder, MACD y SMA, probados contra valores de referencia en `tests/test_estrategia.py`. Cada uno tiene un porqué en un agente. |
| Métricas de desempeño | 4 | Win rate con IC de Wilson, expectativa, profit factor, RR neto, acierto de equilibrio, drawdown máximo y Sharpe por operación, todas netas de costos. |
| Costos | 3–4 | 0.10% de comisión + 0.05% de deslizamiento por lado en cada cálculo, con el acierto de equilibrio publicado. No se separa maker y taker porque todas las órdenes son a mercado (taker). |
| Stop Loss y Take Profit | 4 | SL a 1.5·ATR, por debajo del ruido normal, y TP a 3·ATR, definidos al entrar. Una entrada con RR neto < 1 queda vetada. |
| Tamaño de posición | 4 | 0.5% de riesgo hasta el SL. Kelly ½ sobre la cota inferior de Wilson, solo desde 30 operaciones, con tope por orden y por efectivo. |
| Gestión de riesgo | 4 | Pérdida diaria, drawdown, posiciones máximas y racha de pérdidas. Kill switch automático o manual que solo una persona rearma. |

### 3 · Aprendizaje (ML / DL) y predicciones (20%)
| Indicador | Nivel | Evidencia / por qué |
|---|---|---|
| Problema y objetivo | 4 | Objetivo: y = P(t+1)/P(t) − 1, ligado a la regla «comprar si el estimado > δ». |
| Datos y features | 3–4 | Rezagos y features calculados solo con el pasado y escalados con datos de entrenamiento. Fuente (Yahoo) y periodo (2015–2026) documentados. |
| Validación | 3 | Separación temporal 70/15/15, 3 ventanas expandibles en VA y una sola evaluación en TE. **Falta dejar días de separación** (purga) entre conjuntos: con objetivo a 1 día el traslape es mínimo, pero queda como mejora. |
| Comparación con línea base | 3 | Contra modelo lineal, comprar y mantener, y efectivo (NB1 y NB5). **Faltan** «operar al azar» y cruce de medias: el tiempo de GPU se usó en las 3 configuraciones × 2 redes × 3 semillas. |
| Modelos (ML y DL) | 3–4 | Lineal (ML clásico) contra MLP y LSTM (DL), elegidos por RMSE en VA. Resultado en TE: la IA gana al lineal en 23 de 40 activos. |
| Calibración y probabilidad | 3 | El bot Polymarket mide la calibración con **Brier** y una tabla «predicho vs real». La red DL da un rendimiento, no una probabilidad, y su umbral δ se elige en VA. |
| Del modelo al PnL | 3 | Backtest en TE con comisión 0.10%, capital 100,000 y ejecución en t+1 (+5.76% promedio). El monitoreo de degradación y el reentrenamiento quedan como siguiente paso. |

### 4 · IA como cerebro de operaciones (15%)
| Indicador | Nivel | Evidencia / por qué |
|---|---|---|
| Arquitectura de agentes | 4 | 4 agentes de rol único (modelo DL, técnico, tendencia, volatilidad) y un coordinador con pesos. |
| Salida estructurada | 4 | JSON validado con esquema: decisión, probabilidad 0–1, confianza 0–1, razón y votos. Es determinista (equivale a temperatura 0) y una salida inválida se convierte en «abstenerse». |
| Jerarquía y vetos | 4 | El riesgo determinista tiene veto vinculante. La IA solo propone: no pone órdenes ni cambia tamaños o límites. |
| Robustez | 3–4 | Si falta el pronóstico o las velas, el agente se abstiene; si todos se abstienen, no se opera. Los modelos corren en local. Si un modelo no carga, se usa el pronóstico guardado. |
| Trazabilidad (Reflejo) | 4 | Cada compra guarda los votos, razones y probabilidad de cada agente en la bitácora. Los descartes y vetos también quedan registrados. |
| Aprendizaje de pesos | 4 | Los pesos se auditan contra el PnL real con regla de Laplace y piso de 0.10, desde 10 operaciones. |
| Seguridad del prompt | N/A | **No se usa un LLM**: los agentes son código determinista y verificable, sin costo por llamada y sin riesgo de inyección. Los datos externos (precios, mercados, mensajes) se tratan solo como datos. Los agentes tienen versión (`agentes-v1.0`). |

### 5 · Trading: Entry, Long, Short (20%)
| Indicador | Nivel | Evidencia / por qué |
|---|---|---|
| Tesis de la estrategia | 4 | Reversión a la media intradía, con el régimen donde funciona (lateral o alcista) y donde falla (bajista fuerte, que vigila el agente de tendencia). |
| Entry | 3 | Condiciones medibles: caída ≥ 0.2%, P ≥ 55% y RR neto ≥ 1. Se usa orden a mercado porque son pares muy líquidos y se prioriza entrar en la caída; el deslizamiento se incluye en los costos. Las entradas no expiran. |
| Long | 4 | Señal, confirmación de los agentes, SL bajo el invalidador (1.5·ATR), TP y tiempo máximo. |
| Short | 4 | **Solo Long, con justificación:** MEXC Spot no permite vender en corto. En Polymarket, Down ≈ Short y la IA lo usa. |
| Salidas | 4 | Prioridad definida: SL > TP > señal contraria de la IA > tiempo máximo. |
| Ejecución | 2–3 | Órdenes a mercado con redondeo al paso del par y espera tras un rechazo. **Sin OCO ni bracket**: el bot vigila SL y TP en cada ciclo, que es posible porque la API Spot v3 de MEXC no ofrece OCO. |
| Consistencia backtest ↔ paper | 2 | El backtest del NB5 es diario y el bot es intradía: **las reglas son distintas**. El pronóstico del backtest es uno de los agentes del bot. El deslizamiento real contra el supuesto todavía no se mide (requiere modo real). |

**Estimado propio: ≈ 85–88/100** (factor = nivel ÷ 4, ponderado por criterio; el rango sale de los indicadores calificados «3–4»). Lo que más falta para 100:
- tiempo real con WebSocket;
- días de separación y más líneas base en los notebooks;
- órdenes OCO;
- un backtest intradía de la estrategia Pro con las mismas reglas que el paper.

---

## Rutas de API usadas
| Servicio | Ruta | Método | Para qué | Autenticación |
|---|---|---|---|---|
| MEXC | `/api/v3/ticker/price` | GET | Último precio de los 10 pares | Pública |
| MEXC | `/api/v3/klines` | GET | Velas OHLCV (1m, 5m, 15m, 60m) | Pública |
| MEXC | `/api/v3/ticker/24hr` | GET | Cambio, máximo, mínimo y volumen de 24 h | Pública |
| MEXC | `/api/v3/time` · `/exchangeInfo` | GET | Hora del servidor y paso mínimo por par | Pública |
| MEXC | `/api/v3/account` · `/myTrades` | GET | Saldo y operaciones reales | HMAC-SHA256 |
| MEXC | `/api/v3/order/test` · `/order` | POST | Probar y enviar órdenes a mercado | HMAC-SHA256 |
| Polymarket | `gamma-api.polymarket.com/markets/slug/{slug}` | GET | Mercado de la hora (BTC/ETH Up or Down) | Pública |
| Polymarket | `clob.polymarket.com/book?token_id=` | GET | Libro de órdenes | Pública |
| Polymarket | `polymarket.com/api/geoblock` | GET | Revisa si el país puede operar | Pública |
| Telegram | `api.telegram.org/bot{token}/sendMessage` · `/getUpdates` | POST/GET | Avisos al celular | Token del bot |
| Yahoo Finance | `yfinance.download` | — | Datos diarios de 40 activos y respaldo | Pública |
