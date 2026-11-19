# Mercado IA · App de pronóstico y trading simulado

App en Streamlit del proyecto de Deep Learning para Finanzas. Incluye:
- **40 activos** con datos en vivo de Yahoo Finance.
- **Pronósticos** del siguiente día con tus modelos MLP/LSTM del Notebook 5.
- **Indicadores** del PDF: SMA 20, Bollinger, RSI 14, MACD y ATR 14.
- **Portafolio simulado** con compra y venta.
- **Resultados en TE** del backtest.

## Estructura de la carpeta
```
app_bolsa/
├── app.py              ← la interfaz
├── nucleo.py           ← datos, indicadores, pronóstico y portafolio
├── requirements.txt
├── .streamlit/config.toml   ← tema oscuro
├── modelos/            ← aquí van los modelos de paquete_app.zip
└── resultados/         ← aquí van los CSV de paquete_app.zip
```

## Paso 1 · Copiar tus modelos y resultados
1. En Google Drive, descarga `ProyectoDL_Finanzas/entrega/paquete_app.zip` (lo genera el Notebook 6).
2. Descomprímelo **dentro de esta carpeta** `app_bolsa/`. Debe quedar así:
   - `modelos/AAPL.keras`, `modelos/AAPL_config.joblib`, … (80 archivos más `configuracion_final.json`)
   - `resultados/ficha_activos.csv`, `resultados/simulacion.csv`, …

## Paso 2 · Instalar Python
Si no lo tienes, instala **Python 3.11 o 3.12** desde python.org. En Windows, marca **"Add Python to PATH"** durante la instalación.

## Paso 3 · Usar las mismas versiones que en Colab (importante)
Los modelos se guardaron con la versión de TensorFlow y scikit-learn de Colab.
1. En cualquier notebook de Colab, corre:
   ```python
   import tensorflow as tf, sklearn; print(tf.__version__, sklearn.__version__)
   ```
2. Abre `requirements.txt` en Sublime Text y cambia las dos últimas líneas por esas versiones exactas. Por ejemplo:
   ```
   scikit-learn==1.6.1
   tensorflow==2.19.0
   ```

## Paso 4 · Instalar y correr la app
Abre una **terminal** (Windows: "Símbolo del sistema" o PowerShell; Mac: "Terminal") y escribe:

```bash
cd ruta/a/app_bolsa
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac / Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```
Se abre el navegador en `http://localhost:8501`. Para detener la app, presiona `Ctrl + C` en la terminal.

La siguiente vez solo necesitas: `cd app_bolsa`, activar `.venv` y `streamlit run app.py`.

## Paso 5 (opcional) · Publicarla en internet gratis
1. Crea una cuenta en github.com y un repositorio nuevo. Sube **toda** la carpeta `app_bolsa`, incluidos `modelos/` y `resultados/`.
2. Entra a share.streamlit.io con tu cuenta de GitHub → **Create app** → elige tu repositorio y el archivo `app.py`.
3. En **Advanced settings**, elige Python 3.11 o 3.12 y presiona **Deploy**.
4. Obtienes un link para compartir con tu profesor.

## Problemas comunes
| Problema | Solución |
|---|---|
| «No se encontró resultados/ficha_activos.csv» | Falta el Paso 1: descomprime `paquete_app.zip` dentro de `app_bolsa/`. |
| «No se pudo cargar el modelo…» | Las versiones no coinciden con Colab (Paso 3). Mientras tanto, la app usa los pronósticos guardados. |
| Faltan algunos activos | Yahoo limita las descargas. Espera un minuto y presiona «Actualizar datos». |
| `streamlit` no se reconoce | Activa el entorno `.venv` (Paso 4) o instala con `pip install -r requirements.txt`. |

## Notas
- El portafolio es **simulado**: está en USD, sin apalancamiento ni ventas en corto, y cobra 0.10% por operación. Se guarda en `portafolio.json`.
- USD/MXN, USD/JPY, USD/CAD y USD/CHF son **solo de análisis**: desde una cuenta en dólares su valor es constante.
- Proyecto académico. Las señales **no** son recomendaciones de inversión.
