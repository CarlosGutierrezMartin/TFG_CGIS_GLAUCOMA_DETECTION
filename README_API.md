# API REST — Detección temprana de glaucoma

API en **FastAPI** que sirve el ensemble U-Net (InceptionResNetV2, 5 folds) del TFG
sobre el conjunto **REFUGE Test400**. Devuelve, en JSON, la imagen original, la máscara
real (GT), la predicción del modelo (overlays en base64), una **probabilidad calibrada
en %**, una **banda de riesgo no intrusiva** y la comprobación de **acierto** frente a la
etiqueta clínica real.

> ⚠️ **Aviso clínico.** Sistema de cribado / detección temprana de signos asociados a
> sospecha glaucomatosa. **No es un diagnóstico.** El diagnóstico de glaucoma requiere
> evaluación oftalmológica completa (campo visual, OCT, presión intraocular, historia
> clínica).

## 1. Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10–3.11 recomendado (compatibilidad con TensorFlow 2.15+).

## 2. Artefactos (cópialos de Google Drive a `artifacts/`)

La carpeta `artifacts/` está en `.gitignore` (no se versiona). Estructura esperada:

```
artifacts/
  models/
    model_fold_1.keras ... model_fold_5.keras     # de Models_v5_TverskyCup/
  external_validation_results.csv                  # para re-ajustar la calibración
  Test400/
    images/    # imágenes de REFUGE-Test400
    masks/     # máscaras GT disco/copa
    Glaucoma_label_and_Fovea_location.xlsx         # etiquetas (col. Label(Glaucoma=1))
```

Todas las rutas son configurables por variables de entorno (ver `api/config.py`):
`GLAUCOMA_MODELS_DIR`, `GLAUCOMA_VAL_RESULTS_CSV`, `GLAUCOMA_TEST_IMAGES_DIR`,
`GLAUCOMA_TEST_MASKS_DIR`, `GLAUCOMA_TEST_LABELS_XLSX`, `GLAUCOMA_USE_TTA`.

### ¿Por qué el CSV de validación?

La probabilidad calibrada (Platt) y el umbral del TFG se ajustan en Validation400 y **no
se guardan como artefacto** en el notebook. La API los **re-ajusta al arrancar** desde
`external_validation_results.csv`, reproduciendo exactamente la Sección 15
(corrección afín del vCDR → score `0.78·vCDR + 0.22·rCDR` → Platt → umbral sens≥0.85).
Los parámetros se cachean en `artifacts/calibration_params.json`.

## 3. Ejecución

```bash
uvicorn api.app:app --reload
# Docs interactivas: http://127.0.0.1:8000/docs
```

## 4. Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Estado: modelos cargados, nº de casos, parámetros de calibración. |
| `GET` | `/api/test-cases/random` | Caso aleatorio de Test400 con predicción, probabilidad y acierto. |
| `GET` | `/api/test-cases/{image_name}` | Caso concreto por nombre (p.ej. `T0123.jpg`). |

### Ejemplo de respuesta (`/api/test-cases/random`)

```json
{
  "image_name": "T0123.jpg",
  "probability_percent": 47.3,
  "risk_band": "intermedia",
  "risk_message": "Hallazgos intermedios; se aconseja revisión rutinaria.",
  "images": {
    "original": "<png base64>",
    "prediction_overlay": "<png base64>",
    "ground_truth_overlay": "<png base64>"
  },
  "biomarkers": {
    "vCDR": 0.61, "vCDR_corrected": 0.55, "rCDR": 0.58,
    "area_CDR": 0.34, "isnt_like_risk": 0.5
  },
  "ground_truth": { "true_label": 1, "true_vCDR": 0.58 },
  "decision": { "predicted_label": 1, "threshold": 0.288, "score": 0.41, "correct": true },
  "quality": { "mean_entropy": 0.07, "valid_anatomy": true, "disc_found": true, "cup_found": true },
  "disclaimer": "Sistema de cribado..."
}
```

**Overlays:** disco/anillo en **verde**, copa en **rojo**. Las imágenes son PNG en base64
(sin prefijo data URI); en el frontend usar `<img src="data:image/png;base64,{...}">` o
descodificar el buffer.

**Bandas de riesgo:** `< 30 %` baja · `30–60 %` intermedia · `> 60 %` moderada.

## 5. Verificación

```bash
# Métricas agregadas sobre N casos (contrastar con AUC≈0.86, sens 0.80, spec 0.75).
python -m scripts.smoke_test --n 40

# Paridad con el notebook (vCDR) si tienes el CSV post-hoc.
python -m scripts.smoke_test --n 0 --parity-csv artifacts/test_results_posthoc.csv
```

## 6. Notas de diseño

- Los 5 modelos se cargan **una sola vez en memoria** (el notebook los recargaba por
  petición por límites de RAM de Colab). En CPU una consulta tarda unos segundos; con GPU
  es casi instantánea.
- El núcleo (`api/glaucoma_core.py`) reutiliza *verbatim* las funciones del notebook
  (preprocesado, inferencia, postproceso, biomarcadores, overlay), de modo que la API
  reproduce el pipeline del TFG.
- No hay endpoint de subida de imagen (fuera de alcance), pero
  `glaucoma_core.preprocess_image_only` ya lo permitiría sin cambios.
