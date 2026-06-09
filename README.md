# Segmentacion disco-copa y biomarcadores para cribado de glaucoma

Proyecto de TFG centrado en la segmentacion del disco optico y la copa optica en retinografias, con estimacion de biomarcadores morfologicos asociados a sospecha glaucomatosa. El sistema usa una U-Net con backbone InceptionResNetV2, ensemble de 5 folds y evaluacion sobre REFUGE con separacion estricta entre entrenamiento, validacion y test.

Este repositorio contiene el codigo reproducible, la API REST, una interfaz web ligera y la documentacion tecnica del experimento final. No incluye datasets, modelos entrenados ni artefactos generados.

## Alcance clinico

El proyecto es una herramienta de cribado experimental. No emite diagnosticos ni sustituye una valoracion oftalmologica completa. El diagnostico de glaucoma requiere, como minimo, exploracion clinica, OCT, campo visual, presion intraocular e historia del paciente.

## Resultados principales

Evaluacion final sobre REFUGE-Test400, sin usar el conjunto de test para calibrar umbrales ni ajustar parametros:

| Metrica | Resultado |
|---|---:|
| IoU / Dice disco | 0.824 / 0.901 |
| IoU / Dice copa | 0.604 / 0.743 |
| MAE vCDR crudo | 0.085 |
| MAE vCDR con correccion afin | 0.064 |
| AUC score combinado | 0.862 |
| Punto operativo principal | sensibilidad 0.80, especificidad 0.75 |

La calibracion del umbral, la correccion afin de vCDR y la calibracion de probabilidad se ajustan exclusivamente en REFUGE-Validation400 y se aplican despues a REFUGE-Test400.

## Estructura

```text
.
├── api/                         # FastAPI, inferencia, calibracion y utilidades
├── frontend/                    # Interfaz HTML/CSS/JS sin build
├── scripts/                     # Pruebas offline del pipeline
├── TFG_GLAUCOMA_v7.0.ipynb      # Notebook final sin salidas embebidas
├── PIPELINE_DOCUMENTACION.md    # Documentacion tecnica del experimento
├── ANALISIS_RESULTADOS_FINAL.md # Analisis de resultados final
├── MARCO_TEORICO.md             # Marco teorico
├── README_API.md                # Guia de ejecucion de la API
└── requirements.txt             # Dependencias Python
```

## Artefactos no versionados

Por tamano, licencias y reproducibilidad, los siguientes elementos se mantienen fuera de Git:

- modelos entrenados `.keras`;
- dataset REFUGE;
- resultados intermedios y salidas de evaluacion;
- entornos virtuales y configuraciones locales.

La API espera esta estructura local cuando se ejecuta el pipeline real:

```text
artifacts/
  models/
    model_fold_1.keras
    model_fold_2.keras
    model_fold_3.keras
    model_fold_4.keras
    model_fold_5.keras
  external_validation_results.csv
  Test400/
    images/
    masks/
    Glaucoma_label_and_Fovea_location.xlsx
```

Todas las rutas se pueden sobrescribir con variables de entorno. Consulta `README_API.md` para el detalle.

## Instalacion

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12 es la version recomendada para la API y los modelos exportados con Keras 3.

## Ejecucion rapida

Modo demo, sin modelos ni dataset:

```bash
GLAUCOMA_API_MODE=demo uvicorn api.app:app --reload
```

Interfaz web:

```text
http://127.0.0.1:8000/
```

Pipeline real, con `artifacts/` preparado:

```bash
uvicorn api.app:app --reload
```

Documentacion interactiva de la API:

```text
http://127.0.0.1:8000/docs
```

## Verificacion offline

```bash
python -m scripts.smoke_test --n 40
```

Para evaluar todos los casos disponibles:

```bash
python -m scripts.smoke_test --n 0
```

## Metodologia

- Entrenamiento y validacion interna con K-Fold sobre REFUGE-Training400.
- Calibracion externa en REFUGE-Validation400.
- Evaluacion final una sola vez sobre REFUGE-Test400.
- Verificacion de ausencia de fuga por ruta y hash.
- Ensemble de 5 modelos con test-time augmentation horizontal.
- Postprocesado anatomico para conservar componente principal y contener la copa dentro del disco.
- Extraccion de biomarcadores: vCDR, hCDR, area CDR, rCDR, anillo neurorretiniano e indicador ISNT-like simplificado.

## Documentacion

- `PIPELINE_DOCUMENTACION.md`: decisiones tecnicas, protocolo experimental, resultados y limitaciones.
- `ANALISIS_RESULTADOS_FINAL.md`: analisis final de metricas, intervalos y posicionamiento.
- `MARCO_TEORICO.md`: contexto clinico y tecnico.
- `README_API.md`: instalacion, artefactos, endpoints y pruebas de la API.

## Limitaciones

- Validacion limitada a REFUGE.
- No validado en otras camaras, poblaciones ni condiciones de adquisicion.
- La principal fuente de error esta en la segmentacion de la copa optica.
- El indicador ISNT-like es una aproximacion geometrica y no una evaluacion clinica completa.
- El sistema debe interpretarse como apoyo experimental al cribado, no como diagnostico.
