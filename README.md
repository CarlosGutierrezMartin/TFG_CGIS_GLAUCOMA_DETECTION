# TFG CGIS — Segmentación disco-copa y estimación de biomarcadores asociados a sospecha glaucomatosa

**Notebook principal: `TFG_GLAUCOMA_v6.0.ipynb`**

## Objetivo

Sistema de **segmentación de disco óptico y copa óptica** (U-Net con backbone `inceptionresnetv2`) y de **estimación de biomarcadores morfológicos** (`vCDR` y derivados) **asociados a sospecha glaucomatosa**, evaluado sobre el dataset REFUGE con validación externa reservada. **No es un sistema de diagnóstico automático de glaucoma**: el diagnóstico requiere campo visual, OCT, presión intraocular, historia clínica y exploración oftalmológica.

El diseño prioriza reproducibilidad, trazabilidad metodológica y ausencia de fuga de datos entre entrenamiento, validación externa y test.

> 📘 **Documentación completa:** [`PIPELINE_DOCUMENTACION.md`](PIPELINE_DOCUMENTACION.md) es la referencia técnica detallada (la "biblia" del proyecto), con cada decisión justificada, los resultados completos, el posicionamiento frente al estado del arte y el análisis de auditoría. Este README es solo un resumen operativo.

## Resultados finales (REFUGE-Test400, 400 imágenes nunca vistas)

| Métrica | Valor |
|---|---|
| IoU / Dice Disco | 0.824 / 0.901 |
| IoU / Dice Copa | 0.604 / 0.743 |
| MAE vCDR (crudo → corregido afín) | 0.085 → **0.064** |
| AUC (score combinado) | **0.862** |
| Punto operativo principal `sensitivity_at_least_0.85` | Sens 0.80 · Spec 0.75 |
| Alternativa equilibrada (Youden) | Sens 0.70 · Spec 0.83 |

Las métricas puntuales se acompañan de **intervalos de confianza bootstrap al 95 %** (celda 12.b). El umbral, la corrección afín del vCDR y la calibración de probabilidad se ajustan **siempre en Validation400** y se aplican a Test400 sin recalibrar.

**Posicionamiento (REFUGE, Orlando et al. 2020):** el AUC de clasificación del challenge osciló entre 0.846 y 0.989, con el vCDR de la GT alcanzando 0.947. Este sistema (0.862) se sitúa en la franja baja del leaderboard de 2018, resultado razonable y defendible para un TFG entrenado **solo con 400 imágenes**, sin datos externos y en Colab gratuito. El margen de mejora está en la segmentación de la copa, no en el clasificador. Detalles en la sección 22 de la documentación.

## Estructura del notebook (15 secciones)

1. Configuración del entorno
2. Carga y preparación del dataset
3. Protocolo experimental y preprocesamiento
4. Construcción del pipeline de entrenamiento
5. Definición del modelo
6. Entrenamiento K-Fold (segmentación disco-copa)
7. Carga y verificación de modelos entrenados
8. Inferencia y segmentación (ensemble + TTA)
9. Extracción de biomarcadores clínicos
10. Validación externa (REFUGE-Validation400)
11. Calibración clínica y estudio de ablación
12. Evaluación final en REFUGE-Test400 · **12.b Significancia estadística (IC bootstrap)**
13. Análisis visual de errores y auditoría técnica de ROI
14. Resumen final del experimento y exportación de resultados
15. Refinamiento clínico post-hoc y calibración (corrección afín del vCDR, ablación, calibración Platt + fiabilidad/Brier/ECE, gate de incertidumbre)

## Protocolo experimental

Separación estricta de datos (pilar metodológico del trabajo):

- `REFUGE-Training400`: entrenamiento y validación interna mediante K-Fold (5 folds).
- `REFUGE-Validation400`: validación externa y **calibración** de umbral, corrección de bias y probabilidad. No se usa para entrenar.
- `REFUGE-Test400`: evaluación final. **No se usa para ninguna decisión de diseño ni calibración.**

El K-Fold se aplica solo sobre `train_data`; la validación externa y el test usan conjuntos independientes. Se verifica la ausencia de fuga de datos por ruta y por hash MD5.

## Rutas esperadas en Google Drive

```text
/content/drive/MyDrive/TFG_Glaucoma_CLEAN/
  Refuge.zip                      # respaldo (descompresión solo en primera ejecución)
  Refuge/                         # dataset descomprimido (REFUGE-Training400, -Validation400-GT, Test400, ...)
  Models_v5_TverskyCup/           # modelos finales y resultados (CFG.SAVE_PATH)
    model_fold_1.keras ... model_fold_5.keras
    external_validation_results.csv, test_validation_results.csv
    test_metrics_ci.csv           # intervalos de confianza bootstrap
    calibration/                  # ablación, umbrales, diagrama de fiabilidad, Brier/ECE
    posthoc_refinement/           # corrección afín del vCDR y resumen post-hoc
    final_summary/final_report.md
```

## Preprocesamiento e inferencia

Centralizado en funciones reutilizables para entrenamiento, validación, inferencia y visualización:

- lectura RGB con OpenCV;
- recorte de ROI centrado en la papila (radio 200 px) con localizador robusto y **guarda** (solo interviene si el método original falla);
- redimensionamiento a `512 × 512`;
- conversión de máscaras REFUGE a clases semánticas (`0` fondo, `1` disco, `2` copa);
- CLAHE sobre luminancia;
- preprocesamiento específico del backbone `inceptionresnetv2`.

La inferencia aplica ensemble de los cinco folds, test-time augmentation horizontal y postprocesamiento anatómico (componente conectado principal y copa contenida en el disco).

## Biomarcadores, score y calibración

- diámetros verticales de disco y copa, `vCDR`, `hCDR`, `area_CDR`, `rCDR`, anillo neurorretiniano e indicador `ISNT-like` (aproximación geométrica, no evaluación clínica completa);
- `risk_score_combined = 0.70·norm(vCDR) + 0.20·norm(rCDR) + 0.10·ISNT_like`;
- **corrección afín del vCDR** ajustada en Validation400 (corrige el sesgo +0.077 → 0; mejora el MAE, no el AUC por ser monótona);
- **calibración de probabilidad (Platt)** con diagrama de fiabilidad, Brier y ECE;
- umbral de decisión calibrado en Validation400 (estrategia principal `sensitivity_at_least_0.85`).

## Ejecución en Colab

1. Abrir `TFG_GLAUCOMA_v6.0.ipynb`.
2. Ejecutar la **Sección 1** (entorno, dependencias, semillas, montaje de Drive).
3. **Secciones 2–4** (dataset, protocolo, pipeline).
4. **Sección 6** (entrenamiento K-Fold) **solo si se desea regenerar modelos** (~3–4 h). Si ya existen en `Models_v5_TverskyCup`, saltar a la 7.
5. **Sección 7** (verificación y carga del ensemble).
6. **Secciones 8–10** (inferencia, biomarcadores, validación externa).
7. **Sección 11** (calibración y ablación → fija umbral y puntos operativos).
8. **Sección 12 + 12.b** (test final + intervalos de confianza).
9. **Secciones 13–14** (análisis de errores, auditoría ROI, resumen).
10. **Sección 15** (refinamiento post-hoc y calibración de probabilidad).

> El reentrenamiento es costoso; las secciones de evaluación funcionan cargando los modelos ya guardados sin reentrenar.

## Logs

Prefijos consistentes, sin emojis: `[INFO]`, `[OK]`, `[ADVERTENCIA]`, `[ERROR]`, `[CRITICO]`.

## Nota sobre la pérdida de entrenamiento

La pérdida es **Tversky ponderada + 0.5·Focal**. La copa usa una parametrización asimétrica (α=0.3 < β=0.7) que, por la convención `TI = TP/(TP+α·FP+β·FN)`, es *recall-oriented* (Salehi et al. 2017) y **no reduce la sobre-segmentación** de la copa. La auditoría confirma que el sesgo del vCDR persiste tras el reentreno; por eso **se corrige post-hoc** (corrección afín, Sección 15). Justificación completa en la sección 11 de la documentación. Una variante *precision-oriented* (α>β) que ataque la sobre-segmentación en origen queda como trabajo futuro (requeriría reentrenar).

## Limitaciones

- Estima biomarcadores asociados a sospecha glaucomatosa; **no diagnostica glaucoma**.
- **Sesgo de vCDR (+0.077)** corregido post-hoc para MAE/interpretabilidad (no para AUC, por ser monótona); el margen real está en la segmentación de copa.
- Localización ROI mediante localizador robusto con guarda; un sistema de producción usaría un detector dedicado.
- Punto operativo principal `sensitivity_at_least_0.85` (cribado, prioriza sensibilidad); Youden como alternativa equilibrada.
- Arquitectura U-Net + InceptionResNetV2 y ensemble de 5 folds sin cambios.
- Indicador ISNT simplificado; no sustituye una evaluación oftalmológica completa.
- Validación limitada a REFUGE; **datos externos excluidos** por riesgo de fuga, domain shift y comparabilidad del benchmark (ver sección 23 de la documentación). No validado en otras cámaras, poblaciones ni condiciones de adquisición.
