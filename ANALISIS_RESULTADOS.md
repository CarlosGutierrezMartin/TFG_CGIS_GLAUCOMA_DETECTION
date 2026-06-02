# Análisis de resultados del sistema

> **Actualización (auditoría v6.0 — resultado verificado).** Este documento se redactó sobre v3/v4. Se intentó **reentrenar de forma dirigida** con una **Tversky multiclase** asimétrica en la copa (α=0.3, β=0.7) con la intención de penalizar la sobre-segmentación causante del sesgo +0.077 del vCDR. **La auditoría de v6.0 demuestra que ese reentreno NO corrigió el sesgo**: en la convención `TI = TP/(TP+α·FP+β·FN)`, β>α es *recall-oriented* (penaliza más los FN; Salehi et al. 2017) y por tanto **favorece predecir más copa**, lo contrario de lo buscado. Empíricamente el sesgo se mantuvo en +0.077 y la copa no mejoró (IoU Test400 0.604 vs 0.627 con Dice). **Decisión final adoptada:** el sesgo se corrige **post-hoc** con una calibración afín ajustada en Validation400 (MAE de vCDR en Test400 0.085 → 0.064, sesgo → 0); es una transformación monótona, por lo que mejora MAE/interpretabilidad/calibración pero **no el AUC**. La discriminación queda limitada por la calidad de segmentación de la copa. Una variante *precision-oriented* (α>β) que ataque la sobre-segmentación en origen requeriría reentrenar y queda como trabajo futuro. **Documento coherente con la referencia técnica completa [`PIPELINE_DOCUMENTACION.md`](PIPELINE_DOCUMENTACION.md).** Las dos correcciones de rigor marcadas con ⚠️ siguen vigentes.

---

## 1. Segmentación — ¿qué tan bien está?

### Validación interna (K-Fold, mejor checkpoint por fold)

| Fold | IoU Disco | IoU Copa | MAE vCDR |
|------|-----------|----------|----------|
| 1    | 0.916     | 0.773    | 0.044    |
| 2    | 0.919     | 0.785    | 0.042    |
| 3    | 0.886     | 0.752    | 0.050    |
| 4    | 0.922     | 0.791    | 0.041    |
| 5    | 0.912     | 0.798    | 0.038    |
| **Media** | **0.911** | **0.780** | **0.043** |

### Evaluación final en Test400 (población completa)

| Métrica       | Media | Std   | Mediana |
|---------------|-------|-------|---------|
| IoU Disco     | 0.833 | 0.106 | 0.851   |
| IoU Copa      | 0.625 | 0.152 | 0.627   |
| MAE vCDR      | 0.083 | 0.085 | 0.060   |

Hay una caída apreciable entre validación interna y test. Los folds internos ven solo 80 imágenes de validación cada uno y el checkpoint se selecciona sobre ese conjunto — hay algo de optimismo estadístico. En test la población es de 400 imágenes nunca vistas, y el resultado baja ~0.08 en IoU de disco y ~0.15 en IoU de copa. Esto es normal pero hay que explicarlo en la memoria.

### Contexto con el estado del arte en REFUGE

Los equipos top del challenge (con arquitecturas más complejas, datos extra o entrenamiento multi-dataset) reportaban:

| Métrica   | Top del challenge | Este sistema |
|-----------|-------------------|--------------|
| IoU Disco | 0.88–0.94         | 0.833        |
| IoU Copa  | 0.72–0.83         | 0.625        |
| MAE vCDR  | 0.040–0.060       | 0.083 (mediana: 0.060) |

El sistema está dentro del rango razonable en disco, y en el límite bajo en copa. El MAE de vCDR en test queda por encima del rango top, aunque la mediana sí entra. Hay casos con error muy grande que tiran de la media hacia arriba.

---

## 2. El bias sistemático — el problema más importante

Los datos revelan un sesgo estructural que no es ruido, es un problema real:

| Variable | Media predicha | Media real | Diferencia |
|----------|----------------|------------|------------|
| vCDR     | 0.552          | 0.475      | **+0.077** |
| area_CDR | 0.340          | 0.259      | **+0.081** |
| rCDR     | 0.577          | 0.499      | **+0.078** |

**El modelo sobreestima sistemáticamente el tamaño de la copa en ~8 puntos porcentuales.** No es error aleatorio; es un sesgo direccional consistente en las tres métricas derivadas del área y los diámetros. Las consecuencias son directas:

- El MAE de vCDR se infla artificialmente: parte del 0.083 es este bias, no error de forma.
- El umbral de clasificación seleccionado (0.22) es muy bajo porque los scores de todos los casos están inflados por la sobreestimación.
- La especificidad baja (0.633 en test) se explica en parte aquí: muchos sanos tienen vCDR predicho elevado y rozan el umbral.

La causa probable es que la red aprende a predecir copa de forma levemente expansiva por la combinación de Dice loss (que penaliza no predecir copa más que predecirla de más) y la dificultad de los bordes copa-anillo, que son la región más ambigua de la imagen.

---

## 3. Clasificación — lectura honesta de las métricas

### Ablación en Validation400

| Score                  | AUC       | Sens (Youden) | Spec (Youden) |
|------------------------|-----------|---------------|---------------|
| GT vCDR (techo real)   | **0.899** | 0.692         | 0.955         |
| Score combinado        | **0.878** | 0.800         | 0.831         |
| Solo vCDR predicho     | 0.867     | 0.850         | 0.758         |
| Solo rCDR predicho     | 0.850     | 0.900         | 0.648         |
| ISNT-like              | 0.705     | 0.400         | 0.969         |

**Lo que dice la ablación:**

- **El techo real del sistema es AUC=0.899**, que es lo que se obtendría si el vCDR se calculara perfectamente desde la segmentación GT. Incluso con segmentación perfecta no llega a 1.0 porque hay casos de glaucoma con vCDR bajo y casos sanos con vCDR alto que son intrínsecamente ambiguos.
- El gap entre el techo (0.899) y el combinado (0.878) es **el coste directo de los errores de segmentación**. Reducir el MAE de vCDR mejoraría el AUC de clasificación de forma inmediata.
- El ISNT-like (AUC=0.705) no aporta nada relevante. Sin lateralidad real de la imagen (ojo izquierdo vs derecho), la señal es espuria. Podría eliminarse del score sin pérdida apreciable.
- El score combinado mejora al vCDR solo en ~0.011 AUC. La mejora es real pero modesta; no justifica complejidad adicional por sí sola, pero sí justifica el diseño como decisión conservadora.

### Test400 final

- **AUC = 0.860** con umbral calibrado para sens ≥ 0.90 en validación.
- El umbral da sens = 0.90 en validación pero solo 0.875 en test. El drop de 0.025 es pequeño y esperado (variabilidad muestral entre los 40 glaucomas de validación y los del test).
- **Spec = 0.633** implica que de cada 10 sanos, ~4 son clasificados como sospechosos. Con 360 sanos en test, eso son ~132 falsos positivos. En un sistema de cribado real este coste tiene implicación clínica directa (derivaciones innecesarias al especialista).

---

## 4. Casos anómalos en la auditoría de test

⚠️ **Corrección respecto a la versión previa de este documento.** En v4.0 la auditoría ROI de cobertura total (punto 13 del notebook) reporta para Test400 `roi_crop_missed_gt_disc = 0`, centros idénticos al algoritmo antiguo en 98.25% y desplazamiento p95 = 0 px. Es decir, **ya no hay imágenes con la papila fuera del recorte ROI**: los ~7 casos de ROI inválida que describía la versión anterior de este análisis **están resueltos** por el localizador robusto con guarda. El criterio automático del notebook concluye correctamente que la nueva ROI no obliga a reentrenar *por motivo de ROI*.

De los **22 casos sospechosos** detectados en test, **los 22 son `valid_gt_possible_model_error`**: GT válida pero el modelo falla (papila pequeña, artefactos, baja calidad, copa ambigua). Es decir, el error residual ya **no** procede de la localización, sino de la segmentación de la copa.

La std del MAE (0.087) es casi igual a la media (0.083), lo que señala una distribución de cola derecha pesada dominada por estos casos de copa difícil. En la memoria conviene reportar el MAE global y la mediana (0.060), y describir cualitativamente estos 22 casos. Es precisamente este error de copa el que ataca el reentreno Tversky de v5.

---

## 5. Propuestas de mejora

Las mejoras están ordenadas por **impacto esperado / esfuerzo necesario**, de menor a mayor coste.

---

### Mejora 1 — Corrección del bias de vCDR

**Coste:** ~30 minutos. **Impacto esperado:** reducción del MAE y mejor interpretabilidad clínica del biomarcador.

⚠️ **Corrección de rigor respecto a la versión previa.** La versión anterior afirmaba "+0.03–0.05 AUC" para esta mejora; **es incorrecto**. Una corrección lineal (afín) del vCDR es una transformación **monótona** del score, y el AUC es invariante a transformaciones monótonas → **el AUC no cambia** (solo se observa un efecto despreciable de des-saturación del clip). El beneficio real es **MAE + interpretabilidad**. El AUC solo sube atacando el error en **origen** (el reentreno Tversky de v5) o mediante **predicción selectiva** (gate de la sección 15). Esta corrección post-hoc se implementa igualmente en la sección 15 (C.1), ajustada en Validation400 y aplicada a Test400.

La sobreestimación sistemática de +0.077 en vCDR es corregible post-hoc con una calibración lineal calculada en Validation400, sin tocar los modelos:

```python
# Regresión lineal sobre pares (pred_vCDR, true_vCDR) de Validation400
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(pred_vcdr_val.reshape(-1,1), true_vcdr_val)
vcdr_pred_corregido = reg.predict(vcdr_pred.reshape(-1,1))
```

Esto no requiere reentrenar nada. Mejoraría el MAE directamente y reduciría los falsos positivos en sanos con copa moderada al desinflar el score.

---

### Mejora 2 — Simplificar el score eliminando ISNT-like

**Coste:** ~15 minutos. **Impacto:** limpieza del sistema, más interpretabilidad.

Con AUC = 0.705, el indicador ISNT-like añade ruido al score porque sin lateralidad real la señal es espuria. El score simplificado:

```
risk_score = 0.80 × norm(vCDR, [0.45, 0.85]) + 0.20 × norm(rCDR, [0.50, 0.85])
```

daría un sistema más robusto y más justificable clínicamente que el actual con tres términos.

---

### Mejora 3 — Red de localización de papila dedicada

**Coste:** 1–2 semanas. **Impacto esperado:** eliminar los 7 casos con ROI inválida, +0.03–0.06 AUC.

Los 7 casos persistentes con ROI inválida son el límite del localizador heurístico. Entrenar un detector ligero (U-Net pequeño o YOLOv8-nano) específicamente para localizar la papila sobre las 1200 imágenes de REFUGE resolvería esos casos. Se pueden generar bounding boxes de entrenamiento a partir de las máscaras GT: el centroide del disco más un margen fijo define la región de interés. No requiere anotación manual adicional.

---

### Mejora 4 — Ampliar datos con datasets externos

**Coste:** 2–3 semanas. **Impacto esperado:** +0.05–0.10 AUC, reducción del bias.

400 imágenes de entrenamiento es el principal cuello de botella. Datasets públicos compatibles con REFUGE:

| Dataset     | Imágenes | Máscaras disco/copa | Acceso |
|-------------|----------|---------------------|--------|
| ORIGA       | 650      | Sí                  | Público |
| DRISHTI-GS  | 101      | Sí                  | Público |
| RIM-ONE-DL  | 313      | Sí                  | Público |

Usarlos para preentrenar o hacer fine-tuning aumentaría el conjunto efectivo de entrenamiento a 1000+ imágenes. El efecto sería especialmente notable en IoU de copa y en la reducción del bias de sobreestimación, que es probablemente un artefacto del pequeño tamaño muestral.

---

### Mejora 5 — Calibración de probabilidades

**Coste:** ~1 hora. **Impacto:** mejor interpretabilidad clínica del score.

El sistema produce scores sin calibración probabilística. Un score de 0.7 no equivale a una probabilidad de glaucoma del 70%. Calibrar con Platt scaling o isotonic regression sobre Validation400 mejoraría la interpretabilidad y posiblemente reduciría los falsos positivos al ajustar mejor la forma de la curva de decisión:

```python
from sklearn.calibration import CalibratedClassifierCV
# O directamente con isotonic regression sobre los scores de validación
from sklearn.isotonic import IsotonicRegression
ir = IsotonicRegression(out_of_bounds='clip')
ir.fit(scores_val, labels_val)
scores_test_calibrado = ir.predict(scores_test)
```

---

### Mejora 6 — Arquitectura más moderna (transformer-based)

**Coste:** 2–4 semanas. **Impacto:** variable, no garantizado con datos limitados.

Sustituir U-Net + InceptionResNetV2 por TransUNet, Swin-UNet o SegFormer. Los transformers capturan mejor el contexto global y han demostrado mejoras en copa óptica específicamente, porque la forma de la copa depende de relaciones de largo alcance con el disco. El inconveniente es que estos modelos son mucho más pesados: con Colab free-tier puede ser necesario reducir resolución (384×384) o batch size a 4, lo que no garantiza que el beneficio supere al costo en reproducibilidad y tiempo.

---

## 6. Resumen ejecutivo

El sistema tiene un rendimiento sólido para un TFG con las restricciones de REFUGE-Training400. El AUC de 0.860 en test es competitivo para un sistema de cribado con solo 400 imágenes de entrenamiento.

El problema más urgente es el **bias sistemático de +0.077 en vCDR** y, en general, la **sobre-segmentación de la copa** (IoU 0.604 vs 0.824 del disco en Test400). El reentreno Tversky **no** corrigió el problema en origen porque su parametrización (β>α en copa) es *recall-oriented* y favorece predecir más copa (ver nota superior y sección 11 de `PIPELINE_DOCUMENTACION.md`). La solución adoptada es la **corrección afín post-hoc**, que mejora el MAE y la interpretabilidad pero **no la discriminación** (es monótona). La frontera copa-anillo es inherentemente más ambigua que la frontera disco-fondo y es el límite natural de lo que se puede aprender con 400 imágenes; cerrar el gap hasta el techo GT-vCDR (~0.90–0.95) exige mejorar la segmentación de copa.

| Prioridad | Mejora | Coste | Estado / Impacto |
|-----------|--------|-------|---------|
| **1 (adoptada)** | **Corrección afín del vCDR (post-hoc, fit Val400)** | 30 min | **Hecho. MAE 0.085→0.064, sesgo→0; no cambia AUC (monótona).** |
| **2 (adoptada)** | **Calibración de probabilidad (Platt) + fiabilidad/Brier/ECE** | 1 h | **Hecho. Mejora Brier/interpretabilidad; no AUC.** |
| 3 (adoptada) | Gate de incertidumbre / predicción selectiva | 1 h | Hecho. Única palanca post-hoc que afecta a la discriminación operativa. |
| 4 (descartado) | Reentreno Tversky en copa (β>α) | ~5.5 h Colab | **No funcionó (recall-oriented). La variante precision-oriented α>β es trabajo futuro.** |
| 5 | Eliminar ISNT-like del score | 15 min | Medio (limpieza; el score simplificado vCDR+rCDR está en C.2). |
| 6 | Detector de papila dedicado | 1–2 semanas | Alto. ROI ya cubierta al 100% en Test400 por el localizador con guarda. |
| 7 | Datos externos (ORIGA, DRISHTI, RIM-ONE) | 2–3 semanas | Muy alto pero **excluido** por leakage/domain-shift/comparabilidad (sección 23 doc.); trabajo futuro controlado. |
| 8 | Arquitectura transformer | 2–4 semanas | Variable. |
