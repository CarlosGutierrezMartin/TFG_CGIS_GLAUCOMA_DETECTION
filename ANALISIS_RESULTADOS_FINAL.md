# Análisis de resultados definitivo — versión final (v7.0)

> **Documento definitivo.** Se redacta sobre la **versión final del notebook** ([TFG_GLAUCOMA_v7.0.ipynb](TFG_GLAUCOMA_v7.0.ipynb)) y **sustituye** al análisis previo ([ANALISIS_RESULTADOS.md](ANALISIS_RESULTADOS.md)), que contenía cifras de iteraciones anteriores (v3/v4). Todas las métricas de este documento proceden de los *outputs* reales ejecutados en v7.0 (modelos `Models_v5_TverskyCup`, ensemble de 5 folds + TTA) y son **coherentes con la referencia técnica completa** [PIPELINE_DOCUMENTACION.md](PIPELINE_DOCUMENTACION.md). Encuadre clínico obligado: **sistema de segmentación disco-copa y estimación de biomarcadores asociados a *sospecha* glaucomatosa, no de diagnóstico automático**.

---

## 0. Resumen ejecutivo

El sistema segmenta disco y copa óptica con un ensemble U-Net + InceptionResNetV2 (5 folds, TTA), extrae biomarcadores morfológicos (vCDR, rCDR, área-CDR, ISNT-like) y produce un *score* de sospecha glaucomatosa. Evaluado bajo un protocolo estricto Train400 / Validation400 / Test400 sin fugas de datos:

| Bloque | Métrica (Test400) | Valor | IC95 % bootstrap |
|---|---|---|---|
| Segmentación | Dice Disco | **0.901** | [0.895, 0.906] |
| Segmentación | Dice Copa | **0.743** | [0.733, 0.755] |
| Biomarcador | MAE vCDR (crudo) | **0.085** | [0.077, 0.093] |
| Clasificación | AUC (*score* combinado) | **0.862** | [0.807, 0.910] |
| Operación | Sens / Spec @ sens≥0.85 | **0.80 / 0.75** | sens [0.67, 0.92] |

**Tres conclusiones que vertebran el trabajo:**

1. **El cuello de botella es la copa, no el clasificador.** El disco se segmenta de forma competitiva (Dice 0.901), pero la copa queda en la franja baja (Dice 0.743, IoU 0.604) por la ambigüedad intrínseca del borde copa-anillo. El techo real del sistema —AUC con vCDR calculado sobre la GT— es **0.903**; el AUC operativo (0.862) está a solo ~0.04 de ese techo, lo que confirma que el margen de mejora está en la **segmentación de copa**, no en el discriminador.
2. **Existe un sesgo estructural de sobre-segmentación de copa (+0.074 a +0.077 en vCDR).** Es direccional y consistente en las tres métricas de área. El reentreno dirigido con Tversky asimétrica **no lo corrigió** (su parametrización era *recall-oriented*); se corrige **post-hoc** con una calibración afín ajustada en Validation400 (MAE Test 0.085 → 0.064, sesgo → 0). Esta corrección es **monótona**, por lo que mejora MAE/interpretabilidad pero **no el AUC**.
3. **La única palanca post-hoc que mejora la discriminación operativa es la predicción selectiva** (gate de incertidumbre): abstenerse en el 8.8 % de casos más inciertos sube la sensibilidad de 0.800 a 0.848 y la especificidad de 0.786 a 0.810 sobre el 91.2 % auto-decidido.

El sistema es **sólido y honestamente reportado para un TFG** entrenado solo con las 400 imágenes de Training400, sin datos externos y en Colab gratuito. No persigue el estado del arte, sino el rigor metodológico defendible.

---

## 1. Protocolo y trazabilidad

- **Datos:** REFUGE Challenge, particiones oficiales fijas — Training400 (entrenamiento), Validation400 (calibración/ablación/selección de umbral) y Test400 (evaluación final, nunca vista).
- **Sin fugas:** `verify_no_data_leakage()` comprueba solapamiento por ruta y por hash MD5 entre conjuntos.
- **Modelos:** ensemble de 5 folds (U-Net con backbone InceptionResNetV2) + Test-Time Augmentation, almacenados en `Models_v5_TverskyCup`.
- **Regla de oro del test:** el umbral, el *score*, la corrección afín del vCDR y la calibración provienen **todos de Validation400**. En Test400 **no se recalibra ni se reentrena nada**.
- **Población de test:** 400 imágenes — **40 glaucoma / 360 sanas** (prevalencia 10 %, fuertemente desbalanceada; condiciona precisión y F1).

---

## 2. Segmentación disco-copa

### 2.1. Test400 (población completa, 400 imágenes)

| Métrica | Media | Std | Mediana | IC95 % bootstrap (B=2000) |
|---|---|---|---|---|
| IoU Disco | 0.824 | 0.080 | 0.835 | [0.816, 0.831] |
| Dice Disco | 0.901 | 0.058 | 0.910 | [0.895, 0.906] |
| IoU Copa | 0.604 | 0.139 | 0.601 | [0.591, 0.617] |
| Dice Copa | 0.743 | 0.113 | 0.751 | [0.733, 0.755] |
| IoU global clínico | 0.714 | 0.085 | 0.714 | — |
| Dice global clínico | 0.822 | 0.066 | 0.826 | — |

**Lectura.** El disco está bien resuelto (frontera disco-fondo de alto contraste). La copa es la región difícil: su Std (0.139) es 1.7× la del disco (0.080) y su mínimo cae a IoU 0.121, señal de una cola de casos duros (copas pálidas, bordes ambiguos, mala calidad). Los IC95 son estrechos en segmentación porque se promedian 400 valores por imagen: las medias son estadísticamente firmes.

### 2.2. Validación interna vs. Validation400 vs. Test400

| Métrica | Validación interna (K-Fold) | Validation400 | Test400 |
|---|---|---|---|
| IoU Disco | 0.912 | 0.853 | 0.824 |
| IoU Copa | 0.782 | 0.631 | 0.604 |
| Dice Disco | 0.943 | 0.918 | 0.901 |
| Dice Copa | 0.865 | 0.764 | 0.743 |
| MAE vCDR (crudo) | 0.044 | 0.086 | 0.085 |

La caída validación-interna → test (≈0.09 en IoU disco, ≈0.18 en IoU copa) es esperable: el *checkpoint* de cada fold se selecciona sobre sus 80 imágenes de validación interna (optimismo estadístico), mientras que Test400 son 400 imágenes nunca vistas. La **estabilidad Validation400 ↔ Test400** (apenas 0.03 en IoU, MAE casi idéntico) es la lectura tranquilizadora: el sistema **generaliza de forma consistente** entre dos conjuntos externos independientes.

---

## 3. Biomarcadores y el sesgo de vCDR

### 3.1. El sesgo estructural (el problema técnico central)

| Variable | Media predicha | Media real | Sesgo (pred − real) |
|---|---|---|---|
| vCDR | 0.549 | 0.475 | **+0.074** |
| área-CDR | 0.343 | 0.258 | **+0.084** |
| rCDR | 0.581 | 0.499 | **+0.082** |

*(En Validation400 el sesgo de vCDR es +0.077; ambos conjuntos coinciden en ~8 puntos.)*

El modelo **sobreestima sistemáticamente el tamaño de la copa en ~8 puntos porcentuales**. No es ruido: es un sesgo direccional consistente en las tres métricas derivadas del área y los diámetros. Consecuencias directas:

- **Infla el MAE de vCDR:** parte del 0.085 es sesgo, no error de forma.
- **Baja la especificidad y obliga a umbrales bajos:** muchos sanos quedan con vCDR predicho elevado y rozan el umbral.

**Causa probable:** la combinación de (1) la ambigüedad intrínseca del borde copa-anillo y (2) la parametrización *recall-oriented* de la copa en la pérdida (Tversky con β>α), que favorece predecir copa de más.

### 3.2. El reentreno Tversky no corrigió el sesgo

Se reentrenó de forma dirigida con una **Tversky multiclase asimétrica en la copa** (α=0.3, β=0.7) con la intención de penalizar la sobre-segmentación. Con la convención `TI = TP/(TP+α·FP+β·FN)`, **β>α es *recall-oriented*** (penaliza más los FN; Salehi et al. 2017) y por tanto **favorece predecir más copa** — lo contrario de lo buscado. Empíricamente: el sesgo se mantuvo en +0.074/+0.077 y la copa no mejoró (IoU Test 0.604). Atacar el sesgo en origen exigiría una variante **precision-oriented (α>β)** y reentrenar; queda documentado como **trabajo futuro**.

### 3.3. Corrección afín post-hoc (solución adoptada)

Se ajusta una regresión lineal en Validation400 y se aplica a Test400, sin tocar los modelos:

```
vCDR_corr = 0.597 · vCDR_pred + 0.143      (ajustada en Validation400)
```

| Conjunto | MAE vCDR crudo | MAE vCDR corregido |
|---|---|---|
| Validation400 | 0.086 | **0.057** |
| Test400 | 0.085 | **0.064** |

El **sesgo medio en validación pasa de +0.077 a 0.000**. **Nota de rigor crítica:** la corrección afín es una transformación **monótona**, por lo que **el AUC no cambia** (0.858 crudo = 0.858 corregido en el vCDR). Su beneficio es **MAE, interpretabilidad clínica y calibración del biomarcador**, no la discriminación.

---

## 4. Clasificación

### 4.1. Ablación de *scores* en Validation400 (AUC)

| *Score* | AUC | Sens / Spec (Youden) |
|---|---|---|
| **GT vCDR (techo real)** | **0.903** | 0.700 / 0.955 |
| GT rCDR | 0.894 | 0.700 / 0.933 |
| GT área-CDR | 0.894 | 0.700 / 0.933 |
| *Score* combinado (vCDR+rCDR+ISNT) | 0.862 | 0.800 / 0.828 |
| vCDR dominante | 0.859 | 0.800 / 0.830 |
| Solo vCDR (predicho) | 0.856 | 0.825 / 0.806 |
| vCDR crudo (predicho) | 0.855 | 0.825 / 0.805 |
| Estructural balanceado | 0.852 | 0.850 / 0.755 |
| Solo rCDR / área-CDR (predichos) | 0.819 | 0.850 / 0.699 |
| ISNT-like | **0.678** | 0.300 / 0.975 |

**Lo que dice la ablación:**

- **El techo real del sistema es AUC = 0.903** (vCDR sobre segmentación GT). Incluso con segmentación perfecta no llega a 1.0: hay glaucomas con vCDR bajo y sanos con vCDR alto, intrínsecamente ambiguos.
- **El gap techo (0.903) → combinado (0.862) es el coste directo de los errores de segmentación de copa.** Mejorar la copa sube el AUC; corregir el sesgo post-hoc no (es monótono).
- **El ISNT-like (AUC 0.678) no aporta señal discriminativa fiable.** Sin lateralidad real (OD/OS) la señal es espuria; añade poco al *score* combinado.
- El *score* combinado supera al solo-vCDR en ~0.006 AUC: mejora real pero modesta. Justifica el diseño como decisión conservadora, no como ganancia sustancial.

### 4.2. Test400 — punto operativo principal

*Score* combinado, umbral fijado en Validation400 para **sensibilidad ≥ 0.85** (estrategia de cribado):

| Métrica | Valor | IC95 % bootstrap |
|---|---|---|
| AUC | **0.862** | [0.807, 0.910] |
| Umbral | 0.288 | — |
| Sensibilidad | **0.800** | [0.667, 0.917] |
| Especificidad | 0.753 | [0.707, 0.796] |
| Precisión (VPP) | 0.264 | — |
| F1 | 0.398 | — |
| Exactitud | 0.758 | — |
| TP / FP / FN / TN | 32 / 89 / 8 / 271 | — |

El umbral fijado en validación (sens 0.85 allí) da **sens 0.80 en test**: una caída de 0.05 pequeña y esperable por la variabilidad muestral de solo 40 glaucomas (el ancho del IC de sensibilidad, [0.67, 0.92], lo confirma). La **especificidad 0.753** implica ~89 falsos positivos sobre 360 sanos: en cribado, este coste (derivaciones innecesarias) se acepta a cambio de no perder casos. La **precisión 0.264** es baja por la prevalencia del 10 % (es la métrica más castigada por el desbalance; no debe leerse como fallo del modelo sino como propiedad de la población).

### 4.3. Curva de puntos operativos (Test400, *score* combinado)

| Estrategia | Umbral | Sens | Spec | F1 |
|---|---|---|---|---|
| sens ≥ 0.95 | 0.175 | 0.950 | 0.494 | 0.292 |
| sens ≥ 0.90 | 0.212 | 0.925 | 0.600 | 0.335 |
| **sens ≥ 0.85 (principal)** | **0.288** | **0.800** | **0.753** | **0.398** |
| Youden (equilibrado) | 0.351 | 0.700 | 0.831 | 0.434 |
| spec ≥ 0.85 | 0.383 | 0.650 | 0.861 | 0.448 |
| spec ≥ 0.90 (mejor F1) | 0.435 | 0.575 | 0.903 | 0.469 |

La tabla hace explícito el *trade-off* clínico: el cribado debe operar en la franja de alta sensibilidad (arriba), asumiendo más falsos positivos. Youden ofrece el punto más equilibrado (sens 0.70 / spec 0.83) como alternativa.

---

## 5. Significancia estadística

Las métricas puntuales de Test400 se acompañan de **intervalos de confianza bootstrap al 95 %** (percentil; B = 2000 remuestreos con reemplazo). Con solo 40 glaucomas, una diferencia de pocas centésimas de AUC **no es distinguible del ruido**; reportar el intervalo es más honesto que un único número.

Implicación clave: el IC del AUC es **[0.807, 0.910]** — ancho ~0.10. Por tanto, las diferencias de ~0.01 AUC entre variantes de *score* (combinado 0.862 vs. simplificado 0.852 vs. vCDR-solo 0.856) son **estadísticamente indistinguibles**, lo que justifica preferir el *score* más simple e interpretable sin penalización real. No se realizan comparaciones entre versiones del modelo; se describe únicamente la versión final.

---

## 6. Refinamiento clínico post-hoc (sección 15 del notebook)

Cuatro técnicas post-hoc, todas ajustadas en Validation400 y aplicadas a Test400 sin reinferir:

**C.1 — Corrección afín del vCDR.** Ya descrita (§3.3). MAE Test 0.085 → **0.064**, sesgo → 0; AUC invariante (monótona).

**C.2 — *Score* simplificado sin ISNT.** Comparativa final de *scores* (AUC):

| *Score* | Val400 | Test400 |
|---|---|---|
| vCDR-solo corregido | 0.855 | 0.858 |
| **Simplificado (vCDR+rCDR)** | 0.843 | **0.852** |
| Combinado (con ISNT) | — | 0.862 |

El *score* simplificado vCDR+rCDR (sin ISNT) pierde solo ~0.01 AUC frente al combinado —indistinguible dentro del IC (§5)— a cambio de mayor interpretabilidad clínica y robustez. Con umbral sens≥0.85 (0.143), en Test da **sens 0.800 / spec 0.786**. Es una **simplificación defendible**.

**C.3 — Calibración de probabilidad (Platt) — resultado honesto.**

| Métrica | Crudo | Calibrado (Platt) |
|---|---|---|
| Brier (test) | **0.0725** | 0.0769 |
| ECE (test) | **0.0246** | 0.0707 |

**Hallazgo a reportar con honestidad:** la calibración Platt **no mejoró** la calibración en test; la empeoró ligeramente. La interpretación correcta es que **el *score* crudo ya estaba bien calibrado** (Brier 0.072, ECE 0.025 son buenos) y ajustar una sigmoide de Platt sobre solo 40 positivos de Validation400 introdujo un desplazamiento que no generaliza. Conclusión metodológica: se reporta la **buena calibración del *score* crudo** y se documenta que Platt no aportó. El AUC es invariante (calibración monótona). Este es exactamente el tipo de resultado negativo que conviene reportar en lugar de ocultar.

**C.4 — Predicción selectiva (gate de incertidumbre).** Se abstiene de decidir en los casos de mayor entropía (umbral 0.160 fijado en Val400) y se derivan a revisión manual:

| Configuración | Sens | Spec | Cobertura |
|---|---|---|---|
| Test completo | 0.800 | 0.786 | 100 % (400/400) |
| Test auto-decidido | **0.848** | **0.810** | 91.2 % (365/400) |

Se abstiene en 35 casos (7 glaucoma, 28 sanos). Es la **única palanca post-hoc que mejora la discriminación operativa real** (sens +0.048, spec +0.024), a cambio de derivar el 8.8 % de casos más inciertos a un especialista — un comportamiento clínicamente sensato para un sistema de cribado.

---

## 7. Auditoría de ROI y análisis de errores

### 7.1. Auditoría ROI de cobertura total (1200 imágenes)

| Conjunto | GT válidas | Papila fuera del recorte | Centros idénticos al *legacy* | Desplaz. p95 |
|---|---|---|---|---|
| Training400 | 400 | 9 | 99.75 % | 0 px |
| Validation400 | 400 | 2 | 98.50 % | 0 px |
| **Test400** | 400 | **0** | 98.25 % | 0 px |

El localizador robusto **con guarda** solo interviene cuando el algoritmo original falla (la inmensa mayoría de centros son idénticos), de modo que **no hay distribution shift** que obligue a reentrenar. En Test400, **0 papilas fuera del recorte**: el error residual del sistema **ya no procede de la localización**, sino de la segmentación de la copa.

### 7.2. Casos de error

Los casos de mayor error de vCDR (cola derecha de la distribución de MAE, Std 0.081 ≈ media 0.085) son **`valid_gt_possible_model_error`**: GT válida pero el modelo falla por copa ambigua, papila pálida, artefactos o baja calidad. En la memoria conviene reportar **media + mediana** (0.085 / 0.064) y describir cualitativamente estos casos, ya que la mediana —más representativa— sí entra en el rango competitivo.

---

## 8. Posicionamiento frente al estado del arte (REFUGE)

Leaderboard oficial de REFUGE-Test400 (Orlando et al., 2020):

| Métrica | Top del challenge | Rango (12 equipos) | **Este sistema** |
|---|---|---|---|
| Dice Disco | 0.960 (CUHKMED) | 0.877–0.960 | **0.901** |
| Dice Copa | 0.884 (Masker) | 0.686–0.884 | **0.743** |
| MAE vCDR | 0.041 (Masker) | 0.041–0.154 | **0.085** |
| AUC clasificación | 0.989 (VRT) | 0.846–0.989 | **0.862** |

**Posicionamiento honesto.** El sistema se sitúa en la **franja baja del leaderboard (≈ puesto 11 de 12)** y por encima del último (AIML, AUC 0.846). Es **esperable y defendible**: los equipos del challenge usaron ensembles complejos, entrenamiento multi-dataset, datos adicionales y pipelines a resolución completa, mientras que aquí se entrena **solo con 400 imágenes de Training400**, sin datos externos y en Colab gratuito. El AUC 0.862 queda por debajo del techo GT-vCDR (0.903 interno; 0.947 en el protocolo oficial), confirmando que **el margen está en la segmentación de copa, no en el clasificador**.

> **Caveat de comparabilidad:** el leaderboard mide Dice sobre la imagen completa con el protocolo oficial; aquí el Dice se mide en el espacio del recorte ROI a 512×512. La comparación es orientativa (orden de magnitud), no una participación oficial.

---

## 9. Limitaciones y trabajo futuro

**El sistema hace:** segmentar disco/copa, calcular biomarcadores morfológicos (vCDR, hCDR, área-CDR, rCDR, ISNT-like) y generar un *score* de sospecha para priorizar derivaciones.

**El sistema NO hace:** diagnosticar glaucoma (requiere tonometría, campo visual, OCT, valoración clínica), estimar presión intraocular, ni operar sobre imágenes de muy baja calidad.

**Limitaciones técnicas:**

1. **Sesgo de vCDR (+0.074/+0.077):** sobre-segmentación de copa; se corrige post-hoc para MAE/interpretabilidad, pero el AUC sigue limitado por la calidad de segmentación de copa.
2. **Pérdida *recall-oriented* en la copa:** la Tversky con β>α no reduce la sobre-segmentación; una variante *precision-oriented* (α>β) es trabajo futuro.
3. **Dataset único (REFUGE):** sin validación en otras cámaras/poblaciones; los datos externos se excluyen deliberadamente (leakage / domain-shift / comparabilidad — §23 de la documentación).
4. **Tamaño del dataset:** 400 imágenes de entrenamiento; suficiente para un TFG, pequeño para uso clínico real.
5. **ROI heurística:** el localizador con guarda cubre el 100 % de Test400, pero un sistema de producción usaría un detector dedicado.

**Trabajo futuro priorizado** (impacto esperado / esfuerzo):

| Prioridad | Línea | Coste | Impacto esperado |
|---|---|---|---|
| Alta | Reentreno *precision-oriented* en copa (Tversky α>β) | ~5.5 h Colab | Ataca el sesgo en origen → ↑AUC (única vía real de subir discriminación junto al gate) |
| Alta | Datos externos armonizados (ORIGA, DRISHTI-GS, RIM-ONE) | 2–3 sem | ↑IoU copa, ↓sesgo; requiere control estricto de leakage/domain-shift |
| Media | Detector de papila dedicado (U-Net pequeño / YOLO-nano) | 1–2 sem | Robustez ROI en producción (ya 100 % en Test400) |
| Media | Arquitectura transformer (TransUNet, Swin-UNet, SegFormer) | 2–4 sem | Variable; no garantizado con 400 imágenes |

---

## 10. Tabla consolidada final

| Bloque | Métrica | Val. interna (K-Fold) | Validation400 | **Test400** | IC95 % (Test) |
|---|---|---|---|---|---|
| Segmentación | IoU Disco | 0.912 | 0.853 | **0.824** | [0.816, 0.831] |
| Segmentación | IoU Copa | 0.782 | 0.631 | **0.604** | [0.591, 0.617] |
| Segmentación | Dice Disco | 0.943 | 0.918 | **0.901** | [0.895, 0.906] |
| Segmentación | Dice Copa | 0.865 | 0.764 | **0.743** | [0.733, 0.755] |
| Biomarcador | MAE vCDR (crudo) | 0.044 | 0.086 | **0.085** | [0.077, 0.093] |
| Biomarcador | MAE vCDR (corregido C.1) | — | 0.057 | **0.064** | — |
| Clasificación | AUC (*score* combinado) | — | 0.862 | **0.862** | [0.807, 0.910] |
| Clasificación | Sens / Spec @ sens≥0.85 | — | 0.85 / 0.74 | **0.80 / 0.75** | sens [0.67, 0.92] |
| Predicción selectiva | Sens / Spec (cob. 91.2 %) | — | — | **0.848 / 0.810** | — |
| Calibración | Brier / ECE (crudo) | — | — | **0.072 / 0.025** | — |

**Lectura ejecutiva final.** El sistema alcanza AUC 0.862 en Test400 con segmentación de disco competitiva (Dice 0.901) y copa en el límite bajo (Dice 0.743), entrenado solo con 400 imágenes. El problema técnico central es la **sobre-segmentación de copa y el sesgo de vCDR**, corregido post-hoc para MAE/calibración (no para AUC, por ser monótono). El **techo realista (GT-vCDR) es 0.90–0.95**; cerrar el gap exige mejorar la segmentación de copa —no el clasificador— mediante reentreno *precision-oriented* o datos externos armonizados. El valor del trabajo no reside en superar el leaderboard, sino en un **pipeline completo, modular, auditado y honestamente reportado**, con cuantificación de incertidumbre (IC bootstrap), diagnóstico y corrección de un sesgo, e identificación rigurosa de las palancas que sí y que no mejoran la discriminación.

---

*Referencias: Orlando, J. I., et al. (2020). REFUGE Challenge. Medical Image Analysis, 59, 101570. — Salehi, S. S. M., et al. (2017). Tversky loss function for image segmentation. MICCAI MLMI.*
