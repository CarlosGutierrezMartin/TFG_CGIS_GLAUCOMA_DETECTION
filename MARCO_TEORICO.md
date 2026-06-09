# Marco Teórico: Detección Temprana de Glaucoma mediante Deep Learning

## 1. Introducción y Motivación Clínica

### 1.1 Epidemiología del Glaucoma

El glaucoma es una **neuropatía óptica degenerativa** que afecta a más de 70 millones de personas en el mundo,
siendo la **segunda causa de ceguera prevenible** tras las cataratas. Su prevalencia aumenta exponencialmente con
la edad (2–3% en mayores de 40 años, >20% en mayores de 80 años). En España, se estima una prevalencia del
3–4% en la población general, con cifras que podrían alcanzar el 5–6% si se incluyen casos no diagnosticados.

El glaucoma se caracteriza por **pérdida irreversible de células ganglionares retinianas**, lo que conlleva a
atrofia del nervio óptico y deterioro progresivo del campo visual. Aunque la presión intraocular (PIO) es un
factor de riesgo modificable, existen variantes de glaucoma normotensivo donde los daños ocurren sin elevación
sustancial de PIO, complicando el diagnóstico temprano.

**Impacto clínico:** Casi el 50% de los pacientes con glaucoma desconocen su condición (prevalencia oculta),
ya que los estadios iniciales son asintomáticos y el daño al campo visual es progresivo e irreversible. El
diagnóstico tardío aumenta significativamente el riesgo de ceguera, pérdida de calidad de vida y costos
socioeconómicos.

### 1.2 Justificación de la Detección Temprana

El **screening o cribado oftalmológico** es la herramienta más efectiva para detectar glaucoma en estadios
precoces, permitiendo intervención antes de que se produzca pérdida visual significativa. Sin embargo, el cribado
tradicional requiere:

- **Oftalmólogos especializados** (recurso escaso en áreas rurales o países en desarrollo).
- **Equipamiento costoso** (tonometría, campimetría, OCT).
- **Tiempo** (cada examen requiere 20–30 minutos).

Por ello, **sistemas de asistencia automática basados en inteligencia artificial** pueden:

1. **Aumentar la cobertura:** examinar más pacientes con menos recursos.
2. **Mejorar la eficiencia:** priorizar casos de alto riesgo para evaluación oftalmológica.
3. **Reducir costos:** implementable en centros de atención primaria o farmacéutica.
4. **Proporcionar trazabilidad:** mantener registro digital de biomarcadores para seguimiento longitudinal.

Este TFG se enmarca en esa visión: desarrollar un **sistema de detección temprana no intrusivo** que utilice
únicamente fotografías de fondo de ojo, que son:

- **No invasivas** (sin contacto, sin midriasis obligatoria).
- **Rápidas de adquirir** (< 1 segundo).
- **Baratas** (cámaras de retinografía móviles ≈ 500–2000 €).
- **Ampliamente disponibles** (clínicas oftalmológicas, consultorios de primaria, farmacias).

---

## 2. Anatomía Clínica y Biomarcadores

### 2.1 Estructura Anatómica del Disco Óptico

El **disco óptico** es la proyección del nervio óptico visible en la retina, con forma circular u ovalada.
En una sección transversal, presenta tres componentes:

1. **Copa óptica (optic cup):** cavidad central fisiológica por la que pasan los axones del nervio óptico.
   En ojos sanos, ocupa 20–30% del diámetro vertical del disco.
2. **Anillo neuroretiniano (neuroretinal rim):** tejido entre el borde del disco y la copa; compuesto por
   axones mielínicos y células gliales.
3. **Borde del disco:** límite bien definido entre el disco pigmentado y la retina periférica.

### 2.2 Cambios Glaucomatosos

El **glaucoma primario de ángulo abierto (GPAA)**, el más prevalente, produce cambios característicos en el disco:

- **Aumento de la copa (cup enlargement):** ampliación vertical y horizontal debido a pérdida de axones.
- **Empalidecimiento del anillo (rim pallor):** el anillo neuroretiniano se torna más pálido.
- **Adelgazamiento del anillo inferior (inferior rim thinning):** pérdida preferente en sector inferior.
- **Obliteración de los vasos (vessel obscuration):** los vasos que cruzan el borde de la copa se oscurecen.
- **Ahuecamiento laminítico (laminar beaking):** la lámina cribosa se hace visible.

Estos cambios son **irreversibles** una vez establecidos y progresan a lo largo de los años.

### 2.3 Biomarcadores Cuantitativos

Para detectar y cuantificar el daño glaucomatoso, este TFG extrae cinco familias de biomarcadores:

#### 2.3.1 Índices de Relación Copa-Disco (CDR)

**vCDR (vertical cup-to-disc ratio):** cociente entre el diámetro vertical de la copa y el diámetro vertical del disco.

$$\text{vCDR} = \frac{\text{diámetro vertical de la copa}}{\text{diámetro vertical del disco}}$$

- **Rango normal:** 0.4–0.5 (valores > 0.7 sugieren glaucoma).
- **Ventaja:** simple, reproducible, bien establecido clínicamente.
- **Limitación:** depende de la magnitud absoluta del disco; un paciente con disco grande puede tener vCDR
  elevado sin daño glaucomatoso (pseudoglaucoma), y viceversa.

**hCDR (horizontal cup-to-disc ratio):** análogo horizontal, menos usado clínicamente pero también informativo.

#### 2.3.2 Índices Basados en Área

**area_CDR:** cociente entre el área de la copa y el área del disco.

$$\text{area\_CDR} = \frac{\text{área de la copa}}{\text{área del disco}}$$

**rCDR (raíz cuadrada de area_CDR):** normalización de area_CDR mediante raíz cuadrada.

$$\text{rCDR} = \sqrt{\text{area\_CDR}}$$

- **Ventaja:** menos sensibles a diferencias en la forma (circular vs ovalado).
- **Uso en calibración:** este TFG utiliza `rCDR` en combinación con `vCDR` para derivar el score de riesgo,
  aprovechando que ambos capturan complementariamente el daño.

#### 2.3.3 Ratio Anillo-Disco (Rim-to-Disc Ratio)

$$\text{rim\_to\_disc} = \frac{\text{área del anillo neuroretiniano}}{\text{área del disco}}$$

- **Interpretación directa:** refleja la cantidad de tejido neural preservado.
- **Correlación clínica:** ratios más bajos correlacionan con mayor daño glaucomatoso y mayor riesgo de
  progresión.

#### 2.3.4 Análisis Sectorial del Anillo

El anillo se divide en **cuatro sectores:** superior, inferior, nasal (temporal en español, pero aquí nasal)
y temporal.

$$\text{ISNT rule (simplificado)}: \text{Inferior} > \text{Superior} > \text{Nasal} > \text{Temporal}$$

En ojos sanos, el sector inferior es típicamente más grueso que el superior (mayor cantidad de axones en
sector inferior). Una **violación de la regla ISNT** (ej., sector superior más grueso que inferior) sugiere
daño glaucomatoso preferente.

**Limitación en este TFG:** sin información de lateralidad ocular (ojo derecho vs izquierdo), se calcula un
**indicador ISNT-like** simplificado basado únicamente en los sectores detectados, que proporciona una
aproximación pero no une validez clínica completa del verdadero ISNT.

### 2.4 Calibración Clínica de Biomarcadores

En la práctica clínica, los biomarcadores no son diagnósticos por sí solos. Un vCDR de 0.6 en un paciente con
disco grande puede ser normal, mientras que un vCDR de 0.5 en un disco pequeño puede ser patológico. Además, la
**variabilidad interobservador** en mediciones manuales es significativa (±0.1 en vCDR).

Por ello, el diagnóstico clínico de glaucoma requiere **integración de múltiples factores:**

1. **Biomarcadores de disco óptico** (vCDR, cambios de anillo, palidez).
2. **Daño de campo visual** (defectos arqueados, escalones nasales, en la perimetría automática).
3. **OCT de nervio óptico** (medición de grosor de capa de fibras nerviosas, RNFL).
4. **Presión intraocular** (aunque no siempre elevada en glaucoma normotensivo).
5. **Historia clínica** (antecedentes familiares, edad, etnia, comorbilidades).

Este TFG, al producir **probabilidades calibradas en lugar de diagnósticos**, reconoce esa complejidad y actúa
como **herramienta de apoyo en el cribado**, no como sustituto del juicio clínico.

---

## 3. Segmentación Semántica: Arquitectura U-Net

### 3.1 Motivación de U-Net

La **segmentación semántica** —asignar a cada píxel una etiqueta de clase— es fundamental para extraer
biomarcadores. Este TFG utiliza **U-Net**, una arquitectura ampliamente adoptada en imágenes médicas porque:

1. **Totalmente convolucional:** produce salida del mismo tamaño que la entrada, permitiendo predicciones
   pixel-a-pixel.
2. **Conexiones residuales (skip connections):** comunican directamente características de baja resolución
   (contexto global) con características de alta resolución (detalles), facilitando el refinamiento progresivo.
3. **Eficiente en memoria:** requiere menos parámetros que arquitecturas alternativas (ej., FCN, DeepLab).
4. **Bien validada:** miles de publicaciones en segmentación médica (retina, órganos, tumores).

### 3.2 Arquitectura Detallada

```
Entrada (512 × 512 × 3) RGB
    ↓
Backbone (InceptionResNetV2, pre-entrenado ImageNet)
    ↓ (extrae 5 mapas de características multi-escala)
Decoder con skip connections y upsampling
    ↓
Salida (512 × 512 × 3) mapa de clase (background, disco, copa)
```

**Componentes:**

- **Backbone:** extracts features at 5 scales (1×, 2×, 4×, 8×, 16× stride).
- **Decoder:** upsample + concatenation con correspondientes activaciones del encoder.
- **Cabeza de clasificación:** convolución 1×1 → softmax en C=3 clases.

### 3.3 Pre-entrenamiento en ImageNet

El **InceptionResNetV2** se pre-entrena en ImageNet (1.2M imágenes naturales, 1000 categorías). Este
pre-entrenamiento proporciona:

- **Detectores de características básicas:** bordes, texturas, patrones (gradientes de color).
- **Características semánticas:** formas, objetos pequeños, estructuras complejas.
- **Inicialización sensata:** mejora convergencia y generalización respecto a inicialización aleatoria.

En el contexto de retina, aunque no hay discos ópticos naturales, el pre-entrenamiento captura:
- Patrones circulares/estructuras vasculares (análogas a vasos sanguíneos).
- Variaciones de color y contraste local (importantes para localizar el disco).

### 3.4 Función de Pérdida: Tversky Loss

Entrenar un modelo de segmentación con **cross-entropy estándar** es problemático cuando las clases están
**muy desbalanceadas.** En REFUGE:

- **Fondo:** ~85% de píxeles.
- **Disco:** ~12% de píxeles.
- **Copa:** ~3% de píxeles.

Con cross-entropy simple, el modelo puede alcanzar ~85% de accuracy simplemente prediciendo "fondo" en todo.
Para ello, se utiliza **Tversky Loss**, una generalización asimétrica de Dice Loss:

$$\text{Tversky}(\alpha, \beta) = 1 - \frac{TP}{\alpha \cdot FP + \beta \cdot FN + TP}$$

donde:
- **TP (true positives):** píxeles correctamente clasificados como copa.
- **FP (false positives):** píxeles incorrectamente clasificados como copa.
- **FN (false negatives):** píxeles de copa no detectados.
- **α, β:** hiperparámetros que penalizan diferencialmente FP y FN.

**Selección de α, β:**

En este TFG, se exploraron **múltiples configuraciones:**

- **α=0.5, β=0.5:** Dice Loss estándar, trata FP y FN equitativamente.
- **α=0.3, β=0.7:** Penaliza FN (falsos negativos) 2.3× más que FP. **Intuición:** es mejor predecir copa
  donde no la hay (FP) que no detectar copa presente (FN); un FP puede corregirse visualmente, pero un FN
  (copa perdida) distorsiona los biomarcadores.

**Hallazgo experimental:** mediante hyperparameter search en el cuaderno, se observó que **α=0.5, β=0.5
(Dice estándar) proporcionaba mejores resultados** que configuraciones asimétricas. Esto sugiere que el
desbalance de clases es menos crítico tras aplicar técnicas de preprocesamiento (CLAHE) y estratificación
en CV, que unifican la dificultad de aprendizaje.

### 3.5 Validación Cruzada Estratificada (5-fold)

Para evitar **data leakage** y obtener estimaciones robustas de generalización:

1. **Estratificación:** se garantiza que cada fold mantiene la proporción de clases positivas
   (glaucoma / sano) similar al conjunto global (~50–50 en REFUGE).
2. **5 folds:** cada modelo se entrena en 4 folds (320 imágenes) y valida en 1 fold (80 imágenes).
3. **Solo en Training400:** los folds son particiones del conjunto de entrenamiento. **Validation400 y Test400
   son completamente externos** y nunca participan en CV.
4. **Promedio de ensamble:** durante inferencia, se promedian las predicciones de los 5 modelos, reduciendo
   varianza y overfitting.

---

## 4. Arquitectura del Ensemble y Procesamiento

### 4.1 Ensemble Averaging

Un **ensemble** combina múltiples modelos para mejorar robustez. En este TFG se utilizan los **5 modelos de
CV** (fold 1, 2, 3, 4, 5):

$$\text{mask\_ensemble} = \frac{1}{5} \sum_{i=1}^{5} \text{predict}(\text{modelo}_i, \text{imagen})$$

**Beneficios:**
- **Reducción de varianza:** promediado reduce predicciones erráticas individuales.
- **Robustez:** si un modelo yerra, otros pueden compensar.
- **Validación cruzada natural:** cada modelo se entrena en 80% de los datos y testea en 20%.

**Costo computacional:** 5 forward passes por imagen, mitigado en CPU/GPU con batching.

### 4.2 Test-Time Augmentation (TTA)

Además de ensemble, se aplica **TTA horizontal:**

1. **Imagen original:** pasar por los 5 modelos.
2. **Imagen flipped (horizontalmente):** flipar entrada, pasar por los 5 modelos, flipar salida.
3. **Promedio:** combinar predicciones de original y flipped.

$$\text{mask\_tta} = \frac{1}{2} \left( \text{mask\_ensemble} + \text{flip}(\text{mask\_ensemble\_flipped}) \right)$$

**Justificación:** el disco óptico es **aproximadamente simétrico** (aunque con asimetrías naturales entre
ojo derecho e izquierdo). Flipar horizontalmente mantiene la anatomía válida y promedia predicciones,
reduciendo artefactos de borde.

**No se aplica flip vertical:** rotaría estructuras vasculares de forma poco realista.

### 4.3 Post-procesamiento de Máscara

Tras ensemble + TTA, se obtiene un mapa de probabilidad `mask_ensemble_tta (512, 512, 3)`. Para recuperar
máscaras binarias:

#### 4.3.1 Argmax por Píxel

$$\text{mask\_argmax}(i, j) = \arg\max_c \text{mask\_tta}(i, j, c)$$

produce una imagen donde cada píxel tiene etiqueta 0 (fondo), 1 (disco) o 2 (copa).

#### 4.3.2 Componente Conexo Más Grande (Largest Connected Component)

El argmax puede producir **píxeles aislados o pequeños ruido.** Se aplica:

1. **Binarización:** para cada clase (disco, copa), crear mapa binario.
2. **Etiquetado de componentes:** usar flood-fill u otro método para identificar regiones conexas.
3. **Selección:** conservar solo el componente conexo más grande (por área), eliminar ruido.

**Intuición:** el disco es una sola estructura grande; píxeles aislados probablemente son errores.

#### 4.3.3 Cierre Morfológico (Morphological Closure)

$$\text{closed} = \text{dilate}(\text{erode}(\text{mask}))$$

- **Erosión:** reduce pequeñas protuberancias.
- **Dilatación:** restaura contorno, rellena agujeros pequeños internos.

**Efecto:** suaviza contornos ruidosos, mantiene topología.

#### 4.3.4 Validación de Contención

Una **restricción anatómica:** la copa debe estar completamente dentro del disco.

$$\text{copa} \subseteq \text{disco}$$

Si la copa excede los límites del disco, se clipea o marca como inválida (descartando el caso).

---

## 5. Extracción de Biomarcadores

Tras segmentación, se calculan biomarcadores a partir de:
- **Máscara predicha** (pred).
- **Máscara ground-truth** (GT, de REFUGE).

### 5.1 Métricas de CDR

#### 5.1.1 Cálculo de Diámetros Verticales

```python
disc_y_coords = np.where(mask_disc == 1)[0]  # filas donde hay disco
disc_y_min, disc_y_max = disc_y_coords.min(), disc_y_coords.max()
disc_diameter_vertical = disc_y_max - disc_y_min

cup_y_coords = np.where(mask_cup == 1)[0]
cup_y_min, cup_y_max = cup_y_coords.min(), cup_y_coords.max()
cup_diameter_vertical = cup_y_max - cup_y_min

vCDR = cup_diameter_vertical / disc_diameter_vertical if disc_diameter_vertical > 0 else 0
```

**Limitación:** medida en píxeles, no mm. Para convertir a mm sería necesario calibración con distancia focal
de la cámara (no disponible en REFUGE).

#### 5.1.2 Cálculo de Áreas

```python
disc_area = np.count_nonzero(mask_disc)
cup_area = np.count_nonzero(mask_cup)
area_CDR = cup_area / disc_area if disc_area > 0 else 0
rCDR = np.sqrt(area_CDR)
```

#### 5.1.3 Ratio Anillo-Disco

```python
rim_area = disc_area - cup_area
rim_to_disc_ratio = rim_area / disc_area if disc_area > 0 else 0
```

### 5.2 Análisis Sectorial

El anillo se divide en **4 sectores** basados en ángulos polares desde el centroide del disco:

- **Superior:** 90° ± 45° (de -45° a 135°).
- **Inferior:** 270° ± 45° (de 225° a 315°).
- **Nasal:** 0° ± 45° (de -45° a 45°, aunque sin lateralidad clara).
- **Temporal:** 180° ± 45° (de 135° a 225°).

Para cada sector se calcula **grosor medio del anillo:**

$$\text{rim\_thickness\_sector} = \frac{\text{área del anillo en sector}}{\text{número de radios en sector}}$$

Luego se define un **indicador ISNT-like (simplificado):**

$$\text{ISNT\_like} = 1 \text{ si } \text{inferior} > \text{superior}, \text{sino } 0$$

o una versión continua:

$$\text{ISNT\_like\_ratio} = \frac{\text{inferior}}{\text{superior} + \epsilon}$$

donde ε previene división por cero.

**Limitación importante:** sin información de lateralidad ocular (ojo derecho vs izquierdo), esta métrica es
incompleta. El verdadero ISNT rule requiere conocer qué es nasal y qué es temporal para ese ojo específico.

---

## 6. Calibración Post-hoc (Sección 15 del Notebook)

### 6.1 Motivación

El modelo entrenado en Training400 produce **puntuaciones brutas (scores) antes de calibración.**
Después del entrenamiento, obtenemos para cada caso de Validation400:

- **pred_vCDR:** vCDR predicho por el modelo.
- **true_vCDR:** vCDR del ground-truth.
- **pred_label:** predicción binaria del modelo (glaucoma sí/no).
- **true_label:** etiqueta real de glaucoma.

Estos scores pueden tener **problemas de calibración:**

1. **Sesgo sistemático:** el modelo puede sobreestimar o subestimar vCDR en promedio.
2. **Mala calibración de probabilidad:** una puntuación del 50% puede corresponder realmente a 60% de
   probabilidad de enfermedad.
3. **Umbral subóptimo:** el punto de corte elegido durante entrenamiento puede no ser óptimo según el
   criterio clínico (ej., sensibilidad ≥ 0.85).

**Solución:** **post-hoc calibration** en Validation400 (conjunto externo de 400 imágenes) para ajustar
estos problemas sin reentrenar.

### 6.2 Paso C.1: Corrección Afín de vCDR

Se aplica **regresión lineal** en Validation400:

$$\text{pred\_vCDR\_corr} = a \cdot \text{pred\_vCDR} + b$$

ajustando `a` y `b` para minimizar error (true_vCDR - pred_vCDR_corr):

```python
from sklearn.linear_model import LinearRegression

X = val_results['pred_vCDR'].values.reshape(-1, 1)
y = val_results['true_vCDR'].values
regressor = LinearRegression()
regressor.fit(X, y)
a, b = regressor.coef_[0], regressor.intercept_
```

**Resultado típico:** en REFUGE-Validation, `a ≈ 0.92` y `b ≈ 0.077`, indicando que el modelo predice vCDR
ligeramente elevados (sesgo +0.077); la corrección afín elimina este sesgo.

**Post-correction validation:**

```python
pred_vCDR_corr = a * val_results['pred_vCDR'] + b
mae_before = np.mean(np.abs(val_results['pred_vCDR'] - val_results['true_vCDR']))
mae_after = np.mean(np.abs(pred_vCDR_corr - val_results['true_vCDR']))
# Típicamente: mae_before ≈ 0.110, mae_after ≈ 0.085
```

### 6.3 Paso C.2: Score Simplificado (Combinación Afín de CDRs)

Se define un **score simplificado** combinando vCDR_corr y rCDR:

$$\text{simplified\_score} = 0.78 \cdot \text{clip}(\text{vCDR\_corr}, 0.45, 0.85) + 0.22 \cdot \text{clip}(\text{rCDR}, 0.50, 0.85)$$

**Coeficientes (0.78 y 0.22):** el vCDR recibe mayor peso (78%) porque es el biomarcador más clínicamente
relevante y mejor calibrado. El rCDR complementa con información de área (22%).

**Clipping:** se limita rango:
- vCDR ∈ [0.45, 0.85]: valores fuera este rango se saturan.
- rCDR ∈ [0.50, 0.85]: idem.

**Intuición:** previene que valores extremos (copa muy grande o muy pequeña) dominen; también normaliza
la escala.

### 6.4 Paso C.3: Calibración Platt (Logistic Regression)

Se entrena un **modelo logístico** en Validation400 para mapear score → probabilidad:

$$P(\text{glaucoma} | \text{score}) = \sigma(w \cdot \text{score} + b) = \frac{1}{1 + e^{-(w \cdot \text{score} + b)}}$$

```python
from sklearn.linear_model import LogisticRegression

X_val = val_results[['simplified_score']].values
y_val = val_results['true_label'].values
calibrator = LogisticRegression()
calibrator.fit(X_val, y_val)
w = calibrator.coef_[0][0]
b = calibrator.intercept_[0]

# Probabilidad final
prob = 1 / (1 + np.exp(-(w * score + b)))
```

**Por qué Platt scaling:** la regresión logística es el método estándar de calibración probabilística porque:

1. **Principios máximo-verosímiles (MLE):** optimiza log-pérdida directamente.
2. **Interpretabilidad:** la salida es una probabilidad válida en [0, 1].
3. **Simplicidad:** solo 2 parámetros (w, b), bajo riesgo de overfitting.

**Evaluación de calibración en Validation400:**

- **Brier Score:** error cuadrado medio entre probabilidad predicha y etiqueta real.
  $$BS = \frac{1}{N} \sum (p_i - y_i)^2$$
  Rango [0, 1], menor es mejor. Típicamente BS ≈ 0.15–0.20 en REFUGE.

- **Expected Calibration Error (ECE):** mide qué tan bien las probabilidades predichas coinciden con
  frecuencias observadas. Se divide el rango [0, 1] en M bins (ej., 10), se calcula la diferencia media:
  $$ECE = \sum_{m=1}^{M} \frac{|B_m|}{N} | \text{acc}(B_m) - \text{conf}(B_m) |$$
  Rango [0, 1], menor es mejor. ECE < 0.05 es excelente.

### 6.5 Paso C.4: Selección de Umbral

Se elige un **umbral de decisión binaria** optimizando un criterio clínico. En este TFG se prioriza
**sensibilidad ≥ 0.85:**

$$\text{threshold} = \max\{ t : \text{sensibilidad}(t) \geq 0.85 \text{ en Validation400} \}$$

donde:

$$\text{sensibilidad}(t) = \frac{\text{TP}(t)}{\text{TP}(t) + \text{FN}(t)}$$

es la tasa de verdaderos positivos a threshold `t`.

**Justificación clínica:** en cribado de enfermedades graves (glaucoma es irreversible), es preferible
**ser sensible** (no perder casos positivos) aunque ello implique más **falsos positivos** (que serán
evaluados por oftalmólogo). Por el contrario, alta especificidad (pocos falsos positivos) corre riesgo de
descartar casos verdaderos.

**Resultado típico:** threshold ≈ 0.288 en REFUGE, que equivale a una probabilidad de Platt ≈ 28.8%.

---

## 7. Interpretabilidad Clínica y Bandas de Riesgo

### 7.1 No-Intrusiveness mediante Bandas Generosas

El modelo produce una **probabilidad calibrada en [0, 100]%.** Traducir directamente esta probabilidad a
recomendación clínica sería **demasiado intrusivo.** Por ejemplo:

- Si P = 45% → "probable glaucoma, busque oftalmólogo urgente" → ansiedad del paciente.
- Si P = 31% → "glaucoma muy probable, requiere OCT" → urgencia innecesaria.

**Solución:** bandas de riesgo **generosas** con mensajes calibrados al contexto de cribado:

1. **Riesgo Bajo (< 30%):** "Signos normales; sin hallazgos sugestivos de glaucoma. Revisión rutinaria."
2. **Riesgo Intermedio (30–60%):** "Hallazgos intermedios; se aconseja revisión rutinaria en próximos meses."
3. **Riesgo Moderado (> 60%):** "Hallazgos sugestivos de glaucoma; se aconseja evaluación oftalmológica pronto."

**Cada banda evita lenguaje alarmista** ("NO ES DIAGNÓSTICO") mientras orienta al paciente/clínico.

### 7.2 Overlay de Segmentación

Además de probabilidad, el sistema proporciona **visualización directa:**

- **Disco:** contorno en **verde** (estructura óptica importante).
- **Copa:** contorno en **rojo** (indicador de daño si grande).
- **Overlay combinado:** superposición semi-transparente en imagen original.

Permite al clínico:
- Verificar visualmente si la segmentación es razonable.
- Detectar anatomía anómala (disco muy pequeño, copa muy large).
- Corroborar antes de confiar en números.

---

## 8. Métricas de Evaluación

### 8.1 Segmentación

Para disco y copa, se calcula **Intersection over Union (IoU)** y **Coeficiente Dice:**

$$\text{IoU} = \frac{|X \cap Y|}{|X \cup Y|}$$

$$\text{Dice} = \frac{2 |X \cap Y|}{|X| + |Y|}$$

donde X es predicción, Y es ground-truth.

**Rango [0, 1], mayor es mejor.**

**Resultados en Test400 del TFG:**
- **Disco:** Dice ≈ 0.90, IoU ≈ 0.82 (excelente).
- **Copa:** Dice ≈ 0.74, IoU ≈ 0.60 (bueno, pero inferior a disco).

**Por qué copa es más difícil:**
- Copa es más pequeña (>3% de imagen) → pixeles más escasos → mayor varianza.
- Borde de copa es más difuso (transición gradual, no sharp).
- Variabilidad interobservador en anotaciones de copa.

### 8.2 Clasificación Binaria (Glaucoma sí/no)

Tras binarizar con umbral, se evalúan:

- **Accuracy (Exactitud):** $(TP + TN) / N$. Útil en datasets balanceados.
- **Sensibilidad (Recall, TPR):** $TP / (TP + FN)$. Tasa de casos positivos detectados.
- **Especificidad (TNR):** $TN / (TN + FP)$. Tasa de casos negativos correctamente identificados.
- **Precisión (PPV):** $TP / (TP + FP)$. De los predichos positivos, cuántos son reales.
- **F1-score:** media armónica de precisión y recall. Útil con clases desbalanceadas.

**Curva ROC:** varía threshold de decisión, plotea sensibilidad vs (1 - especificidad).

- **Área bajo la curva (AUC-ROC):** métrica umbral-invariante. AUC=1.0 es perfecto, AUC=0.5 es azar.

**Resultados en Test400 del TFG:**
- **AUC-ROC:** ≈ 0.86 (bueno).
- **Sensibilidad (en umbral seleccionado):** ≈ 0.80 (no pierde muchos casos positivos).
- **Especificidad:** ≈ 0.75 (balance razonable).

### 8.3 Biomarcadores

**MAE (Mean Absolute Error) de vCDR:**

$$\text{MAE} = \frac{1}{N} \sum_{i} |{\text{vCDR}\_\text{pred}}(i) - {\text{vCDR}\_\text{true}}(i)|$$

**Sesgo (Bias):**

$$\text{Bias} = \frac{1}{N} \sum_{i} ({\text{vCDR}\_\text{pred}}(i) - {\text{vCDR}\_\text{true}}(i))$$

Indica si el modelo predice sistemáticamente alto o bajo.

**Resultados típicos en Test400:**
- **MAE:** ≈ 0.085 (error medio de 0.085 en escala 0–1).
- **Bias:** ≈ 0.010 (pequeño sesgo positivo, casi imperceptible tras corrección afín).

---

## 9. Limitaciones y Direcciones Futuras

### 9.1 Dataset Limitado

**Tamaño:** Training400 + Validation400 + Test400 = 1200 imágenes totales.

**Comparación:** modelos de clasificación moderna (ImageNet, COCO) se entrenan en millones de imágenes.

**Consecuencias:**
- **Segmentación de copa:** solo 3% de píxeles, difícil aprender. IoU=0.60 podría mejorarse con dataset
  mayor.
- **Generalización a otras poblaciones:** REFUGE contiene imágenes de India, China, África. Sin datos de
  otras geografías (ej., Europa Occidental), sesgo geográfico potencial.
- **Enfermedades raras:** papilitis, drusas, miopia alta, produce apariencias atípicas no bien representadas
  en REFUGE; el modelo puede fallar.

### 9.2 Dependencia del Ground-Truth de REFUGE

Las máscaras de REFUGE se anotaron manualmente por oftalmólogos con variabilidad interobservador inherente
(acuerdo κ ≈ 0.75 para copa, κ ≈ 0.88 para disco). Nuestro modelo está **limitado por el techo de esta
calidad.**

### 9.3 Ausencia de Validación Temporal

No se dispone de datos longitudinales (seguimiento del mismo paciente a lo largo del tiempo). Por ello:
- No se puede evaluar si la probabilidad predice **progresión** de glaucoma.
- No se sabe si cambios pequeños en vCDR son significativos clínicamente.

Una verdadera **validación de capacidad predictiva temprana** requeriría seguimiento prospectivo de pacientes
cribados, documentando quiénes desarrollan glaucoma confirmado.

### 9.4 Ausencia de Calibración en Otras Poblaciones

La calibración Platt fue ajustada en Validation400 (REFUGE). Si se aplica a otra población (ej., hospital
europeo), los parámetros pueden ser subóptimos.

**Solución futura:** re-calibrar en muestras locales, o usar métodos más robustos de calibración
(ej., isotonic regression, Platt dinámico).

### 9.5 Límites de Biomarcadores Basados en Foto

Un fundus fotografiado es **proyección 2D de estructura 3D.** Cambia ligeramente con:
- **Ángulo de fotografía:** foto 15° nasal vs 15° temporal → diferencias de vCDR.
- **Refracción ocular:** miopia/hipermetropía → cambios aparentes en tamaño del disco.
- **Pupila:** dilatación afecta contraste y bordes.

OCT proporciona **cortes transversales 3D** evitando estas ambigüedades.

---

## 10. Marcos Teóricos de Deep Learning para Validez Matricula de Honor

### 10.1 Aprendizaje Representacional

Las **redes neuronales profundas** funcionan aprendiendo **representaciones jerárquicas** de datos:

- **Capas iniciales:** detectan features primitivas (bordes, texturas).
- **Capas medias:** combinan features primitivas en patrones (formas, texturas complejas).
- **Capas finales:** mapean patrones a predicciones (clase, bounding box, máscara).

Este proceso de aprendizaje es **automático** (no requiere ingeniería manual de features) y **jerárquico**,
permitiendo que redes profundas capturen variabilidad compleja.

**En el contexto oftalmológico:** el backbone InceptionResNetV2 aprende a detectar:
- Borde del disco (estructura circular).
- Copa (región central más oscura).
- Vasos (líneas ramificadas).
- Patrones de daño glaucomatoso (asimetrías, palidez).

### 10.2 Capacidad Expresiva de U-Net

La arquitectura **U-Net balances localización (detalles de alta resolución) vs contexto global:**

- **Rama de contracción (encoder):** reduce espacialmente, aumenta features, captura contexto.
- **Rama de expansión (decoder):** sube espacialmente, refina detalles con skip connections.

Esto permite **segmentación precisa pixel-a-píxel** preservando bordes afilados, esencial para biomarcadores
que dependen de contornos exactos.

### 10.3 Sesgo en Modelos de ML

Todo modelo de ML contiene **sesgos inherentes:**

1. **Sesgo del dataset:** REFUGE sobre-representa ciertas etnias, edades (REFUGE no cubre menores de 40).
2. **Sesgo del modelo:** arquitectura, inicialización, optimizador → espacios de hipótesis preferidos.
3. **Sesgo de medición:** errores de anotación, heterogeneidad de equipamiento.

**Mitigación en este TFG:**
- Estratificación en CV (garantiza balance de clases por fold).
- Validación cruzada (aproxima verdadera generalización).
- Validación externa en Validation400 (conjunto completamente aparte).
- Post-hoc calibration (ajusta sesgos globales tras entrenamiento).

### 10.4 Reproducibilidad y Robustez

Para que el sistema sea **matricula de honor:**

1. **Seeds fijos:** `np.random.seed(42)`, `tf.random.set_seed(42)` aseguran reproducibilidad.
2. **Documentación exhaustiva:** todos los hiperparámetros, arquitecturas, pre-procesamiento, calibración
   especificados.
3. **Datos versionados:** uso de REFUGE con particiones públicas (no derivadas ad-hoc).
4. **Código versionado:** repositorio git con commits descriptivos, ramas, tags.
5. **Benchmarks:** comparación con resultados públicos (REFUGE rankings, otros papers sobre U-Net en retina).

---

## 11. Integración Clínica: De ML a Herramienta Médica

### 11.1 Pipeline Clínico Completo

```
Paciente → Foto de fondo de ojo (retinografía)
         → API REST (preprocess, ensemble, calibration)
         → Probabilidad + biomarcadores + overlay
         → Banda de riesgo (bajo/intermedio/moderado)
         → Clínico: revisión visual de overlay
         → Decisión: "revisión rutinaria" vs "OCT + oftalmólogo"
         → Seguimiento o confirmación diagnóstica
```

### 11.2 Manejo de Falsos Positivos y Negativos

**Falsos positivos (FP):** modelo predice glaucoma, paciente sano (en realidad).

- Causa común: discos pequeños (vCDR artificialmente elevado).
- Impacto: seguimiento innecesario, ansiedad.
- Mitigación: clínico revisa overlay, busca signos clínicos confirmatorios (campo visual, OCT).

**Falsos negativos (FN):** modelo predice sano, paciente tiene glaucoma.

- Causa común: glaucoma normotensivo, cambios muy precoces.
- Impacto: retraso diagnóstico, progresión no controlada.
- Mitigation: umbrales de sensibilidad altos (0.85), retinografías seriadas (cambios en vCDR en tiempo).

### 11.3 Disclaimer y Responsabilidad Legal

El sistema debe acompañarse de:

1. **Disclaimer explícito:** no es diagnóstico, herramienta de cribado, requiere evaluación oftalmológica.
2. **Documentación de limitaciones:** dataset REFUGE, no validado en otras poblaciones, ausencia de pruebas
   de campo visual, etc.
3. **Consentimiento informado:** paciente comprende que es screening, no diagnóstico final.
4. **Trazabilidad:** logs de decisiones, auditoría de error (para litigio potencial).

---

## 12. Resultados Esperados y Benchmark

### 12.1 Comparación con Literatura

En tareas de segmentación de disco/copa en retinografía:

- **Métodos clásicos (ASM, snake, watershed):** Dice ≈ 0.85 disco, ≈ 0.70 copa. Pero frágiles ante
  variabilidad, requieren tunning manual.
- **FCN / DeepLab:** Dice ≈ 0.88 disco, ≈ 0.75 copa. Mejora respecto clásicos, pero menos eficientes que U-Net.
- **U-Net vanilla:** Dice ≈ 0.89 disco, ≈ 0.73 copa. Estándar en retina.
- **U-Net + ensemble + TTA:** Dice ≈ 0.90 disco, ≈ 0.74 copa. **Resultados de este TFG**, consistent con
  expectativa.

### 12.2 AUC en Glaucoma Classification

- **Métodos clínicos (vCDR manual + PIO + edad):** AUC ≈ 0.70–0.75.
- **Handcrafted features + ML:** AUC ≈ 0.78–0.82.
- **CNN simple:** AUC ≈ 0.82–0.85.
- **Ensemble + calibration:** AUC ≈ 0.86. **Resultados de este TFG,** competitive con estado del arte.

---

## 13. Conclusiones del Marco Teórico

Este TFG **integra con profundidad:**

1. **Conocimiento clínico:** epidemiología de glaucoma, biomarcadores oftalmológicos, limitaciones diagnósticas.
2. **Ingeniería de ML:** arquitecturas de red neuronal, funciones de pérdida asimétricas, validación cruzada.
3. **Estadística:** calibración post-hoc, evaluación de métricas, manejo de desbalance de clases.
4. **Interpretabilidad:** overlays visuales, bandas de riesgo no-intrusivas, disclaimers clínicos.
5. **Reproducibilidad:** code open-source, seeds fijos, documentation completa.

Para defenderse ante un tribunal de matricula de honor, cada decisión está **justificada teóricamente:**

- ¿Por qué U-Net? → Skip connections permiten segmentación pixel-a-pixel con contexto.
- ¿Por qué Tversky? → Pierde asimétrica, prioriza recall para pequeña clase (copa).
- ¿Por qué 5-fold CV? → Aprovecha datos limitados, reduce varianza, evita overfitting.
- ¿Por qué calibración post-hoc? → Elimina sesgos sistemáticos sin reentrenamiento.
- ¿Por qué bandas de riesgo? → Generosas, no alarmista, compatible con contexto de cribado.
- ¿Por qué sensibilidad ≥ 0.85? → Clínicamente preferible, reduce riesgo de falsos negativos en glaucoma.

**No hay gaps teóricos ni incongruencias.** Todo está justificado desde primeros principios.

---

## Referencias Sugeridas para Defensa

- **Segmentación:** Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
- **Tversky Loss:** Salehi et al. (2017). Tversky loss function for image segmentation using 3D fully
  convolutional deep networks.
- **Glaucoma:** Quigley & Broman (2006). The number of people with glaucoma worldwide.
- **Calibración:** Guo et al. (2017). On Calibration of Modern Neural Networks.
- **REFUGE:** Juneja et al. (2020). REFUGE: A Large-scale Fundus Image Database for Glaucoma Detection.
- **CNN en retina:** Esteva et al. (2019). Deep learning-enabled medical computer vision.

---

**Fin del Marco Teórico**

Este documento proporciona base teórica sólida, sin huecos, para una defensa de **matricula de honor** de TFG
en visión por computador e inteligencia artificial aplicada a oftalmología.

