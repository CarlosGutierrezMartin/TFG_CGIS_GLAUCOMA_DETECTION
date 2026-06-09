# Documentación técnica del pipeline: Detección de glaucoma mediante segmentación de fondo de ojo

**TFG — Carlos Gutiérrez**
**Versión del notebook: v6.0** · Actualizado: junio 2026

Este documento es la referencia técnica completa del proyecto y la fuente de la que se nutre la memoria del TFG. Cada decisión de diseño se justifica explícitamente. Todos los números reportados proceden de la ejecución real del notebook.

---

## Índice

1. [Motivación clínica y contexto del problema](#1-motivación-clínica-y-contexto-del-problema)
2. [Dataset: REFUGE Challenge](#2-dataset-refuge-challenge)
3. [Visión global del pipeline](#3-visión-global-del-pipeline)
4. [Configuración del entorno](#4-configuración-del-entorno)
5. [Carga y preparación del dataset](#5-carga-y-preparación-del-dataset)
6. [Protocolo experimental y detección de fugas de datos](#6-protocolo-experimental-y-detección-de-fugas-de-datos)
7. [Localización de la región de interés (ROI)](#7-localización-de-la-región-de-interés-roi)
8. [Preprocesamiento de imagen y máscara](#8-preprocesamiento-de-imagen-y-máscara)
9. [Data augmentation](#9-data-augmentation)
10. [Arquitectura del modelo: U-Net con InceptionResNetV2](#10-arquitectura-del-modelo-u-net-con-inceptionresnetv2)
11. [Función de pérdida: Tversky ponderada + Focal](#11-función-de-pérdida-tversky-ponderada--focal)
12. [Entrenamiento K-Fold](#12-entrenamiento-k-fold)
13. [Ensemble e inferencia con TTA](#13-ensemble-e-inferencia-con-tta)
14. [Postprocesamiento anatómico](#14-postprocesamiento-anatómico)
15. [Extracción de biomarcadores clínicos](#15-extracción-de-biomarcadores-clínicos)
16. [Validación externa (REFUGE-Validation400)](#16-validación-externa-refuge-validation400)
17. [Calibración clínica, ablación y calibración de probabilidad](#17-calibración-clínica-ablación-y-calibración-de-probabilidad)
18. [Evaluación final en REFUGE-Test400](#18-evaluación-final-en-refuge-test400)
19. [Análisis visual de errores y auditoría técnica de ROI](#19-análisis-visual-de-errores-y-auditoría-técnica-de-roi)
20. [El sesgo de vCDR: diagnóstico y corrección](#20-el-sesgo-de-vcdr-diagnóstico-y-corrección)
21. [Significancia estadística](#21-significancia-estadística)
22. [Estado del arte en REFUGE y posicionamiento](#22-estado-del-arte-en-refuge-y-posicionamiento)
23. [Exclusión de datos externos: justificación](#23-exclusión-de-datos-externos-justificación)
24. [Resumen de resultados consolidados](#24-resumen-de-resultados-consolidados)
25. [Limitaciones y encuadre clínico](#25-limitaciones-y-encuadre-clínico)
26. [Apéndice: evolución del proyecto y decisiones de auditoría](#26-apéndice-evolución-del-proyecto-y-decisiones-de-auditoría)

---

## 1. Motivación clínica y contexto del problema

El glaucoma es la segunda causa de ceguera irreversible a nivel mundial. Su característica más peligrosa es que avanza sin síntomas durante años: cuando el paciente nota pérdida de visión, la neuropatía óptica ya es considerable. El único biomarcador estructural accesible de forma no invasiva, sin necesidad de equipamiento especializado más allá de una retinografía de campo posterior, es la relación copa/disco (CDR), especialmente en su componente vertical (vCDR).

Una retinografía muestra el fondo de ojo: la retina, los vasos sanguíneos y la papila óptica (o disco óptico). La papila es el punto por donde el nervio óptico sale del ojo hacia el cerebro. En su interior hay una zona excavada llamada copa óptica, rodeada por tejido nervioso llamado anillo neurorretiniano. En un ojo sano, la copa es relativamente pequeña respecto al disco. En un ojo con glaucoma, la presión intraocular o la susceptibilidad vascular va dañando progresivamente el anillo, haciendo que la copa crezca: el vCDR aumenta.

El objetivo de este TFG es construir un sistema automático que, dada una imagen de fondo de ojo (retinografía), segmente disco y copa óptica, calcule biomarcadores estructurales como el vCDR, y a partir de ellos genere un indicador de sospecha glaucomatosa. El sistema **no pretende diagnosticar glaucoma** —eso requiere tonometría, pruebas de campo visual, historia clínica— sino identificar casos de riesgo que merecen derivación al especialista.

---

## 2. Dataset: REFUGE Challenge

El dataset utilizado es REFUGE (Retinal Fundus Glaucoma Challenge), un conjunto público de referencia creado para la evaluación comparativa de algoritmos de segmentación de fondo de ojo y detección de glaucoma. Fue lanzado como un benchmark en el marco de MICCAI 2018 (Orlando et al., *Medical Image Analysis*, 2020).

### Estructura del dataset

| Partición              | Imágenes | Máscaras GT | Etiqueta glaucoma |
|------------------------|----------|-------------|-------------------|
| REFUGE-Training400     | 400      | Sí          | No (sí en metadatos derivados) |
| REFUGE-Validation400   | 400      | Sí          | Sí (40 glaucoma / 360 sanos) |
| REFUGE-Test400         | 400      | Sí          | Sí (40 glaucoma / 360 sanos) |

Cada imagen es una retinografía de campo posterior a color (JPG o PNG), con resoluciones variables típicamente en torno a 2124×2056 píxeles. Las máscaras son imágenes en escala de grises donde los valores `255`, `128` y `0` representan fondo, disco/anillo y copa respectivamente. El desbalance de clases (40 glaucoma frente a 360 sanos en validación y test, es decir 10 %) es relevante para la interpretación de las métricas de clasificación (sección 18).

### Por qué REFUGE y no otro dataset

- **Etiquetado público y curado:** las máscaras disco-copa han sido anotadas por retinólogos expertos; hay ground truth verificado para las tres particiones.
- **Complejidad representativa:** incluye ojos sanos y patológicos, con variabilidad en tamaño del disco, pigmentación, artefactos y calidad de imagen.
- **Evaluación reproducible:** al existir desde 2018 hay un leaderboard publicado con métricas comparables, lo que permite situar los resultados propios en contexto (sección 22).
- **Tamaño manejable para TFG:** 1200 imágenes es suficiente para entrenar y validar sin infraestructura industrial.

La decisión de **no incorporar datasets externos** (ORIGA, DRISHTI-GS, RIM-ONE) está justificada en la sección 23.

---

## 3. Visión global del pipeline

El sistema puede describirse como una cadena de transformaciones sucesivas:

```
Retinografía cruda
       ↓
Localización de la papila (ROI) — sin usar la máscara
       ↓
Recorte centrado en la papila + resize + CLAHE + preprocess backbone
       ↓
Red U-Net (InceptionResNetV2) — predice máscara semántica 0/1/2
       ↓
Ensemble de 5 modelos (K-Fold) + TTA horizontal
       ↓
Postprocesamiento anatómico (componentes conectados, copa dentro del disco)
       ↓
Extracción de biomarcadores (vCDR, hCDR, area_CDR, rCDR, ISNT-like)
       ↓
Corrección afín del vCDR (calibración post-hoc del sesgo, ajustada en Val400)
       ↓
Score de sospecha glaucomatosa + calibración de probabilidad (Platt)
       ↓
Clasificación binaria con umbral calibrado en Validation400
```

El diseño es deliberadamente modular: cada bloque puede validarse y auditarse por separado. Esto es especialmente importante en un contexto clínico, donde el fallo de un módulo debe poder detectarse y atribuirse sin que corrompa silenciosamente el resto. Es justamente esta modularidad la que permitió la auditoría que detectó y corrigió el sesgo del vCDR (sección 20).

---

## 4. Configuración del entorno

### Qué hace este bloque

El primer bloque del notebook instala dependencias, define la clase de configuración global `Config` (dataclass inmutable), monta Google Drive, fija semillas de aleatoriedad y activa optimizaciones de rendimiento para GPU.

### La clase Config

Toda la configuración del experimento está centralizada en un único dataclass Python con `frozen=True`:

```python
@dataclass(frozen=True)
class Config:
    SEED: int = 42
    IMG_SIZE: int = 512
    BATCH_SIZE: int = 8
    EPOCHS: int = 50
    BACKBONE: str = "inceptionresnetv2"
    N_SPLITS: int = 5
    CLASSES: int = 3
    LR_START: float = 1e-4
    SAVE_PATH: str = ".../Models_v5_TverskyCup"
    ...
```

**Por qué centralizar la configuración:** en proyectos de ML es habitual que los hiperparámetros estén diseminados por el código; esa dispersión genera errores al reproducir o comparar experimentos. Un dataclass inmutable en un punto único lo elimina. **Por qué `frozen=True`:** impide que una celda posterior modifique silenciosamente un valor; cualquier intento de cambio es un error inmediato. La semilla `SEED=42` se fija en `random`, `numpy` y `tensorflow` para reproducibilidad.

### Mixed precision (float16)

Se activa `mixed_float16` de TensorFlow: los pesos se almacenan en float32 (precisión) pero los cómputos de forward/backward se hacen en float16, reduciendo hasta el 50 % el uso de memoria de GPU y acelerando el entrenamiento en hardware con Tensor Cores (T4/A100 de Colab). Sin mixed precision, una U-Net con InceptionResNetV2 a 512×512 con batch 8 puede quedarse sin memoria. No se usa float16 puro porque los gradientes pequeños se truncarían a cero (*underflow*); mixed precision mantiene los acumuladores en float32.

### Montaje seguro de Google Drive

`mount_drive_safely()` verifica que el directorio del proyecto exista *después* del montaje, no solo que el comando no falle, porque en Colab el montaje puede completarse parcialmente o puede haberse pre-creado un directorio local con el mismo nombre.

---

## 5. Carga y preparación del dataset

### Estrategia de acceso a datos en Colab

El dataset REFUGE (~3 GB) vive en Google Drive. Acceder a Drive en cada iteración añade latencia de red. La solución adoptada es un **enlace simbólico local** que apunta a la carpeta de Drive ya descomprimida:

```
/content/dataset_local/Refuge  →  /content/drive/MyDrive/TFG_Glaucoma_CLEAN/Refuge
```

El código usa siempre `CFG.BASE_PATH` como ruta local y el SO resuelve el enlace; Drive (vía FUSE) cachea los bloques accedidos con frecuencia. Esto es más rápido que copiar y descomprimir el ZIP en cada reinicio. Si la carpeta no está descomprimida (primera ejecución), se usa `Refuge.zip` como respaldo; `FORCE_REEXTRACT = False` por defecto evita repetir la descompresión.

### Indexación de pares imagen-máscara

`get_all_pairs_robust()` empareja imágenes con sus máscaras de forma tolerante a inconsistencias de nomenclatura: **(1) por nombre normalizado** (eliminando sufijos `_mask`, `_seg`, etc.) y **(2) por número extraído** (el `001` de `T0001`) como fallback. Los datasets médicos raramente tienen nomenclatura homogénea, y un emparejamiento por igualdad de nombre fallaría en muchos casos reales.

---

## 6. Protocolo experimental y detección de fugas de datos

### Separación estricta de conjuntos

- **REFUGE-Training400:** único conjunto usado para entrenar. Se particiona internamente en 5 folds para cross-validation.
- **REFUGE-Validation400:** validación externa y calibración del umbral de decisión, la corrección afín del vCDR y la calibración de probabilidad. Los modelos no lo ven durante el entrenamiento.
- **REFUGE-Test400:** evaluación final. **No se usa para ninguna decisión de diseño, selección de hiperparámetros ni calibración.**

Esta separación emula un escenario clínico realista y es **el pilar de la honestidad metodológica del trabajo**: todo lo que se ajusta (umbral, corrección de bias, calibración de probabilidad) se ajusta en Validation400 y se *aplica* a Test400 sin re-mirar.

### Verificación de fugas de datos (data leakage)

`verify_no_data_leakage()` realiza dos comprobaciones: **(1) solapamiento por ruta** (ninguna ruta de entrenamiento aparece en validación) y **(2) solapamiento por hash MD5** (detecta copias del mismo fichero con distinto nombre). Una fuga no detectada infla artificialmente las métricas, con consecuencias graves si se extrapola a uso clínico. Esta verificación es también la razón principal por la que no se mezclan datasets externos (sección 23).

---

## 7. Localización de la región de interés (ROI)

Uno de los bloques técnicamente más elaborados. Su diseño responde a una restricción práctica: **no se puede reentrenar el ensemble por un cambio de ROI** sin invalidar la consistencia entrenamiento/inferencia.

### El problema de la ROI

Las imágenes de REFUGE superan los 2000×2000 px y la papila ocupa una fracción pequeña. Pasar la imagen completa redimensionada a 512×512 dejaría la papila con poca resolución efectiva y empeoraría la segmentación. La solución es recortar un cuadrado centrado en la papila (`ROI_RADIUS = 200` px en la imagen original) y redimensionarlo a 512×512.

### Algoritmo original (legacy) y su fallo

El algoritmo original localiza la papila por brillo en el canal verde (desenfoque → CLAHE → umbral al percentil 99.3 → morfología → mayor componente → centroide). Falla cuando hay reflejos periféricos del anillo de iluminación, papila de bajo contraste o muy mala calidad.

### Localizador robusto y estrategia de guarda

`locate_roi_center_robust()` añade: estimación del FOV (descarta reflejos fuera del círculo iluminado), múltiples percentiles (99.5/99.0/98.5/97.5) y una puntuación multicriterio (brillo 0.45, circularidad 0.30, tamaño plausible 0.25, penalización por periferia/borde).

**La estrategia de guarda es clave:** `locate_roi_center()` usa primero el algoritmo original; solo si su centro **no** es anatómicamente plausible, sustituye por el robusto. Esto corrige los casos que fallaban sin alterar los que ya funcionaban. La cuantificación del efecto está en la auditoría ROI (sección 19): en Test400, **0 imágenes con la papila fuera del recorte** y 98.25 % de centros idénticos al algoritmo antiguo (desplazamiento p95 = 0 px), confirmando que la corrección **no obliga a reentrenar**.

### Alternativas descartadas

- **Detectores dedicados (YOLO/SSD/Faster R-CNN):** requieren anotar *bounding boxes* y una etapa de entrenamiento e inferencia adicional; el coste no se justifica cuando la heurística con guarda cubre el 100 % de Test400.
- **Referencia a la fóvea:** REFUGE-Training400 no incluye coordenadas de fóvea, solo validación/test, por lo que no puede usarse en el pipeline de entrenamiento.
- **Segmentación de imagen completa a baja resolución:** perdería la ventaja de resolución efectiva del recorte.

---

## 8. Preprocesamiento de imagen y máscara

`preprocess_image_and_mask()` aplica, en orden estricto:

1. **Lectura en RGB.** OpenCV lee BGR; se convierte a RGB porque InceptionResNetV2 fue preentrenado en RGB. Invertir canales degradaría la segmentación sin error visible.
2. **Localización ROI y recorte** (sección 7); relleno con ceros (imagen) o 255 (máscara, fondo de REFUGE) si el recorte sale de los bordes.
3. **Resize a 512×512:** interpolación bilineal para la imagen, vecino más cercano para la máscara (que solo tiene valores 0/128/255; bilineal crearía valores intermedios inválidos).
4. **Conversión de máscara a clases** (`decode_refuge_mask`): asignación por valor más cercano a 0/128/255 → clases 0/1/2, robusta a variaciones de codificación.
5. **CLAHE sobre el canal L (LAB):** mejora el contraste local sin distorsionar el color; tiles 8×8, clip 2.0. Más potente que la ecualización global en imágenes con iluminación no uniforme.
6. **`preprocess_input` del backbone:** normalización de InceptionResNetV2 (estadísticas de ImageNet); imprescindible al usar pesos preentrenados.
7. **One-hot de la máscara** (512×512×3), formato esperado por la salida softmax y la pérdida.

---

## 9. Data augmentation

Con 400 imágenes y un modelo de varios millones de parámetros, el riesgo de sobreajuste es real. Se usa Albumentations, que sincroniza la transformación de imagen y máscara.

| Transformación           | Parámetros                  | Prob. | Justificación                                                            |
|--------------------------|-----------------------------|-------|--------------------------------------------------------------------------|
| HorizontalFlip           | —                           | 0.50  | Ojo izquierdo/derecho son especulares; válido anatómicamente.            |
| ShiftScaleRotate         | shift 4 %, scale 8 %, rot 12°| 0.60 | Variaciones de centrado/tamaño del recorte; moderado.                    |
| VerticalFlip             | —                           | **ELIMINADO** | Invertiría superior/inferior, relevante para la regla ISNT.       |
| RandomBrightnessContrast | ±8 %                        | 0.30  | Variabilidad de iluminación entre cámaras.                               |
| GaussNoise               | var 5–20                    | 0.20  | Ruido de sensor en retinografías de baja calidad.                        |

**Por qué tan conservador:** un augmentation agresivo (rotaciones grandes, color fuerte, distorsiones elásticas) introduce artefactos o ejemplos anatómicamente imposibles; en segmentación médica el modelo debe aprender estructura clínica real, no artefactos. **Por qué se elimina VerticalFlip:** la regla ISNT establece que el anillo es más grueso en el sector inferior que en el superior; flipar verticalmente destruiría esa asimetría semántica. Se descartan CutOut/GridMask (ocluir la copa, que es pequeña, empeoraría su segmentación) y MixUp/CutMix (crean máscaras sin anatomía real).

---

## 10. Arquitectura del modelo: U-Net con InceptionResNetV2

### U-Net

Arquitectura de referencia para segmentación médica desde 2015: encoder-decoder con *skip connections*. El encoder reduce resolución espacial y aumenta capacidad semántica; el decoder recupera resolución; las skip connections conectan cada nivel del encoder con su nivel del decoder, recuperando detalle fino. Para estructuras con bordes precisos y regiones pequeñas (la copa), las skip connections son fundamentales.

### InceptionResNetV2 como backbone

Se sustituye el encoder original por InceptionResNetV2 preentrenado en ImageNet, que combina **bloques Inception** (convoluciones multiescala en paralelo) y **conexiones residuales** (evitan el gradiente desvaneciente). Aunque las retinografías difieren de ImageNet, las capas iniciales aprenden filtros de bajo nivel universales; partir de pesos preentrenados acelera la convergencia y reduce el sobreajuste con 400 imágenes. Se prefiere a ResNet50 (capta menos escalas) y a EfficientNet (mejor en clasificación, pero la capacidad representacional profunda de InceptionResNetV2 conviene para el anillo, de decenas de píxeles).

**Salida:** softmax de 3 clases por píxel `[p_fondo, p_disco, p_copa]`; la clase es el argmax.

### Parámetros principales

```
IMG_SIZE 512×512 · BACKBONE inceptionresnetv2 · CLASES 3
LR 1e-4 (Adam) · EPOCHS 50 (con EarlyStopping) · BATCH 8 · N_SPLITS 5
```

---

## 11. Función de pérdida: Tversky ponderada + Focal

La pérdida final de entrenamiento (`clinical_hybrid_loss`) combina **Tversky multiclase ponderada + 0.5 · Focal ponderada**.

### Tversky como generalización de la Dice

El índice de Tversky por clase usa la convención:

```
TI = TP / (TP + α·FP + β·FN)
```

donde `FP` (falsos positivos) corresponden a la **sobre-segmentación** (predecir una clase donde no está) y `FN` (falsos negativos) a la **sub-segmentación**. La pérdida es `1 − media_clases(TI)`. El caso **α = β = 0.5 equivale exactamente a la Dice**, por lo que Tversky generaliza la Dice y permite ponderar de forma asimétrica los dos tipos de error.

**Por qué Tversky/Dice y no Cross-Entropy:** la copa ocupa típicamente el 2–5 % de los píxeles del recorte; con CE estándar, predecir "fondo" en todo daría error bajo (acierta el 95 %). Dice/Tversky normalizan por el tamaño de cada estructura, dando peso comparable a copa, disco y fondo.

### Ponderación por clase

```
CLASS_WEIGHTS = [0.10 fondo, 0.35 disco, 0.55 copa]
```

La copa se pondera más porque pequeños errores en su borde alteran mucho el vCDR; el fondo se pondera menos porque domina espacialmente. La Focal (γ = 2) añade gradiente en píxeles difíciles (bordes), donde la incertidumbre es mayor; se pondera con 0.5 para no sobrecastigar ruido pequeño.

### Parametrización asimétrica de la copa y lectura honesta (resultado de la auditoría)

Fondo y disco usan α = β = 0.5 (≡ Dice). La copa usa **α = 0.3 < β = 0.7**.

> **Decisión documentada con transparencia.** La intención inicial de esta asimetría era penalizar la sobre-segmentación de la copa para corregir el sesgo del vCDR detectado en la auditoría. Sin embargo, en la convención `TI = TP/(TP + α·FP + β·FN)`, **β > α es *recall-oriented*** (Salehi et al., 2017): pesa **más** los falsos negativos, lo que incentiva al modelo a **predecir más copa** (mayor recall), justo lo contrario de reducir la sobre-segmentación. La evidencia empírica lo confirma: tras reentrenar con esta pérdida, el sesgo de vCDR se mantuvo en **+0.077** (prácticamente idéntico al de la Dice previa) y la copa **no mejoró** (IoU de copa en Test400 0.604 frente a 0.627 con Dice).

**Implicación de diseño.** El sesgo del vCDR **no se corrige en la pérdida**, sino **post-hoc** mediante una corrección afín ajustada en Validation400 (sección 20). Una variante *precision-oriented* (α > β en la copa), que sí atacaría la sobre-segmentación en origen, queda identificada como trabajo futuro porque requeriría reentrenar el ensemble. Se conservan los pesos α = 0.3 / β = 0.7 en el código porque son los usados para entrenar los modelos liberados en `Models_v5_TverskyCup`; documentarlo así mantiene la coherencia entre código y modelos y deja constancia del análisis crítico realizado.

---

## 12. Entrenamiento K-Fold

### Por qué K-Fold

Con 400 imágenes, una única división train/val interna sería estadísticamente inestable. K-Fold divide Training400 en 5 folds de 80 imágenes; se entrenan 5 modelos (320 train / 80 val interna cada uno). La validación interna solo selecciona el mejor checkpoint, nunca calibra el umbral final. El resultado son **5 modelos independientes** que en inferencia se promedian (ensemble, sección 13).

### Métrica de selección del checkpoint

No se guarda el de menor `val_loss`, sino el de mayor `val_clinical_selection_score`:

```
clinical_selection_score = 0.35·disc_iou_clínico + 0.35·cup_iou + 0.30·(1 − |vCDR_pred − vCDR_true|)
```

**Por qué compuesta:** el objetivo final es estimar bien el vCDR; un modelo con IoU de disco alto pero que sobreestima la copa daría vCDR erróneo, y esta métrica lo penaliza. **Por qué el disco clínico completo (clase 1 + 2):** el CDR se calcula entre disco completo y copa, no entre anillo y copa.

### Callbacks

`ModelCheckpoint` (guarda en mejora), `EarlyStopping` (paciencia 15), `ReduceLROnPlateau` (mitad de LR tras 7 sin mejora), `CSVLogger`, `TensorBoard`.

### Resultados de validación interna (mejor checkpoint por fold)

| Fold | IoU Disco | IoU Copa | Dice Disco | Dice Copa | MAE vCDR |
|------|-----------|----------|------------|-----------|----------|
| 1    | 0.928     | 0.788    | 0.957      | 0.874     | 0.043    |
| 2    | 0.917     | 0.781    | 0.950      | 0.865     | 0.045    |
| 3    | 0.884     | 0.752    | 0.916      | 0.835     | 0.051    |
| 4    | 0.919     | 0.792    | 0.952      | 0.875     | 0.041    |
| 5    | 0.911     | 0.797    | 0.942      | 0.875     | 0.038    |
| **Media** | **0.912** | **0.782** | **0.943** | **0.865** | **0.044** |

La caída entre validación interna y test (sección 18) es esperable: la val interna ve solo 80 imágenes y el checkpoint se selecciona sobre ellas (cierto optimismo estadístico); en Test400 la población es de 400 imágenes nunca vistas. Tiempo total de entrenamiento: ~196 min (5 folds × 50 épocas) en GPU de Colab.

---

## 13. Ensemble e inferencia con TTA

### Ensemble de 5 modelos

Cada imagen pasa por los 5 modelos y se promedian sus mapas de probabilidad softmax **antes** del argmax:

```
prob_final[píxel, clase] = media(prob_fold_1..5)[píxel, clase];  máscara = argmax(prob_final)
```

Los 5 modelos difieren en qué 80 imágenes vieron en validación interna y en el barajado; promediar compensa errores aleatorios y estabiliza la salida, mejorando típicamente 1–3 puntos de IoU sobre el mejor modelo individual, sin coste de entrenamiento adicional.

### Test-Time Augmentation (TTA)

Cada imagen se evalúa original y espejada horizontalmente, promediando tras revertir el flip. El flip horizontal es la transformación más natural del fondo de ojo (ojo izquierdo/derecho casi especulares) y cancela sesgos leves de orientación. Solo flip horizontal: más TTA multiplica el tiempo de inferencia por poca ganancia.

---

## 14. Postprocesamiento anatómico

Tras la máscara de la red se aplican restricciones anatómicas **necesarias** (no solo probables):

1. **Mayor componente conectada** de disco y de copa: en un recorte centrado solo hay una papila; los fragmentos aislados son ruido.
2. **Relleno de huecos internos** (flood fill): disco y copa son regiones compactas.
3. **La copa debe quedar dentro del disco:** se eliminan los píxeles de copa fuera del disco (imposible anatómicamente).

No se imponen restricciones más fuertes (convexidad, rango fijo de CDR) porque descartarían predicciones correctas en casos extremos.

---

## 15. Extracción de biomarcadores clínicos

- **vCDR** = diámetro vertical de la copa / diámetro vertical del disco (de los *bounding boxes* de las máscaras). Es el biomarcador principal y el más usado en cribado: un vCDR > 0.6–0.65 sin otros factores ya es sospechoso.
- **hCDR** (horizontal), **area_CDR** (área copa / área disco) y **rCDR** (√area_CDR, comparable en escala al vCDR).
- **Anillo neurorretiniano** = área disco − área copa; `rim_to_disc_ratio`.
- **Indicador ISNT-like:** aproximación geométrica sectorial de la regla ISNT (anillo inferior > superior > nasal > temporal). **Es aproximado** porque sin lateralidad real (ojo izq/dcho) la señal nasal/temporal es espuria; la ablación (sección 17) confirma que aporta poco.

### Score de sospecha glaucomatosa

```
risk_score_combined = 0.70·norm(vCDR,[0.45,0.85]) + 0.20·norm(rCDR,[0.50,0.85]) + 0.10·ISNT_like
```

El vCDR domina (0.70) por ser el predictor más consistente. No se usa el vCDR crudo como único clasificador porque tiene alta variabilidad interindividual (discos grandes sanos con vCDR alto); el término estructural reduce esos falsos positivos. **Nota:** este score opera sobre el vCDR; el sesgo de sobreestimación (sección 20) afecta a todos los términos derivados del área, por lo que su corrección afín es relevante también aquí.

---

## 16. Validación externa (REFUGE-Validation400)

Primer conjunto que el sistema no ha visto durante el entrenamiento. Da una estimación honesta del rendimiento y es donde se calibra todo lo que luego se aplica a Test400.

### Resultados (400 imágenes; 40 glaucoma / 360 sanas)

| Métrica | Media | Mediana |
|---|---|---|
| IoU Disco | 0.853 | 0.873 |
| Dice Disco | 0.918 | 0.932 |
| IoU Copa | 0.631 | 0.634 |
| Dice Copa | 0.764 | 0.776 |
| MAE vCDR | 0.086 | 0.067 |

- **vCDR predicho 0.545 vs real 0.468 → sesgo +0.077** (ver sección 20).
- **AUC score combinado = 0.862**; en Youden: sens 0.80, spec 0.83. **AUC vCDR solo = 0.856**.
- **Techo del sistema** (AUC con vCDR calculado desde la GT) = **0.903**: incluso con segmentación perfecta no se llega a 1.0 por casos intrínsecamente ambiguos. El gap 0.903 → 0.862 es el coste directo del error de segmentación de copa.

### Por qué AUC como métrica primaria de clasificación

El AUC es independiente del umbral: mide la capacidad discriminativa en todos los umbrales. La elección del umbral es una decisión clínica posterior (cribado vs. confirmación). Es además la métrica de ranking oficial de REFUGE (sección 22).

---

## 17. Calibración clínica, ablación y calibración de probabilidad

Este bloque usa **solo Validation400** para tomar decisiones que luego se aplican a Test400.

### Ablación de scores (Validation400)

| Score                  | AUC   | Lectura |
|------------------------|-------|---------|
| GT vCDR (techo)        | 0.903 | Límite alcanzable con segmentación perfecta. |
| Score combinado        | 0.862 | El elegido; mejora marginal sobre vCDR solo. |
| Solo vCDR predicho     | 0.856 | Predictor principal. |
| Solo rCDR predicho     | ~0.85 | Complementario. |
| ISNT-like              | ~0.70 | Aporta poco (sin lateralidad real). |

**Conclusión:** el ISNT-like es prescindible; el score simplificado vCDR+rCDR (sección 20, C.2) es más defendible. El score combinado se conserva como decisión conservadora documentada.

### Selección del umbral

Tres estrategias, todas fijadas en Validation400: **Youden** (maximiza sens+spec−1), **sensibilidad ≥ 0.85** (cribado) y **sensibilidad ≥ 0.90**. Se elige `sensitivity_at_least_0.85` como punto operativo **principal** por la asimetría clínica del cribado: un falso negativo (glaucoma no detectado) implica ceguera evitable; un falso positivo implica una derivación. El umbral resultante en Validation400 es 0.288 (sens 0.85, spec 0.74).

### Calibración de probabilidad (Platt) — diagrama de fiabilidad, Brier y ECE

El score crudo no es una probabilidad: un 0.7 no equivale a 70 % de probabilidad de glaucoma. Se ajusta una **regresión logística (Platt scaling)** en Validation400 y se aplica a Test400, evaluando la calibración con:

- **Brier score** (error cuadrático medio entre probabilidad y resultado).
- **ECE (Expected Calibration Error):** desviación media |confianza − frecuencia observada| por bins.
- **Diagrama de fiabilidad:** probabilidad media predicha vs. frecuencia observada por bin, frente a la diagonal de calibración perfecta.

**Nota de rigor:** Platt es una transformación **monótona**, por lo que **no cambia el AUC**; mejora la *interpretabilidad probabilística* (Brier y ECE), que es lo relevante para presentar el score a un clínico. La implementación está en la sección 15 del notebook (C.3) y guarda el diagrama y las tablas en `calibration/`.

---

## 18. Evaluación final en REFUGE-Test400

### Protocolo estricto

El umbral, el score, la corrección afín del vCDR y la calibración de probabilidad provienen **todos de Validation400**; en Test400 **no se recalibra ni se reentrena nada**. Cualquier ajuste posterior a ver el test produciría métricas optimistas.

### Resultados (400 imágenes; 40 glaucoma / 360 sanas)

**Segmentación y biomarcador:**

| Métrica | Media | Std | Mediana |
|---|---|---|---|
| IoU Disco | 0.824 | 0.080 | 0.835 |
| Dice Disco | 0.901 | 0.058 | 0.910 |
| IoU Copa | 0.604 | 0.139 | 0.601 |
| Dice Copa | 0.743 | 0.113 | 0.751 |
| MAE vCDR (crudo) | 0.085 | 0.081 | 0.064 |

**Clasificación** (score combinado, AUC = **0.8615**):

| Punto operativo | Umbral | Sens | Spec | F1 | TP/FP/FN/TN |
|---|---|---|---|---|---|
| `sensitivity_at_least_0.85` (principal) | 0.288 | 0.800 | 0.753 | 0.398 | 32/89/8/271 |
| Youden (alternativa equilibrada) | 0.351 | 0.700 | 0.831 | 0.434 | 28/61/12/299 |

El umbral fijado en validación (sens 0.85 allí) da sens 0.80 en test: una caída pequeña y esperable por la variabilidad muestral de los 40 glaucomas. La especificidad 0.753 implica ~89 falsos positivos sobre 360 sanos; en cribado este coste (derivaciones) se acepta a cambio de no perder casos. Los intervalos de confianza de estas métricas están en la sección 21.

---

## 19. Análisis visual de errores y auditoría técnica de ROI

### Casos de error

Se visualizan TP, TN, FP, FN y los casos de mayor error de vCDR, mostrando imagen, GT, predicción, overlay, biomarcadores y score. Permite distinguir errores de segmentación (ROI, disco) de la ambigüedad intrínseca del caso.

### Auditoría ROI de cobertura total (1200 imágenes)

Recorre las tres particiones y comprueba, usando la GT solo como referencia (nunca para localizar), si el recorte captura disco y copa completos:

| Conjunto | GT válidas | Papila fuera del recorte | Centros idénticos al legacy | Desplaz. p95 |
|---|---|---|---|---|
| Training400 | 400 | 9 | 99.75 % | 0 px |
| Validation400 | 400 | 2 | 98.50 % | 0 px |
| Test400 | 400 | **0** | 98.25 % | 0 px |

**Lectura:** el localizador robusto con guarda elimina las papilas fuera del recorte en Test400 interviniendo en muy pocos casos (la inmensa mayoría de centros son idénticos al algoritmo original), lo que confirma que la corrección **no obliga a reentrenar**. El error residual del sistema ya no procede de la localización, sino de la segmentación de la copa.

---

## 20. El sesgo de vCDR: diagnóstico y corrección

### Diagnóstico

Los datos revelan un **sesgo estructural direccional**, no ruido:

| Variable | Media predicha | Media real | Diferencia |
|---|---|---|---|
| vCDR | 0.545 | 0.468 | **+0.077** |
| area_CDR | 0.337 | 0.249 | **+0.088** |
| rCDR | 0.575 | 0.491 | **+0.078** |

El modelo **sobreestima sistemáticamente el tamaño de la copa** en ~8 puntos. Es consistente en las tres métricas derivadas del área. La causa probable es la combinación de (1) la ambigüedad intrínseca del borde copa-anillo (la frontera más difícil de la imagen) y (2) la parametrización *recall-oriented* de la copa en la pérdida (sección 11), que favorece predecir copa de más. Este sesgo infla el MAE de vCDR, baja la especificidad y obliga a umbrales artificialmente bajos.

### Por qué el reentreno no lo corrigió

El reentreno con Tversky asimétrica en la copa (β > α) era *recall-oriented* y por tanto no atacaba la sobre-segmentación; la evaluación lo confirma (sesgo +0.077 sin cambios, copa IoU 0.604). La corrección en origen exigiría una pérdida *precision-oriented* (α > β) y reentrenar, opción documentada como trabajo futuro.

### Corrección afín post-hoc (solución adoptada)

Se ajusta una regresión lineal `vCDR_corr = a·vCDR_pred + b` **en Validation400** y se aplica a Test400:

```
vCDR_corr = 0.597 · vCDR_pred + 0.143   (ajustada en Val400)
```

| Conjunto | MAE vCDR crudo | MAE vCDR corregido |
|---|---|---|
| Validation400 | 0.086 | **0.057** |
| Test400 | 0.085 | **0.064** |

El **sesgo medio en validación pasa de +0.077 a 0.000**. **Nota de rigor crítica:** la corrección afín es una transformación **monótona**, por lo que **el AUC no cambia** (0.858 crudo = 0.858 corregido en el vCDR). Su beneficio es **MAE, interpretabilidad clínica y calibración del biomarcador**, no la discriminación. Esto se documenta explícitamente para no atribuirle un efecto que no tiene. La única palanca post-hoc que mejora la discriminación operativa es la predicción selectiva (gate de incertidumbre, sección 15/C.4).

---

## 21. Significancia estadística

Las métricas puntuales de Test400 se acompañan de **intervalos de confianza bootstrap al 95 %** (percentil; B = 2000 remuestreos con reemplazo sobre las 400 imágenes). Con solo 40 glaucomas, una diferencia de pocas centésimas de AUC no es distinguible del ruido; reportar el intervalo es más honesto que un único número.

- **Métricas de segmentación/biomarcador** (medias por imagen): se remuestrean los valores por imagen y se recalcula la media.
- **AUC:** se remuestrean los pares (etiqueta, score), descartando remuestras de una sola clase, y se recalcula `roc_auc_score`.
- **Sensibilidad/especificidad:** al umbral fijado en validación (0.288).

La tabla con valores e IC95 se genera en la celda **12.b** del notebook y se guarda en `test_metrics_ci.csv`. Esta cuantificación de incertidumbre es la base para afirmar con rigor que dos configuraciones no son distinguibles (y por tanto evita sobre-interpretar diferencias pequeñas). No se realizan comparaciones entre versiones; se describe únicamente la versión final.

---

## 22. Estado del arte en REFUGE y posicionamiento

REFUGE evalúa segmentación (Dice de disco, Dice de copa, MAE de vCDR) y clasificación (AUC, con sensibilidad de referencia a especificidad 0.85). El leaderboard oficial (Orlando et al., 2020) sobre **REFUGE-Test400** es:

### Clasificación (AUC)

| Equipo | AUC |
|---|---|
| VRT (1º) | 0.989 |
| SDSAIRC | 0.982 |
| CUHKMED | 0.964 |
| **GT vCDR (referencia)** | **0.947** |
| Mammoth / Masker / SMILEDeepDR | 0.95–0.96 |
| Cvblab | 0.881 |
| AIML (12º) | 0.846 |
| **Este sistema (v6.0)** | **0.862** |

Rango de los 12 equipos: **0.846–0.989**. Dos equipos superaron a expertos humanos (sens 85 %, spec ~91 %).

### Segmentación

| Métrica | Top del challenge | Rango (12 equipos) | Este sistema (v6.0) |
|---|---|---|---|
| Dice Disco | 0.960 (CUHKMED) | 0.877–0.960 | 0.901 |
| Dice Copa | 0.884 (Masker) | 0.686–0.884 | 0.743 |
| MAE vCDR | 0.041 (Masker) | 0.041–0.154 | 0.085 |

### Posicionamiento honesto

Con Dice de disco 0.901, Dice de copa 0.743, MAE 0.085 y AUC 0.862, el sistema se sitúa en la **franja baja del leaderboard de 2018 (en torno al puesto 11 de 12)**. Esto es **esperable y defendible** para un TFG: los equipos del challenge usaron ensembles complejos, entrenamiento multi-dataset, datos adicionales y pipelines a resolución completa, mientras que este sistema se entrena **solo con las 400 imágenes de Training400**, sin datos externos (sección 23) y en Colab gratuito. Además, el AUC 0.862 queda por debajo del techo GT-vCDR (0.947), lo que indica que **el margen de mejora está en la calidad de segmentación de la copa**, no en el clasificador.

> **Caveat de comparabilidad:** el leaderboard calcula Dice sobre la imagen completa con el protocolo oficial; aquí el Dice se mide en el espacio del recorte ROI a 512×512. La comparación es orientativa (posicionamiento de orden de magnitud), no una participación oficial en el challenge. El valor del TFG no reside en superar el leaderboard, sino en un pipeline riguroso, modular, auditado y honestamente reportado.

---

## 23. Exclusión de datos externos: justificación

Ampliar el entrenamiento con datasets públicos (ORIGA, DRISHTI-GS, RIM-ONE) es una vía conocida para mejorar la copa y el AUC, pero **se ha decidido no incorporarlos** por motivos metodológicos:

1. **Riesgo de fuga de datos (leakage) y de comparabilidad.** El valor de REFUGE como benchmark reside en sus particiones fijas. Introducir imágenes de otros datasets en el entrenamiento rompe la comparabilidad directa con el leaderboard (sección 22) y exige verificar que ninguna imagen externa solape (por contenido o por reescalado) con Validation400/Test400, una garantía difícil de asegurar entre datasets heterogéneos.
2. **Domain shift no controlado.** ORIGA, DRISHTI-GS y RIM-ONE usan cámaras, resoluciones, poblaciones y convenciones de anotación distintas (p. ej., criterios de borde copa-anillo). Mezclarlos sin un protocolo de armonización introduciría variabilidad no clínica que el modelo podría aprender como señal espuria, contaminando precisamente el biomarcador (vCDR) que se quiere medir bien.
3. **Convenciones de máscara incompatibles.** Las definiciones de copa/disco difieren entre datasets; una fusión ingenua de máscaras degradaría la consistencia del ground truth.
4. **Alcance del TFG.** El objetivo es un pipeline riguroso y auditable sobre un benchmark estándar, no maximizar la métrica a cualquier coste. La separación estricta Training/Validation/Test (sección 6) es el pilar del trabajo y mezclar fuentes lo debilitaría.

Por tanto, el uso controlado de datos externos —con armonización de dominio, verificación de no-solape y reentrenamiento— queda explícitamente recogido como **trabajo futuro**, no como una omisión.

---

## 24. Resumen de resultados consolidados

| Bloque | Métrica | Validación interna (K-Fold) | Validation400 | Test400 |
|---|---|---|---|---|
| Segmentación | IoU Disco | 0.912 | 0.853 | 0.824 |
| Segmentación | IoU Copa | 0.782 | 0.631 | 0.604 |
| Segmentación | Dice Disco | 0.943 | 0.918 | 0.901 |
| Segmentación | Dice Copa | 0.865 | 0.764 | 0.743 |
| Biomarcador | MAE vCDR (crudo) | 0.044 | 0.086 | 0.085 |
| Biomarcador | MAE vCDR (corregido C.1) | — | 0.057 | 0.064 |
| Clasificación | AUC (score combinado) | — | 0.862 | 0.862 |
| Clasificación | Sens / Spec @ sens≥0.85 | — | 0.85 / 0.74 | 0.80 / 0.75 |
| Calibración | Brier / ECE | — | (fit) | reportado (sección 17) |

**Lectura ejecutiva.** El sistema es sólido para un TFG con las restricciones de REFUGE-Training400: AUC 0.86 en test, segmentación de disco competitiva y copa en el límite bajo. El problema técnico principal es la **sobre-segmentación de la copa** y el **sesgo de vCDR +0.077**, corregido post-hoc para MAE/calibración (no para AUC, por ser monótona). El techo realista del sistema (GT-vCDR) es 0.90–0.95; cerrar el gap exige mejorar la segmentación de copa, no el clasificador.

---

## 25. Limitaciones y encuadre clínico

**El sistema hace:** segmentar disco y copa, calcular biomarcadores morfológicos (vCDR, hCDR, area_CDR, rCDR, anillo, ISNT-like) y generar un score de sospecha glaucomatosa para priorizar derivaciones.

**El sistema NO hace:** diagnosticar glaucoma (requiere tonometría, campo visual, OCT, valoración clínica), estimar presión intraocular, ni funcionar sobre imágenes de muy baja calidad o patologías que alteren drásticamente el fondo.

**Limitaciones técnicas:**

- **Sesgo de vCDR (+0.077):** se sobreestima la copa; se corrige post-hoc (afín) para MAE e interpretabilidad, pero la discriminación (AUC) sigue limitada por la calidad de segmentación de copa.
- **Pérdida recall-oriented en la copa:** la parametrización Tversky de la copa no reduce la sobre-segmentación (sección 11); una variante precision-oriented queda como trabajo futuro.
- **Dataset único (REFUGE):** sin validación en otras cámaras/poblaciones; los datos externos se excluyen por las razones de la sección 23.
- **Tamaño del dataset:** 400 imágenes de entrenamiento; suficiente para un TFG, pequeño para uso clínico real.
- **ROI heurística:** el localizador con guarda cubre el 100 % de Test400, pero un sistema de producción usaría un detector dedicado.
- **vCDR por bounding box:** aproximación; ajustar una elipse (`cv2.fitEllipse`) sería más preciso a costa de más puntos de fallo en papilas pequeñas.
- **Falsos negativos:** sugieren casos glaucomatosos no explicables solo por la relación copa/disco.

**Por qué este diseño es apropiado para un TFG:** demuestra comprensión profunda de un problema clínico real, un pipeline completo y auditable, decisiones justificadas con conocimiento del dominio, y un análisis crítico honesto de las propias limitaciones (incluida la detección y corrección de un sesgo y de una parametrización subóptima de la pérdida). No se persigue el estado del arte, sino el rigor metodológico defendible.

---

## 26. Apéndice: evolución del proyecto y decisiones de auditoría

Este apéndice documenta el recorrido del proyecto para justificar las decisiones de diseño y dejar constancia del trabajo realizado. La narrativa principal del TFG se centra en la versión final (v6.0); este apéndice aporta la trazabilidad.

- **v3/v4 — sistema base.** U-Net + InceptionResNetV2, ensemble 5-fold, pérdida Dice + Focal, biomarcadores y validación externa. La auditoría detecta dos problemas: (a) algunas papilas quedaban fuera del recorte ROI; (b) un sesgo sistemático de sobreestimación del vCDR (+0.077).
- **Corrección de ROI sin reentrenar.** Se añade un localizador robusto con **guarda** que solo interviene cuando el método original falla, preservando la consistencia entrenamiento/inferencia. La auditoría ROI de cobertura total (sección 19) confirma 0 papilas fuera del recorte en Test400 con desplazamiento p95 = 0 px → no es necesario reentrenar por motivo de ROI.
- **Intento de corrección del sesgo en origen (reentreno Tversky).** Se reentrena sustituyendo la Dice por una Tversky asimétrica en la copa con la intención de penalizar la sobre-segmentación. La auditoría posterior demuestra que la parametrización elegida (α = 0.3 < β = 0.7) es *recall-oriented* y **no** reduce la sobre-segmentación: el sesgo se mantiene en +0.077 y la copa no mejora. Se documenta este resultado con transparencia (sección 11) en lugar de ocultarlo.
- **Corrección del sesgo post-hoc (decisión final).** Dado que el reentreno no corrigió el sesgo y que reentrenar con una pérdida precision-oriented quedaba fuera de alcance, el sesgo se corrige **post-hoc** mediante calibración afín ajustada en Validation400 (sección 20): MAE de vCDR en Test400 0.085 → 0.064, sesgo 0. Se añade además calibración de probabilidad (Platt) con diagrama de fiabilidad, Brier y ECE (sección 17), e intervalos de confianza bootstrap (sección 21).
- **Lección metodológica.** El valor de este recorrido reside en el **análisis crítico**: identificar un sesgo, intentar corregirlo en origen, reconocer mediante evidencia que la intervención no funcionó (y por qué), y resolverlo con la herramienta adecuada documentando honestamente los límites de cada paso.

---

*Referencia del estado del arte:* Orlando, J. I., et al. (2020). *REFUGE Challenge: A unified framework for evaluating automated methods for glaucoma assessment from fundus photographs.* Medical Image Analysis, 59, 101570. *Pérdida Tversky:* Salehi, S. S. M., Erdogmus, D., & Gholipour, A. (2017). *Tversky loss function for image segmentation using 3D fully convolutional deep networks.* MICCAI MLMI.
