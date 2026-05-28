# TFG CGIS - Deteccion de glaucoma mediante segmentacion disco-copa

## Objetivo

Este repositorio contiene el notebook principal del TFG para deteccion asistida de glaucoma a partir de imagenes de fondo de ojo del dataset REFUGE. El sistema segmenta disco optico y copa optica mediante U-Net con backbone `inceptionresnetv2`, extrae biomarcadores clinicos simplificados y valida el riesgo glaucomatoso sobre un conjunto externo reservado.

La reorganizacion actual prioriza reproducibilidad, trazabilidad metodologica y ausencia de fuga de datos entre entrenamiento y validacion externa.

## Estructura principal

- `TFG_CGIS_GLAUCOMA_DETECTION/TFG_GLAUCOMA_v2_0.ipynb`: notebook tecnico reorganizado.
- `TFG_CGIS_GLAUCOMA_DETECTION/README.md`: documento interno breve del subproyecto.

El notebook queda organizado en trece bloques:

1. Configuracion del entorno
2. Carga y preparacion del dataset
3. Protocolo experimental y preprocesamiento
4. Construccion del pipeline de entrenamiento
5. Definicion del modelo
6. Entrenamiento K-Fold
7. Carga y verificacion de modelos entrenados
8. Inferencia y segmentacion
9. Extraccion de biomarcadores clinicos
10. Validacion externa
11. Estudio de ablacion
12. Validacion visual
13. Resumen final del experimento

## Rutas esperadas en Google Drive

No se han cambiado nombres de archivos ni rutas del proyecto. El notebook espera la siguiente estructura:

```text
/content/drive/MyDrive/TFG_Glaucoma
/content/drive/MyDrive/TFG_Glaucoma/Refuge.rar
/content/drive/MyDrive/TFG_Glaucoma/Models_vPro_Fixed
```

Los modelos finales se guardan y cargan con estos nombres:

```text
model_fold_1.keras
model_fold_2.keras
model_fold_3.keras
model_fold_4.keras
model_fold_5.keras
```

El dataset REFUGE se conserva con sus nombres originales:

```text
REFUGE-Training400
Annotation-Training400
REFUGE-Validation400
REFUGE-Validation400-GT
Disc_Cup_Masks
```

## Protocolo experimental

El cambio metodologico principal es la separacion estricta de datos:

- `REFUGE-Training400`: entrenamiento y validacion interna mediante K-Fold.
- `REFUGE-Validation400`: validacion externa final. No se usa para entrenar.

El notebook define dos variables independientes:

```python
train_data = get_all_pairs_robust(TRAIN_DIRS)
external_val_data = get_all_pairs_robust(EXTERNAL_VAL_DIRS)
```

El K-Fold se aplica solo sobre `train_data`. La validacion externa se ejecuta solo sobre `external_val_data`.

## Preprocesamiento e inferencia

El preprocesamiento se centraliza en funciones reutilizables para entrenamiento, validacion, inferencia y visualizacion:

- lectura RGB con OpenCV;
- recorte de region de interes centrado en la papila con radio de 200 pixeles;
- redimensionamiento a `512 x 512`;
- conversion de mascaras REFUGE a clases semanticas;
- CLAHE sobre luminancia;
- preprocesamiento especifico del backbone `inceptionresnetv2`.

Las clases de mascara son:

```text
0 = fondo
1 = disco optico
2 = copa optica
```

La inferencia aplica ensemble de los cinco folds, test-time augmentation horizontal y postprocesamiento anatomico basico para conservar el componente conectado principal y asegurar que la copa no quede fuera del disco.

## Biomarcadores y score

La segmentacion termina en una mascara postprocesada. Despues se calculan biomarcadores en funciones separadas:

- diametro vertical del disco;
- diametro vertical de la copa;
- `vCDR`;
- `rCDR`;
- indicador geometrico simplificado inspirado en la regla ISNT.

El indicador ISNT no debe interpretarse como una evaluacion clinica completa de la regla ISNT. Es una aproximacion geometrica para analisis experimental.

El umbral clinico `0.52` se mantiene como criterio heuristico. Antes de presentar resultados definitivos conviene justificarlo formalmente, recalibrarlo con indice de Youden o priorizar sensibilidad segun el criterio clinico elegido.

## Ejecucion en Colab

1. Abrir `TFG_CGIS_GLAUCOMA_DETECTION/TFG_GLAUCOMA_v2_0.ipynb`.
2. Ejecutar configuracion del entorno.
3. Montar Drive, copiar `Refuge.rar`, descomprimir y validar estructura.
4. Indexar `train_data` y `external_val_data`.
5. Entrenar con `run_kfold_training()` solo si se desea regenerar modelos.
6. Cargar modelos con `load_ensemble_models()`.
7. Ejecutar validacion externa con `evaluate_external_validation(models)`.
8. Ejecutar ablacion con `run_ablation_study(external_results)`.
9. Revisar casos visuales con `plot_representative_cases(models, external_results)`.

El entrenamiento completo es costoso y queda comentado por defecto para evitar ejecuciones accidentales.

## Logs

Los mensajes del notebook usan lenguaje tecnico en espanol y prefijos consistentes:

```text
[INFO]
[OK]
[ADVERTENCIA]
[ERROR]
[CRITICO]
```

No se usan emojis en logs ni mensajes tecnicos.

## Validacion y metricas

La validacion externa calcula:

- AUC-ROC;
- umbral aplicado;
- sensibilidad;
- especificidad;
- precision;
- F1-score;
- matriz de confusion.

Las metricas actuales no deben considerarse definitivas hasta recalcularlas con la separacion corregida entre entrenamiento interno y validacion externa.

## Limitaciones

- No se modifica la arquitectura del modelo en esta reorganizacion.
- No se renombran rutas, carpetas ni archivos para mantener compatibilidad con Google Drive.
- Streamlit queda fuera del flujo experimental principal hasta cerrar entrenamiento, validacion externa, score final y umbral clinico.
- El indicador ISNT es simplificado y no sustituye una evaluacion oftalmologica completa.
