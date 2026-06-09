"""
Ensemble de los 5 modelos U-Net en memoria.

A diferencia del notebook (`predict_ensemble_batch_sequential`), que recargaba cada
.keras por petición para no agotar la RAM de Colab, aquí los 5 modelos se cargan una
sola vez al arrancar y se mantienen en memoria. El promedio del ensemble con TTA
horizontal es idéntico al del notebook.
"""

import os
import threading

import numpy as np

# SM_FRAMEWORK ya queda fijado al importar glaucoma_core; lo aseguramos por si acaso.
os.environ.setdefault("SM_FRAMEWORK", "tf.keras")

import tensorflow as tf

from .config import CFG
from .glaucoma_core import predict_model_with_tta


class GlaucomaEnsemble:
    """Mantiene los 5 modelos cargados y expone la predicción del ensemble."""

    def __init__(self):
        self.models = []
        self._lock = threading.Lock()  # las llamadas a predict de TF no son thread-safe

    def load(self):
        """Carga los 5 modelos (compile=False, como load_single_model del notebook)."""
        self.models = []

        for fold in range(1, CFG.N_SPLITS + 1):
            path = CFG.model_path(fold)

            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"No se encuentra el modelo del fold {fold}: {path}. "
                    "Copia los model_fold_*.keras de Drive a GLAUCOMA_MODELS_DIR."
                )

            model = tf.keras.models.load_model(
                path, custom_objects={}, compile=False, safe_mode=False
            )
            self.models.append(model)

        return self

    @property
    def is_loaded(self):
        return len(self.models) == CFG.N_SPLITS

    def predict_proba_map(self, image_preprocessed, use_tta=None):
        """
        Predice el mapa de probabilidades (512,512,3) promediando los 5 folds con TTA.

        `image_preprocessed` es la salida de preprocess_image_only (H,W,3) float32.
        """
        if not self.is_loaded:
            raise RuntimeError("El ensemble no está cargado. Llama a load() primero.")

        use_tta = CFG.USE_TTA if use_tta is None else use_tta

        batch = np.expand_dims(image_preprocessed, axis=0).astype(np.float32)

        ensemble_prediction = None

        with self._lock:
            for model in self.models:
                pred = predict_model_with_tta(model, batch, use_tta=use_tta)

                if ensemble_prediction is None:
                    ensemble_prediction = pred
                else:
                    ensemble_prediction = ensemble_prediction + pred

        ensemble_prediction = ensemble_prediction / CFG.N_SPLITS
        return ensemble_prediction[0].astype(np.float32)


# Instancia única usada por la app.
ensemble = GlaucomaEnsemble()
