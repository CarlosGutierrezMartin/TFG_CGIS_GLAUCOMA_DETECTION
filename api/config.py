"""
Configuración centralizada de la API.

Las rutas a artefactos (modelos, CSV de validación y dataset Test400) se leen de
variables de entorno para no acoplar el código a una ubicación concreta. Por
defecto apuntan a ./artifacts dentro del repositorio.

Las constantes del modelo (IMG_SIZE, ROI_RADIUS, BACKBONE, N_SPLITS, CLASSES)
replican exactamente las del notebook TFG_GLAUCOMA_v7.0.ipynb (dataclass Config).
"""

import os
from dataclasses import dataclass

# Raíz del repositorio (este archivo está en api/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_ARTIFACTS = os.path.join(_REPO_ROOT, "artifacts")


def _env(name, default):
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Config:
    # ---- Constantes del modelo (idénticas al notebook) ----
    SEED: int = 42
    IMG_SIZE: int = 512
    ROI_RADIUS: int = 200
    BACKBONE: str = "inceptionresnetv2"
    N_SPLITS: int = 5
    CLASSES: int = 3

    # ---- Inferencia ----
    USE_TTA: bool = _env("GLAUCOMA_USE_TTA", "1") not in ("0", "false", "False")

    # ---- Rutas de artefactos (configurables por entorno) ----
    ARTIFACTS_DIR: str = _env("GLAUCOMA_ARTIFACTS_DIR", _DEFAULT_ARTIFACTS)
    MODELS_DIR: str = _env("GLAUCOMA_MODELS_DIR", os.path.join(_DEFAULT_ARTIFACTS, "models"))
    VAL_RESULTS_CSV: str = _env(
        "GLAUCOMA_VAL_RESULTS_CSV",
        os.path.join(_DEFAULT_ARTIFACTS, "external_validation_results.csv"),
    )
    CALIBRATION_PARAMS_JSON: str = _env(
        "GLAUCOMA_CALIBRATION_JSON",
        os.path.join(_DEFAULT_ARTIFACTS, "calibration_params.json"),
    )

    # ---- Dataset REFUGE Test400 ----
    TEST_IMAGES_DIR: str = _env(
        "GLAUCOMA_TEST_IMAGES_DIR",
        os.path.join(_DEFAULT_ARTIFACTS, "Test400", "images"),
    )
    TEST_MASKS_DIR: str = _env(
        "GLAUCOMA_TEST_MASKS_DIR",
        os.path.join(_DEFAULT_ARTIFACTS, "Test400", "masks"),
    )
    TEST_LABELS_XLSX: str = _env(
        "GLAUCOMA_TEST_LABELS_XLSX",
        os.path.join(_DEFAULT_ARTIFACTS, "Test400", "Glaucoma_label_and_Fovea_location.xlsx"),
    )
    # Columnas del xlsx de etiquetas de Test400 (confirmadas en la salida del notebook).
    TEST_LABEL_IMG_COL: str = _env("GLAUCOMA_TEST_LABEL_IMG_COL", "ImgName")
    TEST_LABEL_COL: str = _env("GLAUCOMA_TEST_LABEL_COL", "Label(Glaucoma=1)")

    def model_path(self, fold_number):
        return os.path.join(self.MODELS_DIR, f"model_fold_{fold_number}.keras")

    @property
    def model_paths(self):
        return [self.model_path(f) for f in range(1, self.N_SPLITS + 1)]


CFG = Config()


# ============================================================
# Bandas de riesgo (no diagnósticas, generosas)
# ============================================================
# Detección temprana, no intrusiva: probabilidad en % -> banda + mensaje.

RISK_LOW_MAX = 30.0    # < 30 %  -> baja
RISK_MID_MAX = 60.0    # 30-60 % -> intermedia ; > 60 % -> moderada

DISCLAIMER = (
    "Sistema de cribado / detección temprana de signos asociados a sospecha "
    "glaucomatosa. NO constituye un diagnóstico. El diagnóstico de glaucoma "
    "requiere evaluación oftalmológica completa (campo visual, OCT, presión "
    "intraocular e historia clínica)."
)


def risk_band_for_probability(probability_percent):
    """
    Mapea una probabilidad en % a (banda, mensaje) según umbrales generosos.

    < 30 %   -> baja
    30-60 %  -> intermedia
    > 60 %   -> moderada
    """
    if probability_percent < RISK_LOW_MAX:
        return "baja", "Baja probabilidad de signos de glaucoma."
    if probability_percent <= RISK_MID_MAX:
        return "intermedia", "Hallazgos intermedios; se aconseja revisión rutinaria."
    return "moderada", "Probabilidad moderada; se aconseja revisión oftalmológica pronto."
