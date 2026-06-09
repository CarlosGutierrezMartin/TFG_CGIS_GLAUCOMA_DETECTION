"""Modelos Pydantic de respuesta de la API."""

from typing import Optional

from pydantic import BaseModel


class CaseImages(BaseModel):
    original: str                       # PNG base64 (ROI redimensionada)
    prediction_overlay: str             # overlay disco/copa de la predicción
    ground_truth_overlay: Optional[str] = None  # overlay de la máscara real


class Biomarkers(BaseModel):
    vCDR: Optional[float] = None
    vCDR_corrected: Optional[float] = None
    hCDR: Optional[float] = None
    rCDR: Optional[float] = None
    area_CDR: Optional[float] = None
    rim_to_disc_ratio: Optional[float] = None
    isnt_like_risk: Optional[float] = None


class GroundTruth(BaseModel):
    true_label: Optional[int] = None    # 0 sano / 1 glaucoma (None si desconocida)
    true_vCDR: Optional[float] = None


class Decision(BaseModel):
    predicted_label: Optional[int] = None
    threshold: float                    # umbral en espacio de score (Val400)
    score: Optional[float] = None
    correct: Optional[bool] = None      # acierto vs etiqueta real (None si no hay GT)


class Quality(BaseModel):
    mean_entropy: float
    valid_anatomy: bool
    disc_found: bool
    cup_found: bool


class CaseResponse(BaseModel):
    image_name: str
    probability_percent: float
    risk_band: str                      # baja | intermedia | moderada
    risk_message: str
    images: CaseImages
    biomarkers: Biomarkers
    ground_truth: GroundTruth
    decision: Decision
    quality: Quality
    disclaimer: str


class CalibrationInfo(BaseModel):
    affine_a: float
    affine_b: float
    threshold: float
    proba_threshold: float
    n_val_samples: int


class HealthResponse(BaseModel):
    status: str
    mode: str = "real"
    ready: bool = False
    models_loaded: int
    n_test_cases: int
    use_tta: bool
    calibration: Optional[CalibrationInfo] = None
    startup_error: Optional[str] = None
    disclaimer: str
