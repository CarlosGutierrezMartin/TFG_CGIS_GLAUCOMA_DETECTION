"""
Orquestación de un caso completo: preprocesado -> ensemble -> postproceso ->
biomarcadores -> probabilidad calibrada -> overlays -> respuesta.

Reúne las piezas de glaucoma_core, ensemble, calibration y viz en una sola
función reutilizable por los endpoints.
"""

import math

import numpy as np

from .calibration import Calibrator
from .config import risk_band_for_probability, DISCLAIMER
from .dataset import TestCase
from .ensemble import GlaucomaEnsemble
from . import glaucoma_core as core
from .viz import build_case_images


def _clean(value):
    """Convierte NaN/inf en None para que sea serializable como JSON."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else f
    return value


def process_case(case: TestCase, ensemble: GlaucomaEnsemble, calibrator: Calibrator,
                 use_tta=None):
    """
    Ejecuta el pipeline completo sobre un caso de Test400 (con máscara real).

    Devuelve un dict con la estructura de CaseResponse.
    """
    # 1. Preprocesado + inferencia ensemble.
    image_preprocessed, image_resized, _ = core.preprocess_image_only(case.image_path)
    probability_map = ensemble.predict_proba_map(image_preprocessed, use_tta=use_tta)

    # 2. Máscara predicha postprocesada + máscara real alineada al recorte.
    raw_mask = np.argmax(probability_map, axis=-1).astype(np.uint8)
    final_mask, postprocess_diag = core.postprocess_segmentation_mask(raw_mask)
    true_mask = core.true_mask_for_pair(case.image_path, case.mask_path)

    # 3. Biomarcadores (predicción y GT).
    pred_bio = core.compute_biomarkers_from_mask(final_mask)
    true_bio = core.compute_biomarkers_from_mask(true_mask)

    # 4. Probabilidad calibrada (cadena C.1 -> C.3 del notebook).
    calib = calibrator.probability_from_biomarkers(
        pred_vcdr=pred_bio.get("vCDR"), pred_rcdr=pred_bio.get("rCDR")
    )
    proba = calib["proba"]
    probability_percent = 0.0 if (proba is None or math.isnan(proba)) else round(proba * 100.0, 1)

    risk_band, risk_message = risk_band_for_probability(probability_percent)

    # 5. Decisión + acierto/fallo frente a la etiqueta real.
    predicted_label = calib["predicted_label"]
    if predicted_label is not None and case.true_label is not None:
        correct = bool(predicted_label == case.true_label)
    else:
        correct = None

    # 6. Calidad / incertidumbre.
    quality = core.summarize_prediction_quality(probability_map, postprocess_diag)

    # 7. Overlays en base64.
    images = build_case_images(image_resized, final_mask, true_mask=true_mask)

    return {
        "image_name": case.image_name,
        "probability_percent": probability_percent,
        "risk_band": risk_band,
        "risk_message": risk_message,
        "images": images,
        "biomarkers": {
            "vCDR": _clean(pred_bio.get("vCDR")),
            "vCDR_corrected": _clean(calib["vcdr_corrected"]),
            "hCDR": _clean(pred_bio.get("hCDR")),
            "rCDR": _clean(pred_bio.get("rCDR")),
            "area_CDR": _clean(pred_bio.get("area_CDR")),
            "rim_to_disc_ratio": _clean(pred_bio.get("rim_to_disc_ratio")),
            "isnt_like_risk": _clean(pred_bio.get("isnt_like_risk")),
        },
        "ground_truth": {
            "true_label": case.true_label,
            "true_vCDR": _clean(true_bio.get("vCDR")),
        },
        "decision": {
            "predicted_label": predicted_label,
            "threshold": round(calibrator.params.threshold, 4),
            "score": _clean(calib["score"]),
            "correct": correct,
        },
        "quality": {
            "mean_entropy": _clean(quality["mean_entropy"]),
            "valid_anatomy": bool(quality["valid_anatomy"]),
            "disc_found": bool(quality["disc_found"]),
            "cup_found": bool(quality["cup_found"]),
        },
        "disclaimer": DISCLAIMER,
    }
