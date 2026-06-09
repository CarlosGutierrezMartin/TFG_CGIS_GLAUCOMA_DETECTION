"""
Reconstrucción de la cadena de calibración del notebook (Sección 15, celda 32).

Los parámetros (corrección afín del vCDR, Platt y umbral) se ajustan en
Validation400 y NO se persisten como artefacto en el notebook. Aquí se re-ajustan
al arrancar la API leyendo `external_validation_results.csv`, reproduciendo
exactamente:

  C.1  pred_vCDR_corr = a * pred_vCDR + b           (LinearRegression en Val400)
  C.2  score = 0.78 * clip(vCDR_corr, 0.45, 0.85)
             + 0.22 * clip(rCDR,      0.50, 0.85)
  C.3  proba = Platt(score)                          (LogisticRegression en Val400)
       thr   = umbral con sensibilidad >= 0.85 (mayor especificidad) en Val400

Como respaldo, si existe `calibration_params.json` se cargan de ahí los parámetros
sin necesidad del CSV.
"""

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_curve

from .config import CFG

LABEL_COL = "true_glaucoma_label"


@dataclass
class CalibrationParams:
    affine_a: float
    affine_b: float
    platt_coef: float
    platt_intercept: float
    threshold: float          # umbral en espacio de score (sens >= 0.85 en Val400)
    proba_threshold: float     # umbral equivalente en espacio de probabilidad calibrada
    n_val_samples: int

    def to_json(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as fh:
            return cls(**json.load(fh))


def _minmax_clip(values, low, high):
    """Escala a [0,1] con límites clínicos (idéntico a minmax_clip_v5 del notebook)."""
    return np.clip((np.asarray(values, dtype=float) - low) / (high - low), 0.0, 1.0)


def _threshold_at_sensitivity(y, score, target=0.85):
    """Umbral con sensibilidad >= target y la mayor especificidad posible (Val400)."""
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(score)
    y, score = y[mask], score[mask]

    fpr, tpr, thr = roc_curve(y, score)
    idx = np.where(tpr >= target)[0]
    best = idx[np.argmin(fpr[idx])]
    return float(thr[best])


def build_score_from_biomarkers(vcdr_corr, rcdr):
    """
    Reconstruye el score final `simplified_vcdr_rcdr` (C.2) a partir del vCDR ya
    corregido afín y del rCDR crudo. Devuelve NaN si falta el vCDR corregido.
    """
    if vcdr_corr is None or (isinstance(vcdr_corr, float) and np.isnan(vcdr_corr)):
        return float("nan")

    vc = float(_minmax_clip([vcdr_corr], 0.45, 0.85)[0])

    if rcdr is None or (isinstance(rcdr, float) and np.isnan(rcdr)):
        # Si no hay rCDR, el notebook propagaría NaN al combinar; replicamos eso.
        return float("nan")

    rc = float(_minmax_clip([rcdr], 0.50, 0.85)[0])
    return 0.78 * vc + 0.22 * rc


class Calibrator:
    """Encapsula los parámetros ajustados y aplica la cadena C.1 -> C.3."""

    def __init__(self, params: CalibrationParams):
        self.params = params

    # ---- C.1 ----
    def correct_vcdr(self, pred_vcdr):
        if pred_vcdr is None or (isinstance(pred_vcdr, float) and np.isnan(pred_vcdr)):
            return float("nan")
        return self.params.affine_a * float(pred_vcdr) + self.params.affine_b

    # ---- C.3 ----
    def score_to_proba(self, score):
        if score is None or (isinstance(score, float) and np.isnan(score)):
            return float("nan")
        z = self.params.platt_coef * float(score) + self.params.platt_intercept
        return float(1.0 / (1.0 + np.exp(-z)))

    def probability_from_biomarkers(self, pred_vcdr, pred_rcdr):
        """
        Pipeline completo: pred_vCDR/pred_rCDR -> vCDR_corr -> score -> probabilidad.

        Devuelve dict con vcdr_corrected, score, proba y la decisión binaria
        (en espacio de score, como en el notebook).
        """
        vcdr_corr = self.correct_vcdr(pred_vcdr)
        score = build_score_from_biomarkers(vcdr_corr, pred_rcdr)
        proba = self.score_to_proba(score)

        if np.isnan(score):
            predicted_label = None
        else:
            predicted_label = int(score >= self.params.threshold)

        return {
            "vcdr_corrected": vcdr_corr,
            "score": score,
            "proba": proba,
            "predicted_label": predicted_label,
        }


def fit_calibration(val_csv_path=None, save_json=True):
    """
    Ajusta la calibración desde `external_validation_results.csv` (Val400).

    Reproduce C.1 (afín), C.2 (score) y C.3 (Platt + umbral) del notebook.
    """
    val_csv_path = val_csv_path or CFG.VAL_RESULTS_CSV

    if not os.path.exists(val_csv_path):
        raise FileNotFoundError(
            f"No se encuentra el CSV de validación para calibrar: {val_csv_path}. "
            "Cópialo de Drive (Models_v5_TverskyCup/external_validation_results.csv) "
            "o define GLAUCOMA_VAL_RESULTS_CSV."
        )

    df = pd.read_csv(val_csv_path)
    df = df.dropna(subset=[LABEL_COL]).copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    # ---- C.1: corrección afín del vCDR ----
    v = df.dropna(subset=["pred_vCDR", "true_vCDR"])
    reg = LinearRegression().fit(v[["pred_vCDR"]].values, v["true_vCDR"].values)
    a = float(reg.coef_[0])
    b = float(reg.intercept_)

    df["pred_vCDR_corr"] = a * df["pred_vCDR"].values + b

    # ---- C.2: score simplificado vCDR + rCDR ----
    vc = _minmax_clip(df["pred_vCDR_corr"], 0.45, 0.85)
    rc = _minmax_clip(df["pred_rCDR"], 0.50, 0.85)
    score = 0.78 * vc + 0.22 * rc

    # ---- C.3: Platt + umbral en Val400 ----
    y = df[LABEL_COL].values.astype(int)
    fit_mask = np.isfinite(score)

    platt = LogisticRegression().fit(
        np.asarray(score)[fit_mask].reshape(-1, 1), y[fit_mask]
    )
    platt_coef = float(platt.coef_[0][0])
    platt_intercept = float(platt.intercept_[0])

    threshold = _threshold_at_sensitivity(y[fit_mask], np.asarray(score)[fit_mask], 0.85)

    # Umbral equivalente en probabilidad (Platt es monótona creciente).
    z = platt_coef * threshold + platt_intercept
    proba_threshold = float(1.0 / (1.0 + np.exp(-z)))

    params = CalibrationParams(
        affine_a=a,
        affine_b=b,
        platt_coef=platt_coef,
        platt_intercept=platt_intercept,
        threshold=threshold,
        proba_threshold=proba_threshold,
        n_val_samples=int(fit_mask.sum()),
    )

    if save_json:
        try:
            params.to_json(CFG.CALIBRATION_PARAMS_JSON)
        except OSError:
            pass  # el guardado es solo cacheo; no es crítico

    return Calibrator(params)


def load_or_fit_calibration():
    """
    Carga `calibration_params.json` si existe; en caso contrario lo ajusta desde
    el CSV de Val400.
    """
    if os.path.exists(CFG.CALIBRATION_PARAMS_JSON):
        return Calibrator(CalibrationParams.from_json(CFG.CALIBRATION_PARAMS_JSON))
    return fit_calibration()
