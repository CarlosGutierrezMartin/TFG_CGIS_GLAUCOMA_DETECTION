"""
API REST de detección temprana de glaucoma (FastAPI).

Sirve el ensemble U-Net del TFG sobre el conjunto REFUGE Test400. Expone una
consulta aleatoria que devuelve, en JSON: imagen original, máscara real (GT),
predicción del modelo (overlays base64), probabilidad calibrada en %, banda de
riesgo no intrusiva y comprobación de acierto frente a la etiqueta real.

Arranque (lifespan): carga los 5 modelos en memoria, ajusta la calibración desde
Validation400 e indexa Test400.

Ejecutar:  uvicorn api.app:app --reload
"""

import logging
import math
import os
import random
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np

from .calibration import load_or_fit_calibration
from .config import CFG, DISCLAIMER, risk_band_for_probability
from .dataset import test_index
from .ensemble import ensemble
from .schemas import CalibrationInfo, CaseResponse, HealthResponse
from .service import process_case
from .viz import encode_png_base64

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("glaucoma-api")

# Estado compartido inicializado en el arranque.
STATE = {"calibrator": None, "mode": "loading", "startup_error": None}
FRONTEND_INDEX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "frontend", "index.html")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("GLAUCOMA_API_MODE", "").lower() == "demo":
        log.info("Modo demo solicitado por GLAUCOMA_API_MODE=demo.")
        STATE["mode"] = "demo"
        STATE["startup_error"] = None
        yield
        return

    try:
        log.info("Cargando ensemble de modelos (%d folds)...", CFG.N_SPLITS)
        ensemble.load()
        log.info("Modelos cargados: %d", len(ensemble.models))

        log.info("Ajustando calibración desde Validation400...")
        STATE["calibrator"] = load_or_fit_calibration()
        p = STATE["calibrator"].params
        log.info(
            "Calibración lista: afín a=%.4f b=%.4f | umbral=%.4f (proba=%.4f) | n_val=%d",
            p.affine_a, p.affine_b, p.threshold, p.proba_threshold, p.n_val_samples,
        )

        _, unmatched = test_index.load()
        log.info("Test400 indexado: %d casos (%d imágenes sin máscara).", len(test_index), unmatched)
        STATE["mode"] = "real"
        STATE["startup_error"] = None
    except Exception as exc:  # noqa: BLE001 - permite que el HTML abra en modo demo
        log.exception("No se pudo iniciar el pipeline real; se activa el modo demo.")
        STATE["mode"] = "demo"
        STATE["startup_error"] = str(exc)
        STATE["calibrator"] = None

    yield

    ensemble.models.clear()


app = FastAPI(
    title="Detección temprana de glaucoma — API REST",
    description=DISCLAIMER,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def frontend():
    """Sirve la interfaz HTML sin arrancar un servidor aparte."""
    if not os.path.exists(FRONTEND_INDEX):
        raise HTTPException(status_code=404, detail="No se encontró frontend/index.html")
    return FileResponse(FRONTEND_INDEX)


@app.get("/health", response_model=HealthResponse)
def health():
    calibrator = STATE["calibrator"]
    calibration_info = None
    if calibrator is not None:
        p = calibrator.params
        calibration_info = CalibrationInfo(
            affine_a=round(p.affine_a, 6),
            affine_b=round(p.affine_b, 6),
            threshold=round(p.threshold, 6),
            proba_threshold=round(p.proba_threshold, 6),
            n_val_samples=p.n_val_samples,
        )

    return HealthResponse(
        status="ok" if STATE["mode"] in ("real", "demo") else "loading",
        mode=STATE["mode"],
        ready=STATE["mode"] in ("real", "demo"),
        models_loaded=len(ensemble.models),
        n_test_cases=len(test_index),
        use_tta=CFG.USE_TTA,
        calibration=calibration_info,
        startup_error=STATE["startup_error"],
        disclaimer=DISCLAIMER,
    )


@app.get("/api/test-cases/random", response_model=CaseResponse)
def random_test_case():
    """Devuelve un caso aleatorio de Test400 con predicción, probabilidad y acierto."""
    if STATE["mode"] == "demo":
        return _demo_case()
    if not ensemble.is_loaded or STATE["calibrator"] is None:
        raise HTTPException(status_code=503, detail="El servicio aún se está inicializando.")
    if len(test_index) == 0:
        raise HTTPException(status_code=503, detail="No hay casos de Test400 indexados.")

    case = random.choice(test_index.cases)
    return _process(case)


@app.get("/api/test-cases/{image_name}", response_model=CaseResponse)
def test_case_by_name(image_name: str):
    """Devuelve un caso concreto de Test400 por nombre de imagen (p.ej. T0123.jpg)."""
    if STATE["mode"] == "demo":
        raise HTTPException(
            status_code=503,
            detail="El servicio está en modo demo; solo está disponible /api/test-cases/random.",
        )
    if not ensemble.is_loaded or STATE["calibrator"] is None:
        raise HTTPException(status_code=503, detail="El servicio aún se está inicializando.")

    case = test_index.get_by_name(image_name)
    if case is None:
        raise HTTPException(status_code=404, detail=f"Caso no encontrado: {image_name}")

    return _process(case)


def _process(case):
    try:
        return process_case(case, ensemble, STATE["calibrator"])
    except Exception as exc:  # noqa: BLE001 - se devuelve como 500 controlado
        log.exception("Error procesando el caso %s", case.image_name)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _circle_mask(shape, center, radius, value):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius ** 2
    out = np.zeros(shape, dtype=np.uint8)
    out[mask] = value
    return out


def _overlay(image, mask, alpha=0.35):
    colors = np.zeros_like(image)
    colors[mask == 1] = (36, 170, 95)
    colors[mask == 2] = (220, 72, 61)
    painted = mask > 0
    out = image.copy()
    out[painted] = ((1.0 - alpha) * out[painted] + alpha * colors[painted]).astype(np.uint8)
    return out


def _demo_case():
    """Respuesta sintética con el mismo contrato que el pipeline real."""
    size = 512
    y, x = np.mgrid[0:size, 0:size]
    cx, cy = 264, 250
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    fundus = np.zeros((size, size, 3), dtype=np.float32)
    fundus[..., 0] = 95 + 105 * np.exp(-(radius / 245) ** 2)
    fundus[..., 1] = 38 + 65 * np.exp(-(radius / 255) ** 2)
    fundus[..., 2] = 24 + 34 * np.exp(-(radius / 275) ** 2)

    vessel = (np.sin((x - 70) / 19) + np.cos((y + 45) / 29)) > 1.45
    fundus[vessel] *= np.array([0.58, 0.48, 0.45])

    disc = _circle_mask((size, size), (268, 246), 86, 1)
    cup = _circle_mask((size, size), (278, 236), 46, 2)
    pred_mask = np.maximum(disc, cup)
    gt_mask = np.maximum(
        _circle_mask((size, size), (264, 250), 82, 1),
        _circle_mask((size, size), (274, 240), 39, 2),
    )

    image = np.clip(fundus, 0, 255).astype(np.uint8)
    probability_percent = round(46.8 + 7.5 * math.sin(random.random() * math.tau), 1)
    risk_band, risk_message = risk_band_for_probability(probability_percent)

    return {
        "image_name": "DEMO_TFG_0001.png",
        "probability_percent": probability_percent,
        "risk_band": risk_band,
        "risk_message": f"{risk_message} Muestra sintética para comprobar la interfaz.",
        "images": {
            "original": encode_png_base64(image),
            "prediction_overlay": encode_png_base64(_overlay(image, pred_mask)),
            "ground_truth_overlay": encode_png_base64(_overlay(image, gt_mask)),
        },
        "biomarkers": {
            "vCDR": 0.535,
            "vCDR_corrected": 0.558,
            "hCDR": 0.548,
            "rCDR": 0.531,
            "area_CDR": 0.282,
            "rim_to_disc_ratio": 0.718,
            "isnt_like_risk": 0.35,
        },
        "ground_truth": {"true_label": 0, "true_vCDR": 0.476},
        "decision": {
            "predicted_label": 1 if probability_percent >= 30.0 else 0,
            "threshold": 0.288,
            "score": 0.421,
            "correct": None,
        },
        "quality": {
            "mean_entropy": 0.073,
            "valid_anatomy": True,
            "disc_found": True,
            "cup_found": True,
        },
        "disclaimer": DISCLAIMER,
    }
