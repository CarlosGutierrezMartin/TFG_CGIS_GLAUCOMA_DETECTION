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
import random
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .calibration import load_or_fit_calibration
from .config import CFG, DISCLAIMER
from .dataset import test_index
from .ensemble import ensemble
from .schemas import CalibrationInfo, CaseResponse, HealthResponse
from .service import process_case

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("glaucoma-api")

# Estado compartido inicializado en el arranque.
STATE = {"calibrator": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
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

    yield

    ensemble.models.clear()


app = FastAPI(
    title="Detección temprana de glaucoma — API REST",
    description=DISCLAIMER,
    version="1.0.0",
    lifespan=lifespan,
)


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
        status="ok" if ensemble.is_loaded else "loading",
        models_loaded=len(ensemble.models),
        n_test_cases=len(test_index),
        use_tta=CFG.USE_TTA,
        calibration=calibration_info,
        disclaimer=DISCLAIMER,
    )


@app.get("/api/test-cases/random", response_model=CaseResponse)
def random_test_case():
    """Devuelve un caso aleatorio de Test400 con predicción, probabilidad y acierto."""
    if not ensemble.is_loaded or STATE["calibrator"] is None:
        raise HTTPException(status_code=503, detail="El servicio aún se está inicializando.")
    if len(test_index) == 0:
        raise HTTPException(status_code=503, detail="No hay casos de Test400 indexados.")

    case = random.choice(test_index.cases)
    return _process(case)


@app.get("/api/test-cases/{image_name}", response_model=CaseResponse)
def test_case_by_name(image_name: str):
    """Devuelve un caso concreto de Test400 por nombre de imagen (p.ej. T0123.jpg)."""
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
