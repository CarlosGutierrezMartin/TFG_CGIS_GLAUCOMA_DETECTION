"""
Smoke test offline de la API (sin levantar el servidor).

Recorre N casos de Test400, ejecuta el pipeline completo y reporta métricas
agregadas (accuracy, sensibilidad, especificidad, MAE de vCDR) contrastables con
los valores del TFG (AUC≈0.86, sens 0.80, spec 0.75 en el conjunto completo).

Opcionalmente compara la paridad de vCDR/probabilidad frente al CSV post-hoc del
notebook (posthoc_refinement/test_results_posthoc.csv) si se proporciona con
--parity-csv.

Uso:
    python -m scripts.smoke_test --n 40
    python -m scripts.smoke_test --n 0 --parity-csv artifacts/test_results_posthoc.csv
"""

import argparse
import math
import random

import numpy as np

from api.calibration import load_or_fit_calibration
from api.dataset import test_index
from api.ensemble import ensemble
from api.service import process_case


def _safe_div(a, b):
    return a / b if b else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40,
                        help="Número de casos a evaluar (0 = todos).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parity-csv", type=str, default=None,
                        help="CSV post-hoc del notebook para comprobar paridad.")
    args = parser.parse_args()

    print("[INFO] Cargando ensemble...")
    ensemble.load()
    print(f"[OK] Modelos cargados: {len(ensemble.models)}")

    print("[INFO] Ajustando calibración...")
    calibrator = load_or_fit_calibration()
    p = calibrator.params
    print(f"[OK] afín a={p.affine_a:.4f} b={p.affine_b:.4f} | umbral={p.threshold:.4f}")

    _, unmatched = test_index.load()
    print(f"[OK] Test400: {len(test_index)} casos ({unmatched} sin máscara).")

    cases = list(test_index.cases)
    random.seed(args.seed)
    random.shuffle(cases)
    if args.n > 0:
        cases = cases[: args.n]

    parity = {}
    if args.parity_csv:
        import pandas as pd
        pdf = pd.read_csv(args.parity_csv)
        name_col = "image_name" if "image_name" in pdf.columns else pdf.columns[0]
        parity = {str(r[name_col]): r for _, r in pdf.iterrows()}

    tp = fp = tn = fn = 0
    vcdr_abs_errors = []
    parity_diffs = []

    for i, case in enumerate(cases, 1):
        res = process_case(case, ensemble, calibrator)

        pred = res["decision"]["predicted_label"]
        true = res["ground_truth"]["true_label"]

        if pred is not None and true is not None:
            if pred == 1 and true == 1:
                tp += 1
            elif pred == 1 and true == 0:
                fp += 1
            elif pred == 0 and true == 0:
                tn += 1
            else:
                fn += 1

        pv = res["biomarkers"]["vCDR"]
        tv = res["ground_truth"]["true_vCDR"]
        if pv is not None and tv is not None:
            vcdr_abs_errors.append(abs(pv - tv))

        if parity and case.image_name in parity:
            ref = parity[case.image_name]
            if "pred_vCDR" in ref and not math.isnan(float(ref["pred_vCDR"])):
                parity_diffs.append(abs((pv or 0) - float(ref["pred_vCDR"])))

        if i % 10 == 0:
            print(f"  procesados {i}/{len(cases)}")

    n_dec = tp + fp + tn + fn
    print("\n=== Resultados agregados ===")
    print(f"Casos con decisión: {n_dec}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Accuracy    : {_safe_div(tp + tn, n_dec):.3f}")
    print(f"Sensibilidad: {_safe_div(tp, tp + fn):.3f}")
    print(f"Especificidad: {_safe_div(tn, tn + fp):.3f}")
    if vcdr_abs_errors:
        print(f"MAE vCDR    : {np.mean(vcdr_abs_errors):.4f} (n={len(vcdr_abs_errors)})")

    if parity_diffs:
        print(f"\n[PARIDAD] |pred_vCDR API - notebook| medio={np.mean(parity_diffs):.5f} "
              f"max={np.max(parity_diffs):.5f} (n={len(parity_diffs)})")


if __name__ == "__main__":
    main()
