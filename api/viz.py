"""
Generación de imágenes para la respuesta de la API.

Produce los overlays disco (verde) / copa (rojo) reutilizando `create_overlay`
del núcleo y los codifica como PNG en base64 listos para incrustar en un <img> o
descodificar en el frontend.
"""

import base64

import cv2
import numpy as np

from .glaucoma_core import create_overlay


def _to_uint8_rgb(image):
    """Normaliza una imagen float [0,1] o uint8 a uint8 RGB."""
    image = np.asarray(image)

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255.0).clip(0, 255)
        image = image.astype(np.uint8)

    return image


def encode_png_base64(image_rgb):
    """Codifica una imagen RGB como PNG en base64 (sin prefijo data URI)."""
    image_rgb = _to_uint8_rgb(image_rgb)

    # cv2 espera BGR para imencode.
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", image_bgr)

    if not ok:
        raise RuntimeError("No se pudo codificar la imagen a PNG.")

    return base64.b64encode(buffer.tobytes()).decode("ascii")


def build_case_images(image_resized, predicted_mask, true_mask=None, alpha=0.35):
    """
    Genera el conjunto de imágenes base64 de un caso:
      - original (ROI redimensionada),
      - overlay de predicción,
      - overlay de máscara real (si se proporciona).
    """
    images = {
        "original": encode_png_base64(image_resized),
        "prediction_overlay": encode_png_base64(
            create_overlay(image_resized, predicted_mask, alpha=alpha)
        ),
    }

    if true_mask is not None:
        images["ground_truth_overlay"] = encode_png_base64(
            create_overlay(image_resized, true_mask, alpha=alpha)
        )

    return images
