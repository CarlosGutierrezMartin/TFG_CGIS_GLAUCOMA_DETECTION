"""
Núcleo de procesamiento extraído de TFG_GLAUCOMA_v7.0.ipynb.

Contiene, sin cambios de lógica, las funciones de:
  - preprocesado (ROI, CLAHE, resize, preprocess_input del backbone),
  - inferencia/postprocesado anatómico,
  - extracción de biomarcadores clínicos,
  - overlays de visualización.

Se han eliminado únicamente las dependencias del entorno Colab (google.colab.drive,
display(), plt.show()) que no aplican en un servicio. La numeración de las secciones
se conserva para poder rastrear cada función a su celda de origen en el notebook.
"""

import os

# Debe definirse ANTES de importar segmentation_models (igual que en el notebook).
os.environ.setdefault("SM_FRAMEWORK", "tf.keras")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import segmentation_models as sm
import tensorflow as tf

from .config import CFG

# Preprocesamiento específico del backbone seleccionado (idéntico al notebook).
preprocess_input = sm.get_preprocessing(CFG.BACKBONE)


# ============================================================
# 3. PROTOCOLO EXPERIMENTAL Y PREPROCESAMIENTO (celda 6)
# ============================================================

def read_rgb_image(image_path):
    """Lee una imagen en RGB."""
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def read_mask_grayscale(mask_path):
    """Lee una mascara en escala de grises."""
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    if mask is None:
        raise FileNotFoundError(f"No se pudo leer la mascara: {mask_path}")

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    return mask


# ------------------------------------------------------------
# 3.5. Localizacion automatica de la region papilar
# ------------------------------------------------------------

def _legacy_locate_roi_center(image_rgb):
    """
    Algoritmo de localizacion ROI original (canal verde + CLAHE + percentil 99.3
    + mayor componente brillante + centroide).
    """
    green = image_rgb[:, :, 1]
    green = cv2.GaussianBlur(green, (9, 9), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    threshold = np.percentile(enhanced, 99.3)
    bright = (enhanced >= threshold).astype(np.uint8) * 255

    kernel = np.ones((15, 15), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = green.shape

    if len(contours) == 0:
        return w // 2, h // 2

    largest_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest_contour)

    if moments["m00"] == 0:
        return w // 2, h // 2

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return cx, cy


def _largest_component_mask(binary):
    """Devuelve la mascara del mayor componente conectado (ignora el fondo)."""
    binary = (binary > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num <= 1:
        return binary

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def estimate_fov_mask(image_rgb):
    """Estima la mascara del campo de vision (FOV) del fondo de ojo."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    fov = (gray > 12).astype(np.uint8)

    if fov.sum() < 0.05 * h * w:
        fov = np.ones((h, w), dtype=np.uint8)

    fov = _largest_component_mask(fov)

    fov = cv2.morphologyEx(fov, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    ys, xs = np.where(fov > 0)

    if len(xs) == 0:
        return (
            np.ones((h, w), dtype=np.uint8),
            (w / 2.0, h / 2.0),
            min(h, w) / 2.0,
        )

    fx = float(xs.mean())
    fy = float(ys.mean())
    fov_radius = float(np.sqrt(fov.sum() / np.pi))

    return fov, (fx, fy), fov_radius


def is_plausible_center(cx, cy, fov_info, image_shape):
    """Comprueba si un centro candidato es anatomicamente plausible."""
    fov_mask, (fx, fy), fov_radius = fov_info
    h, w = image_shape[:2]

    cx_i = int(round(cx))
    cy_i = int(round(cy))

    if not (0 <= cx_i < w and 0 <= cy_i < h):
        return False

    if fov_mask[cy_i, cx_i] == 0:
        return False

    dist_to_fov_center = np.hypot(cx_i - fx, cy_i - fy)

    if dist_to_fov_center > 0.92 * fov_radius:
        return False

    return True


def locate_roi_center_robust(image_rgb):
    """Localizacion robusta de la papila optica usando SOLO la imagen."""
    green = image_rgb[:, :, 1]
    green = cv2.GaussianBlur(green, (9, 9), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    h, w = enhanced.shape

    fov_mask, (fx, fy), fov_radius = estimate_fov_mask(image_rgb)
    enhanced_fov = np.where(fov_mask > 0, enhanced, 0).astype(np.uint8)

    fov_values = enhanced[fov_mask > 0]

    kernel = np.ones((15, 15), np.uint8)
    candidates = []

    for pct in (99.5, 99.0, 98.5, 97.5):
        if fov_values.size == 0:
            break

        threshold = np.percentile(fov_values, pct)
        bright = ((enhanced_fov >= threshold) & (fov_mask > 0)).astype(np.uint8) * 255
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 50:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            ccx = moments["m10"] / moments["m00"]
            ccy = moments["m01"] / moments["m00"]

            perimeter = cv2.arcLength(contour, True)
            circularity = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
            circularity = float(np.clip(circularity, 0.0, 1.0))

            contour_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
            mean_bright = float(cv2.mean(enhanced, mask=contour_mask)[0])

            candidates.append(
                {
                    "cx": ccx,
                    "cy": ccy,
                    "area": float(area),
                    "circularity": circularity,
                    "mean_bright": mean_bright,
                }
            )

    if len(candidates) == 0:
        diagnostics = {
            "center": (int(round(fx)), int(round(fy))),
            "score": float("nan"),
            "n_candidates": 0,
            "fov_center": (fx, fy),
            "fov_radius": fov_radius,
            "used_fallback": True,
        }
        return int(round(fx)), int(round(fy)), diagnostics

    dedup = []
    for cand in sorted(candidates, key=lambda c: c["area"], reverse=True):
        is_duplicate = False
        for kept in dedup:
            if np.hypot(cand["cx"] - kept["cx"], cand["cy"] - kept["cy"]) < 25:
                is_duplicate = True
                break
        if not is_duplicate:
            dedup.append(cand)

    expected_disc_radius = 0.20 * fov_radius

    for cand in dedup:
        eq_radius = np.sqrt(cand["area"] / np.pi)
        size_ratio = eq_radius / max(expected_disc_radius, 1e-6)
        size_score = float(np.exp(-0.5 * (np.log(max(size_ratio, 1e-6)) / 0.6) ** 2))

        dist = np.hypot(cand["cx"] - fx, cand["cy"] - fy)
        periph = dist / max(fov_radius, 1e-6)
        periph_penalty = float(np.clip((periph - 0.85) / 0.15, 0.0, 1.0))
        border_penalty = 1.0 if periph > 0.95 else 0.0

        bright_norm = cand["mean_bright"] / 255.0

        cand["score"] = (
            0.45 * bright_norm
            + 0.30 * cand["circularity"]
            + 0.25 * size_score
            - 0.50 * periph_penalty
            - 0.50 * border_penalty
        )

    best = max(dedup, key=lambda c: c["score"])

    diagnostics = {
        "center": (int(round(best["cx"])), int(round(best["cy"]))),
        "score": float(best["score"]),
        "n_candidates": len(dedup),
        "fov_center": (fx, fy),
        "fov_radius": fov_radius,
        "used_fallback": False,
    }

    return int(round(best["cx"])), int(round(best["cy"])), diagnostics


def locate_roi_center(image_rgb):
    """
    Localiza el centro de la papila optica usando SOLO la imagen.

    Estrategia de OVERRIDE CON GUARDA: respeta el centro del algoritmo original si
    es plausible; solo lo sustituye por el localizador robusto cuando falla.
    """
    cx_old, cy_old = _legacy_locate_roi_center(image_rgb)

    fov_info = estimate_fov_mask(image_rgb)

    if is_plausible_center(cx_old, cy_old, fov_info, image_rgb.shape):
        return cx_old, cy_old

    cx_new, cy_new, _ = locate_roi_center_robust(image_rgb)
    return cx_new, cy_new


def crop_square_with_padding(array, cx, cy, radius, is_mask=False):
    """Recorta una region cuadrada centrada en (cx, cy) con padding si se sale."""
    h, w = array.shape[:2]

    x1 = cx - radius
    x2 = cx + radius
    y1 = cy - radius
    y2 = cy + radius

    src_x1 = max(0, x1)
    src_x2 = min(w, x2)
    src_y1 = max(0, y1)
    src_y2 = min(h, y2)

    pad_left = max(0, -x1)
    pad_right = max(0, x2 - w)
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - h)

    cropped = array[src_y1:src_y2, src_x1:src_x2]

    pad_value = 255 if is_mask else 0

    if len(array.shape) == 3:
        cropped = cv2.copyMakeBorder(
            cropped, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[pad_value, pad_value, pad_value]
        )
    else:
        cropped = cv2.copyMakeBorder(
            cropped, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=pad_value
        )

    return cropped


# ------------------------------------------------------------
# 3.6. Conversion de mascaras a clases semanticas
# ------------------------------------------------------------

def decode_refuge_mask(mask_gray):
    """
    Convierte la mascara original de REFUGE a clases semanticas:
        0 = fondo, 1 = disco/anillo, 2 = copa.
    Usa asignacion por valor mas cercano (255=fondo, 128=disco, 0=copa).
    """
    mask_gray = mask_gray.astype(np.float32)

    unique_values = np.unique(mask_gray)

    if mask_gray.max() <= 2 and len(unique_values) <= 3:
        return mask_gray.astype(np.uint8)

    dist_to_cup = np.abs(mask_gray - 0)
    dist_to_disc = np.abs(mask_gray - 128)
    dist_to_background = np.abs(mask_gray - 255)

    stacked = np.stack([dist_to_background, dist_to_disc, dist_to_cup], axis=-1)

    return np.argmin(stacked, axis=-1).astype(np.uint8)


# ------------------------------------------------------------
# 3.7. Mejora de contraste y normalizacion
# ------------------------------------------------------------

def apply_clahe_rgb(image_rgb):
    """Aplica CLAHE sobre el canal L del espacio LAB."""
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    lab = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_image_only(image_path):
    """
    Preprocesa una imagen sin mascara para inferencia real.

    Devuelve (image_preprocessed, image_resized, metadata).
    """
    image_rgb = read_rgb_image(image_path)

    cx, cy = locate_roi_center(image_rgb)

    image_crop = crop_square_with_padding(
        image_rgb, cx, cy, CFG.ROI_RADIUS, is_mask=False
    )

    image_resized = cv2.resize(
        image_crop, (CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=cv2.INTER_LINEAR
    )

    image_enhanced = apply_clahe_rgb(image_resized)

    image_preprocessed = preprocess_input(image_enhanced.astype(np.float32))

    metadata = {
        "cx": cx,
        "cy": cy,
        "roi_radius": CFG.ROI_RADIUS,
        "original_shape": image_rgb.shape,
    }

    return image_preprocessed.astype(np.float32), image_resized, metadata


def true_mask_for_pair(image_path, mask_path):
    """
    Reconstruye la mascara real (0/1/2) alineada al recorte ROI de la imagen,
    igual que `infer_pair` en el notebook (Seccion 8.7).
    """
    mask_gray = read_mask_grayscale(mask_path)

    image_rgb = read_rgb_image(image_path)
    cx, cy = locate_roi_center(image_rgb)

    mask_crop = crop_square_with_padding(
        mask_gray, cx, cy, CFG.ROI_RADIUS, is_mask=True
    )

    mask_resized = cv2.resize(
        mask_crop, (CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=cv2.INTER_NEAREST
    )

    return decode_refuge_mask(mask_resized)


# ============================================================
# 8. INFERENCIA Y SEGMENTACION (celda 16)
# ============================================================

# Postprocesamiento anatomico basico.
MIN_DISC_AREA_PIXELS = int(0.0025 * CFG.IMG_SIZE * CFG.IMG_SIZE)
MIN_CUP_AREA_PIXELS = int(0.0005 * CFG.IMG_SIZE * CFG.IMG_SIZE)


def class_mask_to_clinical_masks(mask_class):
    """Convierte una mascara semantica 0/1/2 en mascaras clinicas binarias."""
    mask_class = mask_class.astype(np.uint8)

    optic_disc_mask = np.logical_or(mask_class == 1, mask_class == 2)
    cup_mask = mask_class == 2
    rim_mask = np.logical_and(optic_disc_mask, np.logical_not(cup_mask))

    return {
        "optic_disc": optic_disc_mask.astype(np.uint8),
        "cup": cup_mask.astype(np.uint8),
        "rim": rim_mask.astype(np.uint8),
    }


def clinical_masks_to_class_mask(optic_disc_mask, cup_mask):
    """Reconstruye una mascara semantica 0/1/2 desde mascaras clinicas."""
    optic_disc_mask = optic_disc_mask.astype(bool)
    cup_mask = cup_mask.astype(bool)

    final_mask = np.zeros(optic_disc_mask.shape, dtype=np.uint8)
    final_mask[optic_disc_mask] = 1
    final_mask[cup_mask] = 2

    return final_mask


def largest_connected_component(binary_mask, min_area=0):
    """Conserva el mayor componente conectado de una mascara binaria."""
    binary_mask = binary_mask.astype(np.uint8)

    if binary_mask.max() == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8), {
            "found": False, "area": 0, "num_components": 0,
        }

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    if num_labels <= 1:
        return np.zeros_like(binary_mask, dtype=np.uint8), {
            "found": False, "area": 0, "num_components": 0,
        }

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(component_areas)) + 1
    largest_area = int(stats[largest_idx, cv2.CC_STAT_AREA])

    if largest_area < min_area:
        return np.zeros_like(binary_mask, dtype=np.uint8), {
            "found": False, "area": largest_area,
            "num_components": int(num_labels - 1),
        }

    largest_mask = (labels == largest_idx).astype(np.uint8)

    return largest_mask, {
        "found": True, "area": largest_area,
        "num_components": int(num_labels - 1),
    }


def fill_binary_holes(binary_mask):
    """Rellena huecos internos de una mascara binaria."""
    binary_mask = binary_mask.astype(np.uint8)

    if binary_mask.max() == 0:
        return binary_mask

    h, w = binary_mask.shape

    flood = binary_mask.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(flood, mask, seedPoint=(0, 0), newVal=1)

    flood_inv = 1 - flood
    return np.logical_or(binary_mask, flood_inv).astype(np.uint8)


def postprocess_segmentation_mask(mask_class):
    """Aplica restricciones anatomicas simples a la segmentacion final."""
    clinical_masks = class_mask_to_clinical_masks(mask_class)

    raw_disc = clinical_masks["optic_disc"]
    raw_cup = clinical_masks["cup"]

    disc_lcc, disc_info = largest_connected_component(
        raw_disc, min_area=MIN_DISC_AREA_PIXELS
    )
    disc_lcc = fill_binary_holes(disc_lcc)

    cup_lcc, cup_info = largest_connected_component(
        raw_cup, min_area=MIN_CUP_AREA_PIXELS
    )
    cup_lcc = fill_binary_holes(cup_lcc)

    cup_area_before = int(cup_lcc.sum())

    cup_lcc = np.logical_and(
        cup_lcc.astype(bool), disc_lcc.astype(bool)
    ).astype(np.uint8)

    cup_area_after = int(cup_lcc.sum())

    if cup_area_after < MIN_CUP_AREA_PIXELS:
        cup_lcc = np.zeros_like(cup_lcc, dtype=np.uint8)

    final_mask = clinical_masks_to_class_mask(
        optic_disc_mask=disc_lcc, cup_mask=cup_lcc
    )

    diagnostics = {
        "disc_found": disc_info["found"],
        "disc_area": int(disc_lcc.sum()),
        "disc_components_raw": disc_info["num_components"],
        "cup_found": cup_info["found"] and int(cup_lcc.sum()) >= MIN_CUP_AREA_PIXELS,
        "cup_area_before_intersection": cup_area_before,
        "cup_area": int(cup_lcc.sum()),
        "cup_components_raw": cup_info["num_components"],
        "cup_area_removed_outside_disc": int(cup_area_before - cup_area_after),
        "valid_anatomy": bool(disc_lcc.sum() > 0 and cup_lcc.sum() > 0),
    }

    return final_mask, diagnostics


def compute_confidence_maps(probability_map):
    """Calcula mapas de confianza e incertidumbre desde la salida softmax."""
    probability_map = probability_map.astype(np.float32)

    confidence = np.max(probability_map, axis=-1)

    eps = 1e-7
    entropy = -np.sum(probability_map * np.log(probability_map + eps), axis=-1)
    entropy = entropy / np.log(CFG.CLASSES)

    return {
        "confidence": confidence.astype(np.float32),
        "entropy": entropy.astype(np.float32),
    }


def summarize_prediction_quality(probability_map, postprocess_diagnostics):
    """Resume la calidad interna de una prediccion sin usar etiqueta real."""
    confidence_maps = compute_confidence_maps(probability_map)

    confidence = confidence_maps["confidence"]
    entropy = confidence_maps["entropy"]

    return {
        "mean_confidence": float(np.mean(confidence)),
        "median_confidence": float(np.median(confidence)),
        "mean_entropy": float(np.mean(entropy)),
        "median_entropy": float(np.median(entropy)),
        "disc_found": postprocess_diagnostics["disc_found"],
        "cup_found": postprocess_diagnostics["cup_found"],
        "valid_anatomy": postprocess_diagnostics["valid_anatomy"],
    }


def predict_model_with_tta(model, images_np, use_tta=True):
    """
    Predice con un modelo individual, con TTA por flip horizontal (sin flip vertical
    para preservar la semantica superior/inferior).
    """
    pred = model.predict(images_np, verbose=0).astype(np.float32)

    if use_tta:
        images_flip = images_np[:, :, ::-1, :]
        pred_flip = model.predict(images_flip, verbose=0).astype(np.float32)
        pred_flip = pred_flip[:, :, ::-1, :]
        pred = (pred + pred_flip) / 2.0

    return pred


# ------------------------------------------------------------
# 8.8. Overlays de visualizacion
# ------------------------------------------------------------

def create_overlay(image_rgb, mask_class, alpha=0.35):
    """
    Crea una superposicion de mascara sobre imagen.
        clase 1 = disco/anillo (verde), clase 2 = copa (rojo).
    Devuelve float [0,1].
    """
    image = image_rgb.astype(np.float32)

    if image.max() > 1.0:
        image = image / 255.0

    overlay = image.copy()

    disc_color = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    cup_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    disc_region = mask_class == 1
    cup_region = mask_class == 2

    overlay[disc_region] = (1.0 - alpha) * overlay[disc_region] + alpha * disc_color
    overlay[cup_region] = (1.0 - alpha) * overlay[cup_region] + alpha * cup_color

    return np.clip(overlay, 0.0, 1.0)


# ============================================================
# 9. EXTRACCION DE BIOMARCADORES CLINICOS (celda 18)
# ============================================================

def safe_divide(numerator, denominator, default=np.nan):
    """Division segura para evitar errores por denominador cero."""
    if denominator is None or denominator == 0:
        return default
    return float(numerator) / float(denominator)


def binary_mask_area(mask):
    """Calcula el area en pixeles de una mascara binaria."""
    return int(np.sum(mask.astype(bool)))


def get_binary_bbox(mask):
    """Devuelve bounding box de una mascara binaria (o None si esta vacia)."""
    mask = mask.astype(bool)

    if not np.any(mask):
        return None

    ys, xs = np.where(mask)

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    return {
        "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
        "width": int(x_max - x_min + 1), "height": int(y_max - y_min + 1),
    }


def get_mask_centroid(mask):
    """Calcula el centroide de una mascara binaria (o (nan, nan) si esta vacia)."""
    mask_uint8 = mask.astype(np.uint8)

    if mask_uint8.sum() == 0:
        return np.nan, np.nan

    moments = cv2.moments(mask_uint8)

    if moments["m00"] == 0:
        return np.nan, np.nan

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    return float(cx), float(cy)


def get_clinical_masks_from_class_mask(mask_class):
    """Convierte una mascara 0/1/2 en estructuras clinicas (disco/copa/anillo)."""
    mask_class = mask_class.astype(np.uint8)

    optic_disc_mask = np.logical_or(mask_class == 1, mask_class == 2)
    cup_mask = mask_class == 2
    rim_mask = np.logical_and(optic_disc_mask, np.logical_not(cup_mask))

    return {
        "optic_disc": optic_disc_mask.astype(np.uint8),
        "cup": cup_mask.astype(np.uint8),
        "rim": rim_mask.astype(np.uint8),
    }


def compute_cdr_metrics(optic_disc_mask, cup_mask):
    """Calcula relaciones copa/disco: vCDR, hCDR, area_CDR, rCDR."""
    disc_bbox = get_binary_bbox(optic_disc_mask)
    cup_bbox = get_binary_bbox(cup_mask)

    disc_area = binary_mask_area(optic_disc_mask)
    cup_area = binary_mask_area(cup_mask)

    if disc_bbox is None:
        disc_vertical_diameter = 0
        disc_horizontal_diameter = 0
    else:
        disc_vertical_diameter = disc_bbox["height"]
        disc_horizontal_diameter = disc_bbox["width"]

    if cup_bbox is None:
        cup_vertical_diameter = 0
        cup_horizontal_diameter = 0
    else:
        cup_vertical_diameter = cup_bbox["height"]
        cup_horizontal_diameter = cup_bbox["width"]

    vcdr = safe_divide(cup_vertical_diameter, disc_vertical_diameter, default=np.nan)
    hcdr = safe_divide(cup_horizontal_diameter, disc_horizontal_diameter, default=np.nan)
    area_cdr = safe_divide(cup_area, disc_area, default=np.nan)

    rcdr = np.nan if np.isnan(area_cdr) else float(np.sqrt(max(area_cdr, 0.0)))

    return {
        "disc_vertical_diameter": int(disc_vertical_diameter),
        "cup_vertical_diameter": int(cup_vertical_diameter),
        "disc_horizontal_diameter": int(disc_horizontal_diameter),
        "cup_horizontal_diameter": int(cup_horizontal_diameter),
        "vCDR": float(vcdr) if not np.isnan(vcdr) else np.nan,
        "hCDR": float(hcdr) if not np.isnan(hcdr) else np.nan,
        "area_CDR": float(area_cdr) if not np.isnan(area_cdr) else np.nan,
        "rCDR": float(rcdr) if not np.isnan(rcdr) else np.nan,
    }


def compute_area_metrics(optic_disc_mask, cup_mask, rim_mask):
    """Calcula areas estructurales basicas y ratios anillo/disco, copa/anillo."""
    disc_area = binary_mask_area(optic_disc_mask)
    cup_area = binary_mask_area(cup_mask)
    rim_area = binary_mask_area(rim_mask)

    rim_to_disc_ratio = safe_divide(rim_area, disc_area, default=np.nan)
    cup_to_rim_ratio = safe_divide(cup_area, rim_area, default=np.nan)

    return {
        "disc_area": int(disc_area),
        "cup_area": int(cup_area),
        "rim_area": int(rim_area),
        "rim_to_disc_ratio": float(rim_to_disc_ratio) if not np.isnan(rim_to_disc_ratio) else np.nan,
        "cup_to_rim_ratio": float(cup_to_rim_ratio) if not np.isnan(cup_to_rim_ratio) else np.nan,
    }


def compute_sector_masks(shape, center_x, center_y):
    """Divide la imagen en cuatro sectores angulares alrededor del centro del disco."""
    h, w = shape

    yy, xx = np.indices((h, w))

    dx = xx - center_x
    dy = yy - center_y

    angles = np.degrees(np.arctan2(dy, dx))

    superior = np.logical_and(angles >= -135, angles < -45)
    inferior = np.logical_and(angles >= 45, angles < 135)
    right = np.logical_and(angles >= -45, angles < 45)
    left = np.logical_or(angles >= 135, angles < -135)

    return {"superior": superior, "inferior": inferior, "left": left, "right": right}


def compute_rim_sector_ratios(optic_disc_mask, rim_mask):
    """Calcula proporcion de anillo por sector e indicador ISNT-like aproximado."""
    disc_cx, disc_cy = get_mask_centroid(optic_disc_mask)

    if np.isnan(disc_cx) or np.isnan(disc_cy):
        return {
            "rim_ratio_superior": np.nan, "rim_ratio_inferior": np.nan,
            "rim_ratio_left": np.nan, "rim_ratio_right": np.nan,
            "vertical_rim_ratio_mean": np.nan, "horizontal_rim_ratio_mean": np.nan,
            "vertical_to_horizontal_rim_ratio": np.nan,
            "isnt_like_violation_count": np.nan, "isnt_like_risk": np.nan,
        }

    sector_masks = compute_sector_masks(
        shape=optic_disc_mask.shape, center_x=disc_cx, center_y=disc_cy
    )

    ratios = {}

    for sector_name, sector_mask in sector_masks.items():
        disc_sector = np.logical_and(optic_disc_mask.astype(bool), sector_mask)
        rim_sector = np.logical_and(rim_mask.astype(bool), sector_mask)

        disc_sector_area = binary_mask_area(disc_sector)
        rim_sector_area = binary_mask_area(rim_sector)

        ratio = safe_divide(rim_sector_area, disc_sector_area, default=np.nan)
        ratios[f"rim_ratio_{sector_name}"] = float(ratio) if not np.isnan(ratio) else np.nan

    superior = ratios["rim_ratio_superior"]
    inferior = ratios["rim_ratio_inferior"]
    left = ratios["rim_ratio_left"]
    right = ratios["rim_ratio_right"]

    vertical_mean = np.nanmean([superior, inferior])
    horizontal_mean = np.nanmean([left, right])

    vertical_to_horizontal = safe_divide(vertical_mean, horizontal_mean, default=np.nan)

    violations = []

    if not np.isnan(inferior) and not np.isnan(superior):
        violations.append(int(inferior <= superior))

    lateral_min = np.nanmin([left, right])

    if not np.isnan(superior) and not np.isnan(lateral_min):
        violations.append(int(superior <= lateral_min))

    if not np.isnan(vertical_mean) and not np.isnan(horizontal_mean):
        violations.append(int(vertical_mean <= horizontal_mean))

    if len(violations) == 0:
        violation_count = np.nan
        isnt_like_risk = np.nan
    else:
        violation_count = int(np.sum(violations))
        isnt_like_risk = float(np.mean(violations))

    ratios.update(
        {
            "vertical_rim_ratio_mean": float(vertical_mean) if not np.isnan(vertical_mean) else np.nan,
            "horizontal_rim_ratio_mean": float(horizontal_mean) if not np.isnan(horizontal_mean) else np.nan,
            "vertical_to_horizontal_rim_ratio": float(vertical_to_horizontal) if not np.isnan(vertical_to_horizontal) else np.nan,
            "isnt_like_violation_count": violation_count,
            "isnt_like_risk": isnt_like_risk,
        }
    )

    return ratios


def compute_centroid_metrics(optic_disc_mask, cup_mask):
    """Calcula el desplazamiento relativo entre centroide de disco y copa."""
    disc_cx, disc_cy = get_mask_centroid(optic_disc_mask)
    cup_cx, cup_cy = get_mask_centroid(cup_mask)

    disc_bbox = get_binary_bbox(optic_disc_mask)

    if disc_bbox is None:
        norm_factor = np.nan
    else:
        norm_factor = max(disc_bbox["width"], disc_bbox["height"])

    if (
        np.isnan(disc_cx) or np.isnan(disc_cy)
        or np.isnan(cup_cx) or np.isnan(cup_cy)
        or np.isnan(norm_factor) or norm_factor == 0
    ):
        centroid_offset = np.nan
        centroid_offset_x = np.nan
        centroid_offset_y = np.nan
    else:
        centroid_offset_x = (cup_cx - disc_cx) / norm_factor
        centroid_offset_y = (cup_cy - disc_cy) / norm_factor
        centroid_offset = np.sqrt(centroid_offset_x ** 2 + centroid_offset_y ** 2)

    return {
        "disc_centroid_x": float(disc_cx) if not np.isnan(disc_cx) else np.nan,
        "disc_centroid_y": float(disc_cy) if not np.isnan(disc_cy) else np.nan,
        "cup_centroid_x": float(cup_cx) if not np.isnan(cup_cx) else np.nan,
        "cup_centroid_y": float(cup_cy) if not np.isnan(cup_cy) else np.nan,
        "cup_disc_centroid_offset_x": float(centroid_offset_x) if not np.isnan(centroid_offset_x) else np.nan,
        "cup_disc_centroid_offset_y": float(centroid_offset_y) if not np.isnan(centroid_offset_y) else np.nan,
        "cup_disc_centroid_offset": float(centroid_offset) if not np.isnan(centroid_offset) else np.nan,
    }


def compute_biomarkers_from_mask(mask_class):
    """Calcula todos los biomarcadores clinicos derivados de una mascara 0/1/2."""
    clinical_masks = get_clinical_masks_from_class_mask(mask_class)

    optic_disc_mask = clinical_masks["optic_disc"]
    cup_mask = clinical_masks["cup"]
    rim_mask = clinical_masks["rim"]

    area_metrics = compute_area_metrics(optic_disc_mask, cup_mask, rim_mask)
    cdr_metrics = compute_cdr_metrics(optic_disc_mask, cup_mask)
    sector_metrics = compute_rim_sector_ratios(optic_disc_mask, rim_mask)
    centroid_metrics = compute_centroid_metrics(optic_disc_mask, cup_mask)

    biomarkers = {}
    biomarkers.update(area_metrics)
    biomarkers.update(cdr_metrics)
    biomarkers.update(sector_metrics)
    biomarkers.update(centroid_metrics)

    biomarkers["valid_disc"] = bool(area_metrics["disc_area"] > 0)
    biomarkers["valid_cup"] = bool(area_metrics["cup_area"] > 0)
    biomarkers["valid_biomarkers"] = bool(
        biomarkers["valid_disc"]
        and biomarkers["valid_cup"]
        and not np.isnan(biomarkers["vCDR"])
    )

    return biomarkers
