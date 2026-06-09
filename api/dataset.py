"""
Indexación del conjunto REFUGE Test400.

Empareja cada imagen con su máscara GT y su etiqueta clínica (0 sano / 1 glaucoma)
leída del xlsx. El emparejamiento imagen-máscara replica la lógica robusta del
notebook (por nombre normalizado y, como respaldo, por clave numérica).
"""

import glob
import os
import re

import pandas as pd

from .config import CFG

_IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
_MASK_EXTS = [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]


def _list_files(root_dir, extensions):
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True))
        files.extend(glob.glob(os.path.join(root_dir, "**", f"*{ext.upper()}"), recursive=True))
    return sorted(set(files))


def _normalize_stem(path):
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    for rep in ("_mask", "-mask", "_seg", "-seg", "_disc_cup", "-disc-cup",
                "_disc", "-disc", "_cup", "-cup"):
        stem = stem.replace(rep, "")
    return stem.strip()


def _numeric_key(path):
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    nums = re.findall(r"\d+", stem)
    if not nums:
        return None
    return nums[-1].lstrip("0") or "0"


class TestCase:
    """Un caso de Test400: imagen, máscara GT y etiqueta clínica."""

    __slots__ = ("image_name", "image_path", "mask_path", "true_label")

    def __init__(self, image_name, image_path, mask_path, true_label):
        self.image_name = image_name
        self.image_path = image_path
        self.mask_path = mask_path
        self.true_label = true_label


class Test400Index:
    """Índice de casos de Test400 con acceso aleatorio y por nombre."""

    def __init__(self):
        self.cases = []
        self._by_name = {}

    def load(self):
        images = _list_files(CFG.TEST_IMAGES_DIR, _IMAGE_EXTS)
        masks = _list_files(CFG.TEST_MASKS_DIR, _MASK_EXTS)

        if not images:
            raise FileNotFoundError(
                f"No se encontraron imágenes de Test400 en: {CFG.TEST_IMAGES_DIR}"
            )
        if not masks:
            raise FileNotFoundError(
                f"No se encontraron máscaras de Test400 en: {CFG.TEST_MASKS_DIR}"
            )

        masks_by_stem = {}
        masks_by_num = {}
        for m in masks:
            masks_by_stem[_normalize_stem(m)] = m
            num = _numeric_key(m)
            if num is not None and num not in masks_by_num:
                masks_by_num[num] = m

        labels = self._load_labels()

        self.cases = []
        self._by_name = {}
        unmatched = 0

        for img in images:
            stem = _normalize_stem(img)
            num = _numeric_key(img)

            mask_path = masks_by_stem.get(stem)
            if mask_path is None and num is not None:
                mask_path = masks_by_num.get(num)

            if mask_path is None:
                unmatched += 1
                continue

            image_name = os.path.basename(img)
            true_label = self._lookup_label(labels, image_name)

            case = TestCase(image_name, img, mask_path, true_label)
            self.cases.append(case)
            self._by_name[image_name] = case

        if not self.cases:
            raise RuntimeError(
                "No se pudo emparejar ninguna imagen de Test400 con su máscara."
            )

        return self, unmatched

    def _load_labels(self):
        if not os.path.exists(CFG.TEST_LABELS_XLSX):
            raise FileNotFoundError(
                f"No se encuentra el xlsx de etiquetas de Test400: {CFG.TEST_LABELS_XLSX}"
            )

        df = pd.read_excel(CFG.TEST_LABELS_XLSX)

        if CFG.TEST_LABEL_IMG_COL not in df.columns or CFG.TEST_LABEL_COL not in df.columns:
            raise KeyError(
                f"El xlsx no contiene las columnas esperadas "
                f"('{CFG.TEST_LABEL_IMG_COL}', '{CFG.TEST_LABEL_COL}'). "
                f"Columnas disponibles: {list(df.columns)}"
            )

        # Mapa nombre-de-imagen -> etiqueta. Normaliza a minúsculas para robustez.
        mapping = {}
        for _, row in df.iterrows():
            name = str(row[CFG.TEST_LABEL_IMG_COL]).strip()
            mapping[name.lower()] = int(row[CFG.TEST_LABEL_COL])

        return mapping

    @staticmethod
    def _lookup_label(labels, image_name):
        key = image_name.lower()
        if key in labels:
            return labels[key]
        # Probar emparejando solo el stem (sin extensión).
        stem = os.path.splitext(key)[0]
        for k, v in labels.items():
            if os.path.splitext(k)[0] == stem:
                return v
        return None

    def __len__(self):
        return len(self.cases)

    def get_by_name(self, image_name):
        return self._by_name.get(image_name)


# Instancia única usada por la app.
test_index = Test400Index()
