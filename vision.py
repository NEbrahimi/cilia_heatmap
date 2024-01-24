from typing import Tuple
import cv2
import numpy as np


def apply_morph(image: np.ndarray,
                morph_type=cv2.MORPH_CLOSE,
                kernel_size: Tuple[int, int] = (3, 3),
                make_gaussian: bool = True):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    if make_gaussian:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    return cv2.morphologyEx(image, morph_type, kernel)


def add_images(image1: np.ndarray,
               image2: np.ndarray) -> np.ndarray:
    return np.array(image1, dtype=np.uint64) + np.array(image2, dtype=np.uint64)


def normalize_image(image: np.ndarray) -> np.ndarray:
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def apply_heatmap_colors(image: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(image, cv2.COLORMAP_TURBO)


def superimpose(image1: np.ndarray,
                image2: np.ndarray,
                alpha: float = 0.5) -> np.ndarray:
    return cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0.0)
