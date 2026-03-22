import numpy as np
import cv2


def mask_area(mask: np.ndarray) -> int:
    """
    Aire en pixels (nombre de pixels True).
    """
    return int(np.sum(mask))


def mask_bbox(mask: np.ndarray):
    """
    BBox serrée du masque : (x1, y1, x2, y2).
    Si masque vide, retourner None.
    """
    if mask is None or mask.size == 0 or not mask.any():
        return None

    ys, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2


def mask_perimeter(mask: np.ndarray) -> float:
    """
    Périmètre approximatif via extraction de contours OpenCV.
    Si masque vide, retourner 0.0.
    """
    if mask is None or mask.size == 0 or not mask.any():
        return 0.0

    m = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    per = float(sum(cv2.arcLength(c, closed=True) for c in contours))
    return per