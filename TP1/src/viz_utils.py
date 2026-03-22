import numpy as np
import cv2


def render_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    box_xyxy: np.ndarray,
    alpha: float = 0.5,
):
    """
    Retourne une image RGB uint8 avec :
    - bbox dessinée
    - masque superposé (alpha blending)
    """
    out = image_rgb.copy()

    # 1) Dessiner bbox (en RGB, mais OpenCV dessine en BGR : on dessine sur une version BGR)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = [int(v) for v in box_xyxy.tolist()]
    cv2.rectangle(bgr, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 2) Alpha blending du masque (utiliser une couleur fixe, par ex. rouge)
    if mask is not None and mask.any():
        overlay = out.copy()
        overlay[mask] = (255, 0, 0)  # RGB rouge
        out = (alpha * overlay + (1.0 - alpha) * out).astype(np.uint8)

    return out