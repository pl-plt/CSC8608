import os
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor


def get_device() -> str:
    """
    Retourne 'cuda' si dispo, sinon 'cpu'.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.inference_mode()
def load_sam_predictor(checkpoint_path: str, model_type: str = "vit_h") -> SamPredictor:
    """
    Charge SAM et retourne un SamPredictor prêt pour l'inférence.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint_path}")

    device = get_device()

    print(f'Device utilisé pour SAM: {device}')

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    sam.eval()

    predictor = SamPredictor(sam)
    return predictor

@torch.inference_mode()
def predict_mask_from_box(
    predictor: SamPredictor,
    image_rgb: np.ndarray,
    box_xyxy: np.ndarray,
    multimask: bool = True,
):
    """
    image_rgb: (H,W,3) uint8 en RGB
    box_xyxy: (4,) -> [x1,y1,x2,y2] en pixels
    retourne: (mask_bool(H,W), score_float)
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb doit être (H,W,3)")
    if box_xyxy.shape != (4,):
        raise ValueError("box_xyxy doit être de shape (4,)")

    predictor.set_image(image_rgb)

    # SAM attend une box float32 de shape (1,4)
    box = box_xyxy.astype(np.float32)[None, :]

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box,
        multimask_output=multimask,
    )

    best_idx = int(np.argmax(scores))
    mask = masks[best_idx].astype(bool)
    score = float(scores[best_idx])
    return mask, score

@torch.inference_mode()
def predict_masks_from_box_and_points(
    predictor: SamPredictor,
    image_rgb: np.ndarray,
    box_xyxy: np.ndarray,
    point_coords: np.ndarray | None,
    point_labels: np.ndarray | None,
    multimask: bool = True,
):
    """
    Retourne (masks, scores) où :
      - masks : (K, H, W) bool
      - scores : (K,) float
    """
    predictor.set_image(image_rgb)

    box = box_xyxy.astype(np.float32)[None, :]

    if point_coords is not None:
        pc = point_coords.astype(np.float32)
        pl = point_labels.astype(np.int64)
    else:
        pc, pl = None, None

    masks, scores, _ = predictor.predict(
        point_coords=pc,
        point_labels=pl,
        box=box,
        multimask_output=multimask,
    )

    return masks.astype(bool), scores.astype(float)