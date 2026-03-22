# TP1/src/quick_test_overlay.py (à créer) ou exécuter dans un python interactif
import numpy as np
import cv2
from pathlib import Path

from sam_utils import load_sam_predictor, predict_mask_from_box
from geom_utils import mask_area, mask_bbox, mask_perimeter
from viz_utils import render_overlay

img_paths = Path("TP1/data/images").glob("*lacentrale.png")  # adaptez si besoin
img_paths = list(img_paths)
print("Number of images found:", len(img_paths))
for img_path in img_paths:
    print("Processing:", img_path)
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    ckpt = "TP1/models/sam_vit_h_4b8939.pth"
    pred = load_sam_predictor(ckpt, model_type="vit_h")

    box = np.array([50, 50, 250*4, 250*3], dtype=np.int32)  # adaptez
    mask, score = predict_mask_from_box(pred, rgb, box, multimask=True)

    m_area = mask_area(mask)
    m_bbox = mask_bbox(mask)
    m_per = mask_perimeter(mask)

    overlay = render_overlay(rgb, mask, box, alpha=0.5)

    out_dir = Path("TP1/outputs/overlays")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"overlay_{img_path.stem}.png"

    # Sauvegarde via OpenCV (BGR)
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print("score", score, "area", m_area, "bbox", m_bbox, "perimeter", m_per)
    print("saved:", out_path)