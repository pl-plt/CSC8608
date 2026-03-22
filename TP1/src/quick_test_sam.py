# TP1/src/quick_test_sam.py (à créer) ou exécuter dans un python interactif
import numpy as np
import cv2
from pathlib import Path
from sam_utils import load_sam_predictor, predict_mask_from_box

img_path = next(Path("TP1/data/images").glob("*.jpg"))  # adaptez si besoin
bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

ckpt = "TP1/models/sam_vit_h_4b8939.pth"  # nom du fichier .pth
pred = load_sam_predictor(ckpt, model_type="vit_h") # Attention, ou vit_l ou vit_b

# bbox "à la main" (exemple) : adaptez à votre image
box = np.array([50, 50, 250, 250], dtype=np.int32)

mask, score = predict_mask_from_box(pred, rgb, box, multimask=True)
print("img", rgb.shape, "mask", mask.shape, "score", score, "mask_sum", mask.sum())