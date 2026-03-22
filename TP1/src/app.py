import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from sam_utils import load_sam_predictor, predict_mask_from_box, predict_masks_from_box_and_points
from geom_utils import mask_area, mask_bbox, mask_perimeter
from viz_utils import render_overlay


DATA_DIR = Path("./data/images")
OUT_DIR = Path("./outputs/overlays")
CKPT_PATH = "./models/sam_vit_h_4b8939.pth"   # à adapter
MODEL_TYPE = "vit_h"


def load_image_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Image illisible: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


@st.cache_resource
def get_predictor():
    # Chargement unique : important pour éviter de recharger SAM à chaque interaction UI
    return load_sam_predictor(CKPT_PATH, model_type=MODEL_TYPE)


st.set_page_config(page_title="TP1 - SAM Segmentation", layout="wide")
st.title("TP1 — Segmentation interactive (SAM)")

if "points" not in st.session_state:
    st.session_state["points"] = []  # liste de tuples (x, y, label) label: 1=FG, 0=BG

if "last_pred" not in st.session_state:
    st.session_state["last_pred"] = None  # dict ou None

# 1) Liste d'images
imgs = sorted([p for p in DATA_DIR.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
if len(imgs) == 0:
    st.error("Aucune image trouvée dans TP1/data/images/")
    st.stop()

img_name = st.selectbox("Choisir une image", [p.name for p in imgs])
img_path = DATA_DIR / img_name
img = load_image_rgb(img_path)
H, W = img.shape[:2]

# 2) Affichage image
st.image(img, caption=f"{img_name} ({W}x{H})", use_container_width=True)

# 3) Sliders bbox (bornés)
st.subheader("Bounding box (pixels)")
col1, col2, col3, col4 = st.columns(4)

with col1:
    x1 = st.slider("x1", 0, W - 1, 0)
with col2:
    y1 = st.slider("y1", 0, H - 1, 0)
with col3:
    x2 = st.slider("x2", 0, W - 1, W - 1)
with col4:
    y2 = st.slider("y2", 0, H - 1, H - 1)

st.subheader("Guidage (optionnel) : points FG/BG")

# UI de saisie d'un point
c1, c2, c3 = st.columns(3)
with c1:
    px = st.slider("point x", 0, W - 1, int(W * 0.5))
with c2:
    py = st.slider("point y", 0, H - 1, int(H * 0.5))
with c3:
    ptype = st.selectbox("type", ["FG (objet)", "BG (fond)"])

# Boutons
if st.button("Ajouter point"):
    label = 1 if ptype.startswith("FG") else 0
    st.session_state["points"].append((int(px), int(py), int(label)))

if st.button("Réinitialiser points"):
    st.session_state["points"] = []

# Affichage de l'état courant (utile pour debug)
st.write({
    "n_points": len(st.session_state["points"]),
    "points": st.session_state["points"],
})

# Normaliser bbox (x1<x2, y1<y2)
x_min, x_max = (x1, x2) if x1 < x2 else (x2, x1)
y_min, y_max = (y1, y2) if y1 < y2 else (y2, y1)
box = np.array([x_min, y_min, x_max, y_max], dtype=np.int32)

# 4) Segmentation
do_segment = st.button("Segmenter")
if do_segment:
    predictor = get_predictor()

    pts = st.session_state["points"]
    if len(pts) > 0:
        point_coords = np.array([(x, y) for (x, y, _) in pts], dtype=np.float32)
        point_labels = np.array([lab for (_, _, lab) in pts], dtype=np.int64)
    else:
        point_coords, point_labels = None, None

    t0 = time.time()
    masks, scores = predict_masks_from_box_and_points(
        predictor=predictor,
        image_rgb=img,
        box_xyxy=box,
        point_coords=point_coords,
        point_labels=point_labels,
        multimask=True,
    )
    dt = (time.time() - t0) * 1000.0

    st.session_state["last_pred"] = {
        "img_name": img_name,
        "box": box.copy(),
        "points": list(pts),
        "masks": masks,
        "scores": scores,
        "time_ms": float(dt),
    }

lp = st.session_state["last_pred"]
if lp is not None and lp["img_name"] == img_name:
    masks = lp["masks"]
    scores = lp["scores"]

    st.subheader("Choix du masque candidat (multimask)")
    st.write({"scores": [float(s) for s in scores.tolist()], "time_ms": lp.get("time_ms")})

    default_idx = int(np.argmax(scores))
    idx = st.selectbox("index du masque", list(range(len(scores))), index=default_idx)

    mask = masks[int(idx)].astype(bool)
    overlay = render_overlay(img, mask, box, alpha=0.5)

    # métriques sur le masque choisi
    m_area = mask_area(mask)
    m_bbox = mask_bbox(mask)
    m_per = mask_perimeter(mask)

    st.image(overlay, caption=f"mask_idx={idx} | score={float(scores[idx]):.3f}", use_container_width=True)
    st.write({
        "mask_idx": int(idx),
        "score": float(scores[idx]),
        "area_px": int(m_area),
        "mask_bbox": m_bbox,
        "perimeter": float(m_per),
    })

    if st.button("Sauvegarder overlay (masque sélectionné)"):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"overlay_{img_path.stem}_m{int(idx)}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        st.success(f"Sauvegardé: {out_path}")


# Ajouter une prévisualisation bbox-only

def draw_preview(image_rgb: np.ndarray, box_xyxy: np.ndarray, points):
    preview = image_rgb.copy()
    bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)

    x1, y1, x2, y2 = [int(v) for v in box_xyxy.tolist()]
    cv2.rectangle(bgr, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

    for (x, y, lab) in points:
        # FG = vert, BG = rouge
        color = (0, 255, 0) if lab == 1 else (0, 0, 255)
        cv2.circle(bgr, (int(x), int(y)), radius=6, color=color, thickness=-1)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

preview = draw_preview(img, box, st.session_state["points"])
st.image(preview, caption="Prévisualisation : bbox + points (avant segmentation)", use_container_width=True)

# Bonus simple : avertissement bbox très petite
if (x_max - x_min) < 10 or (y_max - y_min) < 10:
    st.warning("BBox très petite : essayez une bbox plus large.")