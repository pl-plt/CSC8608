# À placer au début de TP2/app.py

import streamlit as st
from PIL import Image

from pipeline_utils import (
    DEFAULT_MODEL_ID,
    load_text2img,
    to_img2img,
    get_device,
    make_generator,
)

st.set_page_config(page_title="TP2 - Diffusion e-commerce", layout="wide")

@st.cache_resource
def get_text2img_pipe(model_id: str, scheduler_name: str):
    # TODO: charger le pipeline text2img
    return load_text2img(model_id, scheduler_name)

st.title("TP2 — Diffusion mini-product (e-commerce)")

mode = st.sidebar.selectbox("Mode", ["Text2Img", "Img2Img"])

model_id = st.sidebar.text_input("Model ID", value=DEFAULT_MODEL_ID)
scheduler_name = st.sidebar.selectbox("Scheduler", ["EulerA", "DDIM", "DPM++"])

seed = st.sidebar.number_input("Seed", min_value=0, max_value=10_000_000, value=42, step=1)
steps = st.sidebar.slider("Steps", 5, 60, 30)
guidance = st.sidebar.slider("Guidance (CFG)", 1.0, 15.0, 7.5, 0.5)

prompt = st.text_area("Prompt", value="ultra-realistic product photo of a backpack on a white background, studio lighting, soft shadow, very sharp")
negative_prompt = st.text_area("Negative prompt", value="text, watermark, logo, low quality, blurry, deformed")

init_image = None
strength = None
if mode == "Img2Img":
    up = st.file_uploader("Input image (img2img)", type=["png", "jpg", "jpeg"])
    strength = st.slider("Strength", 0.0, 0.95, 0.60, 0.05)
    if up is not None:
        init_image = Image.open(up).convert("RGB")
        st.image(init_image, caption="Input image", use_container_width=True)

run = st.button("Generate", type="primary")


if run:
    if mode == "Img2Img" and init_image is None:
        st.error("Please upload an input image for Img2Img.")
        st.stop()

    pipe_t2i = get_text2img_pipe(model_id, scheduler_name)
    device = get_device()
    g = make_generator(int(seed), device)

    if mode == "Text2Img":
        out = pipe_t2i(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=512,
            width=512,
            generator=g,
        )
        img = out.images[0]
        config = {
            "mode": "Text2Img",
            "model_id": model_id,
            "scheduler": scheduler_name,
            "seed": int(seed),
            "steps": int(steps),
            "guidance": float(guidance),
            "height": 512,
            "width": 512,
        }
    else:
        pipe_i2i = to_img2img(pipe_t2i)
        out = pipe_i2i(
            prompt=prompt,
            image=init_image,
            strength=strength,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=g,
        )
        img = out.images[0]
        config = {
            "mode": "Img2Img",
            "model_id": model_id,
            "scheduler": scheduler_name,
            "seed": int(seed),
            "steps": int(steps),
            "guidance": float(guidance),
            "strength": float(strength),
            "height": 512,
            "width": 512,
        }

    st.image(img, caption=f"{config['mode']} | {config['scheduler']} | seed={config['seed']}", use_container_width=True)
    st.subheader("Config (for reproducibility)")
    st.json(config)