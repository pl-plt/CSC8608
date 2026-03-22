from __future__ import annotations
import os
import torch
from diffusers import StableDiffusionPipeline

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[smoke] device={device} dtype={dtype}")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        # safety checker conservé par défaut
    ).to(device)

    # Aide VRAM sur GPU ~11GB
    pipe.enable_attention_slicing()

    prompt = "ultra-realistic product photo of a watch on a white background, studio lighting, soft shadow, very sharp"
    negative = "text, watermark, logo, low quality, blurry, deformed"

    g = torch.Generator(device=device).manual_seed(42)

    out = pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=25,
        guidance_scale=7.5,
        height=512,
        width=512,
        generator=g,
    )

    img = out.images[0]
    os.makedirs("outputs", exist_ok=True)
    path = os.path.join("outputs", "smoke.png")
    img.save(path)
    print(f"[smoke] saved: {path}")

if __name__ == "__main__":
    main()