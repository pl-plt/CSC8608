from __future__ import annotations

from typing import Dict
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

# Choisissez un modèle par défaut
DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"  # ex: "stable-diffusion-v1-5/stable-diffusion-v1-5"

SCHEDULERS: Dict[str, object] = {
    "DDIM": DDIMScheduler,
    "EulerA": EulerAncestralDiscreteScheduler,
    "DPM++": DPMSolverMultistepScheduler,
}

def get_device() -> str:
    # TODO: retourner "cuda" si disponible, sinon "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_dtype(device: str):
    # TODO: fp16 sur cuda, fp32 sinon
    return torch.float16 if device == "cuda" else torch.float32

def make_generator(seed: int, device: str) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g

def set_scheduler(pipe, scheduler_name: str):
    # TODO: remplacer le scheduler courant par celui choisi
    cls = SCHEDULERS[scheduler_name]
    pipe.scheduler = cls.from_config(pipe.scheduler.config)
    return pipe

def load_text2img(model_id: str, scheduler_name: str):
    device = get_device()
    dtype = get_dtype(device)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        # safety checker conservé par défaut
    ).to(device)

    # Aide VRAM (utile sur GPU ~11GB)
    pipe.enable_attention_slicing()

    pipe = set_scheduler(pipe, scheduler_name)
    return pipe

def to_img2img(text2img_pipe):
    # TODO: créer un pipeline img2img qui réutilise exactement les mêmes composants
    return StableDiffusionImg2ImgPipeline(**text2img_pipe.components)