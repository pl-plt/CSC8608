import os
import time

import numpy as np
import torch
from transformers import pipeline
import torchaudio
import soundfile as sf

def main():
    os.makedirs("TP3/outputs", exist_ok=True)

    text = (
        "Thanks for calling. I am sorry your order arrived damaged. "
        "I can offer a replacement or a refund. "
        "Please confirm your preferred option."
    )

    # Modèle TTS léger Facebook MMS (Massively Multilingual Speech)
    tts_model_id = "facebook/mms-tts-eng"

    device = 0 if torch.cuda.is_available() else -1
    tts = pipeline(
        task="text-to-speech",
        model=tts_model_id,
        device=device
    )

    t0 = time.time()
    out = tts(text)
    t1 = time.time()

    audio = np.asarray(out["audio"], dtype=np.float32)                 # numpy array
    sr = int(out["sampling_rate"])
    elapsed_s = t1 - t0
    audio_dur_s = float(audio.shape[-1] / float(sr)) if audio.ndim > 1 else float(len(audio) / float(sr))
    rtf = elapsed_s / max(audio_dur_s, 1e-9)

    # normaliser la forme vers [1, T]
    if audio.ndim == 1:                 # [T]
        audio = audio[None, :]          # [1, T]
    elif audio.ndim == 2:
        # cas [T, 1] -> [1, T]
        if audio.shape[1] == 1:
            audio = audio.T             # [1, T]
        # cas [1, T] déjà OK
        elif audio.shape[0] == 1:
            pass
        else:
            # cas multi-canaux [T, C] -> [C, T]
            audio = audio.T
    else:
        raise ValueError(f"Unexpected audio shape: {audio.shape}")

    out_wav = "TP3/outputs/tts_reply_call_01.wav"

    wav_np = audio[0] if audio.ndim == 2 else audio  # [T], float32
    sf.write(out_wav, wav_np, sr, subtype="PCM_16")

    print("tts_model_id:", tts_model_id)
    print("device:", "cuda" if device == 0 else "cpu")
    print("audio_dur_s:", round(audio_dur_s, 2))
    print("elapsed_s:", round(elapsed_s, 2))
    print("rtf:", round(rtf, 3))
    print("saved:", out_wav)

if __name__ == "__main__":
    main()
