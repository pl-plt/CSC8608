"""
Script helper : génère call_01.wav à partir du texte du TP via TTS (facebook/mms-tts-eng).
À exécuter une seule fois pour créer le fichier d'entrée.
"""
import os
import numpy as np
import torch
import torchaudio
import soundfile as sf
from transformers import pipeline

TEXT = (
    "Hello, thank you for calling customer support. "
    "My name is Alex, and I will help you today. "
    "I am calling about an order that arrived damaged. "
    "The package was delivered yesterday, but the screen is cracked. "
    "I would like a refund or a replacement as soon as possible. "
    "The order number is A X 1 9 7 3 5. "
    "You can reach me at john dot smith at example dot com. "
    "Also, my phone number is 555 0199. "
    "Thank you."
)

def main():
    os.makedirs("TP3/data", exist_ok=True)
    out_path = "TP3/data/call_01.wav"

    print("Generating call_01.wav via TTS (facebook/mms-tts-eng)...")
    tts = pipeline(task="text-to-speech", model="facebook/mms-tts-eng", device=-1)

    out = tts(TEXT)
    audio = np.asarray(out["audio"], dtype=np.float32)
    sr = int(out["sampling_rate"])

    # Normaliser vers [1, T]
    if audio.ndim == 1:
        audio = audio[None, :]
    elif audio.ndim == 2 and audio.shape[1] == 1:
        audio = audio.T
    elif audio.ndim == 2 and audio.shape[0] != 1:
        audio = audio.T

    # Resample à 16000 Hz si nécessaire
    wav_np = audio[0]  # [T], already float32
    if sr != 16000:
        wav_t = torch.from_numpy(audio)
        wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
        wav_np = wav_t[0].numpy()
        sr = 16000

    sf.write(out_path, wav_np, sr, subtype="PCM_16")
    duration_s = len(wav_np) / sr
    print(f"Saved: {out_path}  |  sr={sr}  |  duration={duration_s:.2f}s  |  samples={len(wav_np)}")

if __name__ == "__main__":
    main()
