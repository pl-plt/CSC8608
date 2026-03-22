import os
import torch
import torchaudio
import transformers
import datasets

def main():
    print("=== TP3 sanity check ===")
    print("torch:", torch.__version__)
    print("torchaudio:", torchaudio.__version__)
    print("transformers:", transformers.__version__)
    print("datasets:", datasets.__version__)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print("gpu_name:", gpu_name)
        print("gpu_mem_gb:", round(gpu_mem_gb, 2))

    # Génère un mini signal audio (1 seconde) pour valider torchaudio
    sr = 16000
    t = torch.linspace(0, 1, sr)
    wav = 0.1 * torch.sin(2 * torch.pi * 440.0 * t)  # 440 Hz
    wav = wav.unsqueeze(0)  # [1, time]

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=400,
        hop_length=160,
        n_mels=80
    )(wav)
    logmel = (mel + 1e-6).log()

    print("wav_shape:", tuple(wav.shape))
    print("logmel_shape:", tuple(logmel.shape))

if __name__ == "__main__":
    main()
