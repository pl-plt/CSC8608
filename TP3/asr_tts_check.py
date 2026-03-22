import time
import torch
import soundfile as sf
from transformers import pipeline

def main():
    wav_path = "TP3/outputs/tts_reply_call_01.wav"
    model_id = "openai/whisper-tiny"

    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        device=device
    )

    # Charger l'audio via soundfile pour éviter la dépendance torchcodec
    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    inp = {"array": data, "sampling_rate": sr}

    t0 = time.time()
    generate_kwargs = {
        "language": "english"
    }
    out = asr(inp, generate_kwargs=generate_kwargs)
    t1 = time.time()

    print("model_id:", model_id)
    print("elapsed_s:", round(t1 - t0, 2))
    print("text:", out.get("text", "").strip())

if __name__ == "__main__":
    main()
