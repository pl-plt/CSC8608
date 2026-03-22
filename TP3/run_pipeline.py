import os
import json
import subprocess
from pathlib import Path

def run(cmd: str):
    print(">>", cmd)
    subprocess.run(cmd, shell=True, check=True)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    os.makedirs("TP3/outputs", exist_ok=True)

    # 1) VAD
    run("python TP3/vad_segment.py")
    vad = load_json("TP3/outputs/vad_segments_call_01.json")

    # 2) ASR
    run("python TP3/asr_whisper.py")
    asr = load_json("TP3/outputs/asr_call_01.json")

    # 3) Analytics
    run("python TP3/callcenter_analytics.py")
    summ = load_json("TP3/outputs/call_summary_call_01.json")

    # 4) TTS (optionnel) : si le script existe, on lance
    tts_path = Path("TP3/tts_reply.py")
    tts_done = False
    if tts_path.exists():
        run("python TP3/tts_reply.py")
        tts_done = True

    # Résumé final (léger)
    summary = {
        "audio_path": vad.get("audio_path"),
        "duration_s": vad.get("duration_s"),
        "num_segments": vad.get("stats", {}).get("num_segments"),
        "speech_ratio": vad.get("stats", {}).get("speech_ratio"),
        "asr_model": asr.get("model_id"),
        "asr_device": asr.get("device"),
        "asr_rtf": asr.get("rtf"),
        "intent": summ.get("intent"),
        "pii_stats": summ.get("pii_stats"),
        "tts_generated": tts_done
    }

    out_path = "TP3/outputs/pipeline_summary_call_01.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== PIPELINE SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("saved:", out_path)

if __name__ == "__main__":
    main()
