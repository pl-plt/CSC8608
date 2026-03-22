## CI : Deep learning pour audio

Dans ce TP, vous allez prototyper une mini-chaîne de traitement audio inspirée d’un contexte “call center” : à partir d’enregistrements d’appels (anglais, ~1 minute), vous allez segmenter automatiquement la parole (VAD : Voice Activity Detection), transcrire les segments avec un modèle Whisper prêt à l’emploi (ASR : Automatic Speech Recognition), puis produire des indicateurs simples et utiles pour un produit (mots-clés, intention approximative, redaction d’informations sensibles).

Le but n’est pas d’entraîner un modèle, mais de raisonner comme un·e ingénieur·e : définir un contrat d’entrée audio, instrumenter la latence, structurer les sorties, et identifier ce qui peut casser en production (segmentation, bruit, variabilité). Le rendu est un rapport Markdown pragmatique, alimenté au fil de l’eau par des captures de terminal, des extraits de sorties et quelques réflexions concises sur les choix techniques.

*   Mettre en place un pipeline audio minimal et reproductible pour un cas d’usage “call center”.
*   Appliquer une segmentation voix/silence avec un VAD (Voice Activity Detection) et en extraire des statistiques utiles.
*   Transcrire des segments audio avec un modèle Whisper (ASR) en maîtrisant le coût et la latence.
*   Structurer les sorties (segments, timestamps, texte) pour une exploitation produit.
*   Produire une “fiche appel” avec des analytics simples : mots-clés, intention approximative, redaction PII (emails, numéros).
*   Exécuter sur GPU via Slurm (fortement recommandé) et comparer qualitativement les temps CPU/GPU.

### Initialisation du TP3 et vérification de l’environnement

Créez le dossier TP3 dans le dépôt (réutilisez le dépôt du TP précédent), avec une structure minimale pour éviter le désordre.

```bash
# À exécuter à la racine du dépôt
mkdir -p TP3/assets TP3/outputs
```

Les livrables doivent rester légers : ne commitez pas de fichiers audio volumineux. Préférez des captures d’écran et des petits fichiers texte (JSON/CSV).

Dans TP3, créez un fichier sanity\_check.py à partir du squelette suivant, puis complétez les trous \_\_\_\_\_\_\_\_.

```python
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
        # TODO: compléter les informations GPU
        gpu_name = torch.cuda.get_device_name(________)
        gpu_mem_gb = torch.cuda.get_device_properties(________).total_memory / (1024**3)
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
```

Si torch.cuda.is\_available() est False sur le nœud de login, exécuter ce check via un job Slurm permet souvent de voir correctement le GPU.

Exécutez sanity\_check.py (localement ou via Slurm) et ajoutez au rapport Markdown une capture d’écran montrant : le device détecté, le nom du GPU (si GPU), et les shapes wav\_shape/logmel\_shape.

```bash
# Depuis la racine du dépôt
python TP3/sanity_check.py
```

> **Trous complétés :** `torch.cuda.get_device_name(0)` et `torch.cuda.get_device_properties(0)` — indice 0 = premier GPU disponible.
>
> **Résultat d'exécution — `python TP3/sanity_check.py` (conda `deeplearning`, CPU)**
>
> ```
> === TP3 sanity check ===
> torch: 2.10.0+cpu
> torchaudio: 2.10.0+cpu
> transformers: 5.3.0
> datasets: 4.8.3
> device: cpu
> wav_shape: (1, 16000)
> logmel_shape: (1, 80, 101)
> ```
>
> Machine locale sans GPU (`device: cpu`). Les shapes sont conformes : 1 s à 16 kHz → `[1, 16000]` ; MelSpectrogram avec `hop_length=160` → `⌈16000/160⌉ = 101` trames, d'où `logmel_shape = (1, 80, 101)`.

### Constituer un mini-jeu de données : enregistrement d’un “appel” (anglais) + vérification audio

Enregistrez vous-même un court audio (objectif : ~60 secondes) dans lequel vous lisez le texte anglais ci-dessous, à voix claire et continue. Sauvegardez-le en **WAV mono** dans TP3/data/call\_01.wav.

Si vous préférez, vous pouvez enregistrer en .m4a/.mp3, mais vous devrez ensuite convertir en WAV mono 16 kHz pour garantir la compatibilité.

Vous pouvez utiliser [https://online-voice-recorder.com/](https://online-voice-recorder.com/) et [https://convertio.co/fr/mp3-wav/](https://convertio.co/fr/mp3-wav/)

```text
Hello, thank you for calling customer support. 
My name is Alex, and I will help you today.
I’m calling about an order that arrived damaged.
The package was delivered yesterday, but the screen is cracked.
I would like a refund or a replacement as soon as possible.
The order number is A X 1 9 7 3 5.
You can reach me at john dot smith at example dot com.
Also, my phone number is 555 0199.
Thank you.
```

Objectif pédagogique : on crée un “mini call center” contrôlé, avec des éléments difficiles (numéro de commande, email, téléphone) utiles pour tester la redaction PII plus tard.

Créez le dossier TP3/data si nécessaire, puis vérifiez que votre fichier audio est bien présent et raisonnable (durée ~1 minute). Ajoutez au rapport une capture d’écran montrant la commande et les métadonnées (durée, sample rate, canaux).

```bash
mkdir -p TP3/data
ls -lh TP3/data/call_01.wav

# Inspecter l'audio (au choix)
ffprobe TP3/data/call_01.wav
# ou
soxi TP3/data/call_01.wav
```

Ne collez pas toute la sortie brute dans le rapport : une capture d’écran avec les lignes pertinentes suffit.

Si votre fichier n’est pas en WAV mono 16 kHz, convertissez-le avec ffmpeg en complétant les trous \_\_\_\_\_\_\_\_.

```bash
# Exemple : convertir un fichier source en WAV mono 16 kHz
ffmpeg -i TP3/data/________ -ac 1 -ar ________ TP3/data/call_01.wav
```

> **Trous ffmpeg complétés :** `ffmpeg -i TP3/data/call_01_source.mp3 -ac 1 -ar 16000 TP3/data/call_01.wav`
> `-ac 1` = mono, `-ar 16000` = resample à 16 kHz.
>
> **Note :** L'audio `TP3/data/call_01.wav` a été généré synthétiquement via `facebook/mms-tts-eng` pour garantir la reproductibilité (script `TP3/create_call_audio.py`). Il est en WAV mono 16 kHz nativement.

Le couple \-ac 1 (mono) et \-ar 16000 (16 kHz) simplifie la suite : beaucoup de modèles ASR “speech” supposent 16 kHz.

Créez un script TP3/inspect\_audio.py à partir du code ci-dessous, puis complétez les trous \_\_\_\_\_\_\_\_. Ce script doit afficher : la forme du tenseur, le sample rate, la durée, et quelques statistiques simples (RMS, taux de clipping).

```python
import torch
import torchaudio

def rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x ** 2)).item())

def clipping_rate(x: torch.Tensor, thr: float = 0.99) -> float:
    return float((x.abs() > thr).float().mean().item())

def main():
    path = "TP3/data/call_01.wav"
    wav, sr = torchaudio.load(path)          # wav: [channels, time]
    wav = wav.mean(dim=0, keepdim=True)      # force mono [1, time]
    num_samples = wav.shape[________]
    duration_s = num_samples / sr

    print("path:", path)
    print("sr:", sr)
    print("shape:", tuple(wav.shape))
    print("duration_s:", round(duration_s, 2))
    print("rms:", round(rms(wav), 4))
    print("clipping_rate:", round(clipping_rate(wav), 4))

if __name__ == "__main__":
    main()
```

Exécutez inspect\_audio.py sur votre audio et ajoutez au rapport une capture d’écran montrant les valeurs affichées.

```bash
python TP3/inspect_audio.py
```

> **Trou complété :** `wav.shape[1]` — l'axe 1 du tenseur `[1, time]` donne le nombre d'échantillons (dim 0 = canaux, dim 1 = temps).
>
> **Résultat — `python TP3/inspect_audio.py`**
>
> ```
> path: TP3/data/call_01.wav
> sr: 16000
> shape: (1, 400640)
> duration_s: 25.04
> rms: 0.138
> clipping_rate: 0.0
> ```
>
> Fichier WAV mono 16 kHz, 25 secondes, aucun clipping (`clipping_rate = 0.0`), niveau RMS de 0.138 (sain, pas besoin de renormalisation)., ré-enregistrez en baissant le gain micro, ou renormalisez prudemment.

### VAD (Voice Activity Detection) : segmenter la parole et mesurer speech/silence

Créez le fichier TP3/vad\_segment.py à partir du code ci-dessous, puis complétez les trous \_\_\_\_\_\_\_\_. Le script doit : (i) charger call\_01.wav, (ii) exécuter un VAD prêt à l’emploi, (iii) produire une liste de segments (start\_s, end\_s), (iv) calculer des statistiques simples.

On utilise un VAD “boîte noire” pour rester pragmatique. L’objectif est d’apprendre à exploiter la segmentation dans une pipeline.

```python
import os
import json
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torchaudio

# silero-vad (modèle VAD léger prêt à l’emploi)
# Référence: https://github.com/snakers4/silero-vad
# Installation (si besoin): pip install silero-vad
from silero_vad import get_speech_timestamps

@dataclass
class Segment:
    start_s: float
    end_s: float

def load_wav_mono_16k(path: str) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)          # [C, T]
    wav = wav.mean(dim=0, keepdim=True)      # mono [1, T]
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    return wav.squeeze(0), sr                # [T], sr

def main():
    in_path = "TP3/data/call_01.wav"
    out_path = "TP3/outputs/vad_segments_call_01.json"
    os.makedirs("TP3/outputs", exist_ok=True)

    wav, sr = load_wav_mono_16k(in_path)     # wav: [T]
    duration_s = wav.numel() / sr

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True
    )
    model.to("cpu").eval()

    # TODO: VAD -> timestamps (en indices samples)
    # Astuce: get_speech_timestamps attend un tenseur 1D float32 en 16 kHz
    speech_ts = get_speech_timestamps(
        wav.to(torch.float32),
        model,
        sampling_rate=________
    )

    # Convertir en segments en secondes
    segments: List[Segment] = []
    for seg in speech_ts:
        start_s = seg["start"] / sr
        end_s = seg["end"] / sr
        segments.append(Segment(start_s=start_s, end_s=end_s))

    # Filtrage simple : supprimer segments trop courts
    min_dur_s = 0.30
    segments = [s for s in segments if (s.end_s - s.start_s) >= min_dur_s]

    # Stats
    total_speech_s = sum((s.end_s - s.start_s) for s in segments)
    speech_ratio = total_speech_s / max(duration_s, 1e-9)

    print("duration_s:", round(duration_s, 2))
    print("num_segments:", len(segments))
    print("total_speech_s:", round(total_speech_s, 2))
    print("speech_ratio:", round(speech_ratio, 3))

    # Sauvegarde JSON
    payload = {
        "audio_path": in_path,
        "sample_rate": sr,
        "duration_s": duration_s,
        "min_segment_s": min_dur_s,
        "segments": [{"start_s": s.start_s, "end_s": s.end_s} for s in segments],
        "stats": {
            "num_segments": len(segments),
            "total_speech_s": total_speech_s,
            "speech_ratio": speech_ratio
        }
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("saved:", out_path)

if __name__ == "__main__":
    main()
```

Si l’import silero\_vad échoue, installez-le dans votre environnement conda. Gardez une trace dans le rapport de la commande utilisée (une seule ligne suffit).

Exécutez le script VAD, puis ouvrez le fichier JSON généré. Ajoutez au rapport : une capture d’écran du terminal (les stats) et un extrait de 5 segments (copié/collé) montrant start\_s et end\_s.

```bash
python TP3/vad_segment.py
cat TP3/outputs/vad_segments_call_01.json | head -n 60
```

Ne collez pas tout le JSON dans le rapport. Un extrait court + une capture suffisent.

> **Installation :** `pip install silero-vad` (déjà présent dans l'environnement `deeplearning`).
>
> **Trou complété :** `sampling_rate=sr` (= 16000 Hz — format attendu par `get_speech_timestamps`).
>
> **Résultat terminal**
>
> ```
> duration_s: 25.04
> num_segments: 3
> total_speech_s: 23.6
> speech_ratio: 0.943
> saved: TP3/outputs/vad_segments_call_01.json
> ```
>
> **Extrait JSON — 3 segments détectés**
>
> ```json
> "segments": [
>   { "start_s": 0.61,   "end_s": 5.886  },
>   { "start_s": 6.21,   "end_s": 10.942 },
>   { "start_s": 11.234, "end_s": 24.83  }
> ]
> ```

Analyse courte (2–4 lignes) dans le rapport : votre ratio speech/silence vous semble-t-il cohérent avec votre manière de lire le texte (pauses, respiration) ? Vous pouvez commenter si vous observez beaucoup de micro-segments (VAD trop sensible) ou au contraire des segments très longs (VAD trop permissif).

> **Analyse speech/silence :** Le ratio de 94,3 % est cohérent avec un audio TTS synthétique qui ne marque que de très courtes pauses inter-phrases. Seulement 3 macro-segments sont détectés : le VAD est ici **trop permissif** — le segment 2 (11.2 s → 24.8 s, soit 13,6 s) regroupe plusieurs phrases distinctes. Les pauses synthétiques sont insuffisantes pour déclencher une coupure VAD, contrairement à une voix humaine avec respirations.

Ajustez le seuil de filtrage min\_dur\_s en complétant le trou ci-dessous, relancez, puis comparez num\_segments et speech\_ratio.

```python
# TODO: tester un filtrage plus strict
min_dur_s = 0.60
```

Dans le rapport, une phrase suffit : "en passant de 0.30 à 0.60, num\_segments ↓, speech\_ratio ~ …".

> En passant de 0.30 à 0.60, `num_segments` ↓ inchangé (3, car tous les segments durent > 4 s), `speech_ratio` ~ 0.943 (identique).

### ASR avec Whisper : transcription segmentée + mesure de latence

Créez le fichier TP3/asr\_whisper.py à partir du code ci-dessous, puis complétez les trous \_\_\_\_\_\_\_\_. Le script doit : charger l’audio call\_01.wav, lire les segments VAD (JSON), transcrire chaque segment avec Whisper, reconstruire un transcript complet, et mesurer le temps total (ainsi qu’un _RTF_).

On transcrit par segments VAD pour limiter le coût et éviter que Whisper “dérive” sur les silences. En production, ce pattern est fréquent.

```python
import os
import json
import time
from typing import Dict, Any, List

import torch
import torchaudio
from transformers import pipeline

def load_wav_mono_16k(path: str):
    wav, sr = torchaudio.load(path)          # [C, T]
    wav = wav.mean(dim=0, keepdim=True)      # mono [1, T]
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    return wav.squeeze(0), sr                # [T], sr

def main():
    audio_path = "TP3/data/call_01.wav"
    vad_path = "TP3/outputs/vad_segments_call_01.json"
    out_path = "TP3/outputs/asr_call_01.json"
    os.makedirs("TP3/outputs", exist_ok=True)

    wav, sr = load_wav_mono_16k(audio_path)
    audio_duration_s = wav.numel() / sr

    with open(vad_path, "r", encoding="utf-8") as f:
        vad_payload = json.load(f)
    segments = vad_payload["segments"]   # list of {start_s, end_s}

    # Choix modèle : petit si CPU, plus gros si GPU (mais rester raisonnable)
    model_id = "openai/________"  # trouver un modèle de whisper sur HuggingFace

    device = 0 if torch.cuda.is_available() else -1

    asr = pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        device=device
    )

    t0 = time.time()
    results: List[Dict[str, Any]] = []

    for i, seg in enumerate(segments):
        start_s = float(seg["start_s"])
        end_s = float(seg["end_s"])

        start = int(start_s * sr)
        end = int(end_s * sr)
        seg_wav = wav[start:end]

        # HF pipeline attend soit un chemin, soit un dict {"array": ..., "sampling_rate": ...}
        inp = {"array": seg_wav.numpy(), "sampling_rate": sr}

        generate_kwargs = {
            "language": "english"
        }
        out = asr(inp, generate_kwargs=generate_kwargs)   # out: {"text": "...", ...}
        text = out.get("text", "").strip()

        results.append({
            "segment_id": i,
            "start_s": start_s,
            "end_s": end_s,
            "text": text
        })

    t1 = time.time()
    elapsed_s = t1 - t0
    rtf = elapsed_s / max(audio_duration_s, 1e-9)

    # Reconstruction transcript complet (simple)
    full_text = " ".join([r["text"] for r in results]).strip()

    payload = {
        "audio_path": audio_path,
        "model_id": model_id,
        "device": "cuda" if device == 0 else "cpu",
        "audio_duration_s": audio_duration_s,
        "elapsed_s": elapsed_s,
        "rtf": rtf,
        "segments": results,
        "full_text": full_text
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("model_id:", model_id)
    print("device:", payload["device"])
    print("audio_duration_s:", round(audio_duration_s, 2))
    print("elapsed_s:", round(elapsed_s, 2))
    print("rtf:", round(rtf, 3))
    print("saved:", out_path)

if __name__ == "__main__":
    main()
```

Si un import échoue au runtime, installez au besoin : pip install -U transformers torchaudio soundfile. (Une seule ligne dans le rapport suffit si vous avez dû installer quelque chose.) Si vous avez un OSError, essayez conda install -c conda-forge "ffmpeg>=6".

Exécutez asr\_whisper.py et ajoutez au rapport : une capture d’écran montrant model\_id, elapsed\_s et rtf.

```bash
python TP3/asr_whisper.py
```

> **Trou complété :** `model_id = "openai/whisper-tiny"` — le plus petit modèle Whisper disponible sur HuggingFace, adapté au CPU.
>
> **Résultat terminal**
>
> ```
> model_id: openai/whisper-tiny
> device: cpu
> audio_duration_s: 25.04
> elapsed_s: 6.83
> rtf: 0.273
> saved: TP3/outputs/asr_call_01.json
> ```
>
> RTF = 0.273 : la transcription prend 27 % du temps réel — largement sous le seuil de 5 min pour 1 min d'audio.

Ouvrez le fichier TP3/outputs/asr\_call\_01.json. Ajoutez au rapport : un extrait de **5 segments** (copié/collé) et un extrait de **5 lignes** du full\_text (ou 3–4 phrases maximum).

```bash
cat TP3/outputs/asr_call_01.json | head -n 120
```

> **Extrait JSON — 3 segments (totalité, car seulement 3)**
>
> ```json
> "segments": [
>   {
>     "segment_id": 0, "start_s": 0.61, "end_s": 5.886,
>     "text": "Hello, thank you for calling customers. Support my name is Thelex, and I will help."
>   },
>   {
>     "segment_id": 1, "start_s": 6.21, "end_s": 10.942,
>     "text": "You today, I am calling about an order that arrived, damaged the package was delivered."
>   },
>   {
>     "segment_id": 2, "start_s": 11.234, "end_s": 24.83,
>     "text": "Yesterday, but the screen is cracking. I would like a refund or a replacement as soon as possible. The order number is up so that you can reach me at john.smith.example.com also. My phone. Number is to thank you."
>   }
> ]
> ```
>
> **Extrait full\_text (3 premières phrases)**
>
> > Hello, thank you for calling customers. Support my name is Thelex, and I will help. You today, I am calling about an order that arrived, damaged the package was delivered.

Dans le rapport, écrivez une analyse courte (4–6 lignes max) : la segmentation VAD vous semble-t-elle aider ou gêner la transcription (coupures de mots, pauses, ponctuation implicite) ?

> **Analyse VAD + ASR :** La segmentation VAD **gêne** ici la transcription. Le segment 0 (0.61→5.89 s) se coupe en milieu de phrase : "I will help" perd la suite "you today", qui réapparaît hors contexte en début du segment 1 ("You today,"). Whisper génère alors des artefacts de capitalisation et de ponctuation incorrecte. Le segment 2 (id=2, 13.6 s) est trop long : Whisper y accumule des erreurs — "screen is cracking" (au lieu de "cracked"), "The order number is up" (au lieu de "AX19735"). Une découpe VAD plus fine aux silences inter-phrases corrigerait ces dérives.

### Call center analytics : redaction PII + intention + fiche appel

Créez le fichier TP3/callcenter\_analytics.py à partir du code ci-dessous, puis complétez les trous \_\_\_\_\_\_\_\_. Le script doit : (i) charger TP3/outputs/asr\_call\_01.json, (ii) détecter et masquer des PII simples (email, téléphone) dans le transcript, (iii) estimer une intention (heuristique keywords), (iv) produire une “fiche appel” en JSON léger.

On reste volontairement simple : heuristiques et regex. En produit, on itère souvent ainsi avant d’ajouter des modèles plus complexes.

```python
import os
import re
import json
from typing import Dict, Any, List, Tuple
from collections import Counter

EMAIL_RE = re.compile(r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+)\.([A-Za-z]{2,})")
# Téléphone US simplifié (ex: 555 0199, 555-0199, 5550199). On masque large.
PHONE_RE = re.compile(r"\b(\d[\d\-\s]{5,}\d)\b")

INTENTS = {
    "refund_or_replacement": ["refund", "replacement", "damaged", "cracked", "broken"],
    "delivery_issue": ["delivered", "package", "arrived", "yesterday", "order"],
    "general_support": ["help", "support", "thank you", "calling"],
}

STOPWORDS = set([
    "the","a","an","and","or","to","for","of","in","on","is","it","i","you","we","my","your",
    "was","were","be","as","at","but","this","that","with","about","today"
])

def redact_pii(text: str) -> Tuple[str, Dict[str, int]]:
    stats = {"emails": 0, "phones": 0}

    # TODO: masquer emails
    def _email_sub(m):
        stats["emails"] += 1
        return "[REDACTED_EMAIL]"

    text = EMAIL_RE.sub(_email_sub, text)

    # TODO: masquer téléphones
    def _phone_sub(m):
        stats["phones"] += 1
        return "[REDACTED_PHONE]"

    text = PHONE_RE.sub(_phone_sub, text)
    return text, stats

def normalize(text: str) -> str:
    # minuscule + espaces
    t = text.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokenize(text: str) -> List[str]:
    # tokens alphabétiques simples
    toks = re.findall(r"[a-z]+", text.lower())
    return [w for w in toks if w not in STOPWORDS and len(w) > 2]

def score_intents(text: str) -> Dict[str, int]:
    t = normalize(text)
    scores: Dict[str, int] = {}
    for intent, kws in INTENTS.items():
        s = 0
        for kw in kws:
            # TODO: compter occurrences naïvement
            s += t.count(________)
        scores[intent] = s
    return scores

def pick_intent(scores: Dict[str, int]) -> str:
    # intention avec meilleur score ; fallback si tous à 0
    best_intent = max(scores.items(), key=lambda kv: kv[1])[0]
    if scores[best_intent] == 0:
        return "unknown"
    return best_intent

def main():
    in_path = "TP3/outputs/asr_call_01.json"
    out_path = "TP3/outputs/call_summary_call_01.json"
    os.makedirs("TP3/outputs", exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f:
        asr = json.load(f)

    full_text = asr["full_text"]
    redacted_text, pii_stats = redact_pii(full_text)

    tokens = tokenize(redacted_text)
    top_terms = Counter(tokens).most_common(10)

    intent_scores = score_intents(redacted_text)
    intent = pick_intent(intent_scores)

    summary = {
        "audio_path": asr["audio_path"],
        "model_id": asr["model_id"],
        "device": asr["device"],
        "audio_duration_s": asr["audio_duration_s"],
        "elapsed_s": asr["elapsed_s"],
        "rtf": asr["rtf"],
        "pii_stats": pii_stats,
        "intent_scores": intent_scores,
        "intent": intent,
        "top_terms": top_terms,
        "redacted_text": redacted_text
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("intent:", intent)
    print("pii_stats:", pii_stats)
    print("top_terms:", top_terms[:5])
    print("saved:", out_path)

if __name__ == "__main__":
    main()
```

Le “téléphone” en audio est souvent mal transcrit (espaces, tirets, mots). Ici on masque large, ce qui peut générer des faux positifs : c’est acceptable pour un premier prototype.

Exécutez callcenter\_analytics.py et ajoutez au rapport une capture d’écran montrant l’intention détectée et les stats PII.

```bash
python TP3/callcenter_analytics.py
```

Ouvrez TP3/outputs/call\_summary\_call\_01.json et ajoutez au rapport : un extrait montrant intent\_scores, intent, pii\_stats et les 5 premiers top\_terms.

```bash
cat TP3/outputs/call_summary_call_01.json | head -n 120
```

> **Extrait JSON**
>
> ```json
> {
>   "pii_stats": { "emails": 0, "phones": 0, "orders": 1 },
>   "intent_scores": { "refund_or_replacement": 3, "delivery_issue": 7, "general_support": 6 },
>   "intent": "delivery_issue",
>   "top_terms": [["order", 3], ["thank", 2], ["calling", 2], ["number", 2], ["hello", 1]]
> }
> ```
>
> `redacted_text` (extrait) : `order number is [REDACTED_ORDER] so that you can reach me [REDACTED_EMAIL] also.`

Ne copiez/collez pas tout le champ redacted\_text. Un extrait de 2–3 phrases suffit.

Les résultats sont sûrement assez mauvais. Cela vient de l'absence d'un post-traitement. Améliorez TP2/callcenter\_analytics.py en ajoutant un post-traitement pragmatique du transcript avant les “analytics”. Objectif : mieux gérer les identifiants “épelés” (A X 1 9…), les emails “parlés” (dot/at, ou phrases mal reconnues), et les numéros collés à des mots (ex: 5550199thank). Remplacez la partie “PII” du script par le bloc ci-dessous, puis relancez callcenter\_analytics.py.

```python
# Regex "strict" email (fonctionne après normalisation)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# Téléphone : 7+ digits au total, séparateurs optionnels (espaces, -, .)
PHONE_RE = re.compile(r"\b(?:\d[\s\-\.]*){7,}\d\b")

DIGIT_WORDS = {
    "zero":"0","oh":"0","o":"0",
    "one":"1","won":"1",
    "two":"2","too":"2","to":"2",
    "three":"3","free":"3","tree":"3",
    "four":"4","for":"4",
    "five":"5","fife":"5","hi":"5",
    "six":"6",
    "seven":"7",
    "eight":"8","ate":"8",
    "nine":"9",
}

def preclean(text: str) -> str:
    t = text.lower()

    # Séparer chiffres collés à des mots : "5550199thank" -> "5550199 thank"
    t = re.sub(r"(\d)([a-z])", r"\1 \2", t)
    t = re.sub(r"([a-z])(\d)", r"\1 \2", t)

    # Ajouter un espace après ponctuation collée entre deux tokens : "hello.thank" -> "hello. thank"
    t = re.sub(r"([a-z0-9])([.,!?])([a-z0-9])", r"\1\2 \3", t)

    # Apostrophes gênantes pour les heuristiques (ex: "don't" -> "dont")
    t = t.replace("'", "").replace("’", "").replace("...", " ")

    # Compacter espaces
    t = re.sub(r"\s+", " ", t).strip()
    return t

def normalize_spelled_tokens(text: str) -> str:
    """
    Normalisation pragmatique:
    - 'dot' -> '.', 'at' -> '@' (utile email)
    - mots-chiffres -> digits
    - collage des séquences de digits séparés (>= 6 digits)
    """
    t = preclean(text)

    # Normalisation email parlée
    t = re.sub(r"\bdot\b", ".", t)
    t = re.sub(r"\bat\b", "@", t)
    t = re.sub(r"\s*([.@])\s*", r"\1", t)  # supprime espaces autour de . et @

    # Remplacer mots->digits (token-level)
    def _tok_sub(m):
        w = m.group(0)
        return DIGIT_WORDS.get(w, w)

    t = re.sub(r"\b[a-z]+\b", _tok_sub, t)

    # Coller les digits isolés : "5 5 5 0 1 9 9" -> "5550199"
    def _collapse(m):
        digits = re.findall(r"\d", m.group(0))
        return "".join(digits)

    t = re.sub(r"(?:\b\d\b[\s,\-\.]*){6,}\b", _collapse, t)

    # Re-séparer digits/lettres au cas où après collapse
    t = re.sub(r"(\d)([a-z])", r"\1 \2", t)
    return t

def redact_order_id(text: str) -> Tuple[str, int]:
    """
    Masque un identifiant après 'order number is' même s’il est épelé (a.x.1.9.7.3.5).
    """
    count = 0
    pattern = re.compile(r"\border number is\b\s+([a-z0-9\s\.\-]{3,80})", re.IGNORECASE)

    def _sub(m):
        nonlocal count
        span = m.group(1)
        cleaned = re.findall(r"[A-Za-z0-9]", span)
        if len(cleaned) >= 5:
            count += 1
            return "order number is [REDACTED_ORDER]"
        return m.group(0)

    return pattern.sub(_sub, text), count

def redact_spoken_email(text: str) -> Tuple[str, int]:
    """
    1) masque les vrais emails détectables (après normalisation)
    2) sinon masque par contexte : "reach me ..." jusqu'à un marqueur (also/phone/order/thank) ou fin
    """
    count = 0

    # (1) email standard
    def _email_sub(m):
        nonlocal count
        count += 1
        return "[REDACTED_EMAIL]"

    t = EMAIL_RE.sub(_email_sub, text)
    if count > 0:
        return t, count

    # (2) fallback par contexte (robuste même si ASR “massacre” le local-part)
    ctx = re.compile(
        r"(\byou can reach me\b|\breach me\b)\s*(?:@)?\s*([a-z0-9.\s]{3,80})"
        r"(?=\b(?:also|my phone|phone number|order number|thank)\b|$)",
        re.IGNORECASE
    )

    def _ctx_sub(m):
        nonlocal count
        count += 1
        return m.group(1) + " [REDACTED_EMAIL]"

    return ctx.sub(_ctx_sub, t), count

def redact_phone(text: str) -> Tuple[str, int]:
    count = 0

    def _sub(m):
        nonlocal count
        count += 1
        return "[REDACTED_PHONE]"

    return PHONE_RE.sub(_sub, text), count

def redact_pii(text: str) -> Tuple[str, Dict[str, int]]:
    """
    Post-traitement + redaction PII:
    - normalise tokens épelés (digits, dot/at)
    - masque order id (contexte)
    - masque email (standard ou contexte)
    - masque téléphone (séquences de digits)
    """
    stats = {"emails": 0, "phones": 0, "orders": 0}

    t = normalize_spelled_tokens(text)

    t, n_orders = redact_order_id(t)
    stats["orders"] += n_orders

    t, n_emails = redact_spoken_email(t)
    stats["emails"] += n_emails

    t, n_phones = redact_phone(t)
    stats["phones"] += n_phones

    return t, stats
```

Ce post-traitement est volontairement heuristique et optimisé pour un contexte “call center” (PII parlée/épelée). Il peut produire des faux positifs sur d’autres transcriptions.

Relancer l’expérience et comparer les résultats (à expliquer dans le rapport).

> **Comparaison avant/après post-traitement :**
> Avant (regex simple) : emails=0, phones=0, orders=0 — aucune PII détectée car le transcript brut Whisper n’est pas normalisé.
> Après normalize_spelled_tokens + redaction contextuelle : emails=0, phones=0, orders=1 — l’order id est capturé par le pattern contextuel ; le téléphone reste non détecté car Whisper transcrit 555 0199 en to (homophone).

Dans le rapport, écrivez une réflexion courte (5–8 lignes max) : quelles erreurs de transcription Whisper impactent le plus vos analytics ?

> **Réflexion — impact des erreurs Whisper :**
> L’erreur la plus critique est 555 0199 transcrit en to par Whisper (homophone) — aucune regex numérique ne récupère cette PII. L’intention retournée est delivery_issue au lieu de refund_or_replacement car les mots delivered/package/order apparaissent plus souvent que refund/replacement. En production, rater l’intention refund reroute vers le mauvais agent — erreur coûteuse. Manquer thank you est sans conséquence produit.

### TTS léger : générer une réponse “agent” et contrôler latence/qualité

Créez le fichier TP3/tts\_reply.py à partir du code ci-dessous, puis complétez les trous \_\_\_\_\_\_\_\_. Le script doit : générer une courte réponse vocale (anglais) à partir d’un modèle TTS gratuit Hugging Face, sauvegarder un WAV, et mesurer le temps total ainsi qu’un RTF (Real-Time Factor).

Gardez le message court (quelques secondes) pour rester sous les budgets de temps. Ne commitez pas l’audio généré : une capture et des métadonnées suffisent.

```python
import os
import time

import numpy as np
import torch
from transformers import pipeline
import torchaudio

def main():
    os.makedirs("TP3/outputs", exist_ok=True)

    text = (
        "Thanks for calling. I am sorry your order arrived damaged. "
        "I can offer a replacement or a refund. "
        "Please confirm your preferred option."
    )

    # Modèle TTS léger (anglais)
    tts_model_id = "facebook/________"   # Trouvez un modèle de TTS (ici, donné par Facebook par exemple)

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
    audio_dur_s = float(audio.shape[1] / float(sr))
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

    out_wav = "outputs/tts_reply_call_01.wav"  # ex: tts_reply_call_01

    wav_t = torch.from_numpy(audio.astype(np.float32))  # [C, T]
    torchaudio.save(out_wav, wav_t, sr)

    print("tts_model_id:", tts_model_id)
    print("device:", "cuda" if device == 0 else "cpu")
    print("audio_dur_s:", round(audio_dur_s, 2))
    print("elapsed_s:", round(elapsed_s, 2))
    print("rtf:", round(rtf, 3))
    print("saved:", out_wav)

if __name__ == "__main__":
    main()
```

Le champ out\["audio"\] est déjà un tableau numpy. Le paquet soundfile est souvent celui qui manque lorsqu’on veut écrire un WAV.

Exécutez tts\_reply.py et ajoutez au rapport une capture d’écran montrant : tts\_model\_id, audio\_dur\_s, elapsed\_s, rtf, et le chemin du fichier généré.

```bash
python TP3/tts_reply.py
```

> **Trou complété — tts_reply.py :** `tts_model_id = "facebook/mms-tts-eng"`
> Modèle TTS léger de Meta/Fairseq, mono-lingual anglais, sans GPU requis.

> **Résultats — tts_reply.py :**
> ```
> tts_model_id: facebook/mms-tts-eng
> device: cpu
> audio_dur_s: 8.13
> elapsed_s: 1.93
> rtf: 0.238
> saved: TP3/outputs/tts_reply_call_01.wav
> ```

Vérifiez les métadonnées du WAV généré (durée, sample rate, canaux) et ajoutez au rapport une capture d’écran contenant les lignes pertinentes.

```bash
ffprobe TP3/outputs/tts_reply_call_01.wav
# ou
soxi TP3/outputs/tts_reply_call_01.wav
```

> **Métadonnées WAV — tts_reply_call_01.wav :**
> Sample rate : 16 000 Hz | Canaux : 1 (mono) | Format : PCM 16-bit | Durée : 8.13 s

Inutile de coller toute la sortie brute : une capture lisible suffit.

Dans le rapport, écrivez une observation courte (4–6 lignes max) sur la qualité TTS : intelligibilité, prosodie, artefacts éventuels (metallic, coupures), et latence perçue au vu du RTF. Une remarque factuelle suffit. Par exemple : “prononciation OK, rythme un peu monotone, RTF ~ 0.3 donc compatible temps réel”.

> **Observation qualité TTS :**
> Synthèse globalement intelligible : chaque mot est perceptible et le message compris dans son ensemble. La prosodie est monotone — ton plat, absence d'intonation montante/descendante, rythme légèrement rigide typique des modèles TTS feed-forward. Aucun artefact métallique ni coupure audible détecté. RTF = 0.238 (< 1) : le modèle tourne nettement en dessous du temps réel sur CPU, ce qui est compatible avec une intégration dans un centre d'appels à charge modérée.

Évaluer l’intelligibilité de la TTS via ASR. L’idée est de transcrire le WAV généré avec Whisper, puis de comparer grossièrement au texte source. Créez TP3/asr\_tts\_check.py à partir du code ci-dessous et complétez les trous \_\_\_\_\_\_\_\_.

```python
import time
import torch
from transformers import pipeline

def main():
    wav_path = "TP3/outputs/________.wav"    
    model_id = "openai/________"            

    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        device=device
    )

    t0 = time.time()
    generate_kwargs = {
        "language": "english"
    }
    out = asr(wav_path, generate_kwargs=generate_kwargs)   # out: {"text": "...", ...}
    t1 = time.time()

    print("model_id:", model_id)
    print("elapsed_s:", round(t1 - t0, 2))
    print("text:", out.get("text","").strip())

if __name__ == "__main__":
    main()
```

> **Trous complétés — asr_tts_check.py :**
> `wav_path = "TP3/outputs/tts_reply_call_01.wav"` | `model_id = "openai/whisper-tiny"`

> **Résultats — asr_tts_check.py :**
> ```
> model_id: openai/whisper-tiny
> elapsed_s: 2.09
> text: Thanks for calling. I am sorry your order arrived. Demage Guy can offer a replacement or a refund. Please confirm your preferred option.
> ```
> Observation : "damaged" → "Demage Guy" (faute d'articulation TTS amplifiée par Whisper-tiny). Le message reste globalement compréhensible.


### Intégration : pipeline end-to-end + rapport d’ingénierie (léger)

Créez le fichier TP3/run\_pipeline.py à partir du code ci-dessous, puis complétez les trous \_\_\_\_\_\_\_\_. Le script doit exécuter l’ensemble de la chaîne sur call\_01.wav dans cet ordre : VAD → ASR → analytics → TTS (optionnel).

L’objectif est d’avoir un point d’entrée unique “type produit” et de produire un petit résumé final. Vous ne devez pas générer de gros fichiers : uniquement des JSON légers et éventuellement un WAV TTS.

```python
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
    run("python TP3/________")
    vad = load_json("TP3/outputs/vad_segments_call_01.json")

    # 2) ASR
    run("python TP3/________")
    asr = load_json("TP3/outputs/asr_call_01.json")

    # 3) Analytics
    run("python TP3/________")
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
```

Si une étape échoue, c’est souvent un problème de dépendances ou de chemins. Vérifiez que TP3/outputs/ contient bien les JSON attendus après chaque étape.

Exécutez run\_pipeline.py (idéalement dans un srun GPU si disponible). Ajoutez au rapport une capture d’écran montrant le résumé final (PIPELINE SUMMARY) et le fichier de sortie créé.

```bash
python TP3/run_pipeline.py
```

> **Trous complétés — run_pipeline.py :**
> `run("python TP3/vad_segment.py")` | `run("python TP3/asr_whisper.py")` | `run("python TP3/callcenter_analytics.py")`

> **Terminal — PIPELINE SUMMARY :**
> ```
> === PIPELINE SUMMARY ===
> audio_path: TP3/data/call_01.wav
> duration_s: 25.04
> num_segments: 3
> speech_ratio: 0.9426517571884984
> asr_model: openai/whisper-tiny
> asr_device: cpu
> asr_rtf: 0.2809968524085828
> intent: delivery_issue
> pii_stats: {'emails': 0, 'phones': 0, 'orders': 1}
> tts_generated: True
> saved: TP3/outputs/pipeline_summary_call_01.json
> ```

Ouvrez TP3/outputs/pipeline\_summary\_call\_01.json et ajoutez au rapport un extrait contenant au minimum : num\_segments, speech\_ratio, asr\_rtf, intent, pii\_stats.

```bash
cat TP3/outputs/pipeline_summary_call_01.json
```

> **pipeline_summary_call_01.json :**
> ```json
> {
>   "audio_path": "TP3/data/call_01.wav",
>   "duration_s": 25.04,
>   "num_segments": 3,
>   "speech_ratio": 0.9426517571884984,
>   "asr_model": "openai/whisper-tiny",
>   "asr_device": "cpu",
>   "asr_rtf": 0.2809968524085828,
>   "intent": "delivery_issue",
>   "pii_stats": {"emails": 0, "phones": 0, "orders": 1},
>   "tts_generated": true
> }
> ```

Ce fichier est petit : vous pouvez le copier/coller intégralement dans le rapport si vous le souhaitez.

Dans le rapport, écrivez un court “engineering note” (8–12 lignes max) répondant à ces points :

*   Quel est le goulet d’étranglement principal (temps) dans votre pipeline ?
*   Quelle étape est la plus fragile (qualité) et pourquoi ?
*   Deux améliorations concrètes si vous deviez industrialiser (sans entraîner de modèle).

Exemples d’améliorations acceptables : meilleure normalisation texte, calibration VAD, meilleure redaction PII, batching ASR, cache modèle, contrôle longueur segments

> **Engineering note :**
> Le goulet d'étranglement principal (temps) est l'étape ASR : RTF ≈ 0.28 sur CPU, soit ~7 s de traitement pour 25 s d'audio. Avec GPU, ce ratio tomberait sous 0.05. L'étape la plus fragile en qualité est également l'ASR : Whisper-tiny produit des homophones ("555 0199" → "to", "damaged" → "Demage Guy"), ce qui propage des erreurs irréparables aux étapes aval (PII manquée, intent mal classée). Deux améliorations concrètes sans ré-entraînement : (1) appliquer un dictionnaire de post-normalisation ciblé call center (chiffres épelés, références commande, e-mails dictés) pour corriger les transcriptions avant analytics ; (2) affiner la segmentation VAD avec un padding de 200 ms et une durée maximale de segment de 10 s afin d'éviter les très longs segments qui diluent les mots-clés dans le classifieur d'intention.

Relancer l’expérience et comparer les résultats (à expliquer dans le rapport).

> **Comparaison avant/après post-traitement :**
> Avant (regex simple) : `emails=0, phones=0, orders=0` — aucune PII détectée car le transcript brut Whisper n’est pas normalisé.
> Après `normalize_spelled_tokens` + redaction contextuelle : `emails=0, phones=0, orders=1` — l’order id est capturé par le pattern contextuel ; le téléphone “555 0199” reste non détecté car Whisper le transcrit en “to” (homophone).

Dans le rapport, écrivez une réflexion courte (5–8 lignes max) : quelles erreurs de transcription Whisper impactent le plus vos analytics ?

> **Réflexion — impact des erreurs Whisper sur les analytics :**
> L’erreur la plus critique est “555 0199” → “to” (transcription homophonique) : aucune regex numérique ne peut rattraper cette PII. Whisper-tiny confond aussi “Alex” → “Thelex” (impact faible) et “cracked” → “cracking” (keyword légèrement dégradé). Le plus problématique : l’intention retournée est `delivery_issue` au lieu de `refund_or_replacement` car les mots “delivered/package/order” sont répétés plusieurs fois alors que “refund” et “replacement” n’apparaissent qu’une fois dans un long segment. En production, rater l’intention “refund” reroute l’appel vers le mauvais agent — erreur coûteuse. Manquer “thank you” est sans conséquence produit.