import os
import re
import json
from typing import Dict, Any, List, Tuple
from collections import Counter

# ---- Regex et utilitaires post-traitement ----

# Regex "strict" email (fonctionne après normalisation)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# Téléphone : 7+ digits au total, séparateurs optionnels (espaces, -, .)
PHONE_RE = re.compile(r"\b(?:\d[\s\-\.]*){7,}\d\b")

INTENTS = {
    "refund_or_replacement": ["refund", "replacement", "damaged", "cracked", "broken"],
    "delivery_issue": ["delivered", "package", "arrived", "yesterday", "order"],
    "general_support": ["help", "support", "thank you", "calling"],
}

STOPWORDS = set([
    "the","a","an","and","or","to","for","of","in","on","is","it","i","you","we","my","your",
    "was","were","be","as","at","but","this","that","with","about","today"
])

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
    # Ajouter un espace après ponctuation collée entre deux tokens
    t = re.sub(r"([a-z0-9])([.,!?])([a-z0-9])", r"\1\2 \3", t)
    # Apostrophes gênantes
    t = t.replace("'", "").replace("\u2019", "").replace("...", " ")
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
    Masque un identifiant après 'order number is' même s'il est épelé (a.x.1.9.7.3.5).
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

    def _email_sub(m):
        nonlocal count
        count += 1
        return "[REDACTED_EMAIL]"

    t = EMAIL_RE.sub(_email_sub, text)
    if count > 0:
        return t, count

    # fallback par contexte
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

def normalize(text: str) -> str:
    t = text.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokenize(text: str) -> List[str]:
    toks = re.findall(r"[a-z]+", text.lower())
    return [w for w in toks if w not in STOPWORDS and len(w) > 2]

def score_intents(text: str) -> Dict[str, int]:
    t = normalize(text)
    scores: Dict[str, int] = {}
    for intent, kws in INTENTS.items():
        s = 0
        for kw in kws:
            s += t.count(kw)
        scores[intent] = s
    return scores

def pick_intent(scores: Dict[str, int]) -> str:
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
