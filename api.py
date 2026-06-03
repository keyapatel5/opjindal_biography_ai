import os
import time
import re
import subprocess
import json
import glob
import logging
import hashlib
import random
import threading
import base64
import tempfile
from collections import OrderedDict
from typing import Optional
from difflib import SequenceMatcher

import numpy as np
import fastapi
from fastapi import UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import edge_tts
from faster_whisper import WhisperModel
from mongo_config import db


# ============================================================
# CONFIG
# ============================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR   = os.path.join(BASE_DIR, "tmp_audio")
LOG_DIR    = os.path.join(BASE_DIR, "logs")

for _d in [TEMP_DIR, LOG_DIR]:
    os.makedirs(_d, exist_ok=True)

VOICE_EN = "en-IN-PrabhatNeural"
VOICE_HI = "hi-IN-MadhurNeural"

COLLECTION = "knowledge_base"
EN_Q_KEY      = "question_english"
EN_A_KEY      = "answer_english"
HI_Q_KEY      = "question_hindi"
HI_A_KEY      = "answer_hindi"

MIN_MATCH  = 0.35
SEM_FLOOR  = 0.25
EARLY_EXIT = 0.65
TOP_K      = 50
STT_MODEL  = "base"

HINDI_CONFUSED_LANGS = {
    "ur", "ne", "mr", "pa", "gu", "bn", "ar", "fa", "ja", "zh"
}

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "bot.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("JindalBot")


# ============================================================
# INTENT ENGINE
# Maps every question/answer to a topic so born?death, etc.
# ============================================================

# Keywords that strongly signal each intent.
# These are checked against BOTH query words AND answer words.
_INTENT_GROUPS = {
    "birth": {
        "born", "birth", "janm", "janam", "paida", "janme",
        "birthday", "dob", "janmdin", "paidaish", "janma",
        # Hindi: ?? ???? ???, ????
        "peda", "janamdin",
    },
    "death": {
        "died", "death", "die", "maut", "mrityu", "mara",
        "passed", "killed", "crash", "accident", "plane",
        "helicopter", "martyr", "expired", "nidhan", "guzre",
        "nidhana", "guzra", "intqal", "wafat",
    },
    "birth_place": {
        "birthplace", "native", "hometown", "gaon", "village",
        "janmsthan", "janmbhoomi", "nativetown",
    },
    "age": {
        "age", "umar", "umra", "years old", "kitne saal",
    },
    "education": {
        "education", "school", "college", "study", "studied",
        "degree", "padhai", "shiksha",
    },
    "family": {
        "wife", "husband", "son", "daughter", "children",
        "father", "mother", "brother", "sister", "family",
        "married", "marriage", "patni", "biwi", "beta",
        "beti", "pita", "mata", "bhai", "behen", "parivar",
        "shaadi", "vivah", "shadi", "bachche",
    },
    "business": {
        "business", "company", "steel", "industry", "factory",
        "empire", "trade", "vyapar", "udyog", "karkhana",
        "ispat", "loha", "founded", "established",
    },
    "politics": {
        "political", "politics", "minister", "election",
        "mla", "mp", "party", "congress", "government",
        "rajniti", "mantri", "chunav",
    },
    "award": {
        "award", "achievement", "honor", "prize",
        "recognition", "puraskar", "samman", "medal",
    },
    "wealth": {
        "property", "wealth", "rich", "assets",
        "daulat", "sampatti", "net worth",
    },
}

# Reverse: word ? intent
_KW_TO_INTENT: dict = {}
for _intent, _kws in _INTENT_GROUPS.items():
    for _kw in _kws:
        _KW_TO_INTENT[_kw] = _intent

# Pairs that must NEVER match each other � cross-penalty = 0.05
_HARD_CONFLICTS = {
    frozenset(["birth", "death"]),
    frozenset(["birth", "birth_place"]),
    frozenset(["death", "birth_place"]),
    frozenset(["birth", "wealth"]),
    frozenset(["birth", "politics"]),
    frozenset(["death", "business"]),
    frozenset(["family", "birth"]),
    frozenset(["family", "birth_place"]),
}

# ============================================================
# STT PROMPT — names Whisper must recognise correctly
# ============================================================

STT_INITIAL_PROMPT = (
    "O.P. Jindal,OP Jindal,op Jindal,O P Jindal,Jindal Steel, Savitri Devi Jindal, "
    "Naveen Jindal, Sajjan Jindal, Prithviraj Jindal, Ratan Jindal, "
    "Vidya Devi, Shanti Devi, Kanshi Ram, "
    "Hisar, Haryana, Nalwa, Kurukshetra, Lok Sabha, "
    "parliamentary constituency, personal motto, "
    "Jindal Group, Jindal Industries, JSPL, JSW Steel."
)

def extract_intent(text: str) -> str:
    """Extract topic-intent from any text (question or answer)."""
    q = text.lower().strip()
    words = q.split()
    # Check bigrams first
    for i in range(len(words) - 1):
        bg = words[i] + " " + words[i + 1]
        if bg in _KW_TO_INTENT:
            return _KW_TO_INTENT[bg]
    # Single words
    for w in words:
        w = re.sub(r"[^\w]", "", w)
        if w in _KW_TO_INTENT:
            return _KW_TO_INTENT[w]
    return "unknown"


def intent_score_multiplier(q_intent: str, d_intent: str) -> float:
    """
    Returns a score multiplier based on intent match/mismatch.
    1.0 = neutral, >1.0 = boost, <1.0 = penalty.
    """
    if q_intent == "unknown":
        return 1.0
    if d_intent == "unknown":
        return 0.80
    if d_intent == q_intent:
        return 1.35   # strong boost for exact match
    pair = frozenset([q_intent, d_intent])
    if pair in _HARD_CONFLICTS:
        return 0.05   # near-zero � completely wrong topic
    return 0.40       # general mismatch


# ============================================================
# FALLBACK RESPONSES
# ============================================================

class Fallback:
    def __init__(self):
        self.en = {
            "general": [
                "Please ask me about O.P. Jindal's life, family, business, or achievements!",
                "That's outside my expertise. Try asking about O.P. Jindal's journey!",
                "I don't have info on that. Ask me about O.P. Jindal!",
                "My specialty is O.P. Jindal - try asking about his birth, career, or contributions!",
                "Ask me anything about O.P. Jindal and I'll do my best to answer!",
            ],
            "family": [
                "Try asking about O.P. Jindal's wife, children, or parents!",
                "Ask me about O.P. Jindal's sons, daughters, or his wife Savitri Devi!",
            ],
            "business": [
                "Ask me about Jindal Steel or how O.P. Jindal built his industrial empire!",
                "O.P. Jindal started as a bucket trader! Ask me about his business journey!",
            ],
        }
        self.hi = {
            "general": [
                "????? ????? ?.??. ????? ?? ????, ??????, ??????? ?? ?????????? ?? ???? ??? ?????!",
                "?? ???? ??????? ?? ???? ??? ?.??. ????? ?? ???? ??? ?????!",
                "???? ???? ??????? ???? ??! ?.??. ????? ?? ???? ??? ?????!",
                "?.??. ????? ?? ???? ??? ??? ?? ????? ?? ??? ???? ?????!",
            ],
            "family": [
                "????? ?.??. ????? ?? ?????, ?????? ?? ?????? ?? ???? ??? ?????!",
            ],
            "business": [
                "????? ????? ?? ?.??. ????? ?? ??????? ?? ???? ??? ?????!",
            ],
        }
        self._fam = {
            "family", "wife", "son", "daughter", "children",
            "father", "mother", "patni", "biwi", "beta",
            "beti", "pita", "mata", "bhai", "behen", "parivar",
        }
        self._biz = {
            "business", "company", "steel", "industry",
            "factory", "vyapar", "udyog", "karkhana",
        }
        self._last = {}

    def get(self, query, lang):
        w = set(re.findall(r"\w+", query.lower()))
        cat = "family" if w & self._fam else "business" if w & self._biz else "general"
        hi = lang in ("hi", "hindi", "hinglish")
        pool = (self.hi if hi else self.en).get(cat, (self.hi if hi else self.en)["general"])
        last = self._last.get(f"{lang}_{cat}")
        avail = [r for r in pool if r != last] or pool
        pick = random.choice(avail)
        self._last[f"{lang}_{cat}"] = pick
        log.info(f"FALLBACK [{lang}/{cat}]")
        return pick


fallback = Fallback()


# ============================================================
# HINGLISH TRANSLATOR
# ============================================================

class Translator:
    def __init__(self):
        self.words = {
            "kya": "what", "kab": "when", "kahan": "where",
            "kaun": "who", "kahaan": "where", "kaise": "how",
            "kyun": "why", "kyu": "why", "kitna": "how much",
            "kitne": "how many", "kitni": "how much",
            "kis": "which", "kaunsa": "which",
            "hai": "is", "hain": "are", "ho": "be",
            "hoon": "am", "hu": "am",
            "tha": "was", "the": "were", "thi": "was",
            "hoga": "will be", "hogi": "will be",
            "hua": "happened", "hui": "happened",
            "kiya": "did", "kiye": "did", "ki": "of",
            "karte": "do", "karta": "does", "karti": "does",
            "kar": "do", "karo": "do", "karna": "to do",
            "bana": "made", "banaya": "made",
            "shuru": "start", "khatam": "end",
            "mila": "got", "diya": "gave", "liya": "took",
            "aaya": "came", "gaya": "went",
            "paida": "born", "janma": "born", "janme": "born",
            "mara": "died", "mare": "died", "mar": "died",
            "naam": "name", "nam": "name",
            "pura": "full", "poora": "full",
            "janm": "birth", "janam": "birth",
            "maut": "death", "mrityu": "death",
            "umar": "age", "umra": "age", "saal": "year",
            "pita": "father", "pitaji": "father",
            "papa": "father", "baap": "father",
            "mata": "mother", "mataji": "mother", "maa": "mother",
            "bhai": "brother", "behen": "sister",
            "beta": "son", "beti": "daughter",
            "patni": "wife", "biwi": "wife", "pati": "husband",
            "bachche": "children", "baccha": "child",
            "parivaar": "family", "parivar": "family",
            "ghar": "home", "gaon": "village", "sheher": "city",
            "desh": "country", "jagah": "place",
            "padhai": "education", "shiksha": "education",
            "kaam": "work", "kam": "work",
            "vyapar": "business", "vyapaar": "business",
            "company": "company", "factory": "factory",
            "karkhana": "factory", "naukri": "job",
            "paise": "money", "dhan": "wealth",
            "safalta": "success", "kamyabi": "success",
            "sangharsh": "struggle", "mehnat": "hard work",
            "pehla": "first", "pahla": "first",
            "dusra": "second", "doosra": "second",
            "sabse": "most", "bahut": "very", "bohot": "very",
            "bada": "big", "chota": "small",
            "purana": "old", "naya": "new",
            "accha": "good", "achha": "good",
            "mein": "in", "me": "in", "par": "on",
            "se": "from", "ko": "to",
            "ke": "of", "ka": "of",
            "aur": "and", "ya": "or", "lekin": "but",
            "agar": "if", "jab": "when", "phir": "then",
            "saath": "with", "bina": "without",
            "batao": "tell", "bataiye": "tell",
            "bolo": "speak", "suno": "listen",
            "steel": "steel", "ispat": "steel",
            "loha": "iron", "udyog": "industry",
            "jindal": "jindal", "jinda": "jindal",
            "om": "om", "prakash": "prakash",
            "op": "op", "o.p.": "op",
            "sahab": "sir", "ji": "sir", "shri": "mr",
            "main": "i", "hum": "we", "tum": "you", "aap": "you",
            "yeh": "this", "ye": "this",
            "woh": "that", "wo": "that",
            "unka": "his", "uska": "his", "unki": "his",
            "mera": "my", "meri": "my",
            "tumhara": "your", "aapka": "your", "aapki": "your",
            "log": "people", "baat": "matter",
        }
        self.dev = {
            "?": "ka", "?": "kha", "?": "ga", "?": "gha",
            "?": "cha", "?": "chha", "?": "ja",
            "?": "ta", "?": "tha", "?": "da", "?": "dha", "?": "na",
            "?": "ta", "?": "tha", "?": "da", "?": "dha", "?": "na",
            "?": "pa", "?": "pha", "?": "ba", "?": "bha", "?": "ma",
            "?": "ya", "?": "ra", "?": "la", "?": "va",
            "?": "sha", "?": "sha", "?": "sa", "?": "ha",
            "?": "a", "?": "i", "?": "i", "?": "u", "?": "u",
            "?": "e", "?": "ai", "?": "o", "?": "au", "?": "",
            "?": "m", "?": "h", "?": "n",
            "?": "aa", "?": "i", "?": "ii",
            "?": "u", "?": "uu", "?": "e", "?": "ai",
            "?": "o", "?": "au",
        }
        self.phon = {
            "kya": ["kia", "kyaa"],
            "hai": ["he", "hae"],
            "kahan": ["kaha", "kahaan"],
            "naam": ["nam"],
            "jindal": ["jinda", "jindaal"],
            "paida": ["pada", "payda"],
            "janm": ["janam", "janma"],
        }

    def _translit(self, text):
        return "".join(self.dev.get(c, c) for c in text)

    def translate(self, text):
        text = self._translit(text).lower()
        out = []
        for w in text.split():
            c = re.sub(r"[^\w\s]", "", w)
            if c in self.words:
                out.append(self.words[c])
            else:
                found = False
                for key, variants in self.phon.items():
                    if c in variants or c == key:
                        if key in self.words:
                            out.append(self.words[key])
                            found = True
                            break
                if not found:
                    out.append(w)
        return " ".join(out)

    def detect_translate(self, text):
        has_dev = bool(re.search(r"[\u0900-\u097F]", text))
        words = text.lower().split()
        hc = sum(1 for w in words if re.sub(r"[^\w]", "", w) in self.words)
        total = len(words)
        if has_dev:
            return self.translate(text), "hindi"
        elif total > 0 and (hc / total) > 0.15:
            return self.translate(text), "hinglish"
        return text, "english"


TR = Translator()


# ============================================================
# TEXT PROCESSING
# ============================================================

STOPS = {
    "a", "an", "the", "is", "was", "are", "were", "be",
    "been", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "can", "may",
    "might", "am", "not", "no", "so", "too", "very",
    "i", "me", "my", "we", "our", "you", "your",
    "he", "him", "she", "her", "it", "its",
    "they", "them", "their", "this", "that",
    "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "and", "but", "or", "if", "as", "than",
    "up", "about", "tell", "please", "know", "said", "us",
}

ORDS = {
    "first", "second", "third", "oldest", "youngest",
    "eldest", "pehla", "dusra", "tisra", "sabse",
    "bada", "chota",
}


def norm(text):
    t = text.lower().strip()
    t = re.sub(r"om\s+prakash\s+jindal", "op jindal", t)
    t = re.sub(r"o\.?\s*p\.?\s+jindal", "op jindal", t)
    t = re.sub(r"\bo\s+p\b", "op", t)
    t = re.sub(r"o\.p\.", "op", t)
    t = re.sub(r"o\s*p\s*jindal", "op jindal", t)
    t = re.sub(r"['\u2018\u2019\u0027]s\b", "", t)
    t = re.sub(r"[^\w\s\u0900-\u097F]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def tok(text):
    return [w for w in norm(text).split() if w not in STOPS and len(w) > 1]


def lang_detect(text, stt_hint: str = None):
    hc = len(re.findall(r"[\u0900-\u097F]", text))
    total = len(text.strip())
    if total == 0:
        return "en"
    if (hc / total) > 0.3:
        return "hi"
    if stt_hint == "hi" and hc > 0:
        return "hi"
    words = text.lower().split()
    wc = sum(1 for w in words if re.sub(r"[^\w]", "", w) in TR.words)
    if len(words) > 0 and wc >= 2:
        return "hinglish"
    return "en"


def qword(text):
    q = text.lower().strip()
    if re.match(r"^(when|what year|which year)", q): return "when"
    if re.match(r"^(where|what place)", q):          return "where"
    if re.match(r"^(who|whose)", q):                 return "who"
    if re.match(r"^(how many|how much|how old)", q): return "howmany"
    if re.match(r"^how", q):                         return "how"
    if re.match(r"^(what|which)", q):                return "what"
    if re.match(r"^why", q):                         return "why"
    return "x"


# ============================================================
# FUZZY
# ============================================================

def fuzzy(s1, s2):
    a, b = s1.lower(), s2.lower()
    r = SequenceMatcher(None, a, b).ratio()
    w1, w2 = set(a.split()), set(b.split())
    if not w1 or not w2:
        return r
    j = len(w1 & w2) / len(w1 | w2)
    p = sum(
        1 for x in w1
        if len(x) > 2 and any(
            len(y) > 2 and SequenceMatcher(None, x, y).ratio() > 0.8
            for y in w2
        )
    )
    mx = max(len(w1), len(w2))
    return max(r, j, p / mx if mx else 0)


def overlap(qw, tw):
    if not qw or not tw:
        return 0.0
    qs, ts = set(qw), set(tw)
    e = len(qs & ts)
    p = sum(
        0.5 for q in qs - ts
        if len(q) > 2 and any(
            len(t) > 2 and SequenceMatcher(None, q, t).ratio() > 0.75
            for t in ts
        )
    )
    u = len(qs | ts)
    return (e + p) / u if u else 0.0


# ============================================================
# CACHE
# ============================================================

class Cache:
    def __init__(self):
        self.d = OrderedDict()
        self.h = self.m = 0

    def get(self, t):
        k = hashlib.md5(t.lower().strip().encode()).hexdigest()
        if k in self.d:
            e = self.d[k]
            if time.time() - e["t"] < 3600:
                self.d.move_to_end(k)
                self.h += 1
                return e["v"]
            del self.d[k]
        self.m += 1
        return None

    def put(self, t, v):
        k = hashlib.md5(t.lower().strip().encode()).hexdigest()
        self.d[k] = {"v": v, "t": time.time()}
        if len(self.d) > 500:
            self.d.popitem(last=False)

    def clear(self): self.d.clear()
    def stats(self): return {"hits": self.h, "misses": self.m, "size": len(self.d)}


cache = Cache()


# ============================================================
# SEARCH ENGINE
# ============================================================

class Engine:
    def __init__(self, name):
        self.name  = name
        self.qs    = []
        self.ans   = []
        self.qn    = []
        self.qt    = []
        self.emb   = None
        self.bm25  = None
        self.ok    = False
        self.n     = 0
        self._lock = threading.Lock()
        # Pre-computed intent for every DB question AND answer
        self.q_intents = []
        self.a_intents = []

        self._ref = re.compile(
            r"\byou\b|\byour\b|\byours\b|\byourself\b"
            r"|\bu\b|\bur\b"
            r"|\baapka\b|\baapki\b|\baapke\b|\baap\b"
            r"|\btumhara\b|\btumhari\b|\btumhare\b|\btum\b"
            r"|\btera\b|\bteri\b|\btere\b",
            re.I,
        )
        self._subj = "op jindal"

    def rewrite(self, text):
        if not self._ref.search(text):
            return text
        r = text
        for p, v in [
            (r"\byourself\b", self._subj),
            (r"\byours\b",    self._subj + "'s"),
            (r"\byour\b",     self._subj + "'s"),
            (r"\byou\b",      self._subj),
            (r"\bur\b",       self._subj + "'s"),
        ]:
            r = re.sub(p, v, r, flags=re.I)
        for p in [
            r"\baapka\b", r"\baapki\b", r"\baapke\b",
            r"\btumhara\b", r"\btumhari\b", r"\btumhare\b",
            r"\btera\b", r"\bteri\b", r"\btere\b",
        ]:
            r = re.sub(p, self._subj + "'s", r, flags=re.I)
        for p in [r"\baap\b", r"\btum\b"]:
            r = re.sub(p, self._subj, r, flags=re.I)
        r = re.sub(r"\bwere\s+op\s+jindal\b", "was op jindal", r, flags=re.I)
        r = re.sub(r"\bare\s+op\s+jindal\b",  "is op jindal",  r, flags=re.I)
        return re.sub(r"\s+", " ", r).strip()

    def _is_jindal(self, q, a):
        c = (q + " " + a).lower()
        return any(m in c for m in ["jindal", "o.p.", "om prakash", "op jindal"])

    def load(self, mdb, col, qk, ak):
        t0 = time.time()
        try:
            data = list(mdb[col].find({}, {"_id": 0}))
            if not data:
                log.warning(f"[{self.name}] {col} EMPTY")
                return
            log.info(f"[{self.name}] {col}: {len(data)} records keys={list(data[0].keys())}")

            qs, ans, qn, qt = [], [], [], []
            for item in data:
                q = str(item.get(qk, "")).strip()
                a = str(item.get(ak, "")).strip()
                if q and a and q != "None" and a != "None":
                    qs.append(q); ans.append(a)
                    qn.append(norm(q)); qt.append(tok(q))

            n = len(qs)
            if n == 0:
                log.error(f"[{self.name}] 0 pairs!")
                return

            # Pre-compute intents for all DB entries (question + answer)
            q_intents = [extract_intent(q) for q in qn]
            a_intents = [extract_intent(a) for a in ans]

            bm25 = BM25Okapi(qt)
            emb  = embed_model.encode(
                qn, convert_to_tensor=True,
                                    show_progress_bar=False, batch_size=128,
            )

            with self._lock:
                self.qs = qs; self.ans = ans
                self.qn = qn; self.qt  = qt
                self.n  = n;  self.bm25 = bm25
                self.emb = emb; self.ok = True
                self.q_intents = q_intents
                self.a_intents = a_intents

            log.info(f"[{self.name}] {n} indexed in {time.time()-t0:.1f}s")
        except Exception as e:
            log.error(f"[{self.name}] load error: {e}", exc_info=True)

    def search(self, qn_str, qt_list, boost=False):
        with self._lock:
            if not self.ok or self.n == 0:
                return None, 0.0, 0.0
            n          = self.n
            qs         = self.qs
            ans        = self.ans
            qn_db      = self.qn
            qt_db      = self.qt
            emb        = self.emb
            bm25       = self.bm25
            q_intents  = self.q_intents
            a_intents  = self.a_intents

        t0 = time.time()

        # -- Phase 1: semantic + BM25 over ALL entries --
        qe  = embed_model.encode(qn_str, convert_to_tensor=True)
        sem = util.cos_sim(qe, emb)[0].cpu().numpy()
        bm  = bm25.get_scores(qt_list) if qt_list else np.zeros(n)
        mb  = max(float(bm.max()), 1e-6)
        bn  = bm / mb
        p1  = 0.55 * sem + 0.45 * bn

        # -- Phase 2: fuzzy on top-K --
        k   = min(TOP_K, n)
        top = p1.argsort()[-k:]
        fz  = np.zeros(n)
        for i in top:
            fz[i] = max(fuzzy(qn_str, qn_db[i]), overlap(qt_list, qt_db[i]))

        # -- Phase 3: combined base score --
        comb   = np.zeros(n)
        qs_set = set(qt_list)
        qo     = ORDS & qs_set
        st     = set(self._subj.split())
        hs     = bool(st & qs_set)

        for i in top:
            s, b, f = float(sem[i]), float(bn[i]), float(fz[i])
            if   s > 0.80:               sc = 0.75*s + 0.15*b + 0.10*f
            elif f > 0.85:               sc = 0.25*s + 0.25*b + 0.50*f
            elif b > 0.85:               sc = 0.30*s + 0.55*b + 0.15*f
            elif s > 0.65 and f > 0.70:  sc = 0.55*s + 0.20*b + 0.25*f
            else:                        sc = 0.50*s + 0.35*b + 0.15*f

            if s > 0.6 and b > 0.6: sc += 0.08
            if s > 0.6 and f > 0.7: sc += 0.08
            if f > 0.7 and b > 0.6: sc += 0.05

            if qs_set:
                sc += len(qs_set & set(qt_db[i])) / len(qs_set) * 0.05

            if hs and boost:
                if st & set(qt_db[i]):
                    sc += 0.10
                    if self._is_jindal(qs[i], ans[i]):
                        sc += 0.05

            comb[i] = sc

        if hs and boost:
            for i in top:
                if not self._is_jindal(qs[i], ans[i]):
                    comb[i] *= 0.4

        if qo:
            for i in top:
                mo = ORDS & set(qt_db[i])
                if mo and qo and mo != qo:
                    comb[i] *= 0.5

        # -- qword penalty --
        qw = qword(qn_str)
        if qw != "x":
            for i in top:
                dw = qword(qs[i])
                if qw == dw:
                    comb[i] += 0.08
                elif dw != "x":
                    if {qw, dw} == {"when", "where"}:
                        comb[i] *= 0.4
                    elif qw == "who" and dw in ("when", "where", "what"):
                        comb[i] *= 0.6
                    else:
                        comb[i] *= 0.7

        # -- INTENT SCORING (question-level + answer-level) --
        # We extract intent from the QUERY, then check it against
        # BOTH the stored question intent AND the stored answer intent.
        # If either conflicts ? heavy penalty.
        # If both match ? strong reward.
        q_intent = extract_intent(qn_str)
        log.info(f"  [{self.name}] query_intent={q_intent} qw={qw}")

        if q_intent != "unknown":
            for i in top:
                qi = q_intents[i]   # intent of DB question
                ai = a_intents[i]   # intent of DB answer text

                # Use whichever gives the stronger signal
                # Answer-level intent is often more reliable
                # (answers actually say "born on..." / "died on...")
                best_d_intent = ai if ai != "unknown" else qi

                mul = intent_score_multiplier(q_intent, best_d_intent)
                comb[i] *= mul

                # Extra check: if answer intent directly contradicts
                # query intent, apply an additional penalty
                if ai != "unknown" and ai != q_intent:
                    pair = frozenset([q_intent, ai])
                    if pair in _HARD_CONFLICTS:
                        comb[i] *= 0.10  # double-penalise hard conflicts

        # -- final best --
        bi   = int(comb.argmax())
        bs   = float(comb[bi])
        bsem = float(sem[bi])

        ms = (time.time() - t0) * 1000
        t3 = comb.argsort()[-3:][::-1]
        for r, i in enumerate(t3):
            if comb[i] > 0:
                log.info(
                    f"  [{self.name}][{r+1}] score={comb[i]:.3f} "
                    f"sem={sem[i]:.3f} "
                    f"q_intent={q_intents[i]} a_intent={a_intents[i]} "
                    f"Q:{qs[i][:50]}"
                )
        log.info(f"  [{self.name}] {ms:.0f}ms best={bs:.3f}")

        if bs >= MIN_MATCH and bsem >= SEM_FLOOR:
            return ans[bi], bs, bsem
        return None, bs, bsem


eng_en = Engine("EN")
eng_hi = Engine("HI")


# ============================================================
# DYNAMIC DB WATCHER � auto-reload when DB changes
# ============================================================

class DBWatcher:
    def __init__(self, interval_sec=30):
        self.interval  = interval_sec
        self._stop     = threading.Event()
        self._last_cnt = -1
        self._thread   = None

    def _check(self, mdb):
        try:
            cnt = mdb[COLLECTION].count_documents({})
            if self._last_cnt == -1:
                self._last_cnt = cnt; return
            if cnt != self._last_cnt:
                log.info(f"[DBWatcher] reloading...")
                eng_en.load(mdb, COLLECTION, EN_Q_KEY, EN_A_KEY)
                eng_hi.load(mdb, COLLECTION, HI_Q_KEY, HI_A_KEY)
                cache.clear(); self._last_cnt = cnt
        except Exception as e:
            log.error(f"[DBWatcher] {e}")

    def _run(self, mdb):
        log.info(f"[DBWatcher] started (every {self.interval}s)")
        while not self._stop.wait(self.interval):
            self._check(mdb)

    def start(self, mdb):
        self._check(mdb)
        self._thread = threading.Thread(target=self._run, args=(mdb,), daemon=True)
        self._thread.start()

    def stop(self): self._stop.set()


db_watcher = DBWatcher(interval_sec=30)


# ============================================================
# DUAL SEARCH
# ============================================================

def search(query, stt_lang: str = None):
    lang = lang_detect(query, stt_hint=stt_lang)
    t0   = time.time()

    log.info(f"\n{'='*55}")
    log.info(f"QUERY: {query}")
    log.info(f"LANG:  {lang}  (stt_hint={stt_lang})")

    qr   = eng_en.rewrite(query)
    tq, tl = TR.detect_translate(qr)
    tq   = eng_en.rewrite(tq)

    en_n, en_t = norm(tq), tok(tq)
    hi_n = norm(query)
    hi_t = [w for w in hi_n.split() if len(w) > 1]
    tr_n = norm(TR.translate(query))
    tr_t = tok(TR.translate(query))

    log.info(f"EN_Q: {en_n}")
    log.info(f"HI_Q: {hi_n}")

    def _best_en():
        a1, s1, _ = eng_en.search(en_n, en_t, boost=True)
        a2, s2, _ = eng_en.search(tr_n, tr_t, boost=True)
        if a1 and a2:
            return (a1, s1) if s1 >= s2 else (a2, s2)
        return (a1, s1) if a1 else (a2, s2)

    if lang == "hi":
        a_hi, s_hi, _ = eng_hi.search(hi_n, hi_t, boost=True)
        log.info(f"HI_RESULT: {s_hi:.3f} | {str(a_hi)[:50] if a_hi else '-'}")
        if a_hi and s_hi >= EARLY_EXIT:
            return a_hi, s_hi, "hindi_db", lang
        a_en, s_en = _best_en()
        log.info(f"EN_RESULT: {s_en:.3f} | {str(a_en)[:50] if a_en else '-'}")
        if a_hi and (not a_en or s_hi >= s_en):
            return a_hi, s_hi, "hindi_db", lang
        if a_en:
            return a_en, s_en, "english_db", lang

    elif lang == "hinglish":
        a_en, s_en = _best_en()
        log.info(f"EN_RESULT: {s_en:.3f} | {str(a_en)[:50] if a_en else '-'}")
        if a_en and s_en >= EARLY_EXIT:
            return a_en, s_en, "english_db", lang
        a_hi, s_hi, _ = eng_hi.search(hi_n, hi_t, boost=True)
        log.info(f"HI_RESULT: {s_hi:.3f} | {str(a_hi)[:50] if a_hi else '-'}")
        if a_en and a_hi:
            return (a_en, s_en, "english_db", lang) if s_en >= s_hi else (a_hi, s_hi, "hindi_db", lang)
        if a_en: return a_en, s_en, "english_db", lang
        if a_hi: return a_hi, s_hi, "hindi_db", lang

    else:  # English
        a_en, s_en, _ = eng_en.search(en_n, en_t, boost=True)
        log.info(f"EN_RESULT: {s_en:.3f} | {str(a_en)[:50] if a_en else '-'}")
        if a_en and s_en >= EARLY_EXIT:
            return a_en, s_en, "english_db", lang
        a_hi, s_hi, _ = eng_hi.search(hi_n, hi_t, boost=True)
        log.info(f"HI_RESULT: {s_hi:.3f} | {str(a_hi)[:50] if a_hi else '-'}")
        if a_en and (not a_hi or s_en >= s_hi):
            return a_en, s_en, "english_db", lang
        if a_hi:
            return a_hi, s_hi, "hindi_db", lang

    log.info(f"NO MATCH ({(time.time()-t0)*1000:.0f}ms)")
    return None, 0.0, "none", lang


# ============================================================
# APP INIT
# ============================================================

app = fastapi.FastAPI(title="JindalBot", version="9.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


mongo = db.db
log.info(f"MongoDB: {mongo.name}")
try:
    log.info(f"DB: {mongo[COLLECTION].count_documents({})} records")
except Exception as e:
    log.error(f"MongoDB: {e}")

try:
    subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log.info("ffmpeg OK")
except FileNotFoundError:
    log.error("ffmpeg MISSING")

log.info("Loading SentenceTransformer...")
embed_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
log.info(f"Loading Whisper '{STT_MODEL}'...")
stt_model = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")
log.info("Models ready")

eng_en.load(mongo, COLLECTION, EN_Q_KEY, EN_A_KEY)
eng_hi.load(mongo, COLLECTION, HI_Q_KEY, HI_A_KEY)
log.info(f"EN:{eng_en.ok}/{eng_en.n}  HI:{eng_hi.ok}/{eng_hi.n}")

db_watcher.start(mongo)


# ============================================================
# STT � smarter Hindi detection
# ============================================================

def transcribe(wav_path):
    try:
        t0 = time.time()
        segs, info = stt_model.transcribe(
            wav_path,
            beam_size=3,
            vad_filter=False,
            initial_prompt=STT_INITIAL_PROMPT,   # <-- KEY ADDITION
        )
        text = " ".join(s.text for s in segs).strip()
        det  = info.language
        prob = info.language_probability
        ms   = (time.time() - t0) * 1000
        log.info(f"STT lang={det} prob={prob:.2f} {ms:.0f}ms '{text[:80]}'")

        if not text or len(text) < 2:
            return "", "en"

        # Low confidence English → retry forced English (keep prompt here too)
        if det == "en" and prob < 0.5 and len(text) > 2:
            segs2, _ = stt_model.transcribe(
                wav_path,
                beam_size=3,
                language="en",
                vad_filter=False,
                initial_prompt=STT_INITIAL_PROMPT,   # <-- here too
            )
            t2 = " ".join(s.text for s in segs2).strip()
            if t2 and len(t2) >= len(text):
                text = t2
                log.info(f"STT en-retry: '{text[:80]}'")

        dev = len(re.findall(r"[\u0900-\u097F]", text))
        if dev >= 2 or det in ("hi", "ne", "mr"):
            return text, "hi"
        return text, "en"

    except Exception as e:
        log.error(f"STT error: {e}", exc_info=True)
        return "", "en"

# ============================================================
# TTS
# ============================================================

async def tts(text, lang_hint: str = None):
    try:
        lang  = lang_hint if lang_hint in ("hi", "hinglish") else lang_detect(text)
        voice = VOICE_HI if lang in ("hi", "hinglish") else VOICE_EN
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        await edge_tts.Communicate(text, voice).save(tmp_path)
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            with open(tmp_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            os.remove(tmp_path)
            log.info(f"TTS: voice={voice}")
            return b64
    except Exception as e:
        log.error(f"TTS: {e}")
    return None


# ============================================================
# RESPONSE
# ============================================================

def answer(query, sid="default", stt_lang: str = None):
    c = cache.get(query)
    if c:
        log.info(f"CACHE HIT: '{query[:40]}'")
        return c

    ans_val, score, src, lang = search(query, stt_lang=stt_lang)
    if ans_val:
        r = {"answer": ans_val, "confidence": score, "source": src, "language": lang}
    else:
        fb = fallback.get(query, lang)
        r  = {"answer": fb, "confidence": 0.0, "source": "fallback", "language": lang}
    cache.put(query, r)
    return r


# ============================================================
# ENDPOINTS
# ============================================================

class In(BaseModel):
    text: str
    session_id: Optional[str] = "default"


@app.get("/health")
async def health():
    return {
        "v": "9.0", "stt": STT_MODEL,
        "min": MIN_MATCH, "sem": SEM_FLOOR,
        "en": {"ok": eng_en.ok, "n": eng_en.n},
        "hi": {"ok": eng_hi.ok, "n": eng_hi.n},
        "cache": cache.stats(), "db_watcher": "running",
    }


@app.get("/debug")
async def debug():
    return {
        "en": {"n": eng_en.n, "ok": eng_en.ok, "q": eng_en.qs[:3]},
        "hi": {"n": eng_hi.n, "ok": eng_hi.ok, "q": eng_hi.qs[:3]},
    }


@app.post("/chat/")
@app.post("/chat")
async def chat(req: In):
    if not req.text.strip():
        raise HTTPException(400, "Empty")
    t0 = time.time()
    r  = answer(req.text, req.session_id)
    a  = await tts(r["answer"], lang_hint=r["language"])
    ms = (time.time() - t0) * 1000
    log.info(f"CHAT: {ms:.0f}ms")
    return {
        "text": r["answer"], "response": r["answer"],
        "confidence": r["confidence"], "source": r["source"],
        "detected_language": r["language"],
        "response_time_ms": round(ms),
        "audio_data": a,
        "video_url": None,
    }


@app.post("/chat_voice/")
@app.post("/chat_voice")
async def chat_voice(req: In):
    if not req.text.strip():
        raise HTTPException(400, "Empty")
    r = answer(req.text, req.session_id)
    a = await tts(r["answer"], lang_hint=r["language"])
    return {
        "text": r["answer"], "response": r["answer"],
        "confidence": r["confidence"], "source": r["source"],
        "detected_language": r["language"],
        "audio_data": a,
        "video_url": None,
    }


@app.post("/translate/")
@app.post("/translate")
async def translate(req: In):
    t, l = TR.detect_translate(req.text)
    return {"original": req.text, "translated": t, "language": l}


@app.post("/voice_chat/")
@app.post("/voice_chat")
async def voice_chat(file: UploadFile = File(...)):
    ts  = int(time.time())
    raw = os.path.join(TEMP_DIR, f"in_{ts}.webm")
    wav = os.path.join(TEMP_DIR, f"in_{ts}.wav")
    try:
        t0 = time.time()
        with open(raw, "wb") as f:
            f.write(await file.read())
        if os.path.getsize(raw) < 500:
            raise HTTPException(400, "Too short")
        subprocess.run(
            ["ffmpeg", "-i", raw, "-ar", "16000", "-ac", "1",
             "-c:a", "pcm_s16le", wav, "-y"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
        )

        ut, stt_lang = transcribe(wav)
        log.info(f"Voice transcribed: '{ut}' stt_lang={stt_lang}")

        if not ut:
            return {"text": "I didn't hear anything.", "response": "I didn't hear anything.",
                    "audio_data": None, "video_url": None}

        r  = answer(ut, stt_lang=stt_lang)
        a  = await tts(r["answer"], lang_hint=r["language"])
        ms = (time.time() - t0) * 1000
        log.info(f"VOICE: {ms:.0f}ms stt={stt_lang} lang={r['language']}")
        return {
            "user_text": ut,
            "text": r["answer"], "response": r["answer"],
            "detected_language": r["language"],
            "stt_language": stt_lang,
            "confidence": r["confidence"], "source": r["source"],
            "response_time_ms": round(ms),
            "audio_data": a,
            "video_url": None,
        }
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Voice error: {e}", exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        for p in [raw, wav]:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass


@app.post("/train/")
@app.post("/train")
async def train(file: UploadFile = File(...)):
    try:
        data = json.loads(await file.read())
        if not isinstance(data, list) or not data:
            raise HTTPException(400, "Need list")
        mongo[COLLECTION].delete_many({})
        mongo[COLLECTION].insert_many(data)
        eng_en.load(mongo, COLLECTION, EN_Q_KEY, EN_A_KEY)
        eng_hi.load(mongo, COLLECTION, HI_Q_KEY, HI_A_KEY)
        cache.clear()
        return {"status": "ok", "en": eng_en.n, "hi": eng_hi.n}
    except json.JSONDecodeError:
        raise HTTPException(400, "Bad JSON")
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))


@app.post("/train_hindi/")
@app.post("/train_hindi")
async def train_hindi(file: UploadFile = File(...)):
    try:
        data = json.loads(await file.read())
        if not isinstance(data, list) or not data:
            raise HTTPException(400, "Need list")
        mongo[COLLECTION].delete_many({})
        mongo[COLLECTION].insert_many(data)
        eng_en.load(mongo, COLLECTION, EN_Q_KEY, EN_A_KEY)
        eng_hi.load(mongo, COLLECTION, HI_Q_KEY, HI_A_KEY)
        cache.clear()
        return {"status": "ok", "en": eng_en.n, "hi": eng_hi.n}
    except json.JSONDecodeError:
        raise HTTPException(400, "Bad JSON")
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))


@app.post("/reload/")
@app.post("/reload")
async def reload():
    eng_en.load(mongo, COLLECTION, EN_Q_KEY, EN_A_KEY)
    eng_hi.load(mongo, COLLECTION, HI_Q_KEY, HI_A_KEY)
    cache.clear()
    return {"status": "ok", "en": eng_en.n, "hi": eng_hi.n}


@app.on_event("shutdown")
async def shutdown():
    db_watcher.stop()
    log.info("DBWatcher stopped")


@app.middleware("http")
async def log_req(req: Request, call_next):
    t0  = time.time()
    res = await call_next(req)
    p   = req.url.path
    if not p.startswith("/audio") and p not in ("/health", "/debug"):
        log.info(f"{req.method} {p} {res.status_code} {(time.time()-t0)*1000:.0f}ms")
    return res


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)