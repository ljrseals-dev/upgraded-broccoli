"""Speech synthesis, standalone.

Engines (auto order): espeak-ng, espeak (if installed on the machine), then the built-in formant synthesizer.
The built-in synthesizer is EXPERIMENTAL: rule-based letter-to-sound plus a formant source. It always works with only
numpy, but it is robotic and its intelligibility depends on spelling. Install espeak-ng for clearer speech."""
from __future__ import annotations
import re, shutil, subprocess
import numpy as np
from . import dsp
from .voice import synth_voiced, noise_band
from .wavio import parse_wav_bytes, to_mono

# vowels / approximants / nasals: (F1, F2, F3, base duration ms, gain)
VOICED = {
    "AA": (730, 1090, 2440, 120, 1.0), "AE": (660, 1720, 2410, 120, 1.0), "AH": (640, 1190, 2390, 90, 0.9),
    "AO": (570, 840, 2410, 120, 1.0), "EH": (530, 1840, 2480, 100, 0.95), "ER": (490, 1350, 1690, 120, 0.9),
    "IH": (390, 1990, 2550, 85, 0.9), "IY": (270, 2290, 3010, 110, 0.9), "UH": (440, 1020, 2240, 85, 0.9),
    "UW": (300, 870, 2240, 110, 0.9),
    "W": (300, 610, 2200, 70, 0.7), "Y": (270, 2290, 3010, 60, 0.7), "R": (310, 1060, 1380, 80, 0.75),
    "L": (360, 1300, 2500, 75, 0.75),
    "M": (250, 1100, 2200, 80, 0.5), "N": (250, 1700, 2600, 75, 0.5), "NG": (250, 2000, 2700, 85, 0.5),
}
DIPH = {"AY": ("AA", "IY"), "EY": ("EH", "IY"), "OW": ("AO", "UW"), "AW": ("AA", "UW"), "OY": ("AO", "IY")}
# fricatives: (noise lo, noise hi, duration ms, gain, voiced)
FRIC = {
    "F": (3000, 9000, 90, 0.18, False), "V": (3000, 9000, 70, 0.14, True), "TH": (2000, 8000, 85, 0.14, False),
    "DH": (2000, 8000, 60, 0.12, True), "S": (4500, 10000, 100, 0.42, False), "Z": (4500, 10000, 80, 0.3, True),
    "SH": (2000, 6000, 105, 0.42, False), "ZH": (2000, 6000, 85, 0.3, True), "HH": (400, 4500, 70, 0.14, False),
}
# stops: (burst lo, burst hi, voiced)
STOP = {"P": (400, 1800, False), "B": (400, 1800, True), "T": (3000, 8000, False), "D": (3000, 8000, True),
        "K": (1500, 4500, False), "G": (1500, 4500, True)}

WORDS = {
    "the": "DH AH", "a": "AH", "an": "AE N", "to": "T UW", "of": "AH V", "and": "AE N D", "is": "IH Z",
    "you": "Y UW", "your": "Y AO R", "are": "AA R", "one": "W AH N", "two": "T UW", "three": "TH R IY",
    "four": "F AO R", "five": "F AY V", "six": "S IH K S", "seven": "S EH V AH N", "eight": "EY T",
    "nine": "N AY N", "zero": "Z IY R OW", "ten": "T EH N", "hello": "HH AH L OW", "world": "W ER L D",
    "voice": "V OY S", "magic": "M AE JH IH K", "studio": "S T UW D IY OW", "this": "DH IH S", "that": "DH AE T",
    "with": "W IH TH", "for": "F AO R", "from": "F R AH M", "have": "HH AE V", "was": "W AH Z", "what": "W AH T",
    "who": "HH UW", "how": "HH AW", "do": "D UW", "does": "D AH Z", "said": "S EH D", "come": "K AH M",
    "some": "S AH M", "could": "K UH D", "would": "W UH D", "should": "SH UH D", "there": "DH EH R",
    "their": "DH EH R", "where": "W EH R", "here": "HH IY R", "love": "L AH V", "my": "M AY", "me": "M IY",
    "we": "W IY", "he": "HH IY", "she": "SH IY", "it": "IH T", "i": "AY", "be": "B IY", "yes": "Y EH S",
    "no": "N OW", "not": "N AA T", "all": "AO L", "audio": "AO D IY OW", "music": "M Y UW Z IH K",
    "sound": "S AW N D", "speak": "S P IY K", "create": "K R IY EY T", "control": "K AH N T R OW L",
    "test": "T EH S T", "testing": "T EH S T IH NG", "words": "W ER D Z", "computer": "K AH M P Y UW T ER",
}
LONG = {"a": "EY", "e": "IY", "i": "AY", "o": "OW", "u": "UW"}
SHORT = {"a": "AE", "e": "EH", "i": "IH", "o": "AA", "u": "AH"}
DIGRAPHS = [("tch", "CH"), ("igh", "AY"), ("th", "TH"), ("sh", "SH"), ("ch", "CH"), ("ph", "F"), ("ck", "K"),
            ("ng", "NG"), ("wh", "W"), ("qu", "K W"), ("oo", "UW"), ("ee", "IY"), ("ea", "IY"), ("ai", "EY"),
            ("ay", "EY"), ("oa", "OW"), ("ou", "AW"), ("ow", "OW"), ("oi", "OY"), ("oy", "OY"), ("ar", "AA R"),
            ("or", "AO R"), ("er", "ER"), ("ir", "ER"), ("ur", "ER")]
CONS = {"b": "B", "d": "D", "f": "F", "h": "HH", "j": "JH", "k": "K", "l": "L", "m": "M", "n": "N", "p": "P",
        "r": "R", "t": "T", "v": "V", "w": "W", "z": "Z", "x": "K S", "q": "K"}
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
VOWEL_LETTERS = set("aeiouy")


def _expand_numbers(text: str) -> str:
    def conv(m):
        s = m.group(0)
        return " ".join(ONES[int(c)] for c in s) if len(s) <= 6 else " ".join(ONES[int(c)] for c in s[:6])
    return re.sub(r"\d+", lambda m: " " + conv(m) + " ", text)


def g2p_word(word: str) -> list:
    w = re.sub(r"[^a-z']", "", word.lower())
    if not w:
        return []
    if w in WORDS:
        return WORDS[w].split()
    out, i, n = [], 0, len(w)
    while i < n:
        ch = w[i]
        hit = None
        for g, ph in DIGRAPHS:
            if w.startswith(g, i):
                hit = (g, ph)
                break
        if hit:
            out += hit[1].split()
            i += len(hit[0])
            continue
        nxt = w[i + 1] if i + 1 < n else ""
        if ch == "e" and i == n - 1 and i >= 2:
            i += 1                                     # silent final e
            continue
        if ch in "aeiou":
            magic_e = (i + 2 == n - 1 and w[-1] == "e" and nxt not in VOWEL_LETTERS) or \
                      (i + 2 < n and w[i + 1] not in VOWEL_LETTERS and w[i + 2] == "e" and i + 3 == n)
            out.append(LONG[ch] if magic_e else SHORT[ch])
        elif ch == "y":
            out.append("IY" if i == n - 1 and n > 2 else ("AY" if i == n - 1 else "Y"))
        elif ch == "c":
            out.append("S" if nxt in ("e", "i", "y") else "K")
        elif ch == "g":
            out.append("JH" if nxt in ("e", "i", "y") and i > 0 else "G")
        elif ch == "s":
            prev = w[i - 1] if i else ""
            out.append("Z" if prev in VOWEL_LETTERS and nxt in VOWEL_LETTERS else "S")
        elif ch in CONS:
            out += CONS[ch].split()
        i += 1
    return out


def text_to_phones(text: str):
    """-> list of tokens: phoneme strings, or ('PAUSE', ms). Marks word starts via ('WORD',) and question via ('Q',)."""
    text = _expand_numbers(text)
    toks = []
    for m in re.finditer(r"[A-Za-z']+|[,.;:!?]", text):
        s = m.group(0)
        if s in ",;:":
            toks.append(("PAUSE", 220))
        elif s in ".!":
            toks.append(("PAUSE", 380))
        elif s == "?":
            toks.append(("PAUSE", 380)); toks.append(("Q",))
        else:
            ph = g2p_word(s)
            if ph:
                toks.append(("WORD",)); toks += ph; toks.append(("PAUSE", 55))
    return toks


class BuiltinSpeech:
    name = "builtin"

    def __init__(self, sr=22050, pitch_hz=118.0, rate=1.0, seed=0, formant_scale=1.0):
        self.sr, self.f0, self.rate, self.fs = int(sr), float(pitch_hz), max(0.4, float(rate)), float(formant_scale)
        self.rng = np.random.default_rng(seed)

    def _dur(self, ms):
        return max(int(ms / 1000.0 / self.rate * self.sr), 8)

    def _env(self, n, a=0.012, r=0.02):
        e = np.ones(n, np.float32)
        a, r = min(int(a * self.sr), n // 2), min(int(r * self.sr), n // 2)
        if a:
            e[:a] = np.linspace(0, 1, a, dtype=np.float32)
        if r:
            e[n - r:] = np.linspace(1, 0, r, dtype=np.float32)
        return e

    def _voiced(self, f1, f2, f3, n, f0c, gain, tilt=1.3):
        y = synth_voiced(f0c[:n], self.sr, (f1, f2, f3), tilt=tilt, formant_scale=self.fs)
        pk = float(np.max(np.abs(y))) or 1.0
        return y / pk * gain * self._env(n)

    def synth(self, text: str) -> np.ndarray:
        toks = text_to_phones(text)
        # flatten diphthongs
        flat = []
        for t in toks:
            if isinstance(t, str) and t in DIPH:
                a, b = DIPH[t]
                flat += [(a, 0.6), (b, 0.6)]
            elif isinstance(t, str):
                flat.append((t, 1.0))
            else:
                flat.append(t)
        question = any(t == ("Q",) for t in flat)
        n_ph = sum(1 for t in flat if isinstance(t[0], str) and t[0] != "WORD" and t[0] != "PAUSE" and t[0] != "Q")
        segs, idx, stress_next = [], 0, False
        for t in flat:
            if t[0] in ("WORD",):
                stress_next = True
                continue
            if t[0] == "Q":
                continue
            if t[0] == "PAUSE":
                segs.append(np.zeros(self._dur(t[1]), np.float32))
                continue
            ph, dscale = t
            prog = idx / max(n_ph - 1, 1)
            f0 = self.f0 * (1.0 - 0.12 * prog)                  # declination
            if question:
                f0 *= 1.0 + 0.35 * max(0.0, prog - 0.7) / 0.3
            if stress_next and ph in VOICED and ph not in ("M", "N", "NG", "W", "Y", "R", "L"):
                f0 *= 1.07
                stress_next = False
            idx += 1
            if ph in VOICED:
                f1, f2, f3, ms, g = VOICED[ph]
                n = self._dur(ms * dscale)
                segs.append(self._voiced(f1, f2, f3, n, np.full(n, f0), g))
            elif ph in FRIC:
                lo, hi, ms, g, voiced = FRIC[ph]
                n = self._dur(ms)
                y = noise_band(n, self.sr, lo, hi, self.rng)
                y = y / (float(np.max(np.abs(y))) or 1.0) * g * self._env(n, 0.015, 0.02)
                if voiced:
                    y = y + self._voiced(250, 1300, 2400, n, np.full(n, f0), 0.35)
                segs.append(y.astype(np.float32))
            elif ph in STOP or ph in ("CH", "JH"):
                voiced = ph in ("B", "D", "G", "JH")
                lo, hi = (2000, 6000) if ph in ("CH", "JH") else STOP[ph][:2]
                clo = self._dur(55)
                closure = self._voiced(250, 1300, 2400, clo, np.full(clo, f0), 0.12) if voiced else np.zeros(clo, np.float32)
                nb = self._dur(14 if ph not in ("CH", "JH") else 70)
                b = noise_band(nb, self.sr, lo, hi, self.rng)
                b = b / (float(np.max(np.abs(b))) or 1.0) * (0.35 if ph not in ("CH", "JH") else 0.38) * self._env(nb, 0.002, 0.008)
                parts = [closure, b.astype(np.float32)]
                if not voiced:
                    na = self._dur(22)
                    asp = noise_band(na, self.sr, 800, 4000, self.rng)
                    parts.append((asp / (float(np.max(np.abs(asp))) or 1.0) * 0.1 * self._env(na)).astype(np.float32))
                segs.append(np.concatenate(parts))
        if not segs:
            return np.zeros(int(0.1 * self.sr), np.float32)
        xf = max(int(0.006 * self.sr), 1)
        out = segs[0]
        for s in segs[1:]:
            if len(s) > xf and len(out) > xf:
                fade = np.linspace(0, 1, xf, dtype=np.float32)
                out = np.concatenate([out[:-xf], out[-xf:] * (1 - fade) + s[:xf] * fade, s[xf:]])
            else:
                out = np.concatenate([out, s])
        return dsp.peak_normalize(out, -3.0)


def _espeak_exe():
    return shutil.which("espeak-ng") or shutil.which("espeak")


def engines():
    return [{"name": "espeak", "available": _espeak_exe() is not None, "note": "needs espeak-ng or espeak installed"},
            {"name": "builtin", "available": True, "note": "experimental formant synthesizer (numpy only)"}]


def synth_espeak(text: str, voice="en", rate_wpm=175, pitch=50):
    if not re.fullmatch(r"[A-Za-z0-9_+\-]{1,32}", str(voice)):
        raise ValueError("bad voice name")
    exe = _espeak_exe()
    if exe is None:
        raise RuntimeError("espeak-ng/espeak not found on PATH")
    rate_wpm, pitch = int(max(80, min(450, rate_wpm))), int(max(0, min(99, pitch)))
    r = subprocess.run([exe, "--stdout", "-v", str(voice), "-s", str(rate_wpm), "-p", str(pitch)],
                       input=text.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError("espeak failed: " + r.stderr.decode("utf-8", "replace")[:200])
    a, sr = parse_wav_bytes(r.stdout)
    return to_mono(a).astype(np.float32), sr


def synthesize(text: str, engine: str = "auto", sr: int = 22050, rate: float = 1.0, pitch_hz: float = 118.0,
               voice: str = "en", seed: int = 0, character: bool = False, char_kwargs: dict | None = None):
    """-> (audio float32 mono, sample_rate, engine_used)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("text is empty")
    if len(text) > 5000:
        raise ValueError("text too long (max 5000 characters)")
    used, audio, out_sr = None, None, sr
    if engine in ("auto", "espeak"):
        try:
            audio, out_sr = synth_espeak(text, voice=voice, rate_wpm=int(175 * rate))
            used = "espeak"
        except Exception:
            if engine == "espeak":
                raise
    if audio is None:
        if engine not in ("auto", "builtin"):
            raise ValueError(f"unknown engine {engine!r}; choose auto, espeak, or builtin")
        audio, out_sr, used = BuiltinSpeech(sr, pitch_hz, rate, seed).synth(text), sr, "builtin"
    if character:
        audio = dsp.character(audio, out_sr, **(char_kwargs or {}))
    return dsp.finalize(audio, -1.0), out_sr, used
