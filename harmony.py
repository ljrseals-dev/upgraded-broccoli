"""Scale-aware harmony engine: lead melody -> multi-part arrangements."""
from __future__ import annotations
import copy
from .model import Note, Track, TrackSettings

SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],          # natural minor
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
}

# interval name -> (diatonic scale-degree steps, or None) / fixed semitones
INTERVALS = {
    "unison": ("deg", 0), "3rd_up": ("deg", 2), "5th_up": ("deg", 4), "6th_up": ("deg", 5),
    "3rd_low": ("deg", -2), "4th_low": ("deg", -3), "5th_low": ("deg", -4), "6th_low": ("deg", -5),
    "octave_up": ("semi", 12), "octave_low": ("semi", -12),
}

# part-count presets: (track name, role, interval, level_db, pan)
PRESETS = {
    2: [("Lead", "lead", "unison", 0, 0.0), ("Harm High", "harmony", "3rd_up", -4, 0.35)],
    3: [("Lead", "lead", "unison", 0, 0.0), ("Harm High", "harmony", "3rd_up", -4, -0.4),
        ("Harm Low", "harmony", "3rd_low", -5, 0.4)],
    4: [("Lead", "lead", "unison", 0, 0.0), ("Harm High", "harmony", "3rd_up", -4, -0.35),
        ("Harm Mid", "harmony", "5th_up", -5, 0.35), ("Harm Low", "harmony", "3rd_low", -5, 0.0)],
    6: [("Lead", "lead", "unison", 0, 0.0), ("Harm High", "harmony", "3rd_up", -4, -0.3),
        ("Harm Mid", "harmony", "5th_up", -5, 0.3), ("Harm Low", "harmony", "3rd_low", -5, -0.6),
        ("Octave Up", "harmony", "octave_up", -9, 0.6), ("Octave Low", "harmony", "octave_low", -8, 0.0)],
}


class Key:
    def __init__(self, tonic: int = 0, mode: str = "major"):
        if mode not in SCALES:
            raise ValueError(f"unknown mode {mode!r}; choose from {sorted(SCALES)}")
        self.tonic, self.mode, self.steps = tonic % 12, mode, SCALES[mode]

    def _nearest_degree(self, pitch: float) -> tuple[int, float]:
        """Return (absolute scale index, chromatic offset in semitones) for a pitch."""
        rel = pitch - self.tonic
        octave = int(rel // 12)
        pc = rel - 12 * octave
        best = min(range(7), key=lambda i: abs(self.steps[i] - pc))
        return octave * 7 + best, pc - self.steps[best]

    def degree_pitch(self, idx: int) -> float:
        octave, d = divmod(idx, 7)
        return self.tonic + 12 * octave + self.steps[d]

    def shift(self, pitch: float, degrees: int) -> float:
        """Move `degrees` scale steps, keeping any chromatic inflection/detune."""
        idx, off = self._nearest_degree(pitch)
        return self.degree_pitch(idx + degrees) + off

    def in_scale(self, pitch: float, tol: float = 0.01) -> bool:
        return abs(self._nearest_degree(pitch)[1]) <= tol


def harmonize(notes: list[Note], key: Key, interval: str, lo: float = 40, hi: float = 84) -> list[Note]:
    kind, amt = INTERVALS[interval]
    out = []
    for n in notes:
        p = key.shift(n.pitch, amt) if kind == "deg" else n.pitch + amt
        while p > hi:
            p -= 12
        while p < lo:
            p += 12
        out.append(Note(p, n.start, n.dur, n.vel, n.lyric))
    return out


def build_stack(lead: list[Note], key: Key, parts: int = 4, stereo_width: float = 0.65) -> list[Track]:
    if parts not in PRESETS:
        raise ValueError(f"parts must be one of {sorted(PRESETS)}")
    tracks = []
    for i, (name, role, iv, db, pan) in enumerate(PRESETS[parts]):
        notes = copy.deepcopy(lead) if iv == "unison" else harmonize(lead, key, iv)
        s = TrackSettings(level_db=db, pan=pan * stereo_width / 0.65 if role != "lead" else 0.0, seed=i + 1,
                          timing_ms=0.0 if role == "lead" else 12.0,
                          tuning_cents=0.0 if role == "lead" else 8.0)
        s.pan = max(-1.0, min(1.0, s.pan))
        tracks.append(Track(name, role, notes, s, iv))
    return tracks


def build_doubles(lead_track: Track, count: int = 2, spread: float = 0.8) -> list[Track]:
    """Doubles: same line, loosely re-performed (timing/tuning wobble), panned wide."""
    out = []
    for i in range(count):
        side = -1 if i % 2 == 0 else 1
        s = copy.deepcopy(lead_track.settings)
        s.level_db, s.pan, s.seed = -6.0, side * spread, 100 + i
        s.timing_ms, s.tuning_cents, s.reverb = 18.0, 12.0, 0.15
        out.append(Track(f"Double {i+1}", "double", copy.deepcopy(lead_track.notes), s, "unison"))
    return out


def build_backgrounds(lead: list[Note], key: Key, style: str = "airy", min_len: float = 0.6) -> list[Track]:
    """Backing vocals: sustained pads on long lead notes + call-and-response ad-libs on phrase ends."""
    if style not in ("airy", "soulful", "rhythmic", "powerful"):
        raise ValueError("style must be airy|soulful|rhythmic|powerful")
    cfg = {"airy": (0.45, 0.10, "oo", -10), "soulful": (0.2, 0.05, "ah", -8),
           "rhythmic": (0.1, 0.03, "oh", -9), "powerful": (0.05, 0.02, "ah", -6)}[style]
    breath, vib_scale, vowel, db = cfg
    pads = [Note(key.shift(n.pitch, 4), n.start, max(n.dur, min_len), n.vel * 0.6, vowel)
            for n in lead if n.dur >= min_len] if style != "rhythmic" else \
           [Note(key.shift(n.pitch, 2), n.start, n.dur * 0.5, n.vel * 0.7, vowel) for n in lead]
    # ad-libs: echo of the last note of each phrase (gap > 0.5 s), answering after the phrase
    adlibs, ordered = [], sorted(lead, key=lambda n: n.start)
    for a, b in zip(ordered, ordered[1:] + [None]):
        if b is None or b.start - a.end > 0.5:
            adlibs.append(Note(key.shift(a.pitch, 2) + 12, a.end + 0.15, min(0.5, a.dur), 0.7, "oh"))
    pad_s = TrackSettings(level_db=db, pan=-0.5, reverb=0.7, breath=breath, seed=200,
                          vibrato_depth_cents=15 * (1 + vib_scale), timing_ms=15, tuning_cents=10)
    ad_s = TrackSettings(level_db=db + 1, pan=0.55, reverb=0.5, echo=0.4, seed=201, timing_ms=10)
    tracks = [Track("BG Pad", "background", pads, pad_s, "5th_up")]
    if adlibs:
        tracks.append(Track("Ad-libs", "adlib", adlibs, ad_s, "response"))
    return tracks


PITCH_CLASSES = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7,
                 "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}


def parse_key(text) -> int:
    t = str(text).strip().upper()
    if t in PITCH_CLASSES:
        return PITCH_CLASSES[t]
    if t.isdigit() and 0 <= int(t) < 12:
        return int(t)
    raise ValueError(f"unknown key {text!r}; use C, C#, Db, ... or 0-11")


def build_session(lead, parts=4, key="C", mode="major", doubles=0, bg="airy", tempo=100.0):
    """Lead melody -> a complete Session (stack + optional doubles + backgrounds)."""
    from .model import Session
    if parts not in (2, 3, 4, 6):
        raise ValueError("parts must be 2, 3, 4 or 6")
    tonic = parse_key(key)
    k = Key(tonic, mode)
    s = Session(tempo=float(tempo), key_tonic=tonic, key_mode=mode)
    s.tracks = build_stack(lead, k, parts)
    if doubles:
        s.tracks += build_doubles(s.track("Lead"), int(doubles))
    if bg and bg != "none":
        s.tracks += build_backgrounds(lead, k, bg)
    s.validate()
    return s
