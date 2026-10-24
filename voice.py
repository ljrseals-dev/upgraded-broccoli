"""Harmonic formant voice synthesizer (numpy only) and the replaceable vocal renderer interface.

The built-in renderer is a PREVIEW synthesizer: additive harmonics shaped by formant resonances. It is an
approximation of vocal timbre, not a reconstruction of anyone's voice."""
from __future__ import annotations
import random
from abc import ABC, abstractmethod
import numpy as np
from . import dsp
from .model import Track, Note

# (F1, F2, F3) Hz, adult-ish averages
VOWELS = {
    "a": (800, 1200, 2600), "e": (500, 1900, 2600), "i": (300, 2300, 3000),
    "o": (500, 900, 2500), "u": (320, 800, 2400),
}
BW = (90.0, 110.0, 160.0)
GAIN = (1.0, 0.6, 0.3)


def vowel_of(lyric: str) -> str:
    for ch in lyric.lower():
        if ch in VOWELS:
            return ch
    return "a"


def formant_envelope(freqs, formants, bw=BW, gain=GAIN, widen=2.5):
    f = np.asarray(freqs, np.float64)
    env = np.zeros_like(f)
    for F, b, g in zip(formants, bw, gain):
        env += g / (1.0 + ((f - F) / (b * widen)) ** 2)
    return env


def synth_voiced(f0, sr: int, formants, tilt: float = 1.2, formant_scale: float = 1.0, max_harm: int = 60,
                 bw=BW, gain=GAIN) -> np.ndarray:
    """Additive voiced source. f0: per-sample fundamental in Hz (array), formants: (F1,F2,F3) in Hz."""
    f0 = np.asarray(f0, np.float64)
    if f0.size == 0:
        return np.zeros(0, np.float32)
    base = float(np.median(f0))
    fmax = float(np.max(f0))
    H = int(max(1, min(max_harm, (0.45 * sr) / max(fmax, 20.0))))
    phase = 2.0 * np.pi * np.cumsum(f0) / sr
    fm = [F * formant_scale for F in formants]
    amps = formant_envelope(np.arange(1, H + 1) * base, fm, bw, gain) / np.arange(1, H + 1) ** tilt
    # Chebyshev recurrence: sin((n+1)x) = 2 cos(x) sin(nx) - sin((n-1)x)  (one sin and one cos for all harmonics)
    c2 = 2.0 * np.cos(phase)
    s_prev = np.zeros(len(f0), np.float64)
    s_cur = np.sin(phase)
    y = amps[0] * s_cur
    for n in range(2, H + 1):
        s_next = c2 * s_cur - s_prev
        y += amps[n - 1] * s_next
        s_prev, s_cur = s_cur, s_next
    return y.astype(np.float32)


def noise_band(n: int, sr: int, lo: float, hi: float, rng) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, np.float32)
    x = rng.standard_normal(n).astype(np.float32)
    return dsp.bandpass(x, sr, lo, min(hi, 0.45 * sr))


class VocalRenderer(ABC):
    @abstractmethod
    def render_track(self, track: Track, sr: int, tempo: float) -> np.ndarray:
        """Return a mono float32 signal for the whole track (timeline starts at 0 s)."""


class ParametricRenderer(VocalRenderer):
    label = "preview formant synth"

    def __init__(self, profile: dict | None = None):
        self.profile = profile or {}

    def render_track(self, track: Track, sr: int, tempo: float) -> np.ndarray:
        s = track.settings
        rng = random.Random(s.seed)
        nrng = np.random.default_rng(s.seed)
        fscale = float(self.profile.get("formant_scale", 1.0))
        vrate = float(self.profile.get("vibrato_rate_hz", s.vibrato_rate_hz))
        vdepth = float(self.profile.get("vibrato_depth_cents", s.vibrato_depth_cents))
        breath = float(self.profile.get("breathiness", s.breath))
        if not track.notes:
            return np.zeros(0, np.float32)
        total = max(n.end for n in track.notes) + 1.0 + abs(s.delay_ms) / 1000
        out = np.zeros(int(total * sr) + sr, np.float32)
        for n in track.notes:
            t0 = max(0.0, n.start + s.delay_ms / 1000 + rng.gauss(0, s.timing_ms / 1000))
            cents = rng.gauss(0, s.tuning_cents)
            sig = self._note(n, sr, vrate, vdepth, breath, cents, fscale, s.brightness, nrng, rng)
            i = int(t0 * sr)
            if i >= len(out) or len(sig) == 0:
                continue
            m = min(len(sig), len(out) - i)
            out[i:i + m] += sig[:m]
        return out

    def _note(self, n: Note, sr, vrate, vdepth, breath, cents, fscale, bright, nrng, rng):
        N = max(int((n.dur + 0.08) * sr), 16)
        t = np.arange(N) / sr
        onset = np.clip((t - 0.15) / 0.25, 0, 1)
        vib = vdepth * onset * np.sin(2 * np.pi * vrate * t + rng.random() * 6.28)
        f0 = 440.0 * 2 ** ((n.pitch - 69 + (cents + vib) / 100.0) / 12.0)
        tilt = 0.7 + 1.0 * (1.0 - bright)                       # brighter = flatter spectral tilt
        sig = synth_voiced(f0, sr, VOWELS[vowel_of(n.lyric)], tilt=tilt, formant_scale=fscale)
        peak = float(np.max(np.abs(sig))) or 1.0
        sig = sig / peak
        if breath > 0:
            sig = sig + noise_band(N, sr, 2500, 5500, nrng) * breath * 2.5
        a, r = min(int(0.03 * sr), N // 2), min(int(0.08 * sr), N // 2)
        env = np.ones(N, np.float32)
        env[:a] = np.linspace(0, 1, a, dtype=np.float32)
        env[N - r:] = np.linspace(1, 0, r, dtype=np.float32)
        sig = sig * env
        peak = float(np.max(np.abs(sig))) or 1.0
        return (sig / peak * (0.15 + 0.5 * n.vel)).astype(np.float32)
