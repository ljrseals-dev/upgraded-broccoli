"""Renderer registry. Built-in: 'formant' (preview synth) and 'sample' (your own recording, pitched per note;
requires a consent record for the file)."""
from __future__ import annotations
import numpy as np
from . import dsp
from .consent import require_consent
from .model import Track
from .pitch import estimate_f0, hz_to_midi
from .voice import VocalRenderer, ParametricRenderer
from .wavio import read_wav, to_mono


class SampleRenderer(VocalRenderer):
    """Plays a sustained vowel recording of YOUR voice at the pitch of each note (resampling, with a seamless loop
    for notes longer than the sample). Works best within about +/-7 semitones of the recorded pitch; larger shifts
    change the formants (chipmunk or giant effect). The sample must have a consent record."""
    label = "sample renderer (your recording)"

    def __init__(self, path, root_midi: float | None = None, consents_path=None):
        self.record = require_consent(path, consents_path)
        y, sr0 = read_wav(path)
        self.sample = to_mono(y).astype(np.float32)
        self.sr0 = int(sr0)
        if len(self.sample) < int(0.25 * sr0):
            raise ValueError("sample must be at least 0.25 s long")
        if root_midi is None:
            f = estimate_f0(self.sample, self.sr0)
            if f <= 0:
                raise ValueError("could not detect the pitch of the sample; pass root_midi explicitly")
            root_midi = float(hz_to_midi(f))
        self.root = float(root_midi)
        L = len(self.sample)
        self.ls, self.le = int(0.3 * L), int(0.8 * L)
        xf = min(int(0.02 * self.sr0), self.ls, (self.le - self.ls) // 2)
        block = self.sample[self.ls:self.le].copy()
        if xf > 8:                                   # crossfade the loop tail into the audio before the loop start
            fade = np.linspace(0, 1, xf, dtype=np.float32)
            block[-xf:] = block[-xf:] * (1 - fade) + self.sample[self.ls - xf:self.ls] * fade
        self.block = block

    def _source(self, need: int) -> np.ndarray:
        parts, have = [self.sample[:self.ls]], self.ls
        while have < need:
            parts.append(self.block)
            have += len(self.block)
        return np.concatenate(parts)[:max(need, 1)]

    def render_track(self, track: Track, sr: int, tempo: float) -> np.ndarray:
        s = track.settings
        if not track.notes:
            return np.zeros(0, np.float32)
        total = max(n.end for n in track.notes) + 1.0 + abs(s.delay_ms) / 1000
        out = np.zeros(int(total * sr) + sr, np.float32)
        for n in track.notes:
            ratio = 2.0 ** ((n.pitch - self.root) / 12.0) * (self.sr0 / sr)
            nout = max(int((n.dur + 0.08) * sr), 16)
            src = self._source(int(nout * ratio) + 8)
            seg = dsp.resample(src, ratio, self.sr0)[:nout] if ratio != 1.0 else src[:nout]
            if len(seg) < nout:
                seg = np.pad(seg, (0, nout - len(seg)))
            a, r = min(int(0.02 * sr), nout // 2), min(int(0.08 * sr), nout // 2)
            env = np.ones(nout, np.float32)
            env[:a] = np.linspace(0, 1, a, dtype=np.float32)
            env[nout - r:] = np.linspace(1, 0, r, dtype=np.float32)
            seg = seg * env
            pk = float(np.max(np.abs(seg))) or 1.0
            seg = seg / pk * (0.15 + 0.5 * n.vel)
            i = int(max(0.0, n.start + s.delay_ms / 1000) * sr)
            if i >= len(out):
                continue
            m = min(len(seg), len(out) - i)
            out[i:i + m] += seg[:m]
        return out


def get_renderer(name: str = "formant", **kw) -> VocalRenderer:
    if name in ("formant", "builtin", "preview"):
        return ParametricRenderer(kw.get("profile"))
    if name == "sample":
        if "path" not in kw:
            raise ValueError("sample renderer needs path=<wav>")
        return SampleRenderer(kw["path"], kw.get("root_midi"), kw.get("consents_path"))
    raise ValueError(f"unknown renderer {name!r}; choose formant or sample")


def list_renderers():
    return [{"name": "formant", "needs_consent": False, "note": "preview synth, always available"},
            {"name": "sample", "needs_consent": True, "note": "your own recording, pitched per note"}]
