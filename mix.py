"""Vocal mixer (numpy only): per-track EQ, de-ess, compression, saturation, level, pan, reverb and echo sends;
stems + mix export (16/24-bit WAV). Each track is rendered and processed exactly once per export."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
from . import dsp
from .model import Session, Track
from .voice import VocalRenderer, ParametricRenderer
from .wavio import write_wav


def db(x: float) -> float:
    return 10 ** (x / 20)


def _render_job(args):
    renderer, track, sr, tempo = args
    return renderer.render_track(track, sr, tempo)


class VocalMixer:
    def __init__(self, session: Session, renderer: VocalRenderer | None = None, workers: int = 0):
        self.s, self.renderer, self.workers = session, renderer or ParametricRenderer(), int(workers)
        self._ir = dsp.reverb_ir(session.sr)
        self._cache: dict = {}

    # -- rendering ---------------------------------------------------------------------------------------------
    def _key(self, t: Track):
        return repr((t.notes, t.settings, type(self.renderer).__name__, repr(getattr(self.renderer, "__dict__", {}))))

    def _dry_many(self, tracks):
        todo = [t for t in tracks if t.name not in self._cache or self._cache[t.name][0] != self._key(t)]
        if self.workers > 1 and len(todo) > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=self.workers) as ex:
                res = list(ex.map(_render_job, [(self.renderer, t, self.s.sr, self.s.tempo) for t in todo]))
        else:
            res = [self.renderer.render_track(t, self.s.sr, self.s.tempo) for t in todo]
        for t, y in zip(todo, res):
            self._cache[t.name] = (self._key(t), y)
        return {t.name: self._cache[t.name][1] for t in tracks}

    def _process(self, t: Track, y: np.ndarray) -> np.ndarray:
        sr, st = self.s.sr, t.settings
        if len(y) == 0:
            return np.zeros((0, 2), np.float32)
        y = dsp.eq_vocal(y, sr)
        y = dsp.soft_saturate(dsp.compressor(dsp.deess(y, sr), -12.0, 3.0), 2.3)   # threshold -12 dB, 3:1, 2.3 dB drive
        y = y * db(st.level_db)
        wet = np.zeros(len(y) + len(self._ir), np.float32)
        wet[:len(y)] += y
        if st.reverb > 0:
            rv = dsp.fft_convolve(y, self._ir)
            wet[:len(rv)] += st.reverb * 0.6 * rv[:len(wet)]
        if st.echo > 0:
            ec = dsp.echo(y, sr, st.echo_ms)
            if len(ec) > len(wet):
                wet = np.pad(wet, (0, len(ec) - len(wet)))
            wet[:len(ec)] += st.echo * ec
        return dsp.pan_stereo(wet, st.pan)

    def process_track(self, t: Track) -> np.ndarray:
        return self._process(t, self._dry_many([t])[t.name])

    def process_all(self) -> dict:
        self.s.validate()
        dry = self._dry_many(self.s.tracks)
        return {t.name: self._process(t, dry[t.name]) for t in self.s.tracks}

    # -- mixing ------------------------------------------------------------------------------------------------
    @staticmethod
    def _sum(stereos):
        stereos = [x for x in stereos if len(x)]
        if not stereos:
            return np.zeros((1, 2), np.float32)
        n = max(len(x) for x in stereos)
        mix = np.zeros((n, 2), np.float32)
        for x in stereos:
            mix[:len(x)] += x
        return mix

    @staticmethod
    def _finish(m, normalize: bool):
        m = np.tanh(m * 1.1) / np.tanh(1.1)
        if normalize:
            peak = float(np.max(np.abs(m))) if m.size else 0.0
            if peak > 1e-9:
                m = m / peak * db(-1.0)
        return m.astype(np.float32)

    def mix(self, normalize: bool = True) -> np.ndarray:
        proc = self.process_all()
        active = {t.name for t in self.s.active_tracks()}
        return self._finish(self._sum([proc[n] for n in proc if n in active]), normalize)

    @staticmethod
    def safe_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "track"

    def export(self, outdir, stems: bool = True, normalize: bool = True, bits: int = 16) -> dict:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        proc = self.process_all()
        active = {t.name for t in self.s.active_tracks()}
        full = self._finish(self._sum([proc[n] for n in proc if n in active]), normalize)
        n, paths, used = len(full), {}, set()
        if stems:
            for t in self.s.tracks:
                st = np.pad(proc[t.name], ((0, max(0, n - len(proc[t.name]))), (0, 0)))[:n]
                base = stem = self.safe_name(t.name)
                k = 2
                while stem.lower() in used:
                    stem = f"{base}_{k}"
                    k += 1
                used.add(stem.lower())
                paths[t.name] = write_wav(outdir / f"stem_{stem}.wav", st, self.s.sr, bits)
        paths["mix"] = write_wav(outdir / "vocal_mix.wav", full, self.s.sr, bits)
        return paths
