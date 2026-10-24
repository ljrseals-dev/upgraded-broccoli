"""Voice conversion toward an enrolled reference voice (numpy only).

Engines (registry, tried in priority order for engine="auto"):
  exemplar : kNN-VC-style, numpy only. Each source frame's smooth spectral envelope (cepstral) is replaced by the mean of its
             k nearest reference frames' envelopes (matched on mean-normalised cepstra, so phonetic content, not speaker, drives
             the match); the source's own excitation (pitch, harmonics, phase) is kept. Needs about 8 s+ of reference speech.
  ltas     : static long-term-average-spectrum matching plus pitch transfer. Works from about 3 s of reference speech.
  knn-vc, xtts : neural engines from AOC v3/v5. They need torch and model weights, which this build does not include:
             listed as unavailable; `register_engine()` lets you plug in a real implementation (it then wins under "auto").

This is classical timbre and pitch transfer. It moves speech toward the reference voice; it is NOT a neural clone and will not
fool a listener who knows the real voice. Every conversion requires an active consent record for the reference samples and
writes a provenance sidecar (<out>.aoc.json) marking the audio as synthetic/converted."""
from __future__ import annotations
import hashlib, json, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from . import dsp, pitch, __version__
from . import voiceprofile as vp
from .voiceprofile import VoiceProfile, N_FFT, HOP, ANALYSIS_SR

MAX_SHIFT_ST = 7.0                              # beyond this a resampling-based shift empties the top of the spectrum and drags the formants
MAX_GAIN_LN = 18.0 / 8.6859                       # clamp envelope gain at +-18 dB (natural-log units)


@dataclass
class ConversionResult:
    audio: np.ndarray
    sr: int
    engine: str
    info: dict
    provenance: dict


class ConversionEngine:
    name = "base"
    kind = "builtin"
    priority = 0

    def available(self, profile: VoiceProfile):
        return True, ""

    def run(self, y_shifted: np.ndarray, y_orig: np.ndarray, profile: VoiceProfile, k: int, strength: float) -> np.ndarray:
        """y_shifted: pitch-shifted source to modify; y_orig: the same signal before the shift (time aligned, same length), whose
        frames carry the undistorted phonetic content used for matching."""
        raise NotImplementedError


def _smooth_time(G: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or len(G) == 0:
        return G
    c = np.cumsum(np.concatenate([np.zeros((1, G.shape[1]), G.dtype), G], axis=0), axis=0)
    idx = np.arange(len(G))
    lo, hi = np.clip(idx - n // 2, 0, len(G)), np.clip(idx + (n - n // 2), 0, len(G))
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)[:, None]


def _apply(S, G, y_len, win):
    """Multiply the STFT by exp(G), keep each frame's energy, resynthesize."""
    S2 = S * np.exp(G)
    e_old = np.sqrt(np.sum(np.abs(S) ** 2, axis=1) + 1e-12)
    e_new = np.sqrt(np.sum(np.abs(S2) ** 2, axis=1) + 1e-12)
    S2 = S2 * (e_old / e_new)[:, None]
    return dsp._istft(S2.astype(np.complex64), win, N_FFT, HOP, y_len)


class ExemplarEngine(ConversionEngine):
    name, priority = "exemplar", 50

    def available(self, profile):
        if profile.voiced_seconds >= vp.MIN_SECONDS["exemplar"] and len(profile.bank) >= 200:
            return True, ""
        return False, f"needs {vp.MIN_SECONDS['exemplar']:.0f}+ s of reference speech (profile has {profile.voiced_seconds:.1f} s)"

    def run(self, y_shifted, y_orig, profile, k, strength):
        S, win = dsp._stft(y_shifted, N_FFT, HOP)
        cep_s, logenv = vp.cepstra(S)                                                # envelope of what we will modify
        So, _ = dsp._stft(y_orig, N_FFT, HOP)
        cep_o, _ = vp.cepstra(So)                                                    # undistorted content for matching
        n = min(len(S), len(So))
        S, logenv, cep_s, cep_o = S[:n], logenv[:n], cep_s[:n], cep_o[:n]
        e_db = vp.frame_energy_db(So[:n])
        gate = e_db > e_db.max() - vp.GATE_DB
        bank = profile.bank.astype(np.float32)
        B = bank[:, 1:] - bank[:, 1:].mean(axis=0)
        A_full = cep_o[gate, 1:]
        A = A_full - A_full.mean(axis=0)
        k = int(max(1, min(k, len(B) - 1)))
        Bn = np.sum(B * B, axis=1)
        tgt = np.empty((len(A), cep_s.shape[1]), np.float32)
        for s0 in range(0, len(A), 512):                                             # chunked squared-distance matrix
            a = A[s0:s0 + 512]
            d2 = np.sum(a * a, axis=1)[:, None] + Bn[None, :] - 2.0 * a @ B.T
            idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
            tgt[s0:s0 + 512] = bank[idx].mean(axis=1)
        tgt[:, 0] = cep_s[gate, 0]                                                    # keep the modified frame's own level term
        G = np.zeros_like(logenv)
        G[gate] = np.clip(vp.cep_to_logenv(tgt) - logenv[gate], -MAX_GAIN_LN, MAX_GAIN_LN)
        G = _smooth_time(G, 5) * float(strength)
        return _apply(S, G, len(y_shifted), win)


class LtasEngine(ConversionEngine):
    name, priority = "ltas", 10

    def available(self, profile):
        if profile.voiced_seconds >= vp.MIN_SECONDS["ltas"]:
            return True, ""
        return False, f"needs {vp.MIN_SECONDS['ltas']:.0f}+ s of reference speech"

    def run(self, y_shifted, y_orig, profile, k, strength):
        y22 = y_shifted
        S, win = dsp._stft(y22, N_FFT, HOP)
        cep, _ = vp.cepstra(S)
        e_db = vp.frame_energy_db(S)
        gate = e_db > e_db.max() - vp.GATE_DB
        diff = profile.mean_cep.astype(np.float64) - cep[gate].mean(axis=0)
        diff[0] = 0.0
        g = np.clip(vp.cep_to_logenv(diff[None, :])[0], -MAX_GAIN_LN * 0.67, MAX_GAIN_LN * 0.67)
        G = np.zeros((len(S), S.shape[1]), np.float32)
        G[gate] = g
        G = _smooth_time(G, 5) * float(strength)
        return _apply(S, G, len(y22), win)


class _Unavailable(ConversionEngine):
    kind = "external"

    def __init__(self, name, reason, priority):
        self.name, self._reason, self.priority = name, reason, priority

    def available(self, profile):
        return False, self._reason


_REG: dict = {}


def register_engine(engine: ConversionEngine) -> None:
    """Plug in an engine (e.g. a real kNN-VC or XTTS adapter). Higher priority is tried first under engine="auto"."""
    _REG[engine.name] = engine


for _e in (ExemplarEngine(), LtasEngine(),
           _Unavailable("knn-vc", "neural kNN-VC needs torch + model weights (not bundled); register_engine() to add it", 100),
           _Unavailable("xtts", "neural XTTS cloning needs torch + model weights (not bundled); register_engine() to add it", 90)):
    register_engine(_e)


def engines_status(profile: VoiceProfile | None = None) -> list:
    out = []
    for e in sorted(_REG.values(), key=lambda e: -e.priority):
        ok, why = (e.available(profile) if profile is not None else (not isinstance(e, _Unavailable), ""))
        if isinstance(e, _Unavailable):
            ok, why = e.available(profile)
        out.append({"name": e.name, "kind": e.kind, "available": bool(ok), "reason": why, "priority": e.priority})
    return out


def _choose(engine: str, profile: VoiceProfile):
    notes = []
    if engine != "auto":
        e = _REG.get(engine)
        if e is None:
            raise ValueError(f"unknown engine {engine!r}; choose auto, " + ", ".join(sorted(_REG)))
        ok, why = e.available(profile)
        if not ok:
            raise ValueError(f"engine {engine!r} unavailable: {why}")
        return e, notes
    for e in sorted(_REG.values(), key=lambda e: -e.priority):
        ok, why = e.available(profile)
        if ok:
            return e, notes
        notes.append(f"{e.name}: {why}")
    raise ValueError("no conversion engine is available for this profile: " + "; ".join(notes))


def convert(audio, sr: int, profile: VoiceProfile, engine: str = "auto", pitch_strength: float = 1.0,
            timbre_strength: float = 1.0, k: int = 6, max_shift: float = MAX_SHIFT_ST, consents_path=None) -> ConversionResult:
    """Convert mono speech (float array at `sr`) toward `profile`. Returns audio at the same sample rate and length (+-1 frame)."""
    if not (0.0 <= pitch_strength <= 1.0 and 0.0 <= timbre_strength <= 1.0):
        raise ValueError("strengths must be between 0 and 1")
    profile.check_consent(consents_path)                                           # revoked consent stops conversion
    x = np.asarray(audio, np.float32)
    if x.ndim != 1:
        x = x.mean(axis=1).astype(np.float32)
    if len(x) < int(0.25 * sr):
        raise ValueError("audio is shorter than 0.25 s")
    if not np.isfinite(x).all():
        raise ValueError("audio contains NaN or Inf")
    if float(np.max(np.abs(x))) < 1e-4:
        raise ValueError("audio is silent")
    eng, notes = _choose(engine, profile)
    y_orig = vp.to_analysis(x, sr)
    y22 = y_orig
    info = {"fallback_notes": notes}
    _, f0, _ = pitch.yin_f0(y22, ANALYSIS_SR, fmin=60.0, fmax=600.0, frame=1024, hop=HOP)
    voiced = f0[f0 > 0]
    if len(voiced) >= 10 and pitch_strength > 0:
        want = 12.0 * np.log2(profile.f0["median"] / float(np.median(voiced))) * pitch_strength
        st = float(np.clip(want, -max_shift, max_shift))
        if abs(st - want) > 1e-6:
            info["pitch_note"] = (f"pitch gap of {want:+.1f} semitones limited to {st:+.1f}: larger shifts degrade quality; synthesize "
                                  "speech at the target pitch instead (aoc speak --voice does this automatically)")
        y22 = dsp.pitch_shift(y22, ANALYSIS_SR, st)
        info.update(source_f0_hz=round(float(np.median(voiced)), 2), shift_semitones=round(st, 3))
    else:
        info.update(source_f0_hz=None, shift_semitones=0.0, pitch_note="no stable pitch found in the source; pitch left unchanged")
    y_conv = eng.run(y22, y_orig, profile, k, timbre_strength)
    out = vp.dsp.resample(y_conv, ANALYSIS_SR / sr, ANALYSIS_SR) if sr != ANALYSIS_SR else y_conv
    if len(out) < len(x):
        out = np.pad(out, (0, len(x) - len(out)))
    out = out[:len(x)].astype(np.float32)
    rms_in, rms_out = float(np.sqrt(np.mean(x ** 2))), float(np.sqrt(np.mean(out ** 2))) + 1e-12
    out = out * (rms_in / rms_out)
    pk = float(np.max(np.abs(out)))
    if pk > 0.99:
        out = out * (0.99 / pk)
    prov = {"synthetic_or_converted": True, "tool": f"AOC Magic Studio {__version__} standalone", "engine": eng.name,
            "voice_profile": profile.name, "reference_sha256": [s["sha256"] for s in profile.samples],
            "consent_ids": [s["consent_id"] for s in profile.samples], "source_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "notice": "This audio was generated or altered to resemble an enrolled voice by classical signal processing."}
    return ConversionResult(out, sr, eng.name, info, prov)


def write_provenance(wav_path, prov: dict) -> Path:
    p = Path(str(wav_path) + ".aoc.json")
    p.write_text(json.dumps(prov, indent=1, sort_keys=True))
    return p


def similarity(audio, sr: int, profile: VoiceProfile) -> dict:
    """Objective closeness to the profile: f0 error (cents), long-term spectral distance (dB, shape only, 100 Hz-7 kHz) and the
    mean cepstral distance of each frame to its nearest reference frame (lower = closer). These measure the signal, not how it sounds."""
    y22 = vp.to_analysis(audio, sr)
    S, _ = dsp._stft(y22, N_FFT, HOP)
    cep, _ = vp.cepstra(S)
    gate = vp.frame_energy_db(S) > vp.frame_energy_db(S).max() - vp.GATE_DB
    _, f0, _ = pitch.yin_f0(y22, ANALYSIS_SR, fmin=60.0, fmax=600.0, frame=1024, hop=HOP)
    v = f0[f0 > 0]
    f0_err = float(1200 * np.log2(np.median(v) / profile.f0["median"])) if len(v) else float("nan")
    freqs = np.fft.rfftfreq(N_FFT, 1 / ANALYSIS_SR)
    band = (freqs >= 100) & (freqs <= 7000)
    a = vp.cep_to_logenv(cep[gate].mean(axis=0)[None, :].copy().astype(np.float64) * np.r_[0, np.ones(vp.N_CEP)])[0][band]
    b = vp.cep_to_logenv((profile.mean_cep.astype(np.float64) * np.r_[0, np.ones(vp.N_CEP)])[None, :])[0][band]
    a, b = a * 8.6859, b * 8.6859
    lsd = float(np.sqrt(np.mean(((a - a.mean()) - (b - b.mean())) ** 2)))
    bank = profile.bank[:, 1:].astype(np.float32)
    X = cep[gate, 1:]
    dmin = []
    for s0 in range(0, len(X), 512):
        x = X[s0:s0 + 512]
        d2 = np.sum(x * x, axis=1)[:, None] + np.sum(bank * bank, axis=1)[None, :] - 2.0 * x @ bank.T
        dmin.append(np.sqrt(np.maximum(d2.min(axis=1), 0)))
    return {"f0_median_hz": round(float(np.median(v)), 2) if len(v) else None, "f0_error_cents": round(f0_err, 1),
            "ltas_distance_db": round(lsd, 2), "frame_distance": round(float(np.concatenate(dmin).mean()), 4)}


def speak_as(text: str, profile: VoiceProfile, tts_engine: str = "auto", engine: str = "auto", rate: float = 1.0,
             pitch_strength: float = 1.0, timbre_strength: float = 1.0, consents_path=None, seed: int = 0) -> ConversionResult:
    """Text -> speech -> conversion toward `profile` (the v3/v5 flow: TTS, then VC against the reference voice).
    The built-in synthesizer is run at the profile's median pitch, so little or no pitch shifting is needed (shifting costs quality)."""
    from . import speech
    profile.check_consent(consents_path)
    audio, sr, used = speech.synthesize(text, tts_engine, rate=rate, pitch_hz=float(np.clip(profile.f0["median"], 60.0, 300.0)), seed=seed)
    res = convert(audio, sr, profile, engine, pitch_strength, timbre_strength, consents_path=consents_path)
    res.info["tts_engine"] = used
    res.provenance["tts_engine"] = used
    res.provenance["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return res
