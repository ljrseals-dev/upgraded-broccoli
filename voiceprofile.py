"""Voice profiles: enroll one or more reference WAVs (with consent) into a compact, numpy-only description of a voice.

A profile stores: f0 statistics, the long-term cepstral envelope, and a bank of per-frame cepstra from the reference
speech (the 'exemplars' used by the exemplar converter, a numpy-only cousin of kNN-VC). The reference audio itself is not
copied; the profile records each sample's SHA-256 and consent id, and conversion re-checks that consent is still active."""
from __future__ import annotations
import hashlib, json, re, time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from . import consent as consentmod, dsp, pitch
from .wavio import read_wav, to_mono

ANALYSIS_SR = 22050
N_FFT, HOP, N_CEP, GATE_DB = 1024, 256, 40, 40.0
MAX_BANK = 8000
MIN_SECONDS = {"ltas": 3.0, "exemplar": 8.0}
SCHEMA_VERSION = 1
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class EnrollmentError(ValueError):
    pass


def default_dir() -> Path:
    import os
    return Path(os.environ.get("AOC_VOICES") or (Path.home() / ".aoc_magic" / "voices"))


def to_analysis(y, sr: int) -> np.ndarray:
    y = np.asarray(to_mono(np.asarray(y, np.float32)), np.float32)
    if sr != ANALYSIS_SR:
        y = dsp.resample(y, sr / ANALYSIS_SR, sr)
    return y


def cepstra(S: np.ndarray):
    """-> (cep (frames, N_CEP+1) real cepstrum c0..c40, smooth log-envelope (frames, N_FFT//2+1) in natural-log magnitude)."""
    logm = np.log(np.abs(S) + 1e-8)
    cep = np.fft.irfft(logm, N_FFT, axis=1)
    return cep[:, :N_CEP + 1].astype(np.float32), cep_to_logenv(cep[:, :N_CEP + 1])


def cep_to_logenv(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, np.float64)
    full = np.zeros(c.shape[:-1] + (N_FFT,))
    full[..., 0] = c[..., 0]
    full[..., 1:N_CEP + 1] = c[..., 1:]
    full[..., N_FFT - N_CEP:] = c[..., :0:-1]                       # mirror c40..c1
    return np.fft.rfft(full, axis=-1).real.astype(np.float32)


def frame_energy_db(S: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.sum(np.abs(S) ** 2, axis=1) + 1e-12)


def analyze(y22: np.ndarray):
    """-> S, cep, logenv, gate (bool energetic frames), e_db."""
    S, _ = dsp._stft(y22, N_FFT, HOP)
    cep, logenv = cepstra(S)
    e_db = frame_energy_db(S)
    gate = e_db > (e_db.max() - GATE_DB)
    return S, cep, logenv, gate, e_db


@dataclass
class VoiceProfile:
    name: str
    samples: list                      # [{file_name, sha256, seconds, consent_id}]
    f0: dict                           # median, p10, p90, voiced_ratio
    voiced_seconds: float
    mean_cep: np.ndarray               # (N_CEP+1,)
    bank: np.ndarray                   # (n, N_CEP+1) float32
    warnings: list = field(default_factory=list)
    created: str = ""

    @property
    def engines_ok(self) -> dict:
        return {k: self.voiced_seconds >= v for k, v in MIN_SECONDS.items()}

    def check_consent(self, consents_path=None):
        for s in self.samples:
            consentmod.require_active(s["consent_id"], consents_path)

    # -- persistence ---------------------------------------------------------------------------------------------
    def save(self, root=None) -> Path:
        d = Path(root or default_dir()) / self.name
        d.mkdir(parents=True, exist_ok=True)
        bank_path = d / "bank.npz"
        np.savez_compressed(bank_path, bank=self.bank.astype(np.float32))
        meta = {"schema_version": SCHEMA_VERSION, "name": self.name, "created": self.created, "samples": self.samples,
                "analysis": {"sr": ANALYSIS_SR, "n_fft": N_FFT, "hop": HOP, "n_cep": N_CEP, "gate_db": GATE_DB},
                "f0": self.f0, "voiced_seconds": round(self.voiced_seconds, 3), "mean_cep": [round(float(x), 6) for x in self.mean_cep],
                "bank_file": "bank.npz", "bank_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
                "bank_frames": int(len(self.bank)), "warnings": self.warnings, "engines_ok": self.engines_ok}
        (d / "profile.json").write_text(json.dumps(meta, indent=1))
        return d

    @classmethod
    def load(cls, name: str, root=None) -> "VoiceProfile":
        if not NAME_RE.match(name or ""):
            raise ValueError("bad voice name (use letters, digits, - and _, up to 40 characters)")
        d = Path(root or default_dir()) / name
        meta = json.loads((d / "profile.json").read_text())
        if meta.get("schema_version", 0) > SCHEMA_VERSION:
            raise ValueError("voice profile schema is newer than this build supports")
        a = meta["analysis"]
        if (a["sr"], a["n_fft"], a["hop"], a["n_cep"]) != (ANALYSIS_SR, N_FFT, HOP, N_CEP):
            raise ValueError("voice profile was made with different analysis settings; re-enroll")
        bank_path = d / meta["bank_file"]
        if Path(meta["bank_file"]).name != meta["bank_file"]:
            raise ValueError("unsafe bank path in profile")
        if hashlib.sha256(bank_path.read_bytes()).hexdigest() != meta["bank_sha256"]:
            raise ValueError("voice profile bank was modified (hash mismatch); re-enroll")
        bank = np.load(bank_path)["bank"].astype(np.float32)
        return cls(meta["name"], meta["samples"], meta["f0"], float(meta["voiced_seconds"]),
                   np.asarray(meta["mean_cep"], np.float32), bank, meta.get("warnings", []), meta.get("created", ""))


def list_profiles(root=None) -> list:
    r = Path(root or default_dir())
    out = []
    if r.is_dir():
        for p in sorted(r.iterdir()):
            f = p / "profile.json"
            if f.is_file():
                try:
                    m = json.loads(f.read_text())
                    out.append({"name": m["name"], "voiced_seconds": m["voiced_seconds"], "f0_median_hz": m["f0"]["median"],
                                "engines_ok": m.get("engines_ok", {}), "warnings": len(m.get("warnings", []))})
                except (ValueError, KeyError):
                    out.append({"name": p.name, "error": "unreadable profile"})
    return out


def delete_profile(name: str, root=None) -> bool:
    import shutil
    if not NAME_RE.match(name or ""):
        raise ValueError("bad voice name")
    d = Path(root or default_dir()) / name
    if not (d / "profile.json").is_file():
        return False
    shutil.rmtree(d)
    return True


def enroll(name: str, sample_paths, consents_path=None) -> VoiceProfile:
    """Build a profile from reference WAV file(s). Every file needs an active consent record (checked BEFORE the audio is read)."""
    if not NAME_RE.match(name or ""):
        raise ValueError("bad voice name (use letters, digits, - and _, up to 40 characters)")
    paths = [sample_paths] if isinstance(sample_paths, (str, Path)) else list(sample_paths)
    if not paths:
        raise EnrollmentError("at least one sample WAV is required")
    if len(paths) > 20:
        raise EnrollmentError("at most 20 sample files")
    recs = [consentmod.require_consent(p, consents_path) for p in paths]          # consent first, audio second
    banks, cep_all, f0_all, samples, warns = [], [], [], [], []
    voiced_frames = active_frames = 0
    frame_dbfs, clipped, total = [], 0, 0
    for p, rec in zip(paths, recs):
        try:
            y, sr = read_wav(p)
        except (ValueError, OSError) as e:
            raise EnrollmentError(f"{Path(p).name}: cannot read WAV ({e})")
        if not np.isfinite(y).all():
            raise EnrollmentError(f"{Path(p).name}: contains NaN or Inf")
        if sr < 8000:
            raise EnrollmentError(f"{Path(p).name}: sample rate {sr} Hz is too low (need at least 8000)")
        if sr < 16000:
            warns.append(f"{Path(p).name}: sample rate {sr} Hz is low; high-frequency timbre will be missing")
        mono = to_mono(y)
        if float(np.max(np.abs(mono))) < 1e-4:
            raise EnrollmentError(f"{Path(p).name}: silent")
        total += mono.size
        clipped += int(np.sum(np.abs(mono) >= 0.999))
        y22 = to_analysis(mono, sr)
        S, cep, _, gate, e_db = analyze(y22)
        if not gate.any():
            raise EnrollmentError(f"{Path(p).name}: no speech-level audio found")
        frame_dbfs.append(e_db - 10 * np.log10(N_FFT * 0.375))                  # rough dBFS per frame (Hann window energy)
        banks.append(cep[gate])
        _, f0, _ = pitch.yin_f0(y22, ANALYSIS_SR, fmin=60.0, fmax=600.0, frame=1024, hop=HOP)
        n = min(len(f0), len(gate))
        f0_all.append(f0[:n][gate[:n]])
        voiced_frames += int(np.sum(f0[:n][gate[:n]] > 0))
        active_frames += int(gate.sum())
        samples.append({"file_name": Path(p).name, "sha256": rec["file_sha256"], "seconds": round(len(mono) / sr, 2),
                        "consent_id": rec["consent_id"]})
    voiced_seconds = active_frames * HOP / ANALYSIS_SR
    if voiced_seconds < MIN_SECONDS["ltas"]:
        raise EnrollmentError(f"only {voiced_seconds:.1f} s of speech-level audio; need at least {MIN_SECONDS['ltas']:.0f} s "
                              f"(recommended 30 s or more for the exemplar engine)")
    if voiced_seconds < MIN_SECONDS["exemplar"]:
        warns.append(f"{voiced_seconds:.1f} s of audio: enough for the 'ltas' engine only; 'exemplar' needs {MIN_SECONDS['exemplar']:.0f} s+ (30 s+ recommended)")
    elif voiced_seconds < 30:
        warns.append(f"{voiced_seconds:.1f} s of audio is usable; 30 s or more gives the exemplar engine better phonetic coverage")
    if clipped / max(total, 1) > 0.005:
        warns.append(f"{100 * clipped / total:.2f}% of samples are clipped; re-record at a lower level")
    fd = np.concatenate(frame_dbfs)
    floor_db, speech_db = float(np.percentile(fd, 10)), float(np.percentile(fd, 90))
    snr = speech_db - floor_db
    if snr < 12:
        raise EnrollmentError(f"recording is too noisy or has no pauses (speech-to-floor range {snr:.0f} dB; need 12 dB or more)")
    if snr < 20:
        warns.append(f"speech-to-floor range is only {snr:.0f} dB; a quieter room gives a cleaner profile")
    f0v = np.concatenate(f0_all)
    f0v = f0v[f0v > 0]
    if len(f0v) < 20:
        raise EnrollmentError("could not find a stable pitch in the recording (too little voiced speech)")
    cep_cat = np.concatenate(banks)
    if len(cep_cat) > MAX_BANK:                                                   # deterministic, evenly spaced thinning
        cep_cat = cep_cat[np.linspace(0, len(cep_cat) - 1, MAX_BANK).astype(int)]
    f0s = {"median": round(float(np.median(f0v)), 2), "p10": round(float(np.percentile(f0v, 10)), 2),
           "p90": round(float(np.percentile(f0v, 90)), 2), "voiced_ratio": round(voiced_frames / max(active_frames, 1), 3)}
    return VoiceProfile(name, samples, f0s, voiced_seconds, cep_cat.mean(axis=0), cep_cat, warns,
                        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
