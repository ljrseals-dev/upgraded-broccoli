"""numpy-only DSP. compressor / limiter / peak_normalize / soft_saturate are bit-identical to the AOC v5 chain
(guarded by a golden-hash test). Filters are FFT zero-phase; pitch/time use a phase vocoder."""
from __future__ import annotations
from functools import lru_cache
import numpy as np


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def linear_to_db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-12))


# ---- identical to AOC v5 -----------------------------------------------------------------------------------------
def peak_normalize(y, ceiling_db: float = -1.0):
    y = np.asarray(y)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 0.0:
        return y.astype(np.float32, copy=False)
    return (y * (db_to_linear(ceiling_db) / peak)).astype(np.float32)


def compressor(y, threshold_db: float = -20.0, ratio: float = 4.0):
    ratio = max(float(ratio), 1.0)
    x = np.asarray(y).astype(np.float32, copy=True)
    threshold = db_to_linear(threshold_db)
    mag = np.abs(x)
    over = mag > threshold
    if np.any(over):
        x[over] = np.sign(x[over]) * (threshold + (mag[over] - threshold) / ratio)
    return x


def soft_saturate(y, drive_db: float):
    if abs(drive_db) < 1e-6:
        return y
    return np.tanh(y * db_to_linear(drive_db)).astype(np.float32)


def limiter(y, ceiling_db: float = -1.0):
    ceiling = max(db_to_linear(ceiling_db), 1e-6)
    return (np.tanh(np.asarray(y) / ceiling) * ceiling).astype(np.float32)


# ---- filters (FFT, zero phase) -----------------------------------------------------------------------------------
def _pow2(n: int) -> int:
    return 1 << max(4, int(np.ceil(np.log2(max(n, 2)))))


@lru_cache(maxsize=256)
def _fast_len(n: int) -> int:
    """Smallest 2^a * 3^b * 5^c >= n (pocketfft is fastest on smooth lengths)."""
    best = _pow2(n)
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            p = p35
            while p < n:
                p *= 2
            best = min(best, p)
            p35 *= 3
        p5 *= 5
    return max(best, 16)


@lru_cache(maxsize=32)
def _freqs(nfft: int, sr: int) -> np.ndarray:
    return np.fft.rfftfreq(nfft, 1.0 / sr).astype(np.float32)


def fft_filter(y, sr: int, response) -> np.ndarray:
    """Zero-phase filter in the frequency domain. `response(f_hz_array) -> gain_array`."""
    y = np.asarray(y, np.float32)
    n = len(y)
    if n == 0:
        return y
    nfft = _fast_len(n + min(n, 16384))
    Y = np.fft.rfft(y, nfft)
    H = np.asarray(response(_freqs(nfft, int(sr))), np.float32)
    return np.fft.irfft(Y * H, nfft)[:n].astype(np.float32)


def hp_response(f, fc, order=2):
    f = np.asarray(f, np.float64)
    return 1.0 / np.sqrt(1.0 + (float(fc) / np.maximum(f, 1e-9)) ** (2 * order))


def lp_response(f, fc, order=2):
    f = np.asarray(f, np.float64)
    return 1.0 / np.sqrt(1.0 + (f / float(fc)) ** (2 * order))


def bp_response(f, lo, hi, order=2):
    return hp_response(f, lo, order) * lp_response(f, hi, order)


def eq_vocal(y, sr: int, hp_hz: float = 90.0, presence: float = 0.25, lo: float = 2500.0, hi: float = 6000.0):
    """High-pass then add a presence band, in a single FFT pass (identical response to hp followed by y + p*bp(y))."""
    return fft_filter(y, sr, lambda f: hp_response(f, hp_hz) * (1.0 + presence * bp_response(f, lo, hi)))


def highpass(y, sr: int, cutoff_hz: float, order: int = 2):
    return fft_filter(y, sr, lambda f: hp_response(f, cutoff_hz, order))


def lowpass(y, sr: int, cutoff_hz: float, order: int = 2):
    return fft_filter(y, sr, lambda f: lp_response(f, cutoff_hz, order))


def bandpass(y, sr: int, lo_hz: float, hi_hz: float, order: int = 2):
    return fft_filter(y, sr, lambda f: bp_response(f, lo_hz, hi_hz, order))


def fft_convolve(x, h) -> np.ndarray:
    x = np.asarray(x, np.float32)
    h = np.asarray(h, np.float32)
    if len(x) == 0 or len(h) == 0:
        return np.zeros(0, np.float32)
    n = len(x) + len(h) - 1
    nfft = _fast_len(n)
    return np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(h, nfft), nfft)[:n].astype(np.float32)


@lru_cache(maxsize=8)
def reverb_ir(sr: int, seconds: float = 1.8, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    ir = rng.standard_normal(n) * np.exp(-np.arange(n) / (sr * seconds / 5))
    ir = lowpass(ir.astype(np.float32), sr, 6000, order=1)
    ir = (ir / np.sqrt(np.sum(ir.astype(np.float64) ** 2))).astype(np.float32)
    ir.setflags(write=False)
    return ir


def echo(y, sr: int, ms: float, fb: float = 0.4, taps: int = 4) -> np.ndarray:
    y = np.asarray(y, np.float32)
    out = np.zeros(len(y) + int(taps * ms * sr / 1000), np.float32)
    for k in range(1, taps + 1):
        d = int(k * ms * sr / 1000)
        out[d:d + len(y)] += y * (fb ** k)
    return out


def moving_average(x, n: int) -> np.ndarray:
    n = max(int(n), 1)
    if len(x) == 0:
        return np.asarray(x, np.float32)
    c = np.cumsum(np.concatenate([[0.0], np.asarray(x, np.float64)]))
    idx = np.arange(len(x))
    lo = np.clip(idx - n // 2, 0, len(x))
    hi = np.clip(idx + (n - n // 2), 0, len(x))
    return ((c[hi] - c[lo]) / np.maximum(hi - lo, 1)).astype(np.float32)


def deess(y, sr: int, thresh: float = 0.08) -> np.ndarray:
    y = np.asarray(y, np.float32)
    band = bandpass(y, sr, 5500, min(9500, 0.45 * sr))
    env = moving_average(np.abs(band), int(0.005 * sr))
    gain = np.where(env > thresh, thresh / np.maximum(env, 1e-9), 1.0)
    return (y - band * (1 - gain)).astype(np.float32)


def pan_stereo(y, pan: float) -> np.ndarray:
    a = (float(pan) + 1.0) * np.pi / 4.0
    y = np.asarray(y, np.float32)
    return np.stack([y * np.cos(a), y * np.sin(a)], 1).astype(np.float32)


# ---- pitch / time (phase vocoder) --------------------------------------------------------------------------------
def _stft(y, n_fft=2048, hop=512):
    win = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    pad = np.pad(np.asarray(y, np.float32), (n_fft // 2, n_fft // 2 + n_fft))
    frames = np.lib.stride_tricks.sliding_window_view(pad, n_fft)[::hop]
    return np.fft.rfft(frames * win, axis=1), win


def _istft(S, win, n_fft, hop, length):
    frames = np.fft.irfft(S, n_fft, axis=1) * win
    out = np.zeros(hop * (len(frames) - 1) + n_fft, np.float64)
    wsum = np.zeros_like(out)
    for i, fr in enumerate(frames):
        out[i * hop:i * hop + n_fft] += fr
        wsum[i * hop:i * hop + n_fft] += win ** 2
    out = out / np.maximum(wsum, 1e-8)
    return out[n_fft // 2:n_fft // 2 + length].astype(np.float32)


def time_stretch(y, rate: float, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """rate > 1 shortens (faster) without changing pitch. Plain per-bin phase vocoder.
    (Identity phase locking + transient reset was tried in run 2 and reverted: no gain on stationary harmonics, much worse on
    click trains. See docs/RUN_LOG.md.)"""
    y = np.asarray(y, np.float32)
    if rate <= 0:
        raise ValueError("rate must be > 0")
    if abs(rate - 1.0) < 1e-6 or len(y) < n_fft:
        return y
    S, win = _stft(y, n_fft, hop)
    steps = np.arange(0, S.shape[0] - 1, rate)
    omega = 2 * np.pi * hop * np.arange(S.shape[1]) / n_fft
    Sp = np.concatenate([S, np.zeros((1, S.shape[1]), S.dtype)])
    acc = np.angle(Sp[0])
    out = np.empty((len(steps), S.shape[1]), np.complex64)
    for t, st in enumerate(steps):
        i = int(st)
        f = st - i
        mag = (1 - f) * np.abs(Sp[i]) + f * np.abs(Sp[i + 1])
        out[t] = mag * np.exp(1j * acc)
        d = np.angle(Sp[i + 1]) - np.angle(Sp[i]) - omega
        d -= 2 * np.pi * np.round(d / (2 * np.pi))
        acc = acc + omega + d
    return _istft(out, win, n_fft, hop, int(len(y) / rate))


def _resample_linear(y, ratio, sr):
    if ratio > 1.0:
        y = lowpass(y, sr, 0.45 * sr / ratio, order=6)
    pos = np.arange(0, len(y) - 1, ratio)
    return np.interp(pos, np.arange(len(y)), y).astype(np.float32)


def _resample_sinc(y, ratio, taps=16, beta=8.6, chunk=8192):
    """Band-limited interpolation: windowed-sinc (Kaiser) kernel, cutoff at min(1, 1/ratio) of the input Nyquist, so
    decimation is anti-aliased and interpolation has no imaging. `taps` = zero crossings on each side (at cutoff 1)."""
    fc = min(1.0, 1.0 / ratio)
    L = int(np.ceil(taps / fc))                                  # kernel half-width in input samples
    n_out = int(np.floor((len(y) - 1) / ratio)) + 1
    ypad = np.concatenate([np.zeros(L, np.float32), y.astype(np.float32), np.zeros(L + 1, np.float32)])
    k = np.arange(-L + 1, L + 1)
    out = np.empty(n_out, np.float32)
    for s0 in range(0, n_out, chunk):
        pos = np.arange(s0, min(s0 + chunk, n_out)) * ratio
        n0 = np.floor(pos).astype(np.int64)
        frac = (pos - n0)[:, None]
        t = k[None, :] - frac                                    # input-sample offset of each tap from the output position
        w = fc * np.sinc(fc * t) * _kaiser(t / L, beta)
        out[s0:s0 + len(pos)] = np.einsum("ij,ij->i", ypad[(n0[:, None] + k[None, :] + L)], w.astype(np.float32))
    return out


def _kaiser(x, beta):
    """Kaiser window evaluated at x in [-1, 1] (zero outside)."""
    inside = np.abs(x) <= 1.0
    return np.where(inside, np.i0(beta * np.sqrt(np.clip(1.0 - x * x, 0.0, 1.0))) / np.i0(beta), 0.0)


def resample(y, ratio: float, sr: int = 44100, quality: str = "sinc") -> np.ndarray:
    """Read the signal `ratio` times faster (2.0 = one octave up, half the length).
    quality="sinc" (default): windowed-sinc, band-limited. quality="linear": the older linear interpolation
    (kept for parity tests; aliases and images)."""
    y = np.asarray(y, np.float32)
    if len(y) < 2 or abs(ratio - 1.0) < 1e-9:
        return y
    if ratio <= 0:
        raise ValueError("ratio must be > 0")
    if quality == "linear":
        return _resample_linear(y, ratio, sr)
    if quality != "sinc":
        raise ValueError("quality must be 'sinc' or 'linear'")
    return _resample_sinc(y, ratio)


def pitch_shift(y, sr: int, semitones: float) -> np.ndarray:
    y = np.asarray(y, np.float32)
    if abs(semitones) < 1e-6 or len(y) < 2048:
        return y
    r = 2.0 ** (semitones / 12.0)
    out = resample(time_stretch(y, 1.0 / r), r, sr)
    if len(out) < len(y):
        out = np.pad(out, (0, len(y) - len(out)))
    return out[:len(y)]


def character(audio, sr, *, pitch=2.0, speed=1.15, comp_threshold=-20.0, comp_ratio=4.0, drive_db=1.5,
              highpass_hz=70.0, lowpass_hz=12000.0):
    """AOC character chain (same order as v5): highpass, pitch, time, compressor, saturate, lowpass."""
    y = np.asarray(audio, np.float32)
    y = highpass(y, sr, highpass_hz)
    y = pitch_shift(y, sr, pitch)
    y = time_stretch(y, speed)
    y = compressor(y, threshold_db=comp_threshold, ratio=comp_ratio)
    y = soft_saturate(y, drive_db)
    y = lowpass(y, sr, min(lowpass_hz, 0.45 * sr))
    return y.astype(np.float32)


def finalize(y, ceiling_db: float = -1.0) -> np.ndarray:
    return peak_normalize(limiter(y, ceiling_db), ceiling_db)
