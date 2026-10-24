"""Monophonic pitch tracking (YIN, numpy only) and note segmentation. Works best on clean solo vocals."""
from __future__ import annotations
import numpy as np
from .model import Note


def yin_f0(y, sr: int, fmin=70.0, fmax=1000.0, frame=2048, hop=256, thresh=0.15, smooth=True):
    """-> (times, f0 Hz with 0 for unvoiced, confidence 0..1)."""
    y = np.asarray(y, np.float32)
    tau_min, tau_max = int(sr / fmax), min(int(sr / fmin), frame // 2 - 1)
    if len(y) < frame + hop:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    frames = np.lib.stride_tricks.sliding_window_view(y, frame)[::hop]
    half = frame // 2
    nfft = 1 << int(np.ceil(np.log2(frame * 2)))
    f0 = np.zeros(len(frames)); conf = np.zeros(len(frames))
    for i, fr in enumerate(frames):
        x = fr.astype(np.float64)
        if np.sqrt(np.mean(x ** 2)) < 0.005:
            continue
        X = np.fft.rfft(x, nfft)
        c = np.concatenate([[0.0], np.cumsum(x ** 2)])
        w = half
        # difference function d(tau) = sum_{j<w} (x_j - x_{j+tau})^2 via cross-correlation of the first half
        acf = np.fft.irfft(np.conj(np.fft.rfft(x[:w], nfft)) * np.fft.rfft(x, nfft), nfft)   # sum_j x[j] x[j+tau]
        taus = np.arange(1, tau_max + 1)
        e0 = np.sum(x[:w] ** 2)
        et = c[w + taus] - c[taus]
        d = e0 + et - 2.0 * acf[taus]
        cm = d * taus / np.maximum(np.cumsum(d), 1e-12)
        cand = np.where(cm[tau_min - 1:] < thresh)[0]
        if len(cand):
            k = cand[0] + tau_min - 1
            while k + 1 < len(cm) and cm[k + 1] < cm[k]:
                k += 1
        else:
            k = int(np.argmin(cm[tau_min - 1:])) + tau_min - 1
        if cm[k] > 0.35:
            continue
        t = float(k + 1)
        if 1 <= k < len(cm) - 1:                                   # parabolic interpolation
            a, b, cc = d[k - 1], d[k], d[k + 1]                      # interpolate the raw difference function
            den = a - 2 * b + cc
            if abs(den) > 1e-12:
                t = k + 1 + 0.5 * (a - cc) / den
        f0[i] = sr / t
        conf[i] = float(max(0.0, 1.0 - cm[k]))
    times = (np.arange(len(frames)) * hop + frame / 2) / sr
    if smooth:
        f0 = smooth_f0(f0)
        conf = np.where(f0 > 0, conf, 0.0)
    return times, f0, conf


def smooth_f0(f0, window: int = 9, octave_tol: float = 1.2):
    """Correct isolated octave jumps (a frame whose pitch is about 12 or 24 semitones away from the median of its
    neighbours is moved by whole octaves onto the neighbours) and drop isolated voiced frames. f0 in Hz, 0 = unvoiced."""
    f0 = np.array(f0, dtype=np.float64, copy=True)
    n, half = len(f0), window // 2
    midi = np.where(f0 > 0, 69.0 + 12.0 * np.log2(np.maximum(f0, 1e-9) / 440.0), np.nan)
    out = f0.copy()
    for i in range(n):
        if np.isnan(midi[i]):
            continue
        nb = midi[max(0, i - half):i + half + 1]
        nb = nb[~np.isnan(nb)]
        if len(nb) < 3:
            out[i] = 0.0                                   # isolated voiced frame: not a note
            continue
        med = float(np.median(nb))
        diff = midi[i] - med
        for octv in (12.0, 24.0):
            if abs(abs(diff) - octv) < octave_tol:
                out[i] = f0[i] * 2.0 ** (-np.sign(diff) * octv / 12.0)
                break
    return out


def estimate_f0(y, sr: int) -> float:
    _, f0, _ = yin_f0(y, sr)
    v = f0[f0 > 0]
    return float(np.median(v)) if len(v) else 0.0


def hz_to_midi(f):
    return 69.0 + 12.0 * np.log2(np.maximum(f, 1e-9) / 440.0)


def track_notes(y, sr: int, min_note_s=0.08, semitone_tol=0.6, quantize=True, fmin=70.0, fmax=1000.0):
    """Segment a solo vocal into Note objects (pitch in MIDI, start/dur in seconds, vel from RMS)."""
    times, f0, conf = yin_f0(y, sr, fmin=fmin, fmax=fmax)
    if len(times) == 0:
        return []
    hop = (times[1] - times[0]) if len(times) > 1 else 0.005
    midi = np.where(f0 > 0, hz_to_midi(f0), np.nan)
    notes, i, n = [], 0, len(midi)
    y = np.asarray(y, np.float32)
    while i < n:
        if np.isnan(midi[i]):
            i += 1
            continue
        j, vals = i, [midi[i]]
        while j + 1 < n and not np.isnan(midi[j + 1]) and abs(midi[j + 1] - np.median(vals)) <= semitone_tol:
            j += 1
            vals.append(midi[j])
        dur = (j - i + 1) * hop
        if dur >= min_note_s:
            m = float(np.median(vals))
            s0, s1 = int((times[i] - hop / 2) * sr), int((times[j] + hop / 2) * sr)
            seg = y[max(s0, 0):max(s1, 1)]
            rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
            notes.append(Note(float(round(m)) if quantize else m, float(times[i] - hop / 2 if times[i] > hop else times[i]),
                              float(dur), float(min(1.0, max(0.2, rms * 4))), "ah"))
        i = j + 1
    return notes
