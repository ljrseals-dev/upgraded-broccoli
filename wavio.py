"""WAV reading and writing with the standard library only (PCM 8/16/24/32-bit; writes 16 or 24-bit)."""
from __future__ import annotations
import struct, wave
from pathlib import Path
import numpy as np


def write_wav(path, audio, sr: int, bits: int = 16) -> Path:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or a.shape[1] < 1 or a.shape[1] > 8:
        raise ValueError("audio must be shape (n,) or (n, channels<=8)")
    a = np.nan_to_num(np.clip(a, -1.0, 1.0))
    if bits == 16:
        data = (a * 32767.0).round().astype("<i2").tobytes()
    elif bits == 24:
        v = (a * 8388607.0).round().astype("<i4")
        data = v.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    else:
        raise ValueError("bits must be 16 or 24")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(a.shape[1]); w.setsampwidth(bits // 8); w.setframerate(int(sr))
        w.writeframes(data)
    return p


def _decode(raw: bytes, width: int, ch: int) -> np.ndarray:
    if width == 1:
        a = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        a = np.frombuffer(raw[: len(raw) // 2 * 2], "<i2").astype(np.float32) / 32768.0
    elif width == 3:
        n = len(raw) // 3
        b = np.frombuffer(raw[: n * 3], np.uint8).reshape(-1, 3)
        v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        v = np.where(v & 0x800000, v - 0x1000000, v)
        a = v.astype(np.float32) / 8388608.0
    elif width == 4:
        a = np.frombuffer(raw[: len(raw) // 4 * 4], "<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {width}")
    a = a[: len(a) // ch * ch]
    return a.reshape(-1, ch) if ch > 1 else a


def parse_wav_bytes(blob: bytes):
    """Tolerant RIFF/WAVE parser (handles streaming headers with bogus sizes, as espeak writes to stdout)."""
    if len(blob) < 44 or blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE stream")
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(blob):
        cid = blob[pos:pos + 4]
        size = struct.unpack("<I", blob[pos + 4:pos + 8])[0]
        body = pos + 8
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", blob[body:body + 16])
        elif cid == b"data":
            end = body + size if body + size <= len(blob) else len(blob)
            data = blob[body:end]
            break
        pos = body + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError("WAV missing fmt or data chunk")
    tag, ch, sr, _, _, bits = fmt
    if tag not in (1, 0xFFFE):
        raise ValueError("only PCM WAV is supported (got format tag %d)" % tag)
    return _decode(data, bits // 8, ch), int(sr)


def read_wav(path):
    """Return (audio float32 in [-1,1], sr). Mono -> shape (n,), multichannel -> (n, ch)."""
    return parse_wav_bytes(Path(path).read_bytes())


def to_mono(a: np.ndarray) -> np.ndarray:
    return a if a.ndim == 1 else a.mean(axis=1).astype(np.float32)
