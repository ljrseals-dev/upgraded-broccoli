"""Standard MIDI File writer and reader (stdlib only). Write: one track per vocal part (format 1). Read: format 0/1,
tempo map, running status, note-on velocity 0 as note-off."""
from __future__ import annotations
import struct
from pathlib import Path
from .model import Note, Session

TPQ = 480
_CH = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]            # skip channel 10 (index 9, drums)


def _vlq(n: int) -> bytes:
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def _chunk(tag: bytes, body: bytes) -> bytes:
    return tag + struct.pack(">I", len(body)) + body


def write_midi(session: Session, path, program: int = 52) -> Path:
    """Export every non-empty track as a MIDI track. Fractional pitches are rounded. Program 52 = Choir Aahs."""
    session.validate()
    usec = int(round(60_000_000 / max(session.tempo, 1.0)))
    t0 = _vlq(0) + b"\xff\x51\x03" + usec.to_bytes(3, "big") + _vlq(0) + b"\xff\x58\x04\x04\x02\x18\x08" + _vlq(0) + b"\xff\x2f\x00"
    tracks = [_chunk(b"MTrk", t0)]
    sec_to_tick = session.tempo / 60.0 * TPQ
    idx = 0
    for t in session.tracks:
        if not t.notes:
            continue
        ch = _CH[idx % len(_CH)]
        idx += 1
        name = t.name.encode("utf-8")[:60]
        ev = [(0, 0, b"\xff\x03" + bytes([len(name)]) + name), (0, 1, bytes([0xC0 | ch, program & 0x7F]))]
        for n in t.notes:
            p = max(0, min(127, int(round(n.pitch))))
            v = max(1, min(127, int(round(n.vel * 127))))
            on = int(round(n.start * sec_to_tick))
            off = max(on + 1, int(round(n.end * sec_to_tick)))
            ev.append((on, 3, bytes([0x90 | ch, p, v])))
            ev.append((off, 2, bytes([0x80 | ch, p, 0])))
        ev.sort(key=lambda e: (e[0], e[1]))
        body, last = b"", 0
        for tick, _, data in ev:
            body += _vlq(tick - last) + data
            last = tick
        body += _vlq(0) + b"\xff\x2f\x00"
        tracks.append(_chunk(b"MTrk", body))
    head = _chunk(b"MThd", struct.pack(">HHH", 1, len(tracks), TPQ))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(head + b"".join(tracks))
    return p


def _read_vlq(b: bytes, i: int):
    v = 0
    for _ in range(4):
        c = b[i]
        i += 1
        v = (v << 7) | (c & 0x7F)
        if not c & 0x80:
            return v, i
    raise ValueError("bad variable-length quantity")


def read_midi(path):
    """-> dict(tempo_bpm, tpq, tracks=[{name, notes:[Note,...]}]). Times in seconds using the file's tempo map."""
    b = Path(path).read_bytes()
    if b[:4] != b"MThd":
        raise ValueError("not a MIDI file")
    hlen = struct.unpack(">I", b[4:8])[0]
    fmt, ntrk, div = struct.unpack(">HHH", b[8:14])
    if div & 0x8000:
        raise ValueError("SMPTE time division is not supported")
    pos, raw_tracks, tempos = 8 + hlen, [], [(0, 500000)]
    for _ in range(ntrk):
        if b[pos:pos + 4] != b"MTrk":
            raise ValueError("bad track chunk")
        ln = struct.unpack(">I", b[pos + 4:pos + 8])[0]
        body = b[pos + 8:pos + 8 + ln]
        pos += 8 + ln
        i, tick, status, name, evs = 0, 0, 0, "", []
        while i < len(body):
            dt, i = _read_vlq(body, i)
            tick += dt
            st = body[i]
            if st < 0x80:
                if status == 0:
                    raise ValueError("running status without a previous status")
            else:
                status = st
                i += 1
                if st == 0xFF:
                    mt = body[i]; l, i = _read_vlq(body, i + 1); data = body[i:i + l]; i += l
                    if mt == 0x51 and l == 3:
                        tempos.append((tick, int.from_bytes(data, "big")))
                    elif mt == 0x03:
                        name = data.decode("utf-8", "replace")
                    elif mt == 0x2F:
                        break
                    status = 0
                    continue
                if st in (0xF0, 0xF7):
                    l, i = _read_vlq(body, i); i += l; status = 0
                    continue
            kind = status & 0xF0
            n = 1 if kind in (0xC0, 0xD0) else 2
            args = body[i:i + n]
            i += n
            if kind == 0x90 and args[1] > 0:
                evs.append((tick, "on", args[0], args[1], status & 0x0F))
            elif kind == 0x80 or (kind == 0x90 and args[1] == 0):
                evs.append((tick, "off", args[0], 0, status & 0x0F))
        raw_tracks.append((name, evs))
    tempos = sorted(set(tempos))
    uniq = {}
    for t, u in tempos:
        uniq[t] = u
    tmap = sorted(uniq.items())

    def to_sec(tick):
        sec, last_t, last_u = 0.0, 0, tmap[0][1]
        for t, u in tmap:
            if t >= tick:
                break
            sec += (t - last_t) * last_u / 1e6 / div
            last_t, last_u = t, u
        return sec + (tick - last_t) * last_u / 1e6 / div

    out = []
    for name, evs in raw_tracks:
        open_, notes = {}, []
        for tick, kind, p, v, ch in evs:
            if kind == "on":
                open_.setdefault((ch, p), []).append((tick, v))
            else:
                st = open_.get((ch, p))
                if st:
                    t_on, vel = st.pop(0)
                    notes.append(Note(float(p), to_sec(t_on), max(to_sec(tick) - to_sec(t_on), 0.001), vel / 127.0, "ah"))
        notes.sort(key=lambda n: (n.start, n.pitch))
        out.append({"name": name, "notes": notes})
    return {"tempo_bpm": 60e6 / tmap[0][1], "tpq": div, "tracks": out}


def lead_from_midi(path, track=None, poly: str = "reject"):
    """Extract a monophonic lead. poly='reject' raises on overlapping notes; 'top' keeps the highest pitch."""
    data = read_midi(path)
    cands = [t for t in data["tracks"] if t["notes"]]
    if not cands:
        raise ValueError("no notes in MIDI file")
    if track is None:
        chosen = cands[0]
    else:
        names = [t["name"] for t in cands]
        chosen = cands[track] if isinstance(track, int) and 0 <= track < len(cands) else \
            next((t for t in cands if t["name"] == track), None)
        if chosen is None:
            raise ValueError(f"track {track!r} not found (have {names})")
    notes, out = chosen["notes"], []
    for n in notes:
        if out and n.start < out[-1].end - 1e-6:
            if poly == "reject":
                raise ValueError("lead is polyphonic (overlapping notes); use poly='top' to keep the highest note")
            if n.pitch > out[-1].pitch:
                out[-1] = n
            continue
        out.append(n)
    return out, data["tempo_bpm"]
