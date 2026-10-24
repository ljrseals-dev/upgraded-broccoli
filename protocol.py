"""JSON command protocol for embedding AOC in your own DAW or app (any language).

    Engine(workspace).handle({"op": "...", ...}) -> {"ok": true, "result": {...}} | {"ok": false, "error": {...}}

`serve()` speaks the same protocol as JSON lines on stdin/stdout. All file paths are relative to the workspace
directory and are confined to it. Every numeric field is range-checked; unknown ops are structured errors."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from . import __version__, EDITION, speech, midi as midimod, pitch as pitchmod, consent as consentmod
from . import voiceprofile as vpmod, convert as cvmod
from .harmony import build_session
from .mix import VocalMixer
from .model import Note, Session, TrackSettings
from .project import Project
from .renderers import get_renderer, list_renderers
from .wavio import read_wav, to_mono, write_wav
from . import doctor as doctormod

MAX_NOTES = 5000
_PC = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8,
       "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}
TRACK_FIELDS = {"level_db": (-60.0, 24.0), "pan": (-1.0, 1.0), "delay_ms": (-500.0, 500.0), "reverb": (0.0, 1.0),
                "echo": (0.0, 1.0), "echo_ms": (20.0, 2000.0), "timing_ms": (0.0, 100.0), "tuning_cents": (0.0, 100.0),
                "vibrato_depth_cents": (0.0, 100.0), "vibrato_rate_hz": (0.0, 12.0), "breath": (0.0, 1.0),
                "brightness": (0.0, 1.0)}


class ProtocolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def _num(req, key, default=None, lo=None, hi=None, integer=False):
    v = req.get(key, default)
    if v is None:
        raise ProtocolError("bad_request", f"missing field {key!r}")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ProtocolError("bad_request", f"{key!r} must be a number")
    if not np.isfinite(v):
        raise ProtocolError("bad_request", f"{key!r} must be finite")
    if lo is not None and v < lo or hi is not None and v > hi:
        raise ProtocolError("out_of_range", f"{key!r} must be between {lo} and {hi}")
    return int(v) if integer else float(v)


def _str(req, key, default=None, maxlen=5000):
    v = req.get(key, default)
    if v is None:
        raise ProtocolError("bad_request", f"missing field {key!r}")
    if not isinstance(v, str) or len(v) > maxlen:
        raise ProtocolError("bad_request", f"{key!r} must be a string up to {maxlen} characters")
    return v


def parse_notes(raw) -> list:
    if not isinstance(raw, list) or not raw:
        raise ProtocolError("bad_request", "notes must be a non-empty list")
    if len(raw) > MAX_NOTES:
        raise ProtocolError("out_of_range", f"too many notes (max {MAX_NOTES})")
    out = []
    for i, n in enumerate(raw):
        try:
            if isinstance(n, dict):
                p, s, d = n["pitch"], n["start"], n["dur"]
                v, ly = n.get("vel", 0.8), n.get("lyric", "ah")
            else:
                n = list(n)
                p, s, d = n[0], n[1], n[2]
                v, ly = (n[3] if len(n) > 3 else 0.8), (n[4] if len(n) > 4 else "ah")
            vals = [p, s, d, v]
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) for x in vals):
                raise ValueError("numbers expected")
            if not (0 <= p <= 127 and s >= 0 and 0.01 <= d <= 60 and 0 <= v <= 1 and isinstance(ly, str) and len(ly) <= 16):
                raise ValueError("value out of range (pitch 0-127, start>=0, dur 0.01-60, vel 0-1, lyric<=16 chars)")
            out.append(Note(float(p), float(s), float(d), float(v), ly))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProtocolError("bad_note", f"note #{i}: {exc}") from exc
    return out


def session_dict(s: Session, include_notes=False) -> dict:
    d = s.to_dict()
    if not include_notes:
        for t in d["tracks"]:
            t["note_count"] = len(t.pop("notes"))
    return d


class Engine:
    def __init__(self, workspace="."):
        self.ws = Path(workspace).resolve()
        self.ws.mkdir(parents=True, exist_ok=True)
        self.session = Session()
        self.renderer = get_renderer("formant")
        self.renderer_name = "formant"
        self.project: Project | None = None

    # -- path confinement ----------------------------------------------------------------------------------------
    def path(self, rel, must_exist=False, suffixes=None) -> Path:
        if not isinstance(rel, str) or not rel or "\x00" in rel:
            raise ProtocolError("bad_path", "path must be a non-empty string")
        p = (self.ws / rel).resolve()
        if p != self.ws and self.ws not in p.parents:
            raise ProtocolError("bad_path", "path escapes the workspace")
        if suffixes and p.suffix.lower() not in suffixes:
            raise ProtocolError("bad_path", f"expected {'/'.join(suffixes)} file")
        if must_exist and not p.exists():
            raise ProtocolError("not_found", f"{rel} not found in workspace")
        return p

    def rel(self, p: Path) -> str:
        return str(Path(p).resolve().relative_to(self.ws))

    # -- dispatch ------------------------------------------------------------------------------------------------
    def handle(self, req) -> dict:
        try:
            if not isinstance(req, dict):
                raise ProtocolError("bad_request", "request must be a JSON object")
            op = req.get("op")
            fn = getattr(self, "op_" + str(op).replace(".", "_"), None) if isinstance(op, str) and \
                all(c.isalnum() or c in "._" for c in op) else None
            if fn is None:
                raise ProtocolError("unknown_op", f"unknown op {op!r}")
            res = fn(req)
            out = {"ok": True, "result": res}
        except ProtocolError as e:
            out = {"ok": False, "error": {"code": e.code, "message": e.message}}
        except consentmod.ConsentError as e:
            out = {"ok": False, "error": {"code": "consent_required", "message": str(e)}}
        except (ValueError, KeyError) as e:
            out = {"ok": False, "error": {"code": "invalid", "message": str(e)}}
        except Exception as e:                                       # never leak a traceback to a client
            out = {"ok": False, "error": {"code": "internal", "message": f"{type(e).__name__}"}}
        if "id" in req if isinstance(req, dict) else False:
            out["id"] = req["id"]
        return out

    # -- ops -----------------------------------------------------------------------------------------------------
    def op_ping(self, r):
        return {"pong": True}

    def op_version(self, r):
        return {"version": __version__, "edition": EDITION, "workspace": str(self.ws)}

    def op_doctor(self, r):
        return {"checks": doctormod.run_checks()}

    def op_engines_list(self, r):
        return {"speech": speech.engines(), "renderers": list_renderers(), "active_renderer": self.renderer_name}

    def op_speech_synth(self, r):
        text = _str(r, "text")
        engine = _str(r, "engine", "auto", 16)
        rate = _num(r, "rate", 1.0, 0.4, 2.5)
        pitch_hz = _num(r, "pitch_hz", 118.0, 60.0, 300.0)
        bits = _num(r, "bits", 16, integer=True)
        if bits not in (16, 24):
            raise ProtocolError("out_of_range", "bits must be 16 or 24")
        out = self.path(_str(r, "out", "speech.wav", 200), suffixes=(".wav",))
        extra = {}
        if r.get("voice"):
            prof = vpmod.VoiceProfile.load(_str(r, "voice", maxlen=40), self.voices_dir)
            res_c = cvmod.speak_as(text, prof, engine, _str(r, "voice_engine", "auto", 16), rate,
                                   _num(r, "pitch_strength", 1.0, 0.0, 1.0), _num(r, "timbre_strength", 1.0, 0.0, 1.0))
            a, sr, used = res_c.audio, res_c.sr, res_c.engine
            extra = {"voice": prof.name, "tts_engine": res_c.info["tts_engine"], "provenance": self.rel(cvmod.write_provenance(self._write(out, a, sr, bits), res_c.provenance))}
        else:
            a, sr, used = speech.synthesize(text, engine, rate=rate, pitch_hz=pitch_hz, character=bool(r.get("character", False)))
            write_wav(out, a, sr, bits)
        res = {"file": self.rel(out), "sr": sr, "duration_s": round(len(a) / sr, 3), "engine": used, **extra}
        if self.project is not None and r.get("add_to_project"):
            res["clip"] = self.project.add_clip(a, sr, Path(out).stem, text, used, bits)
        return res

    def _write(self, out, a, sr, bits):
        write_wav(out, a, sr, bits)
        return out

    @property
    def voices_dir(self):
        return self.ws / "voices"

    def op_voice_enroll(self, r):
        samples = r.get("samples")
        if not isinstance(samples, list) or not samples or len(samples) > 20:
            raise ProtocolError("bad_request", "samples must be a list of 1-20 workspace WAV paths")
        paths = [self.path(x, True, (".wav",)) for x in samples]
        if r.get("attested") is True:                                        # explicit, per-request attestation records consent
            owner, scope = _str(r, "owner", maxlen=80), _str(r, "scope", maxlen=200)
            for p in paths:
                consentmod.add_consent(p, owner, scope, True)
        prof = vpmod.enroll(_str(r, "name", maxlen=40), paths)
        prof.save(self.voices_dir)
        return {"name": prof.name, "voiced_seconds": round(prof.voiced_seconds, 2), "f0": prof.f0, "engines_ok": prof.engines_ok,
                "warnings": prof.warnings}

    def op_voice_list(self, r):
        return {"voices": vpmod.list_profiles(self.voices_dir)}

    def op_voice_show(self, r):
        prof = vpmod.VoiceProfile.load(_str(r, "name", maxlen=40), self.voices_dir)
        return {"name": prof.name, "f0": prof.f0, "voiced_seconds": round(prof.voiced_seconds, 2), "samples": prof.samples,
                "engines_ok": prof.engines_ok, "warnings": prof.warnings}

    def op_voice_delete(self, r):
        return {"deleted": vpmod.delete_profile(_str(r, "name", maxlen=40), self.voices_dir)}

    def op_voice_engines(self, r):
        prof = vpmod.VoiceProfile.load(r["name"], self.voices_dir) if r.get("name") else None
        return {"engines": cvmod.engines_status(prof)}

    def op_voice_convert(self, r):
        prof = vpmod.VoiceProfile.load(_str(r, "voice", maxlen=40), self.voices_dir)
        a, sr = read_wav(self.path(_str(r, "path", maxlen=200), True, (".wav",)))
        res = cvmod.convert(to_mono(a), sr, prof, _str(r, "engine", "auto", 16), _num(r, "pitch_strength", 1.0, 0.0, 1.0),
                            _num(r, "timbre_strength", 1.0, 0.0, 1.0))
        out = self.path(_str(r, "out", "converted.wav", 200), suffixes=(".wav",))
        write_wav(out, res.audio, res.sr, 16)
        side = cvmod.write_provenance(out, res.provenance)
        return {"file": self.rel(out), "provenance": self.rel(side), "engine": res.engine, "info": res.info}

    def op_studio_build(self, r):
        parts = _num(r, "parts", 4, integer=True)
        if parts not in (2, 3, 4, 6):
            raise ProtocolError("out_of_range", "parts must be 2, 3, 4 or 6")
        key = _str(r, "key", "C", 3).upper()
        if key not in _PC:
            raise ProtocolError("bad_request", f"unknown key {key!r}")
        mode = _str(r, "mode", "major", 16)
        tempo = _num(r, "tempo", 100.0, 20.0, 300.0)
        doubles = _num(r, "doubles", 0, 0, 8, integer=True)
        bg = _str(r, "bg", "airy", 16)
        if bg not in ("airy", "soulful", "rhythmic", "powerful", "none"):
            raise ProtocolError("bad_request", "bg must be airy, soulful, rhythmic, powerful or none")
        if "lead" in r:
            lead = parse_notes(r["lead"])
        elif "midi" in r:
            lead, bpm = midimod.lead_from_midi(self.path(r["midi"], True, (".mid", ".midi")), r.get("track"),
                                               _str(r, "poly", "reject", 8))
            if "tempo" not in r:
                tempo = float(np.clip(bpm, 20, 300))
        else:
            raise ProtocolError("bad_request", "provide 'lead' (notes) or 'midi' (workspace path)")
        s = build_session(lead, parts, key, mode, doubles, bg, tempo)
        self.session = s
        if self.project is not None:
            self.project.session = s
        return session_dict(s)

    def op_studio_state(self, r):
        return session_dict(self.session, bool(r.get("include_notes", False)))

    def op_studio_set_track(self, r):
        t = self.session.track(_str(r, "name", maxlen=64))
        changed = {}
        for k, v in (r.get("set") or {}).items():
            if k in TRACK_FIELDS:
                lo, hi = TRACK_FIELDS[k]
                setattr(t.settings, k, _num({"v": v}, "v", None, lo, hi))
            elif k in ("mute", "solo"):
                if not isinstance(v, bool):
                    raise ProtocolError("bad_request", f"{k!r} must be true or false")
                setattr(t.settings, k, v)
            elif k == "seed":
                setattr(t.settings, k, _num({"v": v}, "v", None, 0, 2 ** 31 - 1, integer=True))
            else:
                raise ProtocolError("bad_request", f"unknown track field {k!r}")
            changed[k] = getattr(t.settings, k)
        return {"track": t.name, "changed": changed}

    def op_studio_set_notes(self, r):
        t = self.session.track(_str(r, "name", maxlen=64))
        t.notes = parse_notes(r.get("notes"))
        return {"track": t.name, "note_count": len(t.notes)}

    def op_studio_set_renderer(self, r):
        name = _str(r, "name", "formant", 16)
        kw = {}
        if name == "sample":
            kw["path"] = str(self.path(_str(r, "path", maxlen=200), True, (".wav",)))
            if "root_midi" in r:
                kw["root_midi"] = _num(r, "root_midi", None, 0, 127)
        self.renderer, self.renderer_name = get_renderer(name, **kw), name
        return {"renderer": name, "label": getattr(self.renderer, "label", name)}

    def op_studio_render(self, r):
        bits = _num(r, "bits", 16, integer=True)
        if bits not in (16, 24):
            raise ProtocolError("out_of_range", "bits must be 16 or 24")
        workers = _num(r, "workers", 0, 0, 8, integer=True)
        out = self.path(_str(r, "out_dir", "render", 200))
        if not self.session.tracks:
            raise ProtocolError("empty", "no tracks; call studio.build first")
        m = VocalMixer(self.session, self.renderer, workers)
        paths = m.export(out, stems=bool(r.get("stems", True)), normalize=bool(r.get("normalize", True)), bits=bits)
        a, sr = read_wav(paths["mix"])
        return {"files": {k: self.rel(v) for k, v in paths.items()}, "duration_s": round(len(a) / sr, 3), "sr": sr,
                "renderer": getattr(self.renderer, "label", self.renderer_name)}

    def op_midi_export(self, r):
        p = midimod.write_midi(self.session, self.path(_str(r, "out", "arrangement.mid", 200), suffixes=(".mid", ".midi")))
        return {"file": self.rel(p)}

    def op_midi_import_lead(self, r):
        lead, bpm = midimod.lead_from_midi(self.path(_str(r, "path", maxlen=200), True, (".mid", ".midi")),
                                           r.get("track"), _str(r, "poly", "reject", 8))
        return {"tempo_bpm": round(bpm, 3), "notes": [[n.pitch, round(n.start, 4), round(n.dur, 4), round(n.vel, 3)] for n in lead]}

    def op_pitch_track(self, r):
        a, sr = read_wav(self.path(_str(r, "path", maxlen=200), True, (".wav",)))
        notes = pitchmod.track_notes(to_mono(a), sr, quantize=bool(r.get("quantize", True)))
        return {"notes": [[n.pitch, round(n.start, 4), round(n.dur, 4), round(n.vel, 3)] for n in notes]}

    def op_project_create(self, r):
        self.project = Project.create(self.path(_str(r, "dir", maxlen=200)), _str(r, "name", "untitled", 80))
        self.project.session = self.session
        self.project.save()
        return {"dir": self.rel(self.project.root)}

    def op_project_open(self, r):
        self.project = Project.open(self.path(_str(r, "dir", maxlen=200), True))
        self.session = self.project.session
        return {"name": self.project.name, "clips": len(self.project.clips), "tracks": len(self.session.tracks)}

    def op_project_save(self, r):
        if self.project is None:
            raise ProtocolError("no_project", "create or open a project first")
        self.project.session = self.session
        return {"file": self.rel(self.project.save())}

    def op_project_verify(self, r):
        if self.project is None:
            raise ProtocolError("no_project", "create or open a project first")
        probs = self.project.verify()
        return {"ok": not probs, "problems": probs}

    def op_consent_add(self, r):
        rec = consentmod.add_consent(self.path(_str(r, "file", maxlen=200), True, (".wav",)), _str(r, "owner", maxlen=80),
                                     _str(r, "scope", maxlen=200), r.get("attested") is True)
        return {"consent_id": rec["consent_id"], "file_sha256": rec["file_sha256"]}


def serve(workspace=".", stdin=None, stdout=None):
    """JSON lines in, JSON lines out. One request per line (max 1 MB). Ends on EOF."""
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    eng = Engine(workspace)
    for line in stdin:
        if not line.strip():
            continue
        if len(line) > 1_000_000:
            resp = {"ok": False, "error": {"code": "too_large", "message": "request line over 1 MB"}}
        else:
            try:
                resp = eng.handle(json.loads(line))
            except ValueError:
                resp = {"ok": False, "error": {"code": "bad_json", "message": "invalid JSON"}}
        stdout.write(json.dumps(resp) + "\n")
        stdout.flush()
