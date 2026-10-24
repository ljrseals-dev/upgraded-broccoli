"""aoc: command line for AOC Magic Studio (standalone). Exit codes: 0 ok, 1 failure, 2 bad input."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from . import __version__, EDITION

HELP_EPILOG = "No network, GPU, or DAW required. Speech: espeak-ng if installed, else built-in (experimental)."


def _voice_opts(p, with_voice=True):
    if with_voice:
        p.add_argument("--voice", help="enrolled voice profile to speak/convert as (see `aoc voice enroll`)")
    p.add_argument("--voices-dir", help="where voice profiles live (default ~/.aoc_magic/voices, or $AOC_VOICES)")
    p.add_argument("--voice-engine", default="auto", help="auto, exemplar, ltas (or a registered engine)")
    p.add_argument("--pitch-strength", type=float, default=1.0, help="0..1 share of the pitch gap to close")
    p.add_argument("--timbre-strength", type=float, default=1.0, help="0..1 how far to move the timbre toward the reference")


def _p():
    ap = argparse.ArgumentParser(prog="aoc", description=f"AOC Magic Studio {__version__} ({EDITION})", epilog=HELP_EPILOG)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version")
    sub.add_parser("doctor", help="check the environment")
    sub.add_parser("engines", help="list speech engines and vocal renderers")
    sp = sub.add_parser("speak", help="text to speech")
    sp.add_argument("text"); sp.add_argument("--out", required=True)
    sp.add_argument("--engine", default="auto", choices=("auto", "builtin", "espeak"))
    sp.add_argument("--rate", type=float, default=1.0); sp.add_argument("--pitch-hz", type=float, default=118.0)
    sp.add_argument("--character", action="store_true", help="apply the AOC character chain (+2 st, 1.15x, compress, saturate)")
    sp.add_argument("--bits", type=int, default=16, choices=(16, 24))
    _voice_opts(sp)
    st = sub.add_parser("studio", help="vocal arrangement studio")
    ss = st.add_subparsers(dest="scmd", required=True)
    d = ss.add_parser("demo"); d.add_argument("--out", required=True)
    for name in ("build", "render"):
        b = ss.add_parser(name)
        if name == "build":
            g = b.add_mutually_exclusive_group(required=True)
            g.add_argument("--lead", help="JSON list of notes"); g.add_argument("--midi", help="MIDI file with the lead")
            b.add_argument("--track"); b.add_argument("--poly", default="reject", choices=("reject", "top"))
            b.add_argument("--parts", type=int, default=4, choices=(2, 3, 4, 6)); b.add_argument("--key", default="C")
            b.add_argument("--mode", default="major", choices=("major", "minor")); b.add_argument("--doubles", type=int, default=0)
            b.add_argument("--bg", default="airy", choices=("airy", "soulful", "rhythmic", "powerful", "none"))
            b.add_argument("--tempo", type=float)
        else:
            b.add_argument("--session", required=True, help="session.json from a previous build")
        b.add_argument("--out", required=True); b.add_argument("--bits", type=int, default=16, choices=(16, 24))
        b.add_argument("--no-normalize", action="store_true"); b.add_argument("--workers", type=int, default=0)
        b.add_argument("--renderer", default="formant", choices=("formant", "sample"))
        b.add_argument("--sample", help="your own WAV recording (needs a consent record)")
        b.add_argument("--root-midi", type=float); b.add_argument("--midi-out", action="store_true")
    mi = sub.add_parser("midi"); ms = mi.add_subparsers(dest="mcmd", required=True)
    me = ms.add_parser("export"); me.add_argument("session"); me.add_argument("out")
    ml = ms.add_parser("lead"); ml.add_argument("file"); ml.add_argument("--track"); ml.add_argument("--poly", default="reject", choices=("reject", "top"))
    pt = sub.add_parser("pitch", help="track notes in a solo vocal WAV"); pt.add_argument("wav"); pt.add_argument("--no-quantize", action="store_true")
    pj = sub.add_parser("project"); ps = pj.add_subparsers(dest="pcmd", required=True)
    pn = ps.add_parser("new"); pn.add_argument("dir"); pn.add_argument("--name", default="untitled")
    pv = ps.add_parser("verify"); pv.add_argument("dir")
    co = sub.add_parser("consent", help="consent records for reference voices"); cs = co.add_subparsers(dest="ccmd", required=True)
    ca = cs.add_parser("add"); ca.add_argument("--file", required=True); ca.add_argument("--owner", required=True)
    ca.add_argument("--scope", required=True)
    ca.add_argument("--attest", action="store_true", help="I own this voice or have permission for this use")
    cs.add_parser("list"); cr = cs.add_parser("revoke"); cr.add_argument("id")
    cv = sub.add_parser("convert", help="convert a recording toward an enrolled voice (needs consent for that voice)")
    cv.add_argument("input"); cv.add_argument("--out", required=True); cv.add_argument("--bits", type=int, default=16, choices=(16, 24))
    _voice_opts(cv)
    vo = sub.add_parser("voice", help="enroll and manage reference voices"); vs = vo.add_subparsers(dest="vcmd", required=True)
    ve = vs.add_parser("enroll", help="build a voice profile from reference WAV(s)")
    ve.add_argument("--name", required=True); ve.add_argument("--sample", action="append", required=True, help="reference WAV (repeat for several)")
    ve.add_argument("--owner", help="whose voice this is (required with --attest)"); ve.add_argument("--scope", help="what the voice may be used for")
    ve.add_argument("--attest", action="store_true", help="I own this voice or have permission to use it for this scope; records consent")
    ve.add_argument("--voices-dir")
    vs.add_parser("list").add_argument("--voices-dir")
    vsh = vs.add_parser("show"); vsh.add_argument("name"); vsh.add_argument("--voices-dir")
    vd = vs.add_parser("delete"); vd.add_argument("name"); vd.add_argument("--voices-dir")
    vs.add_parser("engines").add_argument("--voices-dir")
    sv = sub.add_parser("serve", help="JSON-lines protocol on stdin/stdout (embed in your own DAW)"); sv.add_argument("--workspace", default=".")
    ui = sub.add_parser("ui", help="local browser UI"); ui.add_argument("--workspace", default="aoc_workspace")
    ui.add_argument("--port", type=int, default=8765); ui.add_argument("--no-browser", action="store_true")
    return ap


def _renderer(ns):
    from .renderers import get_renderer
    if ns.renderer == "sample":
        if not ns.sample:
            raise ValueError("--renderer sample needs --sample FILE.wav")
        return get_renderer("sample", path=ns.sample, root_midi=ns.root_midi)
    return get_renderer("formant")


def _export(sess, ns, outdir):
    from .mix import VocalMixer
    from .midi import write_midi
    paths = VocalMixer(sess, _renderer(ns), ns.workers).export(outdir, normalize=not ns.no_normalize, bits=ns.bits)
    sess.save(Path(outdir) / "session.json")
    if ns.midi_out:
        paths["midi"] = write_midi(sess, Path(outdir) / "arrangement.mid")
    print("renderer:", getattr(_renderer(ns), "label", ns.renderer))
    for k, v in paths.items():
        print(f"  {k}: {v}")


def main(argv=None) -> int:
    ns = _p().parse_args(argv)
    try:
        return _run(ns)
    except ModuleNotFoundError as e:
        if e.name == "numpy":
            import sysconfig
            print("aoc: numpy is required and was not found. Install it with `pip install numpy`, or use a build that "
                  f"includes vendor/{sysconfig.get_platform()}-cp{sys.version_info[0]}{sys.version_info[1]}/. Run `aoc doctor` for details.",
                  file=sys.stderr)
            return 2
        raise
    except BrokenPipeError:                         # e.g. `aoc ... | head`
        try:
            import os
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except Exception:
            pass
        return 0
    except (ValueError, OSError, KeyError) as e:
        print(f"aoc: error: {e}", file=sys.stderr)
        return 2
    except PermissionError as e:
        print(f"aoc: error: {e}", file=sys.stderr)
        return 2


def _run(ns) -> int:
    from .consent import ConsentError
    try:
        return _dispatch(ns)
    except ConsentError as e:
        print(f"aoc: consent: {e}", file=sys.stderr)
        return 2


def _dispatch(ns) -> int:
    c = ns.cmd
    if c == "version":
        print(f"AOC Magic Studio {__version__} {EDITION}")
        return 0
    if c == "doctor":
        from .doctor import run_checks, format_table
        rows = run_checks()
        print(format_table(rows))
        return 1 if any(r["level"] == "FAIL" for r in rows) else 0
    if c == "engines":
        from . import speech
        from .renderers import list_renderers
        print(json.dumps({"speech": speech.engines(), "renderers": list_renderers()}, indent=2))
        return 0
    if c == "speak":
        from . import speech
        from .wavio import write_wav
        if ns.voice:
            from . import convert as cv, voiceprofile as vp
            prof = vp.VoiceProfile.load(ns.voice, ns.voices_dir)
            r = cv.speak_as(ns.text, prof, ns.engine, ns.voice_engine, ns.rate, ns.pitch_strength, ns.timbre_strength)
            write_wav(ns.out, r.audio, r.sr, ns.bits)
            side = cv.write_provenance(ns.out, r.provenance)
            print(f"{ns.out}: {len(r.audio) / r.sr:.2f} s, {r.sr} Hz, tts={r.info['tts_engine']}, conversion={r.engine}, voice={prof.name}")
            print(f"notice: converted/synthetic voice (classical timbre and pitch transfer, not a neural clone); provenance: {side}")
            return 0
        a, sr, used = speech.synthesize(ns.text, ns.engine, rate=ns.rate, pitch_hz=ns.pitch_hz, character=ns.character)
        write_wav(ns.out, a, sr, ns.bits)
        print(f"{ns.out}: {len(a) / sr:.2f} s, {sr} Hz, engine={used}")
        return 0
    if c == "convert":
        from . import convert as cv, voiceprofile as vp
        from .wavio import read_wav, to_mono, write_wav
        if not ns.voice:
            raise ValueError("--voice NAME is required")
        prof = vp.VoiceProfile.load(ns.voice, ns.voices_dir)
        a, sr = read_wav(ns.input)
        r = cv.convert(to_mono(a), sr, prof, ns.voice_engine, ns.pitch_strength, ns.timbre_strength)
        write_wav(ns.out, r.audio, r.sr, ns.bits)
        side = cv.write_provenance(ns.out, r.provenance)
        print(f"{ns.out}: engine={r.engine}, shift={r.info.get('shift_semitones')} st; provenance: {side}")
        for k in ("pitch_note",):
            if r.info.get(k):
                print("note:", r.info[k])
        return 0
    if c == "voice":
        from . import consent as cm, voiceprofile as vp, convert as cv
        d = getattr(ns, "voices_dir", None)
        if ns.vcmd == "enroll":
            if ns.attest:
                if not ns.owner or not ns.scope:
                    raise ValueError("--attest needs --owner and --scope")
                for f in ns.sample:
                    cm.add_consent(f, ns.owner, ns.scope, True)
            prof = vp.enroll(ns.name, ns.sample)
            where = prof.save(d)
            print(f"enrolled {prof.name}: {prof.voiced_seconds:.1f} s of speech, f0 median {prof.f0['median']} Hz, bank {len(prof.bank)} frames -> {where}")
            print("engines:", ", ".join(f"{k}={'ok' if v else 'needs more audio'}" for k, v in prof.engines_ok.items()))
            for w in prof.warnings:
                print("warning:", w)
            return 0
        if ns.vcmd == "list":
            print(json.dumps(vp.list_profiles(d), indent=2))
            return 0
        if ns.vcmd == "show":
            prof = vp.VoiceProfile.load(ns.name, d)
            print(json.dumps({"name": prof.name, "f0": prof.f0, "voiced_seconds": round(prof.voiced_seconds, 2), "samples": prof.samples,
                              "engines_ok": prof.engines_ok, "warnings": prof.warnings, "created": prof.created}, indent=2))
            return 0
        if ns.vcmd == "delete":
            print("deleted" if vp.delete_profile(ns.name, d) else "not found")
            return 0
        print(json.dumps(cv.engines_status(), indent=2))
        return 0

    if c == "studio":
        from .model import Note, Session
        from .harmony import build_session
        if ns.scmd == "demo":
            lead = [Note(p, i * 0.6 + (0.5 if i >= 4 else 0), 0.55, 0.8, l) for i, (p, l) in enumerate(
                [(60, "la"), (62, "la"), (64, "lo"), (67, "lee"), (65, "la"), (64, "lo"), (62, "la"), (60, "loo")])]
            sess = build_session(lead, 4, "C", "major", 2, "airy", 100.0)
            ns2 = argparse.Namespace(renderer="formant", sample=None, root_midi=None, workers=0, no_normalize=False, bits=16, midi_out=True)
            _export(sess, ns2, ns.out)
            return 0
        if ns.scmd == "build":
            from .protocol import parse_notes, ProtocolError
            from .midi import lead_from_midi
            tempo = ns.tempo
            if ns.lead:
                try:
                    lead = parse_notes(json.loads(Path(ns.lead).read_text()))
                except ProtocolError as e:
                    raise ValueError(e.message)
            else:
                lead, bpm = lead_from_midi(ns.midi, ns.track, ns.poly)
                tempo = tempo or bpm
            sess = build_session(lead, ns.parts, ns.key, ns.mode, ns.doubles, ns.bg, tempo or 100.0)
        else:
            sess = Session.load(ns.session)
        _export(sess, ns, ns.out)
        return 0
    if c == "midi":
        from .midi import write_midi, lead_from_midi
        if ns.mcmd == "export":
            from .model import Session
            print(write_midi(Session.load(ns.session), ns.out))
        else:
            notes, bpm = lead_from_midi(ns.file, ns.track, ns.poly)
            print(json.dumps({"tempo_bpm": round(bpm, 3), "notes": [[n.pitch, round(n.start, 4), round(n.dur, 4), round(n.vel, 3)] for n in notes]}))
        return 0
    if c == "pitch":
        from .pitch import track_notes
        from .wavio import read_wav, to_mono
        a, sr = read_wav(ns.wav)
        notes = track_notes(to_mono(a), sr, quantize=not ns.no_quantize)
        print(json.dumps([[n.pitch, round(n.start, 4), round(n.dur, 4), round(n.vel, 3)] for n in notes]))
        return 0
    if c == "project":
        from .project import Project
        if ns.pcmd == "new":
            print(Project.create(ns.dir, ns.name).file)
            return 0
        probs = Project.open(ns.dir).verify()
        print(json.dumps({"ok": not probs, "problems": probs}))
        return 0 if not probs else 1
    if c == "consent":
        from . import consent as cm
        if ns.ccmd == "add":
            rec = cm.add_consent(ns.file, ns.owner, ns.scope, ns.attest)
            print(json.dumps(rec, indent=2))
        elif ns.ccmd == "list":
            print(json.dumps(cm._records(None), indent=2))
        else:
            print("revoked" if cm.revoke(ns.id) else "not found")
            return 0 if cm.revoke is not None else 1
        return 0
    if c == "serve":
        from .protocol import serve
        serve(ns.workspace)
        return 0
    if c == "ui":
        from .ui import run
        run(ns.workspace, ns.port, not ns.no_browser)
        return 0
    return 2
