"""Environment checks with honest PASS / WARN / FAIL and a remediation line for anything not PASS."""
from __future__ import annotations
import os, platform, shutil, sys, tempfile
from . import __version__, EDITION


def run_checks() -> list:
    out = []

    def add(name, level, detail, fix=""):
        out.append({"check": name, "level": level, "detail": detail, "fix": fix})

    v = sys.version_info
    add("python", "PASS" if v >= (3, 9) else "FAIL", platform.python_version(), "" if v >= (3, 9) else "install Python 3.9+")
    try:
        import numpy as np
    except ImportError:
        import sysconfig
        tag = f"{sysconfig.get_platform()}-cp{sys.version_info[0]}{sys.version_info[1]}"
        add("numpy", "FAIL", "not importable and no bundled copy for " + tag,
            "pip install numpy   (or run tools/vendor_numpy.py on a machine that has it, to create vendor/" + tag + ")")
        return out
    nv = tuple(int(x) for x in np.__version__.split(".")[:2] if x.isdigit())
    src = "bundled vendor copy" if os.sep + "vendor" + os.sep in os.path.abspath(np.__file__) else "system install"
    add("numpy", "PASS" if nv >= (1, 21) else "FAIL", f"{np.__version__} ({src})", "" if nv >= (1, 21) else "pip install -U numpy")
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    add("espeak", "PASS" if exe else "WARN", exe or "not found (built-in experimental speech will be used)",
        "" if exe else "install espeak-ng for clearer speech (apt install espeak-ng / brew install espeak-ng)")
    try:
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "t"), "w").write("x")
        add("temp dir writable", "PASS", tempfile.gettempdir())
    except OSError as e:
        add("temp dir writable", "FAIL", str(e), "check TMPDIR permissions")
    try:
        from .speech import BuiltinSpeech
        y = BuiltinSpeech(16000).synth("test")
        ok = len(y) > 1000 and bool(np.isfinite(y).all()) and float(np.max(np.abs(y))) > 0.05
        add("speech self-test", "PASS" if ok else "FAIL", f"{len(y)} samples", "" if ok else "report this build")
        from .model import Note, Track, TrackSettings
        from .voice import ParametricRenderer
        z = ParametricRenderer().render_track(Track("t", "lead", [Note(60, 0, 0.2)], TrackSettings()), 22050, 100)
        ok2 = len(z) > 1000 and bool(np.isfinite(z).all())
        add("vocal renderer self-test", "PASS" if ok2 else "FAIL", f"{len(z)} samples", "" if ok2 else "report this build")
    except Exception as e:                                   # a failing self-test is itself the finding
        add("self-test", "FAIL", f"{type(e).__name__}: {e}", "report this build")
    try:
        import tkinter  # noqa: F401
        add("tkinter (not required)", "PASS", "available")
    except ImportError:
        add("tkinter (not required)", "PASS", "not installed; the UI is browser based and does not need it")
    cons = os.environ.get("AOC_CONSENTS") or os.path.join(os.path.expanduser("~"), ".aoc_magic", "consents.jsonl")
    add("consent store", "PASS", cons + (" (exists)" if os.path.exists(cons) else " (none yet; created on first use)"))
    add("edition", "PASS", f"AOC Magic Studio {__version__} {EDITION}")
    return out


def format_table(rows) -> str:
    w = max(len(r["check"]) for r in rows)
    lines = []
    for r in rows:
        lines.append(f"[{r['level']:4}] {r['check']:<{w}}  {r['detail']}")
        if r["level"] != "PASS" and r["fix"]:
            lines.append(f"       fix: {r['fix']}")
    return "\n".join(lines)
