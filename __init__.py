"""AOC Magic Studio, standalone edition: numpy-only audio engine (speech, vocal arrangement, mixing, MIDI, pitch)."""
__version__ = "5.1.0"
EDITION = "standalone"


def _bootstrap_vendor():
    """If numpy is not importable, use the bundled copy in vendor/<platform>-cp<XY>/ when one exists for this OS/arch/Python.
    Uses find_spec so numpy is NOT imported here (keeps `aoc --help` fast). Override the search root with AOC_VENDOR."""
    import importlib.util, os, sys, sysconfig
    if importlib.util.find_spec("numpy") is not None:
        return
    tag = f"{sysconfig.get_platform()}-cp{sys.version_info[0]}{sys.version_info[1]}"
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # source tree, or the .pyz file itself
    argv0 = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else here
    roots = [os.environ.get("AOC_VENDOR"), os.path.join(here, "vendor"), os.path.join(argv0, "vendor"),
             os.path.join(os.path.dirname(argv0), "vendor")]
    for r in roots:
        if r and os.path.isdir(os.path.join(r, tag, "numpy")):
            sys.path.insert(0, os.path.join(r, tag))
            return


_bootstrap_vendor()
