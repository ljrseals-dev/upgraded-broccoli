# AOC Magic Studio: Standalone Edition 5.1.0

A self-contained audio engine and tool set. **No DAW, no GPU, no network, no scipy/torch/librosa.** Python 3.9+ and
**numpy** are the only requirements. The application itself is about **42 KB** (`dist/aoc.pyz`); numpy is the only
sizeable dependency. The *bundled* edition also ships numpy + OpenBLAS for Linux x86_64 / CPython 3.12 in `vendor/` (about 50 MB
unzipped), so it runs with no numpy installed; elsewhere `pip install numpy` or run `tools/vendor_numpy.py`. No models are included or needed.

Built to be the engine under **your own DAW**: call it in-process from Python, or drive it from any language over a
JSON-lines protocol. It is an **offline render engine** (it produces audio buffers and files); it is not a real-time
audio callback engine.

## What it does
| Area | Commands | Notes |
|------|----------|-------|
| Speech | `aoc speak` | espeak-ng if installed, else the built-in formant synthesizer (**experimental, robotic**) |
| Vocal arrangement | `aoc studio build/render/demo` | lead melody to 2/3/4/6-part harmony, doubles, backgrounds, ad-libs; every part is an editable track |
| Mixing | (inside studio) | per-track EQ, de-ess, compression, saturation, level, pan, reverb and echo sends; stems + mix, 16/24-bit WAV |
| MIDI | `aoc midi export/lead` | write each part as a MIDI track; read a lead from any .mid file (tempo map, running status) |
| Pitch tracking | `aoc pitch` | YIN, solo voices; note segmentation |
| Your own voice | `--renderer sample --sample me.wav` | pitches a recording of **your** voice per note; requires a consent record |
| Reference-voice flow | `aoc voice enroll`, `aoc speak --voice`, `aoc convert` | enroll a reference WAV (consent required), then TTS or a recording is moved toward that voice: classical pitch + timbre transfer, **not a neural clone** (`docs/VOICE.md`) |
| Projects | `aoc project new/verify` | folder with `project.json` + `assets/`, SHA-256 drift detection |
| Embedding | `aoc serve`, `aoc.protocol.Engine` | JSON protocol, path-confined, range-checked |
| UI | `aoc ui` | local browser UI (127.0.0.1, token, Host/Origin checks) |
| Health | `aoc doctor` | PASS/WARN/FAIL with a fix line for anything not PASS |

## Quick start
    pip install numpy
    python -m aoc doctor
    python -m aoc studio demo --out demo          # 8 stems + mix + session.json + arrangement.mid
    python -m aoc speak "Hello world" --out hello.wav
    python -m aoc ui                              # opens the browser UI

One-file build (needs numpy on the target machine): `python tools/build_pyz.py` then `python dist/aoc.pyz doctor`.
A bundled executable (Python + numpy inside) is described in `docs/BUILD.md`; it has **not** been built here.

## Evidence
`audit/ledger.jsonl` + `audit/evidence/` (hash chained; `python audit/verify_ledger.py`), `docs/RUN_LOG.md` (what run 1 proved and what it did not).

## Read next
`docs/VOICE.md` (reference-voice flow and its measured limits) · `docs/LIMITATIONS.md` (what is not done or not verified) · `docs/ARCHITECTURE.md` · `docs/API.md` (protocol) ·
`docs/EMBEDDING.md` (use it from your DAW) · `docs/BUILD.md` · `docs/VERIFICATION.md` (what was run).
**Handing it to Claude:** open the folder in Claude Code (it reads `CLAUDE.md`) or paste `docs/CLAUDE_PROMPT.md`; `audit/` has the evidence-ledger tools it uses.

## Responsible use
Voice features only for voices you own or have permission to use. The sample renderer refuses to run without a consent
record for that exact file (matched by SHA-256). The built-in synthesizers are previews, not realistic voice clones.
Choose and add a license before distributing.
