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



# Email Correspondence

---

## Auto-Reply: Sony Music

**From:** Sony Music  
**To:** Me  
**Received:** 20 minutes ago  
**Status:** Details

Hello,

Thank you for reaching out to us. We have received your inquiry, and our team is reviewing it. An agent will be with you shortly.

**Office Hours:** 9:00 a.m.–6:00 p.m. EST, Monday–Friday. If you contacted us outside these hours, we will respond as soon as possible when we return.

Please note that we are currently handling a high volume of inquiries. We appreciate your patience and will assist you promptly.

Please feel free to reach out with any additional questions in the meantime. We are here to help.

Best regards,  
Sony Music

---

## Original Message

**From:** Lewis Jr. `<ljrseals@gmail.com>`  
**Date:** Friday, October 23, 2026, 22:00:35 GMT  
**To:** Sony Records / Sony Music Entertainment — A&R and Business Development  
**Location:** New York, New York

**Date:** 11-23-2026

Dear Sony Records Team,

I am writing to introduce my music-technology work and to ask for guidance on the appropriate person or department at Sony to review it.

I have uploaded AOC Magic and AOC Magic Studio. The core function of the program is to take WAV files and clone the voice. I am also developing Authority, a chatbot, and a speech-to-State-forms software package. In addition, I have ElevenX and Aura G, technologies that I believe should advance music creation and production.

I also understand that Beyoncé Knowles Carter would like for me to release a book on the Endless Engine, Total Beats, and halving the cat techniques. I would welcome Sony’s guidance on the proper publishing, rights, and business-development path for that project.

My work and public repositories are available here: [github.com/ljrseals-dev](https://github.com/ljrseals-dev).

I am not asking Sony to endorse any claim without review. I am prepared to provide a technical brief, demonstration materials, and documentation, and I am committed to responsible voice-cloning practices, including consent and rights clearance.

Could you please advise me on the correct contact for technology evaluation, licensing, A&R, or publishing discussions? I would be grateful for a brief call or for direction to the appropriate department.

Thank you for your time and consideration.

Sincerely,

Lewis Seals
```
