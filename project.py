"""Project folder: project.json (session, clips, metadata) + assets/*.wav with SHA-256 manifest and drift check."""
from __future__ import annotations
import json, os, re, time
from pathlib import Path
import numpy as np
from .consent import file_sha256
from .model import Session
from .wavio import write_wav

SCHEMA_VERSION = 1


class Project:
    def __init__(self, root, session: Session | None = None, name: str = "untitled"):
        self.root = Path(root)
        self.name = name
        self.session = session or Session()
        self.clips: list[dict] = []
        self.created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    @classmethod
    def create(cls, root, name="untitled") -> "Project":
        p = cls(root, name=name)
        (p.root / "assets").mkdir(parents=True, exist_ok=True)
        p.save()
        return p

    @property
    def file(self) -> Path:
        return self.root / "project.json"

    def add_clip(self, audio, sr: int, name: str, text: str = "", engine: str = "", bits: int = 16) -> dict:
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "clip"
        base, k, used = safe, 2, {c["file"] for c in self.clips}
        while f"assets/{safe}.wav" in used:
            safe = f"{base}_{k}"
            k += 1
        rel = f"assets/{safe}.wav"
        write_wav(self.root / rel, audio, sr, bits)
        clip = {"name": name, "file": rel, "sr": int(sr), "text": text, "engine": engine,
                "samples": int(np.asarray(audio).shape[0]), "sha256": file_sha256(self.root / rel)}
        self.clips.append(clip)
        return clip

    def to_dict(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, "name": self.name, "created": self.created,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "session": self.session.to_dict(), "clips": self.clips}

    def save(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        os.replace(tmp, self.file)                                 # atomic replace: never a half-written project
        return self.file

    @classmethod
    def open(cls, root) -> "Project":
        root = Path(root)
        d = json.loads((root / "project.json").read_text())
        ver = d.get("schema_version", 0)
        if ver > SCHEMA_VERSION:
            raise ValueError(f"project schema_version {ver} is newer than supported ({SCHEMA_VERSION})")
        p = cls(root, Session.from_dict(d["session"]), d.get("name", "untitled"))
        p.clips = list(d.get("clips", []))
        p.created = d.get("created", p.created)
        for c in p.clips:                                           # reject path escapes in a hand-edited project
            if Path(c["file"]).is_absolute() or ".." in Path(c["file"]).parts:
                raise ValueError(f"unsafe clip path {c['file']!r}")
        return p

    def verify(self) -> list:
        """Return a list of problems (empty = every asset exists and matches its recorded SHA-256)."""
        out = []
        for c in self.clips:
            f = self.root / c["file"]
            if not f.is_file():
                out.append({"file": c["file"], "problem": "missing"})
            elif file_sha256(f) != c["sha256"]:
                out.append({"file": c["file"], "problem": "modified"})
        return out
