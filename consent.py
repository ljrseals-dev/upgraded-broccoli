"""Consent records for reference voices. A renderer that uses a recording of a person's voice must find a record for
that exact file (matched by SHA-256) before any audio is processed. Records are appended to a JSONL file."""
from __future__ import annotations
import hashlib, json, os, time, uuid
from pathlib import Path


class ConsentError(PermissionError):
    pass


def default_path() -> Path:
    return Path(os.environ.get("AOC_CONSENTS") or (Path.home() / ".aoc_magic" / "consents.jsonl"))


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _records(path):
    p = Path(path) if path else default_path()
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def add_consent(file, owner: str, scope: str, attested: bool, consents_path=None) -> dict:
    """`attested` must be True: the caller asserts the speaker is the owner or has given permission for this scope."""
    if attested is not True:
        raise ConsentError("consent requires an explicit attestation (attested=True)")
    if not owner or not scope:
        raise ValueError("owner and scope are required")
    rec = {"consent_id": uuid.uuid4().hex, "file_sha256": file_sha256(file), "file_name": Path(file).name,
           "owner": owner, "scope": scope, "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "revoked": False}
    p = Path(consents_path) if consents_path else default_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def revoke(consent_id: str, consents_path=None) -> bool:
    p = Path(consents_path) if consents_path else default_path()
    recs = _records(p)
    hit = False
    for r in recs:
        if r.get("consent_id") == consent_id:
            r["revoked"] = True
            hit = True
    if hit:
        p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in recs))
    return hit


def find_consent(file, consents_path=None):
    sha = file_sha256(file)
    for r in reversed(_records(consents_path)):
        if r.get("file_sha256") == sha and not r.get("revoked"):
            return r
    return None


def require_consent(file, consents_path=None) -> dict:
    rec = find_consent(file, consents_path)
    if rec is None:
        raise ConsentError(f"no consent record for {Path(file).name}; add one with `aoc consent add` "
                           "(only for voices you own or have permission to use)")
    return rec


def get_record(consent_id: str, consents_path=None):
    """The record with this id, or None."""
    for r in _records(consents_path):
        if r.get("consent_id") == consent_id:
            return r
    return None


def require_active(consent_id: str, consents_path=None) -> dict:
    """Raise ConsentError if the record is missing or has been revoked (so revocation takes effect immediately)."""
    r = get_record(consent_id, consents_path)
    if r is None or r.get("revoked"):
        raise ConsentError(f"consent {consent_id[:8]}... is missing or revoked")
    return r
