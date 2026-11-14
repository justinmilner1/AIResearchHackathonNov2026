"""Manifests: the reproducibility contract.

A manifest records, for every file a run depends on, its repo-relative path, byte size and
sha256, plus the seeds and parameters that produced it. `verify_manifest` fails loudly if any
file is missing or its hash differs. Tests call it; the train loop calls it before starting.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clharness.config import REPO_ROOT


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def file_entry(path: Path, root: Path = REPO_ROOT) -> dict[str, Any]:
    path = Path(path)
    return {
        # abspath, not resolve: data/ may be a symlinked mount (a cloud volume), and the manifest
        # path must stay relative to the repo root either way
        "path": str(Path(os.path.abspath(path)).relative_to(os.path.abspath(root))),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


@dataclass
class ManifestMismatch(Exception):
    problems: list[str]

    def __str__(self) -> str:
        return "manifest mismatch:\n  " + "\n  ".join(self.problems)


def _iter_file_entries(obj: Any):
    """Yield every dict that looks like a file entry, anywhere in the manifest tree."""
    if isinstance(obj, dict):
        if {"path", "bytes", "sha256"} <= set(obj):
            yield obj
        for v in obj.values():
            yield from _iter_file_entries(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_file_entries(v)


def verify_manifest(manifest: dict[str, Any], root: Path = REPO_ROOT) -> None:
    problems: list[str] = []
    n = 0
    for e in _iter_file_entries(manifest):
        n += 1
        p = root / e["path"]
        if not p.exists():
            problems.append(f"missing: {e['path']}")
            continue
        if p.stat().st_size != e["bytes"]:
            problems.append(f"size differs: {e['path']} ({p.stat().st_size} != {e['bytes']})")
            continue
        if sha256_file(p) != e["sha256"]:
            problems.append(f"sha256 differs: {e['path']}")
    if n == 0:
        problems.append("manifest contains no file entries")
    if problems:
        raise ManifestMismatch(problems)


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def read_manifest(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p) as f:
        return json.load(f)
