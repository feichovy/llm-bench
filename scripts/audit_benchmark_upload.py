#!/usr/bin/env python3
"""Read-only audit of benchmark files and commits before a Git push.

Run from the project repository root. --base-ref must identify the fetched
remote branch. This script does not stage, unstage, commit, or rewrite files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import PurePosixPath
import re
import subprocess
import sys


MAX_BLOB_BYTES = 90 * 1024 * 1024
STAGED_DIRECTORIES = frozenset(("configs", "logs", "results", "scripts"))
FORBIDDEN_DIRECTORIES = frozenset(
    ("models", "engines", "weights", "checkpoints", "build", "cmakefiles",
     ".venv", "venv", "__pycache__", ".cache")
)
FORBIDDEN_SUFFIXES = frozenset(
    (".gguf", ".ggml", ".safetensors", ".pt", ".pth", ".ckpt", ".bin",
     ".o", ".obj", ".a", ".lib", ".dll", ".dylib", ".exe", ".cubin",
     ".fatbin", ".ptx", ".whl", ".model", ".pyc", ".pyo", ".onnx", ".npy", ".npz", ".pkl",
     ".pickle", ".h5", ".hdf5", ".zip", ".gz", ".bz2", ".xz", ".7z",
     ".tar", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp")
)
TOKEN_PATTERN = re.compile(
    rb"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{36,255}|"
    rb"github_pat_[A-Za-z0-9_]{50,255})(?![A-Za-z0-9_])"
)
PRIVATE_KEY_PATTERN = re.compile(
    rb"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----|"
    rb"-----BEGIN PGP " + rb"PRIVATE KEY BLOCK-----"
)


class AuditError(Exception):
    pass


def git(*arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        # Do not echo arbitrary configuration, command arguments, or stderr:
        # these can contain credential-bearing URLs in a misconfigured repo.
        raise AuditError(f"Git {arguments[0]} failed; no repository state was changed.")
    return result.stdout


def resolve_commit(ref: str) -> str:
    return git("rev-parse", "--verify", "--end-of-options", ref + "^{commit}").decode("ascii").strip()


def paths_from(data: bytes):
    return (item.decode("utf-8", "surrogateescape") for item in data.split(b"\0") if item)


def staged_path_allowed(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return path == ".gitignore" or (len(parts) > 1 and parts[0] in STAGED_DIRECTORIES)


def forbidden_path(path: str) -> bool:
    parts = PurePosixPath(path.lower()).parts
    if any(
        part in FORBIDDEN_DIRECTORIES
        or part.startswith(("build-", "build_", "cmake-build-", ".venv-", ".venv_", "venv-", "venv_"))
        for part in parts[:-1]
    ):
        return True
    # A Git submodule can occupy a directory path without a trailing filename.
    if parts and parts[0] in ("models", "engines"):
        return True
    return PurePosixPath(path.lower()).suffix in FORBIDDEN_SUFFIXES or bool(
        re.search(r"\.so(?:\.[0-9]+)*$", path.lower())
    )


class Audit:
    def __init__(self):
        self.blob_findings: dict[str, tuple[str, ...]] = {}
        self.issues: dict[tuple[str, str], str] = {}

    def reject(self, path: str, reason: str, location: str):
        self.issues.setdefault((path, reason), location)

    def inspect_blob(self, oid: str) -> tuple[str, ...]:
        if oid in self.blob_findings:
            return self.blob_findings[oid]
        size = int(git("cat-file", "-s", oid))
        findings = []
        if size > MAX_BLOB_BYTES:
            findings.append("blob exceeds the 90 MiB upload limit")
        else:
            data = git("cat-file", "blob", oid)
            if b"\0" in data:
                findings.append("binary content (NUL byte)")
            else:
                try:
                    data.decode("utf-8")
                except UnicodeDecodeError:
                    findings.append("non-UTF-8 or binary content")
            if TOKEN_PATTERN.search(data):
                findings.append("possible GitHub authentication token")
            if PRIVATE_KEY_PATTERN.search(data):
                findings.append("private-key header")
        self.blob_findings[oid] = tuple(findings)
        return self.blob_findings[oid]

    def inspect_entry(self, mode: str, oid: str, path: str, location: str):
        if forbidden_path(path):
            self.reject(path, "model, build, environment, or binary artifact path", location)
            return
        if mode == "160000":
            self.reject(path, "nested repository/submodule entry", location)
            return
        if mode == "120000":
            self.reject(path, "symbolic-link entry", location)
            return
        if mode not in ("100644", "100755"):
            self.reject(path, "unsupported Git entry mode", location)
            return
        for reason in self.inspect_blob(oid):
            self.reject(path, reason, location)

    def report(self) -> bool:
        if not self.issues:
            return True
        print("FAIL: upload audit rejected these paths:", file=sys.stderr)
        for (path, reason), location in list(self.issues.items())[:30]:
            print(f"  {json.dumps(path, ensure_ascii=True)}: {reason} ({location})", file=sys.stderr)
        if len(self.issues) > 30:
            print(f"  ... {len(self.issues) - 30} additional findings", file=sys.stderr)
        print("No file contents or credentials were printed; Git state was not changed.", file=sys.stderr)
        return False


def run(base_ref: str) -> int:
    if git("rev-parse", "--show-prefix").strip():
        raise AuditError("Run this audit from the project repository root.")
    base = resolve_commit(base_ref)
    head = resolve_commit("HEAD")
    audit = Audit()
    index_entries = git("ls-files", "--stage", "-z").split(b"\0")
    index_count = 0
    unmerged = False
    for record in index_entries:
        if not record:
            continue
        fields, raw_path = record.split(b"\t", 1)
        mode, oid, stage = fields.decode("ascii").split()
        path = raw_path.decode("utf-8", "surrogateescape")
        if stage != "0":
            audit.reject(path, "unresolved merge conflict", "index")
            unmerged = True
        else:
            index_count += 1
            audit.inspect_entry(mode, oid, path, "index")

    if unmerged:
        audit.report()
        return 1

    for path in paths_from(git("diff", "--cached", "--no-renames", "--name-only", "-z", head, "--")):
        if not staged_path_allowed(path):
            audit.reject(path, "staged change is outside the benchmark upload scope", "index")

    commits = git("rev-list", "--reverse", f"{base}..{head}").decode("ascii").splitlines()
    history_count = 0
    for commit in commits:
        for record in git("ls-tree", "-r", "-z", "--full-tree", commit).split(b"\0"):
            if not record:
                continue
            fields, raw_path = record.split(b"\t", 1)
            mode, _kind, oid = fields.decode("ascii").split()
            path = raw_path.decode("utf-8", "surrogateescape")
            history_count += 1
            audit.inspect_entry(mode, oid, path, f"outgoing commit {commit[:12]}")

    if not audit.report():
        return 1
    print(
        f"PASS: {index_count} index files; {len(commits)} outgoing commits "
        f"({history_count} tree entries); {len(audit.blob_findings)} unique blobs checked."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", required=True, help="Existing fetched remote branch, e.g. origin/main")
    args = parser.parse_args()
    try:
        return run(args.base_ref)
    except (AuditError, OSError, ValueError, UnicodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
