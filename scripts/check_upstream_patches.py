"""Report whether the upstream submodule's tracked files equal its pinned commit plus our patch series.

Applies patches/*.patch in name order to the pinned tree in a throwaway index and compares
the result with the working tree, so patches that touch the same file are handled. Prints
JSON; exit code 0 if the working tree matches the series, 1 otherwise.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = REPO_ROOT / "third_party" / "rosetta-neuron-scaling"


def git(*args: str, env: dict | None = None) -> str:
    return subprocess.run(["git", "-C", str(UPSTREAM), *args], capture_output=True, text=True, check=True,
                          env={**os.environ, **(env or {})}).stdout.strip()


def main() -> int:
    patches = sorted(REPO_ROOT.glob("patches/*.patch"))
    with tempfile.TemporaryDirectory() as tmp:
        series = {"GIT_INDEX_FILE": f"{tmp}/series"}
        git("read-tree", "HEAD", env=series)
        try:
            for p in patches:
                git("apply", "--cached", str(p), env=series)
            series_tree = git("write-tree", env=series)
        except subprocess.CalledProcessError as e:
            series_tree = f"error: {e.stderr.strip()}"
        work = {"GIT_INDEX_FILE": f"{tmp}/work"}
        git("read-tree", "HEAD", env=work)
        git("add", "-u", ".", env=work)  # tracked files only; ignores __pycache__
        work_tree = git("write-tree", env=work)
    out = {
        "upstream_sha": git("rev-parse", "HEAD"),
        "patches": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12] for p in patches},
        "matches_patch_series": series_tree == work_tree,
    }
    print(json.dumps(out))
    return 0 if out["matches_patch_series"] else 1


if __name__ == "__main__":
    sys.exit(main())
