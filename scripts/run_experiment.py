"""Run every pair in an experiment YAML through scripts/run_pair.sh, in order.

Each pair's env is the experiment `defaults` overlaid with the pair's own `env`; values
already set in the calling environment win, so `NUM_SAMPLES=1000000 run_experiment.py ...`
overrides the YAML. Finished pairs (DONE marker) are skipped by run_pair.sh.

If the experiment sets `token_cache_world_size: N`, each pair's token sample is built on
CPU by scripts/build_token_cache.py as an N-rank run would draw it, and passed to the
pipeline as TOKEN_CACHE. This makes the sample independent of the local GPU count; N=8
matches the authors' runs.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

from rosetta_repl import REPO_ROOT


def run_name(models: list[str], tag: str | None) -> str:
    name = "__".join(Path(os.path.expandvars(m)).name for m in models)
    return f"{name}__{tag}" if tag else name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--include-cloud", action="store_true", help="also run pairs marked requires: cloud")
    parser.add_argument("--only", help="substring filter on run names")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failed pair")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--caches-only", action="store_true", help="build missing token caches, run nothing")
    args = parser.parse_args()

    exp = yaml.safe_load(args.experiment.read_text())
    failures = []
    for pair in exp["pairs"]:
        name = run_name(pair["models"], pair.get("tag"))
        if args.only and args.only not in name:
            continue
        if pair.get("requires") == "cloud" and not args.include_cloud:
            print(f"[experiment] skip (needs cloud): {name}")
            continue
        env = {k: str(v) for k, v in {**exp.get("defaults", {}), **pair.get("env", {})}.items()}
        env = {**env, **os.environ}
        env.setdefault("ROSETTA_DATA", str(REPO_ROOT / "data"))
        models = [os.path.expandvars(m.replace("$ROSETTA_DATA", env["ROSETTA_DATA"])) for m in pair["models"]]
        if "TOKEN_CACHE" in pair.get("env", {}):  # reuse another experiment's cache, e.g. for diagnostics
            env["TOKEN_CACHE"] = os.path.expandvars(env["TOKEN_CACHE"].replace("$ROSETTA_DATA", env["ROSETTA_DATA"]))
            if not Path(env["TOKEN_CACHE"]).exists():
                sys.exit(f"[experiment] {name}: TOKEN_CACHE does not exist: {env['TOKEN_CACHE']}")
        elif ws := exp.get("token_cache_world_size"):
            cache = REPO_ROOT / "data" / "token_caches" / exp["name"] / f"{name}__ws{ws}__n{env['NUM_SAMPLES']}.pt"
            env["TOKEN_CACHE"] = str(cache)
            if not cache.exists() and not args.dry_run:
                build = [sys.executable, str(REPO_ROOT / "scripts" / "build_token_cache.py"), models[0], models[1],
                         "--world-size", str(ws), "--num-samples", env["NUM_SAMPLES"], "--out", str(cache)]
                print(f"[experiment] building token cache: {cache}", flush=True)
                if subprocess.run(build, env=env).returncode != 0:
                    failures.append(name)
                    if not args.keep_going:
                        break
                    continue
        if args.caches_only:
            continue
        cmd = ["bash", str(REPO_ROOT / "scripts" / "run_pair.sh"), *models, "--", name, exp["name"]]
        knobs = {k: env[k] for k in {**exp.get("defaults", {}), **pair.get("env", {})}}
        print(f"[experiment] {name} {knobs}", flush=True)
        if args.dry_run:
            print("  " + " ".join(cmd))
            continue
        if subprocess.run(cmd, env=env).returncode != 0:
            failures.append(name)
            if not args.keep_going:
                break
    if failures:
        print(f"[experiment] failed: {failures}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
