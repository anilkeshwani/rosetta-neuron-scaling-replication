import importlib.util
import json

from rosetta_repl import REPO_ROOT

_spec = importlib.util.spec_from_file_location(
    "collect_rosetta_counts", REPO_ROOT / "scripts" / "collect_rosetta_counts.py")
collect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect)


def _write(path, n):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([[i, i] for i in range(n)]))


def test_primary_file_wins_and_stale_sweeps_ignored(tmp_path):
    corr = tmp_path / "correlations"
    _write(corr / "best_buddies.json", 10)
    _write(corr / "best_buddies_k1.json", 99)
    _write(corr / "best_buddies_k5.json", 50)
    _write(corr / "best_buddies_k10.json", 70)  # stale, not in K_NEIGHBORS_LIST
    k_sweep = collect.sweep_ks({"K_NEIGHBORS_LIST": "1 5"})
    assert collect.counts_for_run(tmp_path, 1, k_sweep) == {1: 10, 5: 50}


def test_no_sweep_list_reports_primary_only(tmp_path):
    corr = tmp_path / "correlations"
    _write(corr / "best_buddies.json", 10)
    _write(corr / "best_buddies_k2.json", 20)
    assert collect.sweep_ks({"K_NEIGHBORS_LIST": None}) == set()
    assert collect.counts_for_run(tmp_path, 1, collect.sweep_ks({})) == {1: 10}


def test_anchor_dirs_filtered_by_sweep_list(tmp_path):
    _write(tmp_path / "rosetta" / "rosetta_anchors.json", 3)
    _write(tmp_path / "rosetta_k2" / "rosetta_anchors.json", 4)
    _write(tmp_path / "rosetta_k10" / "rosetta_anchors.json", 5)
    assert collect.counts_for_run(tmp_path, 1, {2}) == {1: 3, 2: 4}
