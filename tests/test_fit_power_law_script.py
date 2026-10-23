import importlib.util
import json
import sys

import pandas as pd

from rosetta_repl import REPO_ROOT

_spec = importlib.util.spec_from_file_location("fit_power_law_script", REPO_ROOT / "scripts" / "fit_power_law.py")
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)


def test_skips_k_group_with_one_distinct_size(tmp_path, monkeypatch, capsys):
    csv = tmp_path / "rosetta_counts.csv"
    pd.DataFrame({
        "k": [1, 1, 1, 5, 5],
        "x_mean_neurons": [1e5, 4e5, 1e6, 1e5, 1e5],
        "n_rosetta": [100, 200, 300, 50, 60],
        "done": True,
    }).to_csv(csv, index=False)
    monkeypatch.setattr(sys, "argv", ["fit_power_law.py", str(csv)])
    script.main()
    fits = json.loads((tmp_path / "power_law_fit.json").read_text())["fits"]
    assert set(fits) == {"k1"}
    assert "skipping k=5" in capsys.readouterr().out
