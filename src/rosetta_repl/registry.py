"""Load the model registry in configs/models.yaml and resolve model ids to entries."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from rosetta_repl import REPO_ROOT

REGISTRY_PATH = REPO_ROOT / "configs" / "models.yaml"
_SEED_SUFFIX = re.compile(r"-seed\d+$")


@dataclass(frozen=True)
class ModelEntry:
    id: str
    family: str
    params_b: float
    n_layers: int
    d_mlp: int
    mlp_type: str
    distilled_from: str | None
    gated_repo: bool

    @property
    def n_neurons(self) -> int:
        return self.n_layers * self.d_mlp

    @property
    def short_name(self) -> str:
        return self.id.split("/")[-1]


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, ModelEntry]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return {m["id"]: ModelEntry(**m) for m in raw["models"]}


def resolve(model: str, registry: dict[str, ModelEntry] | None = None) -> ModelEntry:
    """Map a Hub id, or a local random-init directory like .../pythia-160m-seed0, to its entry."""
    registry = registry or load_registry()
    if model in registry:
        return registry[model]
    base = _SEED_SUFFIX.sub("", Path(os.path.expandvars(model)).name)
    matches = [e for e in registry.values() if e.short_name.lower() == base.lower()]
    if len(matches) != 1:
        raise KeyError(f"Cannot resolve {model!r} to a single registry entry (got {len(matches)})")
    return matches[0]
