import pytest

from rosetta_repl.registry import load_registry, resolve


def test_registry_ids_unique_and_complete():
    reg = load_registry()
    assert len(reg) > 20
    for e in reg.values():
        assert e.n_neurons > 0
        assert e.mlp_type in {"gelu", "relu", "gated"}


def test_resolve_random_init_dir():
    assert resolve("/data/random_init/pythia-160m-seed2").id == "EleutherAI/pythia-160m"
    assert resolve("$ROSETTA_DATA/random_init/opt-1.3b-seed0").id == "facebook/opt-1.3b"


def test_resolve_unknown_raises():
    with pytest.raises(KeyError):
        resolve("nobody/not-a-model")


def test_upstream_family_dispatch_is_unambiguous():
    # Upstream picks hook points by substring of the model id, checked in this order.
    order = ["pythia", "gpt2", "opt", ("llama", "gemma", "qwen")]
    expected = {"pythia": "pythia", "gpt2": "gpt2", "opt": "opt"}
    for e in load_registry().values():
        name = e.id.lower()
        hit = next(k for k in order if (any(s in name for s in k) if isinstance(k, tuple) else k in name))
        want = expected.get(e.family, ("llama", "gemma", "qwen"))
        assert hit == want, f"{e.id} would dispatch to {hit}"
