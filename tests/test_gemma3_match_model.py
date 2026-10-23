"""CPU checks that upstream's match_model.py (with patch 0002) handles Gemma 3.

Gemma 3 at 4B and up ships as Gemma3ForConditionalGeneration; the patch loads only the
text decoder as Gemma3ForCausalLM. The 270M and 1B checkpoints are text-only already.
The real 4B test loads ~17 GB on CPU and only runs with RUN_GEMMA3_4B=1.
"""

import copy
import importlib.util
import os
from pathlib import Path

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

MATCH_MODEL = (
    Path(__file__).resolve().parents[1]
    / "third_party/rosetta-neuron-scaling/language/find_pairs/match_model.py"
)
B, T = 2, 16


def _load_match_model():
    spec = importlib.util.spec_from_file_location("upstream_match_model", MATCH_MODEL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mm = _load_match_model()


def _hub_config(repo_id):
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(repo_id)
    except (OSError, ValueError) as e:  # offline, no token, gated
        pytest.skip(f"cannot fetch {repo_id} config: {e}")


def _capture(model, model_id, input_ids, attention_mask):
    layers = mm.get_mlp_post_activation_modules(model, model_id)
    cap = mm.MultiActivationCapture([m for _, m, _ in layers], [c for _, _, c in layers]).register()
    with torch.no_grad():
        mm.make_backbone_forward(model)(input_ids=input_ids, attention_mask=attention_mask)
    acts = cap.get_and_clear(expected_batch=input_ids.shape[0], expected_seq=input_ids.shape[1])
    cap.remove()
    return layers, acts


def _inputs(vocab, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, min(vocab, 32000), (B, T), generator=g)
    mask = torch.ones(B, T, dtype=torch.long)
    mask[1, -4:] = 0
    return ids, mask


def _decoder_hooks(model):
    """Hook the down_proj input directly on a Gemma3ForConditionalGeneration, for reference."""
    return [layer.mlp.down_proj for layer in model.model.language_model.layers]


def test_tiny_multimodal_gemma3_loads_text_only(tmp_path):
    from transformers import Gemma3ForCausalLM, Gemma3ForConditionalGeneration

    cfg = copy.deepcopy(_hub_config("google/gemma-3-4b-pt"))
    tc = cfg.text_config
    tc.num_hidden_layers, tc.hidden_size, tc.intermediate_size = 3, 64, 96
    tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 2, 1, 32
    tc.sliding_window = 8
    tc.layer_types = tc.layer_types[: tc.num_hidden_layers]
    vc = cfg.vision_config
    vc.num_hidden_layers, vc.hidden_size, vc.intermediate_size = 1, 32, 64
    vc.num_attention_heads, vc.image_size, vc.patch_size = 2, 56, 14
    cfg.mm_tokens_per_image = 4

    torch.manual_seed(0)
    full = Gemma3ForConditionalGeneration(cfg).eval()
    path = tmp_path / "gemma-3-tiny-pt"
    full.save_pretrained(path)

    # The stand-in for the 4B checkpoint fails upstream's lookup.
    with pytest.raises(ValueError):
        mm.get_mlp_post_activation_modules(full, str(path))

    loader = mm._gemma3_text_only_loader(str(path), None)
    assert loader is not None
    _, info = loader(str(path), dtype=torch.float32, output_loading_info=True)
    assert info["missing_keys"] == []
    assert all(k.startswith(("vision_tower.", "multi_modal_projector.")) for k in info["unexpected_keys"])
    assert info["mismatched_keys"] == []

    model = mm._load_model(str(path), torch.float32, torch.device("cpu"))
    assert type(model) is Gemma3ForCausalLM
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()

    ids, mask = _inputs(tc.vocab_size)
    layers, acts = _capture(model, str(path), ids, mask)
    assert len(layers) == tc.num_hidden_layers
    assert all(a.shape == (B, T, tc.intermediate_size) for a in acts)

    # Same activations as hooking the decoder inside the full multimodal model.
    ref = mm.MultiActivationCapture(_decoder_hooks(full)).register()
    with torch.no_grad():
        full(input_ids=ids, attention_mask=mask, use_cache=False)
    ref_acts = ref.get_and_clear(B, T)
    ref.remove()
    for a, r in zip(acts, ref_acts):
        torch.testing.assert_close(a, r, rtol=0, atol=0)

    classes = mm._resolve_transformer_layer_classes(model, str(path))
    assert any(c.__name__ == "Gemma3DecoderLayer" for c in classes)
    assert any(isinstance(m, tuple(classes)) for m in model.modules())


def test_gemma3_270m_text_only_path_unchanged():
    from transformers import Gemma3ForCausalLM

    model_id = "google/gemma-3-270m"
    cfg = _hub_config(model_id)
    assert cfg.model_type == "gemma3_text"
    assert mm._gemma3_text_only_loader(model_id, None) is None
    model = mm._load_model(model_id, torch.float32, torch.device("cpu"))
    assert type(model) is Gemma3ForCausalLM
    ids, mask = _inputs(cfg.vocab_size)
    layers, acts = _capture(model, model_id, ids, mask)
    assert len(layers) == cfg.num_hidden_layers
    assert all(a.shape == (B, T, cfg.intermediate_size) for a in acts)
    assert all(torch.isfinite(a).all() for a in acts)


def test_non_gemma_ids_skip_the_gemma3_branch():
    for model_id in ["EleutherAI/pythia-70m", "gpt2", "facebook/opt-125m", "Qwen/Qwen2.5-0.5B"]:
        assert mm._gemma3_text_only_loader(model_id, None) is None


@pytest.mark.skipif(
    os.environ.get("RUN_GEMMA3_4B") != "1", reason="set RUN_GEMMA3_4B=1 (loads ~17 GB on CPU)"
)
def test_real_gemma3_4b_matches_full_model():
    from transformers import AutoModelForCausalLM, Gemma3ForCausalLM

    model_id = "google/gemma-3-4b-pt"
    cfg = _hub_config(model_id)
    loader = mm._gemma3_text_only_loader(model_id, None)
    _, info = loader(model_id, dtype=torch.bfloat16, output_loading_info=True)
    assert info["missing_keys"] == []
    assert all(k.startswith(("vision_tower.", "multi_modal_projector.")) for k in info["unexpected_keys"])
    del _

    model = mm._load_model(model_id, torch.bfloat16, torch.device("cpu"))
    assert type(model) is Gemma3ForCausalLM
    ids, mask = _inputs(cfg.text_config.vocab_size)
    ids[:, 0] = cfg.text_config.bos_token_id
    layers, acts = _capture(model, model_id, ids, mask)
    assert len(layers) == cfg.text_config.num_hidden_layers == 34
    assert all(a.shape == (B, T, 10240) for a in acts)
    assert all(torch.isfinite(a.float()).all() for a in acts)
    del model

    full = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).eval()
    ref = mm.MultiActivationCapture(_decoder_hooks(full)).register()
    with torch.no_grad():
        full(input_ids=ids, attention_mask=mask, use_cache=False)
    ref_acts = ref.get_and_clear(B, T)
    for a, r in zip(acts, ref_acts):
        torch.testing.assert_close(a, r, rtol=0, atol=0)
