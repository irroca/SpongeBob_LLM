import argparse
import os
import warnings

import pytest
import torch

from config import LLMConfig
from model import Whetstone
from train_utils import (
    MODEL_ARCH_FIELDS,
    add_model_args,
    describe_model,
    infer_arch_from_state_dict,
    model_arch_defaults,
    read_checkpoint_arch,
    resolve_model_config,
    save_checkpoint,
)

VOCAB = 64


def _parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seq_len", type=int, default=128)
    add_model_args(parser)
    return parser.parse_args(argv)


def _tiny(dim=32, n_layers=2, n_heads=4, n_kv_heads=2, **kwargs):
    return LLMConfig(
        dim=dim, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads,
        vocab_size=VOCAB, max_seq_len=64, dropout=0.0, **kwargs,
    )


def test_arch_defaults_track_the_config_signature():
    defaults = model_arch_defaults()
    reference = LLMConfig()
    assert set(defaults) == set(MODEL_ARCH_FIELDS)
    for name, value in defaults.items():
        if name != "hidden_dim":  # derived in __init__ when left as None
            assert getattr(reference, name) == value


def test_arch_flags_default_to_none_so_unset_is_distinguishable():
    args = _parse([])
    assert all(getattr(args, name) is None for name in MODEL_ARCH_FIELDS)


def test_resolve_falls_back_to_library_defaults():
    config = resolve_model_config(_parse([]), VOCAB)
    reference = LLMConfig()
    assert (config.dim, config.n_layers, config.n_heads) == (
        reference.dim, reference.n_layers, reference.n_heads,
    )
    assert config.vocab_size == VOCAB
    assert config.max_seq_len == 128


def test_explicit_flags_win_over_defaults():
    config = resolve_model_config(_parse(["--dim", "128", "--n_layers", "3"]), VOCAB)
    assert (config.dim, config.n_layers) == (128, 3)


def test_infer_arch_from_state_dict_recovers_shape_facts():
    model = Whetstone(_tiny(dim=32, n_layers=3, n_heads=4, n_kv_heads=2))

    arch = infer_arch_from_state_dict(model.state_dict())

    assert arch["dim"] == 32
    assert arch["n_layers"] == 3
    assert arch["vocab_size"] == VOCAB
    assert arch["hidden_dim"] == model.params.hidden_dim
    # n_heads is unrecoverable (wq is always dim x dim); only the kv/q ratio shows.
    assert "n_heads" not in arch
    assert arch["kv_q_ratio"] == pytest.approx(0.5)


def test_bare_state_dict_warns_that_n_heads_was_assumed(tmp_path):
    """A raw state_dict cannot pin down n_heads: wq is dim x dim for every head count,
    so only the kv/q ratio survives. Shapes still load, but head grouping differs."""
    path = tmp_path / "legacy_final.pth"
    torch.save(Whetstone(_tiny(dim=32, n_heads=4, n_kv_heads=2)).state_dict(), path)

    with pytest.warns(RuntimeWarning, match="n_heads"):
        config = resolve_model_config(_parse([]), VOCAB, checkpoint_path=str(path))

    assert config.dim == 32
    assert config.n_kv_heads / config.n_heads == pytest.approx(0.5)


def test_explicit_n_heads_resolves_the_ambiguity(tmp_path):
    path = tmp_path / "legacy_final.pth"
    torch.save(Whetstone(_tiny(dim=32, n_heads=4, n_kv_heads=2)).state_dict(), path)

    with pytest.warns(RuntimeWarning):
        config = resolve_model_config(_parse(["--n_heads", "4"]), VOCAB, checkpoint_path=str(path))

    assert (config.n_heads, config.n_kv_heads) == (4, 2)


def test_final_weights_sidecar_makes_the_architecture_exact(tmp_path):
    """save_final_weights writes *.config.json next to the bare state_dict, which is
    what removes the n_heads guesswork for *_final.pth files."""
    from train_utils import config_sidecar_path, save_final_weights

    path = tmp_path / "sft_final.pth"
    original = _tiny(dim=32, n_layers=3, n_heads=4, n_kv_heads=2)
    save_final_weights(str(path), Whetstone(original), original)

    assert os.path.exists(config_sidecar_path(str(path)))
    # Bare state_dict preserved, so anything already reading these files still works.
    assert isinstance(torch.load(path, weights_only=False), dict)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "n_heads assumed" warning this time
        config = resolve_model_config(_parse([]), VOCAB, checkpoint_path=str(path))

    assert (config.dim, config.n_layers, config.n_heads, config.n_kv_heads) == (32, 3, 4, 2)


def test_resolve_adopts_architecture_from_a_training_checkpoint(tmp_path):
    path = tmp_path / "latest_checkpoint.pth"
    config_in = _tiny(dim=64, n_layers=4, n_heads=8, n_kv_heads=4)
    save_checkpoint(str(path), Whetstone(config_in), None, None, 0, 0, 0, 0.0, config_in)

    config = resolve_model_config(_parse([]), VOCAB, checkpoint_path=str(path))

    assert (config.dim, config.n_layers, config.n_heads, config.n_kv_heads) == (64, 4, 8, 4)


def test_explicit_flag_overrides_the_checkpoint(tmp_path):
    path = tmp_path / "ckpt.pth"
    config_in = _tiny(dim=32, n_layers=2)
    save_checkpoint(str(path), Whetstone(config_in), None, None, 0, 0, 0, 0.0, config_in)

    config = resolve_model_config(_parse(["--dropout", "0.1"]), VOCAB, checkpoint_path=str(path))

    assert config.dropout == 0.1
    assert config.dim == 32  # untouched flags still come from the checkpoint


def test_missing_checkpoint_path_is_ignored():
    config = resolve_model_config(_parse([]), VOCAB, checkpoint_path="does/not/exist.pth")
    assert config.dim == LLMConfig().dim


def test_vocab_mismatch_is_an_error_not_a_silent_half_load(tmp_path):
    """Retraining the tokenizer invalidates old checkpoints; say so instead of crashing later."""
    path = tmp_path / "old.pth"
    torch.save(Whetstone(_tiny()).state_dict(), path)

    with pytest.raises(ValueError, match="vocab_size"):
        resolve_model_config(_parse([]), VOCAB + 1, checkpoint_path=str(path))


def test_loading_a_shallower_checkpoint_warns_instead_of_silently_random_init(tmp_path):
    """strict=False tolerates absent keys, so a depth mismatch would otherwise leave
    whole layers randomly initialized with no indication anything went wrong."""
    from train_utils import load_weights

    path = tmp_path / "shallow.pth"
    torch.save(Whetstone(_tiny(n_layers=1)).state_dict(), path)
    deep = Whetstone(_tiny(n_layers=3))

    with pytest.warns(RuntimeWarning, match="does not match"):
        load_weights(str(path), deep, "cpu", strict=False)


def test_resolved_config_round_trips_through_a_checkpoint(tmp_path):
    """The whole point: train at one size, reload at that size without restating flags."""
    from train_utils import save_final_weights

    path = tmp_path / "final.pth"
    original = _tiny(dim=64, n_layers=3, n_heads=8, n_kv_heads=2)
    save_final_weights(str(path), Whetstone(original), original)

    config = resolve_model_config(_parse([]), VOCAB, checkpoint_path=str(path))
    reloaded = Whetstone(config)
    missing, unexpected = reloaded.load_state_dict(torch.load(path, weights_only=False), strict=True)

    assert not missing and not unexpected
    assert (config.n_heads, config.n_kv_heads) == (8, 2)


def test_describe_model_reports_the_tied_parameter_count():
    """state_dict holds tok_embeddings.weight and output.weight as separate keys even
    though they are one tensor, so summing over it overcounts by vocab*dim."""
    config = _tiny()
    model = Whetstone(config)

    text = describe_model(model, config, "student")

    real = sum(p.numel() for p in model.parameters())
    from_state_dict = sum(v.numel() for v in model.state_dict().values())
    assert from_state_dict == real + config.vocab_size * config.dim
    assert text.startswith("student: dim=32")
    assert f"{real / 1e6:.3f}M" in text
