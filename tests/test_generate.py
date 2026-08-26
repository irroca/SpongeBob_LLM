import torch
import pytest

from Config import LLMConfig
from model import SpongeBob
from transformers.modeling_outputs import CausalLMOutputWithPast


def _tiny_model(**kwargs):
    defaults = dict(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64, vocab_size=10, dropout=0.0)
    defaults.update(kwargs)
    return SpongeBob(LLMConfig(**defaults))


def _patch_forward(model, eos_token_id, other_token_id, vocab_size, n_layers, row0_eos=True, row1_eos=False):
    """Force deterministic argmax per row: row0 -> eos (or other), row1 -> other (or eos)."""

    def fake_forward(input_ids, past_key_values=None, use_cache=False, attention_mask=None, **kwargs):
        bsz, seq_len = input_ids.shape[0], input_ids.shape[1]
        logits = torch.full((bsz, seq_len, vocab_size), -10.0)
        logits[0, :, eos_token_id if row0_eos else other_token_id] = 10.0
        if bsz > 1:
            logits[1, :, eos_token_id if row1_eos else other_token_id] = 10.0
        return CausalLMOutputWithPast(logits=logits, past_key_values=[None] * n_layers)

    model.forward = fake_forward


@torch.no_grad()
def test_generate_b1_no_crash_and_shape():
    """B=1: baseline sanity check, generate runs to completion without error.

    Non-stream `generate()` returns only the newly generated suffix (mirrors the
    existing yield contract used by `chat.py`'s streaming loop), so with an
    eos_token_id that never matches, exactly max_new_tokens tokens come back.
    """
    model = _tiny_model()
    model.eval()
    input_ids = torch.randint(0, 10, (1, 4))
    out = model.generate(input_ids, eos_token_id=-1, max_new_tokens=5, temperature=0, pad_token_id=0)
    assert out.shape[0] == 1
    assert out.shape[1] == 5


@torch.no_grad()
def test_generate_b2_no_crash():
    """B=2 previously raised on `.item()` for a multi-element EOS tensor; must no longer crash."""
    model = _tiny_model()
    model.eval()
    input_ids = torch.randint(0, 10, (2, 4))
    # eos_token_id=-1 never matches, so both rows run to max_new_tokens (stress the loop end condition too).
    out = model.generate(input_ids, eos_token_id=-1, max_new_tokens=5, temperature=0, pad_token_id=0)
    assert out.shape[0] == 2


@torch.no_grad()
def test_generate_finished_rows_get_pad_token():
    """Row 0 hits EOS immediately; row 1 never does. Row 0's trailing tokens must be pad_token_id,
    and generation must still proceed for row 1 up to max_new_tokens (not stopped by row 0 alone)."""
    n_layers = 2
    model = _tiny_model(n_layers=n_layers)
    model.eval()
    eos_token_id = 2
    other_token_id = 5
    pad_token_id = 0
    _patch_forward(model, eos_token_id, other_token_id, vocab_size=10, n_layers=n_layers,
                    row0_eos=True, row1_eos=False)

    input_ids = torch.randint(0, 10, (2, 3))
    max_new_tokens = 4
    generated = model.generate(
        input_ids,
        eos_token_id=eos_token_id,
        max_new_tokens=max_new_tokens,
        temperature=0,
        pad_token_id=pad_token_id,
    )

    assert generated.shape[1] == max_new_tokens  # row 1 never finishes -> loop runs full max_new_tokens

    # Row 0: first generated token is the real EOS token, all subsequent tokens are pad.
    assert generated[0, 0].item() == eos_token_id
    assert torch.all(generated[0, 1:] == pad_token_id)

    # Row 1: never hits eos, all tokens are the deterministic "other" token, never pad.
    assert torch.all(generated[1, :] == other_token_id)


@torch.no_grad()
def test_generate_stops_early_when_all_finished():
    """When every row hits EOS on the same step, generation must stop before max_new_tokens."""
    n_layers = 2
    model = _tiny_model(n_layers=n_layers)
    model.eval()
    eos_token_id = 2
    pad_token_id = 0
    _patch_forward(model, eos_token_id, other_token_id=5, vocab_size=10, n_layers=n_layers,
                    row0_eos=True, row1_eos=True)

    input_ids = torch.randint(0, 10, (2, 3))
    generated = model.generate(
        input_ids,
        eos_token_id=eos_token_id,
        max_new_tokens=10,
        temperature=0,
        pad_token_id=pad_token_id,
    )

    assert generated.shape[1] == 1  # stopped after the first step since finished.all() was True
    assert torch.all(generated == eos_token_id)
