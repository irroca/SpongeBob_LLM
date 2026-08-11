import torch
import pytest

from Config import LLMConfig
from model import SpongeBob


def _tiny_model(**kwargs):
    defaults = dict(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64, vocab_size=128, dropout=0.0)
    defaults.update(kwargs)
    return SpongeBob(LLMConfig(**defaults))


@torch.no_grad()
def test_prefill_forward_shape():
    model = _tiny_model()
    model.eval()
    x = torch.randint(0, 128, (2, 8))
    out = model(x)
    assert out.logits.shape == (2, 8, 128)


@torch.no_grad()
def test_decode_with_cache_single_token():
    model = _tiny_model()
    model.eval()
    x = torch.randint(0, 128, (1, 6))
    out1 = model(x, use_cache=True)
    assert out1.past_key_values is not None
    out2 = model(x[:, -1:], past_key_values=out1.past_key_values, use_cache=True, start_pos=5)
    assert out2.logits.shape == (1, 1, 128)


@torch.no_grad()
def test_multitoken_continuation_with_cache():
    """Classic bug: mask used q_len x q_len instead of q_len x kv_len."""
    model = _tiny_model()
    model.eval()
    prefix = torch.randint(0, 128, (1, 5))
    out1 = model(prefix, use_cache=True)
    cont = torch.randint(0, 128, (1, 4))
    out2 = model(cont, past_key_values=out1.past_key_values, use_cache=True, start_pos=5)
    assert out2.logits.shape == (1, 4, 128)


@torch.no_grad()
def test_gqa_shapes():
    model = _tiny_model(n_heads=8, n_kv_heads=2)
    model.eval()
    x = torch.randint(0, 128, (1, 7))
    out = model(x, use_cache=True)
    assert out.logits.shape == (1, 7, 128)
    k, v = out.past_key_values[0]
    assert k.shape[2] == 2  # n_kv_heads


@torch.no_grad()
def test_attention_mask_changes_logits():
    model = _tiny_model()
    model.eval()
    x = torch.randint(0, 128, (1, 6))
    # Mask an earlier key position so later queries change
    attn = torch.ones(1, 6)
    attn[0, 1] = 0
    out_masked = model(x, attention_mask=attn)
    out_full = model(x, attention_mask=None)
    assert out_masked.logits.shape == out_full.logits.shape
    assert not torch.allclose(out_masked.logits[0, 3], out_full.logits[0, 3])


@torch.no_grad()
def test_kv_cache_chunk_mask_pads_on_left():
    """attention_mask covering only the new chunk (len==q_len) during cached decoding must be
    left-padded with ones for the cached past positions, i.e. equivalent to manually passing a
    full kv_len mask with ones prepended. Classic bug: padding the short mask on the right
    instead misapplies it to the start of the cached prefix rather than the new chunk."""
    model = _tiny_model()
    model.eval()
    prefix = torch.randint(0, 128, (1, 5))
    out1 = model(prefix, use_cache=True)
    cont = torch.randint(0, 128, (1, 3))
    chunk_mask = torch.tensor([[1.0, 0.0, 1.0]])  # mask out the middle new-chunk token as a key

    out_chunk = model(cont, past_key_values=out1.past_key_values, use_cache=True, start_pos=5,
                       attention_mask=chunk_mask)

    full_mask = torch.cat([torch.ones(1, 5), chunk_mask], dim=-1)
    out_full_equiv = model(cont, past_key_values=out1.past_key_values, use_cache=True, start_pos=5,
                            attention_mask=full_mask)

    assert torch.allclose(out_chunk.logits, out_full_equiv.logits)


@torch.no_grad()
def test_kv_cache_full_length_mask_used_as_is():
    model = _tiny_model()
    model.eval()
    prefix = torch.randint(0, 128, (1, 5))
    out1 = model(prefix, use_cache=True)
    cont = torch.randint(0, 128, (1, 3))
    kv_len = 8
    full_mask = torch.ones(1, kv_len)
    full_mask[0, 1] = 0
    out = model(cont, past_key_values=out1.past_key_values, use_cache=True, start_pos=5,
                attention_mask=full_mask)
    assert out.logits.shape == (1, 3, 128)


@torch.no_grad()
def test_kv_cache_mask_wrong_length_raises():
    model = _tiny_model()
    model.eval()
    prefix = torch.randint(0, 128, (1, 5))
    out1 = model(prefix, use_cache=True)
    cont = torch.randint(0, 128, (1, 3))
    bad_mask = torch.ones(1, 4)  # neither q_len (3) nor kv_len (8)
    with pytest.raises(ValueError):
        model(cont, past_key_values=out1.past_key_values, use_cache=True, start_pos=5,
              attention_mask=bad_mask)


@torch.no_grad()
def test_rope_overflow_raises():
    model = _tiny_model(max_seq_len=16)
    model.eval()
    x = torch.randint(0, 128, (1, 4))
    with pytest.raises(ValueError):
        model(x, start_pos=14)  # 14+4 > 16
