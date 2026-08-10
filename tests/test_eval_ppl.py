from transformers import AutoTokenizer

from eval_ppl import wrap_pretrain_text


def test_wrap_pretrain_text_adds_bos_and_eos_strings():
    tokenizer = AutoTokenizer.from_pretrained("./spongebob_tokenizer")
    text = "hello world"

    wrapped = wrap_pretrain_text(text, tokenizer)

    assert wrapped == f"{tokenizer.bos_token}{text}{tokenizer.eos_token}"
    assert wrapped.startswith(tokenizer.bos_token)
    assert wrapped.endswith(tokenizer.eos_token)


def test_wrap_pretrain_text_token_ids_differ_from_raw():
    tokenizer = AutoTokenizer.from_pretrained("./spongebob_tokenizer")
    text = "hello world"

    raw_ids = tokenizer(text).input_ids
    wrapped_ids = tokenizer(wrap_pretrain_text(text, tokenizer)).input_ids

    assert wrapped_ids != raw_ids
    bos_id = tokenizer(tokenizer.bos_token, add_special_tokens=False).input_ids
    eos_id = tokenizer(tokenizer.eos_token, add_special_tokens=False).input_ids
    assert wrapped_ids[: len(bos_id)] == bos_id
    assert wrapped_ids[-len(eos_id):] == eos_id
