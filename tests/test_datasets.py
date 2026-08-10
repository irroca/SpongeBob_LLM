from transformers import AutoTokenizer

from dataset import PreferenceDataset, SFTDataset


def test_sft_loss_mask_includes_first_assistant_token():
    tokenizer = AutoTokenizer.from_pretrained("./spongebob_tokenizer")
    ds = SFTDataset("tests/fixtures/sft_tiny.jsonl", tokenizer, max_length=256)
    X, Y, loss_mask = ds[0]
    assert loss_mask.sum() > 0
    sample = ds.samples[0]
    prompt = ds._create_chat_prompt(sample["conversations"])
    input_ids = tokenizer(prompt).input_ids[:256]
    bos = ds.bos_id
    start = None
    for i in range(len(input_ids) - len(bos) + 1):
        if input_ids[i : i + len(bos)] == bos:
            start = i + len(bos)
            break
    assert start is not None
    full_mask = ds._generate_loss_mask(input_ids + [tokenizer.pad_token_id] * (256 - len(input_ids)))
    assert full_mask[start] == 1, "first assistant content token must be in the loss"


def _assert_prompt_masked_and_response_starts(ds, prompt, answer, mask):
    """Shared assertion for PreferenceDataset: prompt tokens must be 0 and the
    first assistant content token must be 1, mirroring the SFT mask test."""
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]
    text = ds.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    input_ids = ds.tokenizer(text).input_ids[: ds.max_length]
    bos = ds.bos_id
    start = None
    for i in range(len(input_ids) - len(bos) + 1):
        if input_ids[i : i + len(bos)] == bos:
            start = i + len(bos)
            break
    assert start is not None
    assert all(m == 0 for m in mask[:start]), "prompt tokens must not be in the loss"
    assert mask[start] == 1, "first assistant content token must be in the loss"


def test_preference_dataset_masks_responses():
    tokenizer = AutoTokenizer.from_pretrained("./spongebob_tokenizer")
    ds = PreferenceDataset("tests/fixtures/preference_tiny.jsonl", tokenizer, max_length=256)
    cX, cY, cM, rX, rY, rM = ds[0]
    assert cX.shape == cY.shape == cM.shape
    assert rX.shape == rY.shape == rM.shape

    sample = ds.samples[0]
    # cM/rM are shifted by one (aligned to Y) relative to the raw input_ids, so
    # prepend the dropped position 0 to compare against the un-shifted mask below.
    full_cmask = [0] + cM.tolist()
    full_rmask = [0] + rM.tolist()
    _assert_prompt_masked_and_response_starts(ds, sample["prompt"], sample["chosen"], full_cmask)
    _assert_prompt_masked_and_response_starts(ds, sample["prompt"], sample["rejected"], full_rmask)
