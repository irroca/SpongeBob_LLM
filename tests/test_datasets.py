from transformers import AutoTokenizer

from dataset import PreferenceDataset, SFTDataset, assistant_loss_mask


def test_assistant_loss_mask_no_pad_leak_after_eos():
    """After a normal reply, tokens past the </s>\\n span must stay 0, even
    though the current implementation over-marks by one extra index."""
    bos_id = [9, 9]
    eos_id = [8, 8]
    pad_token_id = 0
    # bos(0,1) content(2,3,4) eos(5,6) pad(7,8,9)
    input_ids = [9, 9, 3, 4, 5, 8, 8, 0, 0, 0]
    max_length = len(input_ids)

    mask = assistant_loss_mask(input_ids, bos_id, eos_id, max_length, pad_token_id)

    assert mask[2] == 1, "first assistant content token must be in the loss"
    assert mask[5] == 1 and mask[6] == 1, "eos span must be marked inclusive"
    assert mask[7:] == [0, 0, 0], "pad tokens after eos must never be marked"


def test_assistant_loss_mask_truncated_without_eos_marks_last_non_pad_only():
    """When no eos is found (truncated generation), only content up to the
    last non-pad token should be marked; trailing pads must stay 0."""
    bos_id = [9, 9]
    eos_id = [8, 8]
    pad_token_id = 0
    # bos(0,1) content(2..6) pad(7,8,9) -- no eos anywhere
    input_ids = [9, 9, 101, 102, 103, 104, 105, 0, 0, 0]
    max_length = len(input_ids)

    mask = assistant_loss_mask(input_ids, bos_id, eos_id, max_length, pad_token_id)

    assert mask[2] == 1, "first assistant content token must be in the loss"
    assert mask[6] == 1, "last non-pad content token must be marked"
    assert mask[7:] == [0, 0, 0], "trailing pad tokens must never be marked"


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
