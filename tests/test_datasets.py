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


def test_preference_dataset_masks_responses():
    tokenizer = AutoTokenizer.from_pretrained("./spongebob_tokenizer")
    ds = PreferenceDataset("tests/fixtures/preference_tiny.jsonl", tokenizer, max_length=256)
    cX, cY, cM, rX, rY, rM = ds[0]
    assert cM.sum() > 0 and rM.sum() > 0
    assert cX.shape == cY.shape == cM.shape
    assert rX.shape == rY.shape == rM.shape
