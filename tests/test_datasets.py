import torch
from transformers import AutoTokenizer

from dataset import SFTDataset


def test_sft_loss_mask_includes_first_assistant_token():
    tokenizer = AutoTokenizer.from_pretrained("./spongebob_tokenizer")
    # Write is already in fixtures; construct dataset from fixture path
    ds = SFTDataset("tests/fixtures/sft_tiny.jsonl", tokenizer, max_length=256)
    X, Y, loss_mask = ds[0]
    assert loss_mask.sum() > 0
    # Rebuild prompt and locate assistant span start
    sample = ds.samples[0]
    prompt = ds._create_chat_prompt(sample["conversations"])
    input_ids = tokenizer(prompt).input_ids[:256]
    bos = ds.bos_id
    # find first bos occurrence
    start = None
    for i in range(len(input_ids) - len(bos) + 1):
        if input_ids[i : i + len(bos)] == bos:
            start = i + len(bos)
            break
    assert start is not None
    # loss_mask aligns to Y = input_ids[1:], so index start-1 in loss_mask corresponds to token at start
    # The first assistant content token is at index `start` in input_ids; it should be supervised
    # as a prediction target when previous token is at start-1 -> mask at position start-1? 
    # Looking at __getitem__: loss_mask = loss_mask[1:], and Y = input_ids[1:].
    # So for predicting input_ids[j], mask index is j-1 in returned loss_mask, value from original[j].
    # First assistant content token index = start; original loss_mask[start] must be 1.
    full_mask = ds._generate_loss_mask(input_ids + [tokenizer.pad_token_id] * (256 - len(input_ids)))
    assert full_mask[start] == 1, "first assistant content token must be in the loss"
