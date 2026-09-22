"""Train a byte-level BPE tokenizer on prepared corpus files.

The vocabulary is a load-bearing choice, not a detail. Measured on the legacy
6400-token Chinese vocabulary in ``tokenizer/zh_6400``, code compresses at 2.23
chars/token against 4.00 for English prose even though both are ASCII — so a
bilingual corpus with code needs its own vocabulary. But the embedding table is
``vocab_size x dim``, which at 32k tokens is 39.5% of a 29M model and 25.3% of
a 100M one, so vocabulary size and model size have to be chosen together. See
``docs/corpus-plan.md`` and compare candidates with ``datatools.tokenizer_stats``.

::

    python3 train_tokenizer.py --data datasets/prepared/train.jsonl \\
        --out tokenizer/v1_32k --vocab_size 32768
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterator, Sequence

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<unk>", "<s>", "</s>"]
CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set system_message = messages[0]['content'] %}"
    "{{ '<s>system\\n' + system_message + '</s>\\n' }}"
    "{% else %}{{ '<s>system\\n你是 Whetstone，是一个有用的人工智能助手。</s>\\n' }}{% endif %}"
    "{% for message in messages %}{% set content = message['content'] %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}"
    "{% endif %}{% endfor %}"
)


def iter_texts(paths: Sequence[str], max_records: int | None = None) -> Iterator[str]:
    """Stream text out of prepared JSONL, handling every schema in the repo.

    Reuses ``datatools.records`` so the tokenizer sees exactly the text the
    training stages will see, including flattened conversations.
    """
    from datatools.records import read_jsonl, record_text

    emitted = 0
    for path in paths:
        for record in read_jsonl(path):
            text = record_text(record.data)
            if not text.strip():
                continue
            yield text
            emitted += 1
            if max_records is not None and emitted >= max_records:
                return


def build_config(vocab_size: int) -> dict:
    return {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            str(index): {
                "content": token,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            }
            for index, token in enumerate(SPECIAL_TOKENS)
        },
        "additional_special_tokens": [],
        "bos_token": "<s>",
        "eos_token": "</s>",
        "clean_up_tokenization_spaces": False,
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<unk>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "chat_template": CHAT_TEMPLATE,
    }


def train_tokenizer(
    paths: Sequence[str],
    out_dir: str,
    vocab_size: int = 32768,
    max_records: int | None = None,
) -> str:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(iter_texts(paths, max_records), trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()

    # The dataset code and chat template assume these exact ids.
    for index, token in enumerate(SPECIAL_TOKENS):
        actual = tokenizer.token_to_id(token)
        if actual != index:
            raise RuntimeError(f"expected {token!r} at id {index}, got {actual}")

    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))
    tokenizer.model.save(out_dir)
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as fh:
        json.dump(build_config(vocab_size), fh, ensure_ascii=False, indent=2)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer")
    parser.add_argument("--data", nargs="+", required=True, help="Prepared JSONL corpus file(s)")
    parser.add_argument("--out", type=str, required=True, help="Output directory, e.g. tokenizer/v1_32k")
    parser.add_argument("--vocab_size", type=int, default=32768)
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Cap the sample; BPE converges well before the whole corpus is read",
    )
    args = parser.parse_args()

    out_dir = train_tokenizer(args.data, args.out, args.vocab_size, args.max_records)
    print(f"saved tokenizer (vocab {args.vocab_size}) -> {out_dir}")
    print(f"compare it with: python3 -m datatools.tokenizer_stats --probe --tokenizer {out_dir}")


if __name__ == "__main__":
    main()
